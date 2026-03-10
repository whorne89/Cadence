"""
Cadence - Meeting Transcription
Main entry point that orchestrates all components.
"""

import sys
import ctypes
import logging
import time
import re

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, QThread, Qt
import numpy as np


def set_windows_app_id():
    """Set Windows AppUserModelID for proper taskbar/tray display."""
    try:
        app_id = "Cadence.MeetingTranscription.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Not on Windows or API not available


from core.audio_recorder import AudioRecorder
from core.transcriber import Transcriber
from core.session_manager import SessionManager
from gui.system_tray import SystemTrayIcon
from gui.main_window import MainWindow
from gui.settings_dialog import SettingsDialog
from gui.theme import apply_theme
from utils.config import ConfigManager
from utils.logger import setup_logger


logger = logging.getLogger("Cadence")


class TranscriptionWorker(QObject):
    """
    Worker that transcribes audio using energy-based silence detection.

    Polls for new mic audio every 200ms and uses RMS energy to detect
    speech pauses. Transcribes accumulated speech when silence is detected
    (or when max_speech_s safety valve triggers).
    """

    segment_ready = Signal(str, float)  # text, timestamp_seconds
    finished = Signal()

    def __init__(self, transcriber, audio_recorder,
                 silence_threshold=0.005, min_silence_ms=400,
                 min_speech_s=0.3, max_speech_s=30.0,
                 language=None):
        super().__init__()
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_s = min_speech_s
        self.max_speech_s = max_speech_s
        self.language = language
        self._running = False
        self._poll_interval = 0.2  # 200ms

    def run(self):
        """Main loop — polls audio and transcribes on silence boundaries."""
        self._running = True
        sr = self.audio_recorder.sample_rate

        # Track offset into frame list
        mic_offset = 0

        from core.silence_detector import SilenceDetector
        detector = SilenceDetector(self.silence_threshold, self.min_silence_ms, sr)

        # Speech buffer start index
        speech_start = 0

        mic_frames = self.audio_recorder._mic_frames

        logger.info(
            f"Transcription worker started ("
            f"threshold={self.silence_threshold}, silence={self.min_silence_ms}ms, "
            f"min_speech={self.min_speech_s}s, max_speech={self.max_speech_s}s, "
            f"language={self.language})"
        )

        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break

            mic_frames = self.audio_recorder._mic_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for frame in new_frames:
                    detector.feed(frame)
                mic_offset = mic_len

                speech_frames = mic_frames[speech_start:mic_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    detector.is_silent() and detector._has_had_speech
                    and speech_duration > self.min_speech_s
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    start_sample = sum(len(f) for f in mic_frames[:speech_start])
                    timestamp = start_sample / sr

                    text = self._transcribe_audio(speech_frames)
                    if text:
                        self.segment_ready.emit(text, timestamp)

                    speech_start = mic_offset
                    detector.reset()
                elif detector.is_silent() and not detector._has_had_speech:
                    speech_start = mic_offset
                    detector.reset()

        # Flush remaining audio when stopping
        mic_remaining = mic_frames[speech_start:mic_offset] if mic_offset > speech_start else []
        if mic_remaining:
            start_sample = sum(len(f) for f in mic_frames[:speech_start])
            timestamp = start_sample / sr
            text = self._transcribe_audio(mic_remaining)
            if text:
                self.segment_ready.emit(text, timestamp)

        logger.info("Transcription worker stopped")
        self.finished.emit()

    def _transcribe_audio(self, frames):
        """Concatenate frames and transcribe to text. Returns None on failure."""
        try:
            audio = np.concatenate(frames)
            if len(audio) > 0:
                text = self.transcriber.transcribe_text(audio, language=self.language)
                if text and text.strip():
                    return text.strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return None

    def stop(self):
        self._running = False


def _filter_hallucinations(segments):
    """
    Remove segments that are likely Whisper hallucinations.

    Detects:
    - Non-English text (high ratio of non-ASCII or accented characters)
    - ASCII foreign language text (low English vocabulary coverage)
    - Filler-only segments (just "um", "the", "so", etc.)
    """
    FILLER_WORDS = {"um", "uh", "the", "a", "so", "and", "but", "or", "like"}

    COMMON_ENGLISH = {
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
        "the", "a", "an", "this", "that", "these", "those",
        "is", "am", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must",
        "not", "no", "yes", "yeah", "ok", "okay",
        "and", "or", "but", "if", "so", "because", "when", "while", "then",
        "than", "that", "which", "who", "what", "where", "how", "why",
        "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
        "up", "out", "off", "over", "into", "through", "between", "after",
        "before", "during", "around", "down",
        "all", "each", "every", "both", "some", "any", "many", "much", "more",
        "most", "other", "another", "such", "own",
        "just", "also", "very", "really", "actually", "right", "well", "now",
        "here", "there", "still", "already", "even", "only", "too",
        "go", "going", "get", "got", "make", "take", "come", "see", "know",
        "think", "say", "said", "tell", "give", "want", "need", "use", "try",
        "look", "like", "good", "new", "first", "last", "long", "great",
        "little", "big", "old", "next", "back", "way", "time", "thing",
        "people", "work", "day", "part", "let", "put",
    }

    filtered = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        # Check for text with no alphabetic content (e.g., "...", "!?!?")
        letters = re.findall(r'[a-zA-Z\u00C0-\u024F]', text)
        if not letters:
            logger.info(f"Hallucination filtered (no letters): '{text}'")
            continue

        # Check for non-English: if >30% of letters are non-ASCII, likely hallucination
        ascii_letters = sum(1 for c in letters if ord(c) < 128)
        if ascii_letters / len(letters) < 0.7:
            logger.info(f"Hallucination filtered: '{text}'")
            continue

        # Check for filler-only: strip punctuation, check if all words are fillers
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        if clean and len(clean) <= 2 and all(w in FILLER_WORDS for w in clean):
            logger.info(f"Filler filtered: '{text}'")
            continue

        # Check for ASCII foreign language
        if len(clean) >= 4:
            english_count = sum(1 for w in clean if w in COMMON_ENGLISH)
            if english_count == 0:
                logger.info(f"Hallucination filtered (no English words in "
                            f"{len(clean)} words): '{text}'")
                continue

        filtered.append(seg)
    return filtered


def merge_segments(segments, max_gap_s=2.0):
    """
    Merge consecutive segments that are close in time.

    Improves readability by combining fragments that were split by
    the batch polling window into natural utterances.
    """
    if not segments:
        return segments

    merged = [dict(segments[0])]
    last_merge_start = segments[0]["start"]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - last_merge_start <= max_gap_s:
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            last_merge_start = seg["start"]
        else:
            merged.append(dict(seg))
            last_merge_start = seg["start"]

    return merged


class PostProcessWorker(QObject):
    """
    Post-processes audio after recording stops.

    Re-transcribes the full mic recording for better quality,
    filters hallucinations, and merges consecutive segments.
    """

    segments_ready = Signal(list)  # list of {text, start} dicts
    progress = Signal(str)  # status message
    finished = Signal()

    def __init__(self, transcriber, mic_audio, language=None):
        super().__init__()
        self.transcriber = transcriber
        self.mic_audio = mic_audio
        self.language = language

    def run(self):
        """Re-transcribe full recording for cleaner output."""
        segments = []

        try:
            if len(self.mic_audio) > 0:
                self.progress.emit("Processing audio...")

                def on_progress(p):
                    pct = int(p * 100)
                    self.progress.emit(f"Processing audio... {pct}%")

                raw_segments = self.transcriber.transcribe(
                    self.mic_audio, language=self.language,
                    progress_callback=on_progress,
                )
                for seg in raw_segments:
                    segments.append({
                        "text": seg["text"],
                        "start": seg["start"],
                    })

                # Filter hallucinations
                segments = _filter_hallucinations(segments)

                # Merge consecutive segments for readability
                segments = merge_segments(segments)

        except Exception as e:
            logger.error(f"Post-processing error: {e}")

        self.segments_ready.emit(segments)
        self.finished.emit()


class CadenceApp(QObject):
    """Main application controller that wires all components together."""

    def __init__(self):
        super().__init__()

        # Initialize logger
        self.logger = setup_logger()
        self.logger.info("Starting Cadence...")

        # Configuration
        self.config = ConfigManager()

        # Session management
        self.session_manager = SessionManager()

        # Audio recorder — single mic stream
        self.audio_recorder = AudioRecorder()

        # Apply saved mic device setting
        mic_device = self.config.get_mic_device()
        if mic_device is not None:
            self.audio_recorder.set_mic_device(mic_device)

        # Streaming transcriber (lightweight model for real-time)
        streaming_model_size = self.config.get_streaming_model_size()
        self.streaming_transcriber = Transcriber(model_size=streaming_model_size)

        # GUI components (set after creation in main())
        self.tray_icon = None
        self.main_window = None

        # Transcription worker thread state
        self._transcription_thread = None
        self._transcription_worker = None

        # Current recording segments
        self._current_segments = []
        self._selected_folder = None

        self._postprocess_thread = None
        self._postprocess_worker = None
        self._recording_duration = 0.0
        self._pending_mic_audio = None

        self.logger.info("Application initialized")

    def _on_segment(self, text, timestamp):
        """
        Handle a transcript segment from the transcription worker.
        Called on the main thread via Qt signal (thread-safe).
        """
        segment = {"text": text, "start": timestamp}
        self._current_segments.append(segment)
        if self.main_window is not None:
            self.main_window.append_segment(text, timestamp)

    def start_recording(self):
        """Start a new recording session."""
        self.logger.info("Starting recording...")

        # Reset segments for new recording
        self._current_segments = []

        # Start audio capture
        self.audio_recorder.start_recording()

        # Start transcription worker in a QThread
        self._start_transcription_worker()

        # Update UI
        if self.main_window is not None:
            self.main_window.set_recording_state()
        if self.tray_icon is not None:
            self.tray_icon.set_recording_state()
            from datetime import datetime
            start_time = datetime.now().strftime("%I:%M %p").lstrip("0")
            self.tray_icon.show_notification(
                "Recording", "Recording started",
                details=f"Started at {start_time}",
            )

    def _start_transcription_worker(self):
        """Launch the transcription worker in a background QThread."""
        self._stop_transcription_worker()

        self._transcription_thread = QThread()
        language = self.config.get("whisper", "language", default="en")
        self._transcription_worker = TranscriptionWorker(
            self.streaming_transcriber,
            self.audio_recorder,
            silence_threshold=0.005,
            min_silence_ms=400,
            min_speech_s=0.3,
            max_speech_s=30.0,
            language=language,
        )
        self._transcription_worker.moveToThread(self._transcription_thread)

        # Connect signals
        self._transcription_thread.started.connect(self._transcription_worker.run)
        self._transcription_worker.segment_ready.connect(self._on_segment)
        self._transcription_worker.finished.connect(
            self._on_transcription_finished, Qt.ConnectionType.QueuedConnection
        )

        self._transcription_thread.start()

    def _stop_transcription_worker(self):
        """Signal the transcription worker to stop."""
        if self._transcription_worker is not None:
            self._transcription_worker.stop()
        if self._transcription_thread is not None and self._transcription_thread.isRunning():
            self._transcription_thread.quit()
            self._transcription_thread.wait(3000)

    def _cleanup_transcription_thread(self):
        """Clean up transcription thread resources."""
        if self._transcription_thread is not None:
            self._transcription_thread.quit()
            self._transcription_thread.wait(2000)
            self._transcription_thread.deleteLater()
            self._transcription_thread = None
        if self._transcription_worker is not None:
            self._transcription_worker.deleteLater()
            self._transcription_worker = None

    def _on_transcription_finished(self):
        """Called when the transcription worker finishes."""
        self._cleanup_transcription_thread()
        if self._pending_mic_audio is not None:
            self._begin_postprocess()

    def _begin_postprocess(self):
        """Start post-processing after the transcription worker has fully stopped."""
        mic_audio = self._pending_mic_audio
        self._pending_mic_audio = None

        if self.main_window is not None:
            self.main_window.set_processing_state("Cleaning up transcript...")

        self._start_postprocess(mic_audio)

    def stop_recording(self):
        """Stop recording and launch post-processing after worker finishes."""
        self.logger.info("Stopping recording...")

        # Signal transcription worker to stop (non-blocking — let it flush)
        if self._transcription_worker is not None:
            self._transcription_worker.stop()

        # Stop audio capture — returns full audio array
        mic_audio = self.audio_recorder.stop_recording()
        self._recording_duration = self.audio_recorder.get_duration()

        # Store mic audio; post-processing starts when worker emits finished
        self._pending_mic_audio = mic_audio

        # Update UI
        if self.main_window is not None:
            self.main_window.set_processing_state("Finishing transcription...")

        # If no worker running, start post-processing immediately
        if self._transcription_thread is None or not self._transcription_thread.isRunning():
            self._begin_postprocess()

    def _start_postprocess(self, mic_audio):
        """Launch PostProcessWorker in a background QThread."""
        self._postprocess_thread = QThread()
        language = self.config.get("whisper", "language", default="en")
        self._postprocess_worker = PostProcessWorker(
            self.streaming_transcriber, mic_audio,
            language=language,
        )
        self._postprocess_worker.moveToThread(self._postprocess_thread)

        self._postprocess_thread.started.connect(self._postprocess_worker.run)
        self._postprocess_worker.progress.connect(self._on_postprocess_progress)
        self._postprocess_worker.segments_ready.connect(self._on_postprocess_done)
        self._postprocess_worker.finished.connect(self._cleanup_postprocess_thread)

        self._postprocess_thread.start()

    def _on_postprocess_progress(self, message):
        """Update UI with post-processing progress."""
        if self.main_window is not None:
            self.main_window.status_label.setText(message)

    def _on_postprocess_done(self, segments):
        """Replace live transcript with post-processed segments and save."""
        self.logger.info(f"Post-processing complete: {len(segments)} segments")

        # If post-processing failed (empty result but we had live segments), keep live
        if not segments and self._current_segments:
            self.logger.warning("Post-processing returned empty; keeping live transcript")
            segments = self._current_segments

        # Replace live segments with cleaned version
        self._current_segments = segments

        # Update transcript display
        if self.main_window is not None:
            self.main_window.set_transcript(segments)
            self.main_window.set_done_state()

        # Save transcript
        model = self.config.get_streaming_model_size()
        self.session_manager.save_transcript(
            segments, duration=self._recording_duration, model=model,
            folder=self._selected_folder,
        )

        # Refresh folder/transcript lists
        self._refresh_folders()
        if self._selected_folder:
            self.on_folder_selected(self._selected_folder)

        if self.tray_icon is not None:
            self.tray_icon.set_idle_state()
            dur = int(self._recording_duration)
            dur_m, dur_s = dur // 60, dur % 60
            dur_str = f"{dur_m}m {dur_s}s" if dur_m > 0 else f"{dur_s}s"
            word_count = sum(len(s["text"].split()) for s in segments)
            seg_count = len(segments)
            from datetime import datetime
            end_time = datetime.now().strftime("%I:%M %p").lstrip("0")
            self.tray_icon.show_notification(
                "Transcription Complete",
                "Recording saved",
                details=(
                    f"Duration: {dur_str}\n"
                    f"Segments: {seg_count} | Words: {word_count:,}\n"
                    f"Finished at {end_time}"
                ),
            )

    def _cleanup_postprocess_thread(self):
        """Clean up post-processing thread resources."""
        if self._postprocess_thread is not None:
            self._postprocess_thread.quit()
            self._postprocess_thread.wait(2000)
            self._postprocess_thread.deleteLater()
            self._postprocess_thread = None
        if self._postprocess_worker is not None:
            self._postprocess_worker.deleteLater()
            self._postprocess_worker = None

    def show_settings(self):
        """Open the settings dialog."""
        self.logger.info("Opening settings dialog")
        try:
            dialog = SettingsDialog(
                self.config, self.audio_recorder,
                session_manager=self.session_manager, parent=self.main_window,
            )
            dialog.settings_changed.connect(self._on_settings_changed)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"Failed to show settings dialog: {e}")
            if self.tray_icon is not None:
                self.tray_icon.show_error(f"Failed to open settings: {e}")

    def _on_settings_changed(self):
        """Apply updated settings from the dialog."""
        self.logger.info("Settings changed - applying updates")

        # Update mic device
        mic_device = self.config.get_mic_device()
        self.audio_recorder.set_mic_device(mic_device)

        # Update streaming model
        streaming_model = self.config.get_streaming_model_size()
        if streaming_model != self.streaming_transcriber.model_size:
            self.streaming_transcriber.change_model(streaming_model)

    def on_folder_selected(self, folder_name):
        """Load and display transcripts for the selected folder."""
        self._selected_folder = folder_name
        sort_desc = self.main_window._sort_descending if self.main_window else True
        transcripts = self.session_manager.list_transcripts(folder_name, sort_descending=sort_desc)
        if self.main_window:
            self.main_window.set_transcripts(transcripts)

    def on_folder_created(self, name):
        """Create a new folder and refresh the folder list."""
        self.session_manager.create_folder(name)
        self._refresh_folders()

    def on_folder_renamed(self, old_name, new_name):
        """Rename a folder and refresh the folder list."""
        self.session_manager.rename_folder(old_name, new_name)
        self._refresh_folders()

    def on_folder_deleted(self, name):
        """Delete a folder and refresh the folder list."""
        self.session_manager.delete_folder(name)
        self._refresh_folders()

    def on_transcript_selected(self, folder, name):
        """Load and display the selected transcript."""
        transcripts = self.session_manager.list_transcripts(folder)
        for t in transcripts:
            if t["name"] == name:
                data = self.session_manager.load_transcript(t["path"])
                if self.main_window:
                    self.main_window.set_transcript(data["segments"])
                break

    def on_transcript_renamed(self, folder, old_name, new_name):
        """Rename a transcript and refresh the transcript list."""
        self.session_manager.rename_transcript(folder, old_name, new_name)
        self.on_folder_selected(folder)

    def on_transcript_deleted(self, folder, name):
        """Delete a transcript and refresh the transcript list."""
        self.session_manager.delete_transcript(folder, name)
        self.on_folder_selected(folder)

    def on_transcript_moved(self, src_folder, name, dest_folder):
        """Move a transcript to another folder and refresh the transcript list."""
        self.session_manager.move_transcript(src_folder, name, dest_folder)
        self.on_folder_selected(src_folder)

    def on_sort_order_changed(self, descending):
        """Re-sort the transcript list when sort order changes."""
        if self._selected_folder:
            self.on_folder_selected(self._selected_folder)

    def _refresh_folders(self):
        """Refresh the folder list in the main window."""
        folders = self.session_manager.list_folders()
        if self.main_window:
            self.main_window.set_folders(folders)

    def show_window(self):
        """Show and raise the main window."""
        if self.main_window is not None:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()

    def quit(self):
        """Clean shutdown of the application."""
        self.logger.info("Shutting down Cadence...")

        # Stop transcription worker
        self._stop_transcription_worker()

        # Stop post-processing if running
        if self._postprocess_thread is not None and self._postprocess_thread.isRunning():
            self._postprocess_thread.quit()
            self._postprocess_thread.wait(3000)

        # Stop recording if active
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()

        QApplication.quit()


def main():
    """Application entry point."""
    # Set Windows app ID before creating QApplication
    set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName("Cadence")
    app.setApplicationDisplayName("Cadence")
    app.setQuitOnLastWindowClosed(False)  # Keep running with system tray

    # Apply dark theme
    apply_theme(app)

    # Create application controller
    cadence_app = CadenceApp()

    # Create GUI components
    tray_icon = SystemTrayIcon()
    main_window = MainWindow()

    # Attach GUI to app controller
    cadence_app.tray_icon = tray_icon
    cadence_app.main_window = main_window

    # Wire tray icon signals
    tray_icon.start_requested.connect(cadence_app.start_recording)
    tray_icon.stop_requested.connect(cadence_app.stop_recording)
    tray_icon.window_requested.connect(cadence_app.show_window)
    tray_icon.settings_requested.connect(cadence_app.show_settings)
    tray_icon.quit_requested.connect(cadence_app.quit)

    # Wire main window signals
    main_window.start_requested.connect(cadence_app.start_recording)
    main_window.stop_requested.connect(cadence_app.stop_recording)
    main_window.folder_selected.connect(cadence_app.on_folder_selected)
    main_window.folder_created.connect(cadence_app.on_folder_created)
    main_window.folder_renamed.connect(cadence_app.on_folder_renamed)
    main_window.folder_deleted.connect(cadence_app.on_folder_deleted)
    main_window.transcript_selected.connect(cadence_app.on_transcript_selected)
    main_window.transcript_renamed.connect(cadence_app.on_transcript_renamed)
    main_window.transcript_deleted.connect(cadence_app.on_transcript_deleted)
    main_window.transcript_moved.connect(cadence_app.on_transcript_moved)
    main_window.sort_order_changed.connect(cadence_app.on_sort_order_changed)
    main_window.settings_requested.connect(cadence_app.show_settings)

    # Load folders on startup
    cadence_app._refresh_folders()

    # Show the main window
    main_window.show()

    # Startup toast
    model_id = cadence_app.config.get_streaming_model_size()
    model_names = {"tiny": "Fastest", "base": "Balanced", "small": "Accurate", "medium": "Precision"}
    model_label = model_names.get(model_id, model_id)
    tray_icon.show_notification(
        "Cadence Ready",
        "Ready to record",
        details=f"Model: {model_label}",
    )

    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
