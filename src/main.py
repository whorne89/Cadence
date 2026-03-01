"""
Cadence - Meeting Transcription
Main entry point that orchestrates all components.
"""

import sys
import ctypes
import logging
import time

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

    Instead of a fixed timer, polls for new audio every 200ms and uses
    RMS energy to detect speech pauses. Transcribes accumulated speech
    when silence is detected (or when max_speech_s safety valve triggers).
    """

    segment_ready = Signal(str, str, float)  # speaker, text, timestamp_seconds
    finished = Signal()

    def __init__(self, transcriber, audio_recorder,
                 silence_threshold=0.01, min_silence_ms=500,
                 max_speech_s=30.0):
        super().__init__()
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.max_speech_s = max_speech_s
        self._running = False
        self._poll_interval = 0.2  # 200ms

    def run(self):
        """Main loop — polls audio and transcribes on silence boundaries."""
        self._running = True
        sr = self.audio_recorder.sample_rate

        # Track offsets into frame lists
        mic_offset = 0
        sys_offset = 0

        # Per-source silence detectors
        from core.silence_detector import SilenceDetector
        from core.echo_gate import get_audio_for_sample_range
        mic_detector = SilenceDetector(self.silence_threshold, self.min_silence_ms, sr)
        sys_detector = SilenceDetector(self.silence_threshold, self.min_silence_ms, sr)

        # Per-source speech buffer start indices
        mic_speech_start = 0
        sys_speech_start = 0

        # Initialize frame references before loop (avoids NameError if loop never runs)
        mic_frames = self.audio_recorder._mic_frames
        sys_frames = self.audio_recorder._system_frames

        logger.info(
            f"Transcription worker started (silence_threshold={self.silence_threshold}, "
            f"min_silence={self.min_silence_ms}ms, max_speech={self.max_speech_s}s)"
        )

        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break

            # --- Process mic audio ---
            mic_frames = self.audio_recorder._mic_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for frame in new_frames:
                    mic_detector.feed(frame)
                mic_offset = mic_len

                speech_frames = mic_frames[mic_speech_start:mic_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    mic_detector.is_silent() and mic_detector._has_had_speech
                    and speech_duration > 0.5
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    mic_start_sample = sum(len(f) for f in mic_frames[:mic_speech_start])
                    timestamp = mic_start_sample / sr

                    # Echo gate: suppress mic if it's just speaker bleed
                    # Speaker bleed is much quieter than direct speech into mic.
                    # If system audio is active and mic isn't significantly louder,
                    # it's just bleed — suppress it.
                    echo_detected = False
                    mic_audio = np.concatenate(speech_frames)
                    mic_end_sample = mic_start_sample + len(mic_audio)
                    sys_audio = get_audio_for_sample_range(
                        self.audio_recorder._system_frames,
                        mic_start_sample, mic_end_sample,
                    )
                    if len(sys_audio) > 0:
                        sys_rms = float(np.sqrt(np.mean(sys_audio.astype(np.float64) ** 2)))
                        mic_rms = float(np.sqrt(np.mean(mic_audio.astype(np.float64) ** 2)))
                        if sys_rms > 0.005:
                            # Dual check: ratio detects bleed pattern, mic_rms floor
                            # prevents suppressing user speech. Bleed keeps mic_rms
                            # below ~0.013; real speech pushes it above 0.014 even
                            # when speaking over loud system audio.
                            ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
                            echo_detected = ratio < 1.5 and mic_rms < 0.014
                            logger.info(
                                f"Echo gate at {timestamp:.1f}s: "
                                f"mic_rms={mic_rms:.4f}, sys_rms={sys_rms:.4f}, "
                                f"ratio={ratio:.2f}, suppressed={echo_detected}"
                            )

                    if echo_detected:
                        logger.info(f"Echo suppressed at {timestamp:.1f}s")
                    else:
                        self._transcribe_frames(speech_frames, "you", timestamp)

                    mic_speech_start = mic_offset
                    mic_detector.reset()
                elif mic_detector.is_silent() and not mic_detector._has_had_speech:
                    # Pure silence — skip ahead without transcribing
                    mic_speech_start = mic_offset
                    mic_detector.reset()

            # --- Process system audio ---
            sys_frames = self.audio_recorder._system_frames
            sys_len = len(sys_frames)
            if sys_len > sys_offset:
                new_frames = sys_frames[sys_offset:sys_len]
                for frame in new_frames:
                    sys_detector.feed(frame)
                sys_offset = sys_len

                speech_frames = sys_frames[sys_speech_start:sys_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    sys_detector.is_silent() and sys_detector._has_had_speech
                    and speech_duration > 0.5
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
                    timestamp = prev_samples / sr
                    self._transcribe_frames(speech_frames, "them", timestamp)
                    sys_speech_start = sys_offset
                    sys_detector.reset()
                elif sys_detector.is_silent() and not sys_detector._has_had_speech:
                    # Pure silence — skip ahead without transcribing
                    sys_speech_start = sys_offset
                    sys_detector.reset()

        # --- Flush remaining audio when stopping ---
        mic_remaining = mic_frames[mic_speech_start:mic_offset] if mic_offset > mic_speech_start else []
        if mic_remaining:
            prev_samples = sum(len(f) for f in mic_frames[:mic_speech_start])
            self._transcribe_frames(mic_remaining, "you", prev_samples / sr)

        sys_remaining = sys_frames[sys_speech_start:sys_offset] if sys_offset > sys_speech_start else []
        if sys_remaining:
            prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
            self._transcribe_frames(sys_remaining, "them", prev_samples / sr)

        logger.info("Transcription worker stopped")
        self.finished.emit()

    def _transcribe_frames(self, frames, speaker, timestamp):
        """Concatenate frames and transcribe."""
        try:
            audio = np.concatenate(frames)
            if len(audio) > 0:
                text = self.transcriber.transcribe_text(audio)
                if text and text.strip():
                    self.segment_ready.emit(speaker, text.strip(), timestamp)
        except Exception as e:
            logger.error(f"Transcription error ({speaker}): {e}")

    def stop(self):
        self._running = False


class PostProcessWorker(QObject):
    """
    Re-transcribes full audio after recording stops.

    Runs in a QThread. Takes complete mic and system audio arrays,
    transcribes each with vad_filter=True for clean segment boundaries,
    and emits the merged result.
    """

    segments_ready = Signal(list)  # list of {speaker, text, start} dicts
    progress = Signal(str)  # status message
    finished = Signal()

    def __init__(self, transcriber, mic_audio, system_audio):
        super().__init__()
        self.transcriber = transcriber
        self.mic_audio = mic_audio
        self.system_audio = system_audio

    def run(self):
        """Transcribe full audio and emit cleaned segments."""
        segments = []
        has_both = len(self.mic_audio) > 0 and len(self.system_audio) > 0

        try:
            # Transcribe mic audio
            if len(self.mic_audio) > 0:
                self.progress.emit("Processing microphone audio...")
                def mic_progress(p):
                    pct = int(p * (50 if has_both else 100))
                    self.progress.emit(f"Processing microphone audio... {pct}%")
                mic_segments = self.transcriber.transcribe(
                    self.mic_audio, progress_callback=mic_progress,
                )
                for seg in mic_segments:
                    segments.append({
                        "speaker": "you",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Transcribe system audio
            if len(self.system_audio) > 0:
                self.progress.emit("Processing system audio...")
                def sys_progress(p):
                    pct = int((50 if has_both else 0) + p * (50 if has_both else 100))
                    self.progress.emit(f"Processing system audio... {pct}%")
                sys_segments = self.transcriber.transcribe(
                    self.system_audio, progress_callback=sys_progress,
                )
                for seg in sys_segments:
                    segments.append({
                        "speaker": "them",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Sort by timestamp, then remove echo duplicates
            segments.sort(key=lambda s: s["start"])
            from core.echo_gate import deduplicate_segments
            segments = deduplicate_segments(segments)

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

        # Audio recorder — no callback, just records
        self.audio_recorder = AudioRecorder()

        # Apply saved audio device settings
        mic_device = self.config.get_mic_device()
        if mic_device is not None:
            self.audio_recorder.set_mic_device(mic_device)
        system_device = self.config.get_system_device()
        if system_device is not None:
            self.audio_recorder.set_system_device(system_device)

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

        self.logger.info("Application initialized")

    def _on_segment(self, speaker, text, timestamp):
        """
        Handle a transcript segment from the transcription worker.
        Called on the main thread via Qt signal (thread-safe).
        """
        segment = {"speaker": speaker, "text": text, "start": timestamp}
        self._current_segments.append(segment)
        if self.main_window is not None:
            self.main_window.append_segment(speaker, text, timestamp)

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
            self.tray_icon.show_notification("Recording", "Recording started")

    def _start_transcription_worker(self):
        """Launch the transcription worker in a background QThread."""
        # Clean up any previous worker
        self._stop_transcription_worker()

        self._transcription_thread = QThread()
        self._transcription_worker = TranscriptionWorker(
            self.streaming_transcriber,
            self.audio_recorder,
            silence_threshold=0.01,
            min_silence_ms=500,
            max_speech_s=30.0,
        )
        self._transcription_worker.moveToThread(self._transcription_thread)

        # Connect signals
        self._transcription_thread.started.connect(self._transcription_worker.run)
        self._transcription_worker.segment_ready.connect(self._on_segment)
        self._transcription_worker.finished.connect(
            self._cleanup_transcription_thread, Qt.ConnectionType.QueuedConnection
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

    def stop_recording(self):
        """Stop recording and launch post-processing."""
        self.logger.info("Stopping recording...")

        # Stop transcription worker first
        self._stop_transcription_worker()

        # Stop audio capture — returns full audio arrays
        mic_audio, system_audio = self.audio_recorder.stop_recording()
        self._recording_duration = self.audio_recorder.get_duration()

        # Update UI to processing state
        if self.main_window is not None:
            self.main_window.set_processing_state("Cleaning up transcript...")

        # Launch post-processing in background thread
        self._start_postprocess(mic_audio, system_audio)

    def _start_postprocess(self, mic_audio, system_audio):
        """Launch PostProcessWorker in a background QThread."""
        self._postprocess_thread = QThread()
        self._postprocess_worker = PostProcessWorker(
            self.streaming_transcriber, mic_audio, system_audio
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
            self.tray_icon.show_notification(
                "Recording Complete",
                f"Duration: {int(self._recording_duration)}s"
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
            dialog = SettingsDialog(self.config, self.audio_recorder, parent=self.main_window)
            dialog.settings_changed.connect(self._on_settings_changed)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"Failed to show settings dialog: {e}")
            if self.tray_icon is not None:
                self.tray_icon.show_error(f"Failed to open settings: {e}")

    def _on_settings_changed(self):
        """Apply updated settings from the dialog."""
        self.logger.info("Settings changed - applying updates")

        # Update audio devices
        mic_device = self.config.get_mic_device()
        self.audio_recorder.set_mic_device(mic_device)
        system_device = self.config.get_system_device()
        self.audio_recorder.set_system_device(system_device)

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

    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
