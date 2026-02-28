"""
Cadence - Meeting Transcription
Main entry point that orchestrates all components.
"""

import sys
import os
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
from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.resource_path import get_app_data_path


logger = logging.getLogger("Cadence")


class TranscriptionWorker(QObject):
    """
    Worker that periodically transcribes accumulated audio from AudioRecorder.

    Runs in a QThread. Reads new frames from the recorder's frame lists,
    transcribes them, and emits results via Qt signals (thread-safe).
    """

    segment_ready = Signal(str, str, float)  # speaker, text, timestamp_seconds
    finished = Signal()

    def __init__(self, transcriber, audio_recorder, interval=5.0):
        super().__init__()
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.interval = interval
        self._running = False
        self._mic_offset = 0
        self._system_offset = 0

    def run(self):
        """Main loop — polls for new audio and transcribes it."""
        self._running = True
        self._mic_offset = 0
        self._system_offset = 0

        logger.info(f"Transcription worker started (interval={self.interval}s)")

        while self._running:
            time.sleep(self.interval)

            if not self._running:
                break

            # Transcribe new mic audio
            mic_frames = self.audio_recorder._mic_frames
            mic_len = len(mic_frames)
            if mic_len > self._mic_offset:
                new_frames = mic_frames[self._mic_offset:mic_len]
                # Compute timestamp: samples before this chunk / sample_rate
                prev_samples = sum(len(f) for f in mic_frames[:self._mic_offset])
                timestamp = prev_samples / self.audio_recorder.sample_rate
                self._mic_offset = mic_len
                try:
                    audio = np.concatenate(new_frames)
                    if len(audio) > 0:
                        text = self.transcriber.transcribe_text(audio)
                        if text and text.strip():
                            self.segment_ready.emit("you", text.strip(), timestamp)
                except Exception as e:
                    logger.error(f"Mic transcription error: {e}")

            # Transcribe new system audio
            sys_frames = self.audio_recorder._system_frames
            sys_len = len(sys_frames)
            if sys_len > self._system_offset:
                new_frames = sys_frames[self._system_offset:sys_len]
                prev_samples = sum(len(f) for f in sys_frames[:self._system_offset])
                timestamp = prev_samples / self.audio_recorder.sample_rate
                self._system_offset = sys_len
                try:
                    audio = np.concatenate(new_frames)
                    if len(audio) > 0:
                        text = self.transcriber.transcribe_text(audio)
                        if text and text.strip():
                            self.segment_ready.emit("them", text.strip(), timestamp)
                except Exception as e:
                    logger.error(f"System transcription error: {e}")

        logger.info("Transcription worker stopped")
        self.finished.emit()

    def stop(self):
        self._running = False


class ReprocessWorker(QObject):
    """Worker for batch reprocessing transcription in a background QThread."""

    finished = Signal(list)  # Emits list of transcript segments
    error = Signal(str)      # Emits error message

    def __init__(self, transcriber, mic_audio, system_audio):
        super().__init__()
        self.transcriber = transcriber
        self.mic_audio = mic_audio
        self.system_audio = system_audio

    def run(self):
        """Transcribe both channels, label speakers, and sort by time."""
        try:
            segments = []

            # Transcribe microphone audio (speaker = "you")
            if len(self.mic_audio) > 0:
                logger.info(f"Reprocessing mic audio: {len(self.mic_audio)} samples")
                mic_segments = self.transcriber.transcribe(self.mic_audio)
                for seg in mic_segments:
                    seg["speaker"] = "you"
                    segments.append(seg)

            # Transcribe system audio (speaker = "them")
            if len(self.system_audio) > 0:
                logger.info(f"Reprocessing system audio: {len(self.system_audio)} samples")
                system_segments = self.transcriber.transcribe(self.system_audio)
                for seg in system_segments:
                    seg["speaker"] = "them"
                    segments.append(seg)

            # Sort all segments by start time
            segments.sort(key=lambda s: s["start"])

            logger.info(f"Reprocessing complete: {len(segments)} segments")
            self.finished.emit(segments)

        except Exception as e:
            logger.error(f"Reprocessing failed: {e}")
            self.error.emit(str(e))


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

        # Reprocess transcriber (higher quality model for batch reprocessing)
        reprocess_model_size = self.config.get_reprocess_model_size()
        self.reprocess_transcriber = Transcriber(model_size=reprocess_model_size)

        # GUI components (set after creation in main())
        self.tray_icon = None
        self.main_window = None

        # Transcription worker thread state
        self._transcription_thread = None
        self._transcription_worker = None

        # Reprocessing thread state
        self._reprocess_thread = None
        self._reprocess_worker = None

        # Saved audio for reprocessing
        self._mic_audio = None
        self._system_audio = None

        self.logger.info("Application initialized")

    def _on_segment(self, speaker, text, timestamp):
        """
        Handle a transcript segment from the transcription worker.
        Called on the main thread via Qt signal (thread-safe).
        """
        self.session_manager.add_segment(speaker, text, start=timestamp)
        if self.main_window is not None:
            self.main_window.append_segment(speaker, text, timestamp)

    def start_recording(self):
        """Start a new recording session."""
        self.logger.info("Starting recording...")

        # Create a new session
        self.session_manager.create_session()
        self.session_manager.set_model(self.config.get_streaming_model_size())

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
            interval=5.0,
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
        """Stop recording, finalize transcription, and save the session."""
        self.logger.info("Stopping recording...")

        # Stop transcription worker first
        self._stop_transcription_worker()

        # Stop audio capture and get raw audio
        mic_audio, system_audio = self.audio_recorder.stop_recording()
        duration = self.audio_recorder.get_duration()

        # Save raw audio for potential reprocessing
        self._mic_audio = mic_audio
        self._system_audio = system_audio

        # Update session duration
        self.session_manager.set_duration(duration)

        # Save audio files if configured
        if self.config.get("session", "save_audio", default=True):
            audio_dir = get_app_data_path("audio")
            session_id = self.session_manager.active_session["id"][:8]
            if len(mic_audio) > 0:
                mic_path = os.path.join(audio_dir, f"{session_id}_mic.wav")
                self.audio_recorder.save_audio(mic_audio, mic_path)
                self.logger.info(f"Mic audio saved: {mic_path}")
            if len(system_audio) > 0:
                system_path = os.path.join(audio_dir, f"{session_id}_system.wav")
                self.audio_recorder.save_audio(system_audio, system_path)
                self.logger.info(f"System audio saved: {system_path}")

        # Save session transcript
        self.session_manager.save_session()

        # Update UI
        if self.main_window is not None:
            self.main_window.set_done_state()
        if self.tray_icon is not None:
            self.tray_icon.set_idle_state()
            self.tray_icon.show_notification(
                "Recording Stopped",
                f"Duration: {int(duration)}s"
            )

        self.logger.info(f"Recording stopped. Duration: {duration:.1f}s")

    def reprocess(self):
        """Run batch reprocessing of the last recording in a background thread."""
        if self._mic_audio is None and self._system_audio is None:
            self.logger.warning("No audio data available for reprocessing")
            if self.tray_icon is not None:
                self.tray_icon.show_error("No recording to reprocess")
            return

        if self._reprocess_thread is not None and self._reprocess_thread.isRunning():
            self.logger.warning("Reprocessing already in progress")
            return

        self.logger.info("Starting reprocessing...")

        # Update UI to processing state
        if self.main_window is not None:
            self.main_window.set_processing_state()
        if self.tray_icon is not None:
            self.tray_icon.set_processing_state()

        # Clean up previous thread if needed
        if self._reprocess_thread is not None:
            self._reprocess_thread.quit()
            self._reprocess_thread.wait()
            self._reprocess_thread.deleteLater()
            self._reprocess_thread = None
        if self._reprocess_worker is not None:
            self._reprocess_worker.deleteLater()
            self._reprocess_worker = None

        # Ensure we have valid arrays
        mic = self._mic_audio if self._mic_audio is not None else np.array([], dtype=np.float32)
        sys_audio = self._system_audio if self._system_audio is not None else np.array([], dtype=np.float32)

        # Create worker and thread
        self._reprocess_thread = QThread()
        self._reprocess_worker = ReprocessWorker(
            self.reprocess_transcriber, mic, sys_audio
        )
        self._reprocess_worker.moveToThread(self._reprocess_thread)

        # Connect signals
        self._reprocess_thread.started.connect(self._reprocess_worker.run)
        self._reprocess_worker.finished.connect(self._on_reprocess_finished)
        self._reprocess_worker.error.connect(self._on_reprocess_error)

        # Cleanup connections (queued to run on main thread)
        self._reprocess_worker.finished.connect(
            self._cleanup_reprocess_thread, Qt.ConnectionType.QueuedConnection
        )
        self._reprocess_worker.error.connect(
            self._cleanup_reprocess_thread, Qt.ConnectionType.QueuedConnection
        )

        # Start
        self._reprocess_thread.start()

    def _on_reprocess_finished(self, segments):
        """Handle completed reprocessing."""
        self.logger.info(f"Reprocessing finished: {len(segments)} segments")

        # Update session with reprocessed transcript
        if self.session_manager.active_session is not None:
            self.session_manager.active_session["transcript"] = segments
            self.session_manager.mark_reprocessed(self.config.get_reprocess_model_size())
            self.session_manager.save_session()

        # Update main window with new transcript
        if self.main_window is not None:
            self.main_window.set_transcript(segments)
            self.main_window.set_done_state()

        if self.tray_icon is not None:
            self.tray_icon.set_idle_state()
            self.tray_icon.show_notification(
                "Reprocessing Complete",
                f"{len(segments)} segments transcribed"
            )

    def _on_reprocess_error(self, error_msg):
        """Handle reprocessing failure."""
        self.logger.error(f"Reprocessing error: {error_msg}")

        if self.main_window is not None:
            self.main_window.set_done_state()
        if self.tray_icon is not None:
            self.tray_icon.set_idle_state()
            self.tray_icon.show_error(f"Reprocessing failed: {error_msg}")

    def _cleanup_reprocess_thread(self, _result=None):
        """Clean up reprocessing thread resources."""
        if self._reprocess_thread is not None:
            self._reprocess_thread.quit()
            self._reprocess_thread.wait(2000)
            self._reprocess_thread.deleteLater()
            self._reprocess_thread = None
        if self._reprocess_worker is not None:
            self._reprocess_worker.deleteLater()
            self._reprocess_worker = None

    def show_settings(self):
        """Open the settings dialog."""
        self.logger.info("Opening settings dialog")
        try:
            dialog = SettingsDialog(
                self.config,
                self.audio_recorder,
                self.reprocess_transcriber,
            )
            dialog.settings_changed.connect(self._on_settings_changed)
            dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
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

        # Update reprocess model
        reprocess_model = self.config.get_reprocess_model_size()
        if reprocess_model != self.reprocess_transcriber.model_size:
            self.reprocess_transcriber.change_model(reprocess_model)

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

        # Stop recording if active
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()

        # Clean up reprocessing thread
        if self._reprocess_thread is not None and self._reprocess_thread.isRunning():
            self._reprocess_thread.quit()
            self._reprocess_thread.wait(2000)

        QApplication.quit()


def main():
    """Application entry point."""
    # Set Windows app ID before creating QApplication
    set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName("Cadence")
    app.setApplicationDisplayName("Cadence")
    app.setQuitOnLastWindowClosed(False)  # Keep running with system tray

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
    main_window.reprocess_requested.connect(cadence_app.reprocess)

    # Show the main window
    main_window.show()

    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
