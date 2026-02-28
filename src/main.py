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
from utils.config import ConfigManager
from utils.logger import setup_logger


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
            interval=self.config.get_transcription_interval(),
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
        """Stop recording, finalize transcription, and save the transcript."""
        self.logger.info("Stopping recording...")

        # Stop transcription worker first
        self._stop_transcription_worker()

        # Stop audio capture
        self.audio_recorder.stop_recording()
        duration = self.audio_recorder.get_duration()

        # Save transcript
        transcript = self._current_segments
        model = self.config.get_streaming_model_size()
        self.session_manager.save_transcript(transcript, duration=duration, model=model)

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

    def show_settings(self):
        """Open the settings dialog."""
        self.logger.info("Opening settings dialog")
        try:
            dialog = SettingsDialog(self.config, self.audio_recorder)
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

        # Note: transcription interval is read fresh each time recording starts

    def on_folder_selected(self, folder_name):
        """Load and display transcripts for the selected folder."""
        transcripts = self.session_manager.list_transcripts(folder_name)
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

    # Load folders on startup
    cadence_app._refresh_folders()

    # Show the main window
    main_window.show()

    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
