"""
Settings dialog for Cadence.
"""

import os
import time

from PySide6.QtWidgets import (
    QVBoxLayout, QFormLayout, QHBoxLayout, QGridLayout,
    QComboBox, QGroupBox, QPushButton, QLabel, QLineEdit, QFrame,
    QCheckBox, QWidget, QProgressBar, QScrollArea,
)
from PySide6.QtCore import Signal, Qt, QTimer, QThread, QObject
from PySide6.QtGui import QFont, QGuiApplication

from gui.theme import (
    RoundedDialog, MessageBox, BG_SURFACE, BORDER, ACCENT, TEXT_SECONDARY,
)


class _NoWheelComboBox(QComboBox):
    """ComboBox that ignores mouse wheel to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


# ---------------------------------------------------------------------------
# Audio Level Meter Dialog
# ---------------------------------------------------------------------------

class AudioLevelMeterDialog(RoundedDialog):
    """Dialog showing real-time microphone level via sounddevice."""

    def __init__(self, mic_device_index=None, parent=None):
        super().__init__(parent)
        self._mic_device_index = mic_device_index
        self._stream = None
        self._current_level = 0.0

        self.setWindowTitle("Microphone Test")
        self.setMinimumWidth(400)
        self.setMinimumHeight(220)

        self._init_ui()
        self._start_monitoring()

    def _init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Microphone Level Monitor")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        instructions = QLabel("Speak into your microphone to see the audio level.")
        instructions.setStyleSheet("color: rgba(255, 255, 255, 140); margin-bottom: 10px;")
        layout.addWidget(instructions)

        self.level_bar = QProgressBar()
        self.level_bar.setMinimum(0)
        self.level_bar.setMaximum(100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(True)
        self.level_bar.setFormat("%v%")
        self.level_bar.setMinimumHeight(40)
        layout.addWidget(self.level_bar)

        self.status_label = QLabel("Microphone Quality: Testing...")
        self.status_label.setStyleSheet("font-size: 12px; margin-top: 10px;")
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._close_and_stop)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _start_monitoring(self):
        """Open a sounddevice InputStream and poll RMS via QTimer."""
        try:
            import sounddevice as sd
            import numpy as np

            device = self._mic_device_index  # None = default

            self._stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=1600,   # 100 ms of audio at 16 kHz
                device=device,
            )
            self._stream.start()

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(100)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 12px;")

    def _tick(self):
        """Read available audio and update the level bar."""
        try:
            import numpy as np

            if self._stream is None:
                return
            avail = self._stream.read_available
            if avail <= 0:
                return
            data, _ = self._stream.read(avail)
            rms = float(np.sqrt(np.mean(data ** 2)))
            # Map RMS to 0-100 scale (calibrated for normal speech)
            level = min(100, int(rms * 3500))
            self._current_level = level
            self.level_bar.setValue(level)

            if level < 5:
                self.status_label.setText("Microphone Quality: No signal detected")
                self.status_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            elif level < 20:
                self.status_label.setText("Microphone Quality: Very weak signal")
                self.status_label.setStyleSheet("color: #f39c12; font-size: 12px;")
            elif level < 40:
                self.status_label.setText("Microphone Quality: Good")
                self.status_label.setStyleSheet("color: #2ecc71; font-size: 12px; font-weight: bold;")
            else:
                self.status_label.setText("Microphone Quality: Excellent")
                self.status_label.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;")
        except Exception:
            pass  # Ignore transient read errors

    def _close_and_stop(self):
        self._stop_stream()
        self.accept()

    def _stop_stream(self):
        if hasattr(self, "_timer") and self._timer.isActive():
            self._timer.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def closeEvent(self, event):
        self._stop_stream()
        event.accept()

    def reject(self):
        self._stop_stream()
        super().reject()


# ---------------------------------------------------------------------------
# Model Download Worker + Dialog
# ---------------------------------------------------------------------------

class _DownloadWorker(QObject):
    """Background worker that triggers a model download via huggingface_hub."""

    finished = Signal()
    error = Signal(str)

    def __init__(self, transcriber, model_size):
        super().__init__()
        self.transcriber = transcriber
        self.model_size = model_size

    def run(self):
        try:
            # Clean up any partial/failed downloads first
            self.transcriber.clean_partial_download(self.model_size)

            from huggingface_hub import snapshot_download

            repo_id = f"Systran/faster-whisper-{self.model_size}"
            snapshot_download(repo_id, cache_dir=self.transcriber.models_dir)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ModelDownloadDialog(RoundedDialog):
    """Progress dialog shown while downloading a Whisper model."""

    def __init__(self, display_name, model_size, expected_mb, transcriber, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Model")
        self.setMinimumWidth(420)
        self._success = False
        self._expected_bytes = expected_mb * 1024 * 1024

        # Path that huggingface_hub downloads into
        cache_name = f"models--Systran--faster-whisper-{model_size}"
        self._cache_path = os.path.join(transcriber.models_dir, cache_name)
        self._start_time = time.time()

        layout = QVBoxLayout()
        layout.setSpacing(12)

        self._status = QLabel(f"Downloading {display_name}...")
        self._status.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%v%")
        self._bar.setMinimumHeight(28)
        layout.addWidget(self._bar)

        self._time_label = QLabel("Elapsed: 0:00")
        self._time_label.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        layout.addWidget(self._time_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Background download thread
        self._thread = QThread()
        self._worker = _DownloadWorker(transcriber, model_size)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Poll directory size for progress
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._tick)
        self._poll_timer.start(500)

        self._thread.start()

    def _tick(self):
        current = self._dir_size()
        if self._expected_bytes > 0:
            pct = min(99, int(current / self._expected_bytes * 100))
        else:
            pct = 0
        self._bar.setValue(pct)

        elapsed = time.time() - self._start_time
        m, s = divmod(int(elapsed), 60)
        self._time_label.setText(f"Elapsed: {m}:{s:02d}")

    def _dir_size(self):
        total = 0
        if os.path.exists(self._cache_path):
            for dirpath, _, filenames in os.walk(self._cache_path):
                for f in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
        return total

    def _on_finished(self):
        self._poll_timer.stop()
        self._bar.setValue(100)
        self._success = True
        self._cleanup()
        self.accept()

    def _on_error(self, msg):
        self._poll_timer.stop()
        self._cleanup()
        MessageBox.critical(self, "Download Failed", f"Failed to download model:\n\n{msg}")
        self.reject()

    def _cleanup(self):
        self._poll_timer.stop()
        if self._worker:
            try:
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except RuntimeError:
                pass
        if self._thread:
            self._thread.quit()
            if not self._thread.wait(3000):
                self._thread.finished.connect(self._thread.deleteLater)
                if self._worker:
                    self._thread.finished.connect(self._worker.deleteLater)
                self._thread = None
                self._worker = None
                return
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def reject(self):
        self._cleanup()
        super().reject()

    def closeEvent(self, event):
        self._cleanup()
        event.accept()

    @property
    def succeeded(self):
        return self._success


# ---------------------------------------------------------------------------
# Main Settings Dialog
# ---------------------------------------------------------------------------

class SettingsDialog(RoundedDialog):
    """Settings dialog for model, audio device, and preferences."""

    settings_changed = Signal()
    check_for_updates = Signal()

    def __init__(self, config, audio_recorder, session_manager=None,
                 transcriber=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_recorder = audio_recorder
        self.session_manager = session_manager
        self.transcriber = transcriber
        self.setWindowTitle("Cadence Settings")
        self.setFixedWidth(800)
        self._setup_ui()
        self._load_current_settings()

    def showEvent(self, event):
        """Size dialog to fit content width and constrain height to screen."""
        screen = self.screen()
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            hint = self.sizeHint()
            new_width = min(hint.width(), geo.width() - 40)
            new_height = min(hint.height(), geo.height() - 40)
            self.resize(new_width, new_height)
        super().showEvent(event)

    def _setup_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setSpacing(8)

        # Scroll area wrapping all group boxes (vertical only)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_area.viewport().setStyleSheet("background: transparent;")

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 8, 0)
        layout.setSpacing(12)

        # Metrics Dashboard (at top)
        if self.session_manager is not None:
            metrics_group = QGroupBox("Activity")
            metrics_layout = QVBoxLayout()
            metrics_grid = QGridLayout()
            metrics_grid.setSpacing(6)
            self._metric_labels = {}
            self._setup_metrics(metrics_grid)
            metrics_layout.addLayout(metrics_grid)

            # Reset Statistics button row
            reset_row = QHBoxLayout()
            reset_row.addStretch()
            reset_btn = QPushButton("Reset Statistics")
            reset_btn.setFixedWidth(150)
            reset_btn.clicked.connect(self._reset_statistics)
            reset_row.addWidget(reset_btn)
            metrics_layout.addLayout(reset_row)

            metrics_group.setLayout(metrics_layout)
            layout.addWidget(metrics_group)

        # Speech Recognition
        model_group = QGroupBox("Speech Recognition")
        model_layout = QFormLayout()
        model_layout.setVerticalSpacing(4)

        self.model_combo = _NoWheelComboBox()
        models = [
            ("Fastest",   "tiny"),
            ("Balanced",  "base"),
            ("Accurate",  "small"),
            ("Precision", "medium"),
        ]
        for display_name, model_id in models:
            self.model_combo.addItem(display_name, model_id)

        quality_col = QVBoxLayout()
        quality_col.setSpacing(2)
        quality_col.addWidget(self.model_combo)
        quality_desc = QLabel("Select the quality level for speech recognition.")
        quality_desc.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        quality_col.addWidget(quality_desc)
        model_layout.addRow("Quality:", quality_col)

        info_label = QLabel(
            "Fastest \u2014 Whisper Tiny (~70 MB), sub-second\n"
            "Balanced \u2014 Whisper Base (~140 MB), sub-second\n"
            "Accurate \u2014 Whisper Small (~500 MB), ~2s\n"
            "Precision \u2014 Whisper Medium (~1.5 GB), ~5s"
        )
        info_label.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        model_layout.addRow("", info_label)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Audio Devices
        audio_group = QGroupBox("Audio Devices")
        audio_layout = QFormLayout()
        audio_layout.setVerticalSpacing(4)

        self.mic_combo = _NoWheelComboBox()
        self._populate_mic_devices()
        audio_layout.addRow("Microphone:", self.mic_combo)

        self.system_combo = _NoWheelComboBox()
        self._populate_system_devices()
        audio_layout.addRow("System audio:", self.system_combo)

        # Test Microphone button
        test_row = QHBoxLayout()
        test_row.addStretch()
        test_btn = QPushButton("Test Microphone")
        test_btn.setFixedWidth(150)
        test_btn.clicked.connect(self._test_microphone)
        test_row.addWidget(test_btn)
        audio_layout.addRow("", test_row)

        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # Your Profile
        profile_group = QGroupBox("Your Profile")
        profile_layout = QFormLayout()
        profile_layout.setVerticalSpacing(4)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your name")
        profile_col = QVBoxLayout()
        profile_col.setSpacing(2)
        profile_col.addWidget(self.name_edit)
        profile_desc = QLabel("Shown as your speaker label in transcripts (instead of \"You\").")
        profile_desc.setStyleSheet(f"color: rgba(255, 255, 255, 140); font-size: 11px;")
        profile_col.addWidget(profile_desc)
        profile_layout.addRow("Name:", profile_col)
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)

        # Sound Effects
        sound_group = QGroupBox("Sound Effects")
        sound_layout = QVBoxLayout()
        sound_layout.setSpacing(4)
        self.sound_effects_cb = QCheckBox("Play sounds on recording start/stop")
        sound_layout.addWidget(self.sound_effects_cb)
        sound_desc = QLabel("Short piano chime when recording starts or stops.")
        sound_desc.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        sound_layout.addWidget(sound_desc)
        sound_group.setLayout(sound_layout)
        layout.addWidget(sound_group)

        # Bug Report
        bug_group = QGroupBox("Bug Report")
        bug_layout = QVBoxLayout()
        bug_info = QLabel("Experiencing an issue? Submit a bug report with your system info and recent logs.")
        bug_info.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        bug_layout.addWidget(bug_info)
        bug_btn_row = QHBoxLayout()
        bug_btn_row.addStretch()
        report_btn = QPushButton("Report Bug...")
        report_btn.clicked.connect(self._open_bug_report)
        bug_btn_row.addWidget(report_btn)
        bug_layout.addLayout(bug_btn_row)
        bug_group.setLayout(bug_layout)
        layout.addWidget(bug_group)

        # Updates
        update_group = QGroupBox("Updates")
        update_layout = QVBoxLayout()
        update_info = QLabel("Check for a newer version of Cadence on GitHub.")
        update_info.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        update_layout.addWidget(update_info)

        from version import __version__
        version_label = QLabel(f"Current version: {__version__}")
        version_label.setStyleSheet("color: rgba(255, 255, 255, 180); font-size: 11px;")
        update_layout.addWidget(version_label)

        update_btn_row = QHBoxLayout()
        update_btn_row.addStretch()
        self._update_btn = QPushButton("Check for Updates")
        self._update_btn.clicked.connect(self._on_check_for_updates)
        update_btn_row.addWidget(self._update_btn)
        update_layout.addLayout(update_btn_row)
        update_group.setLayout(update_layout)
        layout.addWidget(update_group)

        # Debug
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout()
        debug_layout.setSpacing(4)

        self.debug_master_cb = QCheckBox("Enable Debug Mode")
        debug_layout.addWidget(self.debug_master_cb)
        debug_master_desc = QLabel("Enables advanced diagnostic features for troubleshooting.")
        debug_master_desc.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        debug_layout.addWidget(debug_master_desc)

        # Sub-options container (indented, hidden when master off)
        self.debug_sub_widget = QWidget()
        debug_sub_layout = QVBoxLayout()
        debug_sub_layout.setContentsMargins(20, 4, 0, 0)
        debug_sub_layout.setSpacing(4)

        self.echo_diag_cb = QCheckBox("Echo Diagnostics")
        debug_sub_layout.addWidget(self.echo_diag_cb)
        echo_diag_desc = QLabel("Saves WAV audio chunks to .cadence/echo_debug/ for analysis.")
        echo_diag_desc.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        debug_sub_layout.addWidget(echo_diag_desc)

        self.echo_log_cb = QCheckBox("Echo Gate Logging")
        debug_sub_layout.addWidget(self.echo_log_cb)
        echo_log_desc = QLabel("Verbose energy gate logging (mic/sys RMS, ratio, suppression).")
        echo_log_desc.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 11px;")
        debug_sub_layout.addWidget(echo_log_desc)

        self.debug_sub_widget.setLayout(debug_sub_layout)
        debug_layout.addWidget(self.debug_sub_widget)

        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)

        # Wire master toggle
        self.debug_master_cb.toggled.connect(self._on_debug_toggled)

        layout.addStretch()
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        # Buttons (fixed at bottom, outside scroll area)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("Save")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(self._save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        outer_layout.addLayout(btn_layout)

        self.setLayout(outer_layout)

    # -- Metrics dashboard ---------------------------------------------------

    def _make_metric_card(self, value, label):
        """Create a small metric card widget."""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color:{BG_SURFACE}; border:1px solid {BORDER};"
            f" border-radius:4px; padding:4px; }}"
        )
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 4, 8, 4)
        vbox.setSpacing(1)
        val_lbl = QLabel(str(value))
        val_lbl.setFont(QFont("Calibri", 14, QFont.Weight.Bold))
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(val_lbl)
        desc_lbl = QLabel(label)
        desc_lbl.setFont(QFont("Calibri", 8))
        desc_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; border:none;")
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(desc_lbl)
        return card, val_lbl

    def _setup_metrics(self, grid):
        """Populate the 2x4 metrics grid."""
        metrics = self.session_manager.get_metrics()

        # Format duration
        def fmt_dur(s):
            s = int(s)
            if s >= 3600:
                return f"{s // 3600}h {(s % 3600) // 60}m"
            if s >= 60:
                return f"{s // 60}m {s % 60}s"
            return f"{s}s"

        cards = [
            (str(metrics["total_recordings"]), "Recordings"),
            (fmt_dur(metrics["total_duration_s"]), "Total Duration"),
            (f"{metrics['total_words']:,}", "Total Words"),
            (str(metrics["recordings_this_week"]), "This Week"),
            (fmt_dur(metrics["avg_duration_s"]), "Avg Duration"),
            (f"{metrics['avg_words']:,}", "Avg Words"),
            (f"{metrics['you_words']:,}", "Your Words"),
            (f"{metrics['speaker_words']:,}", "Speaker Words"),
        ]

        for i, (value, label) in enumerate(cards):
            row, col = divmod(i, 4)
            card, val_lbl = self._make_metric_card(value, label)
            self._metric_labels[label] = val_lbl
            grid.addWidget(card, row, col)

    def _reset_statistics(self):
        """Ask for confirmation, then reset all config-based statistics."""
        result = MessageBox.question(
            self,
            "Reset Statistics",
            "Are you sure you want to reset all usage statistics?\n\n"
            "This cannot be undone.",
        )
        if result != MessageBox.Yes:
            return

        self.config.reset_statistics()

        # Update the dashboard labels to show zeros
        if hasattr(self, "_metric_labels"):
            zero_map = {
                "Recordings": "0",
                "Total Duration": "0s",
                "Total Words": "0",
                "This Week": "0",
                "Avg Duration": "0s",
                "Avg Words": "0",
                "Your Words": "0",
                "Speaker Words": "0",
            }
            for label_text, val_lbl in self._metric_labels.items():
                if label_text in zero_map:
                    val_lbl.setText(zero_map[label_text])

        MessageBox.flash(self, "Statistics Reset", "All usage statistics have been cleared.")

    # -- Audio level meter ---------------------------------------------------

    def _test_microphone(self):
        """Open the audio level meter dialog for the currently selected mic."""
        mic_device = self.mic_combo.currentData()  # None = auto
        dlg = AudioLevelMeterDialog(mic_device_index=mic_device, parent=self)
        dlg.exec()

    # -- Model download ------------------------------------------------------

    def _maybe_download_model(self, model_size, display_name):
        """
        If the selected model is not yet downloaded, show a download dialog.

        Returns True if the model is available (already downloaded or
        successfully downloaded), False if the user cancelled or an error
        occurred.
        """
        if self.transcriber is None:
            # No transcriber reference — skip check
            return True

        if self.transcriber.is_model_downloaded(model_size):
            return True

        # Model not found locally — offer to download
        result = MessageBox.question(
            self,
            "Download Model",
            f"The {display_name} model (~{self.transcriber.MODEL_SIZES_MB.get(model_size, '?')} MB) "
            f"is not downloaded yet.\n\n"
            f"Download it now?",
        )
        if result != MessageBox.Yes:
            return False

        expected_mb = self.transcriber.MODEL_SIZES_MB.get(model_size, 200)
        dlg = ModelDownloadDialog(
            display_name, model_size, expected_mb, self.transcriber, parent=self,
        )
        dlg.exec()
        return dlg.succeeded

    # -- Bug report ----------------------------------------------------------

    def _open_bug_report(self):
        """Collect system info and open a pre-filled GitHub issue."""
        import os
        import platform
        import webbrowser
        from urllib.parse import quote
        from version import __version__

        model_id = self.config.get_streaming_model_size()
        model_names = {"tiny": "Fastest", "base": "Balanced", "small": "Accurate", "medium": "Precision"}
        model_display = f"{model_names.get(model_id, model_id)} ({model_id})"

        mic = self.config.get_mic_device()
        mic_display = "Auto-detect" if mic is None else str(mic)

        from utils.resource_path import get_app_data_path
        log_path = os.path.join(get_app_data_path("logs"), "cadence.log")
        log_lines = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                log_lines = "".join(all_lines[-50:])
        except (OSError, FileNotFoundError):
            log_lines = "(no log file found)"

        body = (
            "## Description\n"
            "[Describe the issue here]\n\n"
            "## Steps to Reproduce\n"
            "1. \n2. \n3. \n\n"
            "## Expected Behavior\n\n\n"
            "## System Info\n"
            f"- **App Version**: {__version__}\n"
            f"- **OS**: {platform.platform()}\n"
            f"- **Python**: {platform.python_version()}\n"
            f"- **Model**: {model_display}\n"
            f"- **Microphone**: {mic_display}\n\n"
            "## Recent Logs\n"
            "```\n"
            f"{log_lines}"
            "```\n"
        )

        max_body = 6000
        if len(body) > max_body:
            truncation_note = "\n... (truncated)\n```\n"
            overhead = len(body) - len(log_lines)
            allowed_log = max_body - overhead - len(truncation_note)
            log_lines = log_lines[:allowed_log] + truncation_note
            body = (
                "## Description\n"
                "[Describe the issue here]\n\n"
                "## Steps to Reproduce\n"
                "1. \n2. \n3. \n\n"
                "## Expected Behavior\n\n\n"
                "## System Info\n"
                f"- **App Version**: {__version__}\n"
                f"- **OS**: {platform.platform()}\n"
                f"- **Python**: {platform.python_version()}\n"
                f"- **Model**: {model_display}\n"
                f"- **Microphone**: {mic_display}\n\n"
                "## Recent Logs\n"
                "```\n"
                f"{log_lines}"
                "```\n"
            )

        title = quote("Bug: ")
        encoded_body = quote(body)
        url = f"https://github.com/whorne89/Cadence/issues/new?title={title}&body={encoded_body}"
        webbrowser.open(url)

    # -- Update check -------------------------------------------------------

    def _on_check_for_updates(self):
        """Emit signal to trigger manual update check and close dialog."""
        self.check_for_updates.emit()
        self.accept()

    # -- Debug toggle --------------------------------------------------------

    def _on_debug_toggled(self, checked):
        """Show/hide debug sub-options and reset checkboxes when disabled."""
        self.debug_sub_widget.setVisible(checked)
        if not checked:
            self.echo_diag_cb.setChecked(False)
            self.echo_log_cb.setChecked(False)

    # -- Device population ---------------------------------------------------

    def _populate_mic_devices(self):
        self.mic_combo.addItem("Auto-detect", None)
        for dev in self.audio_recorder.list_mic_devices():
            self.mic_combo.addItem(dev['name'], dev['index'])

    def _populate_system_devices(self):
        self.system_combo.addItem("Auto-detect", None)
        for dev in self.audio_recorder.list_system_devices():
            self.system_combo.addItem(dev['name'], dev['index'])

    # -- Load / Save ---------------------------------------------------------

    def _load_current_settings(self):
        streaming = self.config.get_streaming_model_size()
        idx = self.model_combo.findData(streaming)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        mic_device = self.config.get_mic_device()
        if mic_device is not None:
            idx = self.mic_combo.findData(mic_device)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)

        system_device = self.config.get_system_device()
        if system_device is not None:
            idx = self.system_combo.findData(system_device)
            if idx >= 0:
                self.system_combo.setCurrentIndex(idx)

        self.name_edit.setText(self.config.get_first_name())

        # Sound effects
        self.sound_effects_cb.setChecked(self.config.is_sound_effects_enabled())

        # Debug settings
        debug_enabled = self.config.is_debug_enabled()
        self.debug_master_cb.setChecked(debug_enabled)
        self.debug_sub_widget.setVisible(debug_enabled)
        self.echo_diag_cb.setChecked(self.config.is_echo_debug_enabled())
        self.echo_log_cb.setChecked(self.config.is_echo_gate_logging_enabled())

    def _save_settings(self):
        changes = []

        new_model = self.model_combo.currentData()
        old_model = self.config.get_streaming_model_size()
        model_changed = new_model != old_model

        # If model changed, check if we need to download it
        if model_changed and self.transcriber is not None:
            display_name = self.model_combo.currentText()
            if not self._maybe_download_model(new_model, display_name):
                # User cancelled download — revert combo to current model
                idx = self.model_combo.findData(old_model)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                return  # Don't save anything yet

        if model_changed:
            changes.append(f"Quality: {self.model_combo.currentText()}")
        self.config.set("whisper", "streaming_model_size", value=new_model)

        new_mic = self.mic_combo.currentData()
        if new_mic != self.config.get_mic_device():
            changes.append(f"Microphone: {self.mic_combo.currentText()}")
        self.config.set("audio", "mic_device_index", value=new_mic)

        new_sys = self.system_combo.currentData()
        if new_sys != self.config.get_system_device():
            changes.append(f"System audio: {self.system_combo.currentText()}")
        self.config.set("audio", "system_device_index", value=new_sys)

        new_name = self.name_edit.text().strip()
        if new_name != self.config.get_first_name():
            changes.append(f"Name: {new_name or '(cleared)'}")
        self.config.set("user", "first_name", value=new_name)

        # Sound effects
        new_sound = self.sound_effects_cb.isChecked()
        if new_sound != self.config.is_sound_effects_enabled():
            changes.append(f"Sound Effects: {'On' if new_sound else 'Off'}")
        self.config.set("ui", "sound_effects_enabled", value=new_sound)

        # Debug settings
        new_debug = self.debug_master_cb.isChecked()
        if new_debug != self.config.is_debug_enabled():
            changes.append(f"Debug Mode: {'On' if new_debug else 'Off'}")
        self.config.set("debug", "enabled", value=new_debug)

        new_echo_diag = self.echo_diag_cb.isChecked()
        if new_echo_diag != self.config.is_echo_debug_enabled():
            changes.append(f"Echo Diagnostics: {'On' if new_echo_diag else 'Off'}")
        self.config.set("debug", "echo_diagnostics", value=new_echo_diag)

        new_echo_log = self.echo_log_cb.isChecked()
        if new_echo_log != self.config.is_echo_gate_logging_enabled():
            changes.append(f"Echo Gate Logging: {'On' if new_echo_log else 'Off'}")
        self.config.set("debug", "echo_gate_logging", value=new_echo_log)

        self.config.save()
        self.settings_changed.emit()

        if changes:
            summary = "\n".join(f"\u2022 {c}" for c in changes)
            MessageBox.flash(self, "Settings Saved", summary)

        self.accept()
