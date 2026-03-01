"""
Settings dialog for Cadence.
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QFormLayout, QHBoxLayout, QGridLayout,
    QComboBox, QGroupBox, QPushButton, QLabel, QLineEdit, QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from gui.theme import RoundedDialog, MessageBox, BG_SURFACE, BORDER, ACCENT, TEXT_SECONDARY


class _NoWheelComboBox(QComboBox):
    """ComboBox that ignores mouse wheel to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class SettingsDialog(RoundedDialog):
    """Settings dialog for model, audio device, and preferences."""

    settings_changed = Signal()

    def __init__(self, config, audio_recorder, session_manager=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_recorder = audio_recorder
        self.session_manager = session_manager
        self.setWindowTitle("Cadence Settings")
        self.setMinimumWidth(460)
        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Metrics Dashboard (at top)
        if self.session_manager is not None:
            metrics_group = QGroupBox("Activity")
            metrics_grid = QGridLayout()
            metrics_grid.setSpacing(6)
            self._metric_labels = {}
            self._setup_metrics(metrics_grid)
            metrics_group.setLayout(metrics_grid)
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

        # Buttons
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
        layout.addLayout(btn_layout)

        self.setLayout(layout)

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

    def _populate_mic_devices(self):
        self.mic_combo.addItem("Auto-detect", None)
        for dev in self.audio_recorder.list_mic_devices():
            self.mic_combo.addItem(dev['name'], dev['index'])

    def _populate_system_devices(self):
        self.system_combo.addItem("Auto-detect", None)
        for dev in self.audio_recorder.list_system_devices():
            self.system_combo.addItem(dev['name'], dev['index'])

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

    def _save_settings(self):
        changes = []

        new_model = self.model_combo.currentData()
        if new_model != self.config.get_streaming_model_size():
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

        self.config.save()
        self.settings_changed.emit()

        if changes:
            summary = "\n".join(f"\u2022 {c}" for c in changes)
            MessageBox.flash(self, "Settings Saved", summary)

        self.accept()
