"""
Settings dialog for Cadence.
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QFormLayout, QHBoxLayout,
    QComboBox, QGroupBox, QPushButton, QLabel,
)
from PySide6.QtCore import Signal

from gui.theme import RoundedDialog


class _NoWheelComboBox(QComboBox):
    """ComboBox that ignores mouse wheel to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class SettingsDialog(RoundedDialog):
    """Settings dialog for model, audio device, and preferences."""

    settings_changed = Signal()

    def __init__(self, config, audio_recorder, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_recorder = audio_recorder
        self.setWindowTitle("Cadence Settings")
        self.setMinimumWidth(420)
        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

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

    def _save_settings(self):
        self.config.set("whisper", "streaming_model_size",
                        value=self.model_combo.currentData())
        self.config.set("audio", "mic_device_index",
                        value=self.mic_combo.currentData())
        self.config.set("audio", "system_device_index",
                        value=self.system_combo.currentData())
        self.config.save()
        self.settings_changed.emit()
        self.accept()
