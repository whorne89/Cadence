"""
Settings dialog for Cadence.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout,
    QComboBox, QGroupBox, QDialogButtonBox, QSpinBox, QLabel,
)
from PySide6.QtCore import Signal


class SettingsDialog(QDialog):
    """Settings dialog for model, audio device, and preferences."""

    settings_changed = Signal()

    def __init__(self, config, audio_recorder, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_recorder = audio_recorder
        self.setWindowTitle("Cadence Settings")
        self.setMinimumWidth(400)
        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Model settings
        model_group = QGroupBox("Transcription")
        model_layout = QFormLayout()
        self.streaming_model_combo = QComboBox()
        for m in ["tiny", "base", "small"]:
            self.streaming_model_combo.addItem(m.capitalize(), m)
        model_layout.addRow("Model:", self.streaming_model_combo)

        # Transcription interval
        interval_layout = QHBoxLayout()
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(2, 8)
        self.interval_spin.setSuffix(" seconds")
        interval_layout.addWidget(self.interval_spin)
        model_layout.addRow("Update interval:", interval_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Audio settings
        audio_group = QGroupBox("Audio Devices")
        audio_layout = QFormLayout()
        self.mic_combo = QComboBox()
        self._populate_mic_devices()
        audio_layout.addRow("Microphone:", self.mic_combo)
        self.system_combo = QComboBox()
        self._populate_system_devices()
        audio_layout.addRow("System audio:", self.system_combo)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_mic_devices(self):
        self.mic_combo.addItem("Default", None)
        for dev in self.audio_recorder.list_mic_devices():
            self.mic_combo.addItem(dev['name'], dev['index'])

    def _populate_system_devices(self):
        self.system_combo.addItem("Default (auto-detect)", None)
        for dev in self.audio_recorder.list_system_devices():
            self.system_combo.addItem(dev['name'], dev['index'])

    def _load_current_settings(self):
        streaming = self.config.get_streaming_model_size()
        idx = self.streaming_model_combo.findData(streaming)
        if idx >= 0:
            self.streaming_model_combo.setCurrentIndex(idx)
        interval = int(self.config.get_transcription_interval())
        self.interval_spin.setValue(interval)

    def _save_settings(self):
        self.config.set("whisper", "streaming_model_size",
                        value=self.streaming_model_combo.currentData())
        self.config.set("whisper", "transcription_interval",
                        value=float(self.interval_spin.value()))
        self.config.set("audio", "mic_device_index",
                        value=self.mic_combo.currentData())
        self.config.set("audio", "system_device_index",
                        value=self.system_combo.currentData())
        self.config.save()
        self.settings_changed.emit()
        self.accept()
