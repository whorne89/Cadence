"""
System tray interface for Cadence.
"""

from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PySide6.QtCore import Signal, Qt

from gui.toast_notification import ToastNotification
from gui.theme import RoundedDialog
from utils.resource_path import get_resource_path
from pathlib import Path


class SystemTrayIcon(QSystemTrayIcon):
    """System tray icon with context menu and status updates."""

    start_requested = Signal()
    stop_requested = Signal()
    window_requested = Signal()
    settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_recording = False
        self.icon_dir = Path(get_resource_path("icons"))
        self._create_icons()
        self._create_menu()
        self.set_idle_state()
        self.setToolTip("Cadence - Meeting Transcription")
        self._toast = ToastNotification()
        self.activated.connect(self._on_activated)
        self.show()

    def _create_icons(self):
        self.icon_idle = self._make_circle_icon(QColor(128, 128, 128))
        self.icon_recording = self._make_circle_icon(QColor(220, 50, 50))
        self.icon_processing = self._make_circle_icon(QColor(220, 180, 50))

    def _make_circle_icon(self, color):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()
        return QIcon(pixmap)

    def _create_menu(self):
        menu = QMenu()
        self.action_toggle = QAction("Start Recording", self)
        self.action_toggle.triggered.connect(self._on_toggle)
        menu.addAction(self.action_toggle)
        menu.addSeparator()
        action_window = QAction("Open Window", self)
        action_window.triggered.connect(self.window_requested.emit)
        menu.addAction(action_window)
        action_settings = QAction("Settings", self)
        action_settings.triggered.connect(self.settings_requested.emit)
        menu.addAction(action_settings)
        menu.addSeparator()
        action_about = QAction("About", self)
        action_about.triggered.connect(self._show_about)
        menu.addAction(action_about)
        menu.addSeparator()
        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.quit_requested.emit)
        menu.addAction(action_exit)
        self.setContextMenu(menu)

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.window_requested.emit()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.window_requested.emit()

    def _on_toggle(self):
        if self._is_recording:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_idle_state(self):
        self._is_recording = False
        self.setIcon(self.icon_idle)
        self.setToolTip("Cadence - Ready")
        self.action_toggle.setText("Start Recording")

    def set_recording_state(self):
        self._is_recording = True
        self.setIcon(self.icon_recording)
        self.setToolTip("Cadence - Recording...")
        self.action_toggle.setText("Stop Recording")

    def set_processing_state(self):
        self._is_recording = False
        self.setIcon(self.icon_processing)
        self.setToolTip("Cadence - Processing...")
        self.action_toggle.setText("Processing...")

    def showMessage(self, *args, **kwargs):
        """Override Qt native notification — redirect to custom toast."""
        if len(args) >= 2:
            self._toast.show_toast(args[1])
        elif len(args) >= 1:
            self._toast.show_toast(args[0])

    def show_notification(self, title, message, icon_type=None, duration=3000, details=""):
        """Show a toast notification overlay."""
        self._toast.show_toast(message, details=details)

    def show_error(self, message):
        self._toast.show_toast(message)

    def _show_about(self):
        dialog = AboutDialog()
        dialog.exec()


class AboutDialog(RoundedDialog):
    """Custom rounded About dialog for Cadence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Cadence")
        self.setFixedWidth(380)

        from version import __version__

        layout = QVBoxLayout()
        layout.setSpacing(12)

        title = QLabel("Cadence")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Meeting Transcription")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 13px; color: rgba(255, 255, 255, 160);")
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        desc = QLabel(
            "Real-time meeting transcription with speaker attribution\n"
            "powered by Whisper AI.\n\n"
            "Records mic + system audio, transcribes locally,\n"
            "and separates speakers automatically.\n\n"
            "No internet required \u2014 completely private and secure."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: rgba(255, 255, 255, 180); line-height: 1.3;")
        layout.addWidget(desc)

        author = QLabel("Created by William Horne")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author.setStyleSheet("color: rgba(255, 255, 255, 140);")
        layout.addWidget(author)

        version = QLabel(f"Version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: rgba(255, 255, 255, 140);")
        layout.addWidget(version)

        layout.addSpacing(4)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)
