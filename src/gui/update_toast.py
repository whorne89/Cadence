"""
Interactive update toast notification for Cadence.
Shows a Yes/No prompt with changelog when a new version is available.
"""

from PySide6.QtWidgets import QWidget, QPushButton, QScrollArea, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QRect
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPainterPath, QFont, QFontMetrics, QGuiApplication,
)

from gui.theme import (
    BG_PRIMARY, BG_SURFACE, BORDER, ACCENT, ACCENT_HOVER,
    TEXT_PRIMARY, TEXT_SECONDARY,
)


class UpdateToast(QWidget):
    """
    Interactive toast notification for update prompts.

    Same visual style as ToastNotification (dark pill, bottom-right, accent bar,
    "Cadence" header) but with clickable Yes/No buttons and a scrollable
    changelog area.

    Auto-dismisses after 15 seconds if user doesn't interact.
    """

    accepted = Signal()
    dismissed = Signal()

    # Dimensions
    WIDTH = 380
    HEIGHT = 280
    RADIUS = 12
    MARGIN = 20

    # Colors (matching ToastNotification)
    BG_COLOR = QColor(26, 26, 46, 230)
    BORDER_COLOR = QColor(45, 45, 78, 128)
    ACCENT_COLOR = QColor(52, 152, 219)

    # Timing
    AUTO_DISMISS_MS = 15000
    FADE_IN_MS = 200
    FADE_OUT_MS = 300

    def __init__(self, version_str, release_body="", parent=None):
        super().__init__(parent)

        self._version = version_str
        self._release_body = release_body

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

        # --- Changelog scroll area ---
        # Positioned inside the painted toast area
        changelog_top = 72
        changelog_height = self.HEIGHT - changelog_top - 50  # room for buttons

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setGeometry(16, changelog_top, self.WIDTH - 32, changelog_height)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet(
            f"QScrollArea {{ background-color: {BG_SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px; }}"
            f"QScrollArea > QWidget > QWidget {{ background-color: {BG_SURFACE}; }}"
        )

        changelog_label = QLabel(self._format_changelog(release_body))
        changelog_label.setWordWrap(True)
        changelog_label.setTextFormat(Qt.TextFormat.PlainText)
        changelog_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        changelog_label.setStyleSheet(
            f"QLabel {{ color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background-color: {BG_SURFACE}; padding: 6px; }}"
        )
        self._scroll_area.setWidget(changelog_label)

        # --- Buttons ---
        self._yes_btn = QPushButton("Update", self)
        self._yes_btn.setFixedSize(72, 28)
        self._yes_btn.setStyleSheet(
            f"QPushButton {{ background-color: #2ecc71; color: #fff; border: none;"
            f" border-radius: 4px; font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: #27ae60; }}"
        )
        self._yes_btn.clicked.connect(self._on_accept)

        self._no_btn = QPushButton("Dismiss", self)
        self._no_btn.setFixedSize(72, 28)
        self._no_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BG_SURFACE}; color: {TEXT_SECONDARY}; border: none;"
            f" border-radius: 4px; font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {BORDER}; }}"
        )
        self._no_btn.clicked.connect(self._on_dismiss)

        # Position buttons at the bottom-right of the toast
        btn_y = self.HEIGHT - 38
        self._no_btn.move(self.WIDTH - 16 - 72, btn_y)
        self._yes_btn.move(self.WIDTH - 16 - 72 - 8 - 72, btn_y)

        # Animation
        self._fade_anim = None

        # Auto-dismiss timer
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._on_dismiss)

    @staticmethod
    def _format_changelog(body):
        """Clean up GitHub release markdown for display as plain text."""
        if not body:
            return "(no release notes)"
        # Strip common markdown formatting for plain-text display
        lines = body.strip().splitlines()
        cleaned = []
        for line in lines:
            # Remove markdown header markers
            line = line.lstrip("#").strip()
            cleaned.append(line)
        return "\n".join(cleaned)

    def show_toast(self):
        """Show the update toast with fade-in and start auto-dismiss timer."""
        self._position_on_screen()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(self.FADE_IN_MS)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_anim.finished.connect(lambda: self._dismiss_timer.start(self.AUTO_DISMISS_MS))
        self._fade_anim.start()

    def _position_on_screen(self):
        """Position at bottom-right of primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + geom.width() - self.WIDTH - self.MARGIN
            y = geom.y() + geom.height() - self.HEIGHT - self.MARGIN
            self.move(x, y)

    def _on_accept(self):
        self._dismiss_timer.stop()
        self._fade_out(self.accepted)

    def _on_dismiss(self):
        self._dismiss_timer.stop()
        self._fade_out(self.dismissed)

    def _fade_out(self, signal_to_emit):
        """Fade out then emit the given signal."""
        if self._fade_anim is not None:
            self._fade_anim.stop()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(self.FADE_OUT_MS)
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.finished.connect(signal_to_emit.emit)
        self._fade_anim.start()

    def paintEvent(self, event):
        """Draw the toast background, accent bar, header, and subtitle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rounded rectangle
        path = QPainterPath()
        path.addRoundedRect(
            0.5, 0.5,
            self.WIDTH - 1, self.HEIGHT - 1,
            self.RADIUS, self.RADIUS,
        )

        painter.setPen(QPen(self.BORDER_COLOR, 1))
        painter.setBrush(QBrush(self.BG_COLOR))
        painter.drawPath(path)

        # Left accent bar
        painter.setClipPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.ACCENT_COLOR))
        painter.drawRoundedRect(0, 0, 3, self.HEIGHT, 1, 1)
        painter.setClipping(False)

        # Header: "Cadence"
        content_x = 16
        header_y = 22

        title_font = QFont()
        title_font.setPixelSize(15)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255, 220))
        painter.drawText(content_x, header_y, "Cadence")

        # Subtitle: version + instruction
        body_font = QFont()
        body_font.setPixelSize(12)
        painter.setFont(body_font)
        painter.setPen(QColor(255, 255, 255, 190))

        subtitle = f"Version {self._version} is available. Update now?"
        max_width = self.WIDTH - content_x - 16
        subtitle_top = header_y + 10
        msg_rect = QRect(content_x, subtitle_top, max_width, 30)
        painter.drawText(msg_rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, subtitle)

        # "What's new:" label
        label_font = QFont()
        label_font.setPixelSize(11)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(QColor(255, 255, 255, 160))
        painter.drawText(content_x, subtitle_top + 32, "What's new:")

        painter.end()
