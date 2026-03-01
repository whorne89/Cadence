"""
Toast notification overlay for Cadence.
Auto-dismissing dark pill toast positioned at the bottom-right of the screen.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QFont, QFontMetrics, QGuiApplication


class ToastNotification(QWidget):
    """
    Frameless, transparent, always-on-top toast notification.

    Shows a dark pill with title and message text.
    Auto-dismisses after 3 seconds with fade animation.
    """

    WIDTH = 320
    BASE_HEIGHT = 80
    RADIUS = 12
    MARGIN = 20

    BG_COLOR = QColor(26, 26, 46, 217)
    BORDER_COLOR = QColor(45, 45, 78, 128)
    ACCENT_COLOR = QColor(52, 152, 219)

    HOLD_MS = 3000
    FADE_IN_MS = 200
    FADE_OUT_MS = 300

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.BASE_HEIGHT)

        self._message = ""
        self._details = ""
        self._height = self.BASE_HEIGHT
        self._fade_anim = None
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._fade_out)

    def _position_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + geom.width() - self.WIDTH - self.MARGIN
            y = geom.y() + geom.height() - self._height - self.MARGIN
            self.move(x, y)

    def show_toast(self, message, details=""):
        self._message = message
        self._details = details

        body_font = QFont()
        body_font.setPixelSize(13)
        fm = QFontMetrics(body_font)
        content_x = 16
        max_width = self.WIDTH - content_x - 16

        msg_rect = fm.boundingRect(
            0, 0, max_width, 999,
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            message,
        )
        total_body = msg_rect.height()

        if details:
            bold_font = QFont()
            bold_font.setPixelSize(13)
            bold_font.setBold(True)
            bfm = QFontMetrics(bold_font)
            det_rect = bfm.boundingRect(
                0, 0, max_width, 999,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                details,
            )
            total_body += 4 + det_rect.height()

        self._height = max(self.BASE_HEIGHT, 44 + total_body + 12)
        self.setFixedSize(self.WIDTH, self._height)

        self._stop_fade()
        self._hold_timer.stop()

        self._position_on_screen()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.update()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(self.FADE_IN_MS)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_anim.finished.connect(self._start_hold)
        self._fade_anim.start()

    def _start_hold(self):
        self._hold_timer.start(self.HOLD_MS)

    def _fade_out(self):
        self._stop_fade()
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(self.FADE_OUT_MS)
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()

    def _stop_fade(self):
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        h = self._height
        path = QPainterPath()
        path.addRoundedRect(
            0.5, 0.5,
            self.WIDTH - 1, h - 1,
            self.RADIUS, self.RADIUS,
        )

        painter.setPen(QPen(self.BORDER_COLOR, 1))
        painter.setBrush(QBrush(self.BG_COLOR))
        painter.drawPath(path)

        # Left accent line
        painter.setClipPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.ACCENT_COLOR))
        painter.drawRoundedRect(0, 0, 3, h, 1, 1)
        painter.setClipping(False)

        # Header: "Cadence"
        header_y = 22
        content_x = 16

        title_font = QFont()
        title_font.setPixelSize(15)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255, 220))
        painter.drawText(content_x, header_y, "Cadence")

        # Message body
        body_font = QFont()
        body_font.setPixelSize(12)
        painter.setFont(body_font)
        painter.setPen(QColor(255, 255, 255, 190))

        max_width = self.WIDTH - content_x - 16
        body_top = header_y + 14
        msg_rect = QRect(content_x, body_top, max_width, self._height - body_top)
        painter.drawText(msg_rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, self._message)

        # Bold details below message
        if self._details:
            fm = QFontMetrics(body_font)
            msg_bound = fm.boundingRect(
                0, 0, max_width, 999,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                self._message,
            )
            det_top = body_top + msg_bound.height() + 4

            bold_font = QFont()
            bold_font.setPixelSize(13)
            bold_font.setBold(True)
            painter.setFont(bold_font)
            painter.setPen(QColor(255, 255, 255, 230))

            det_rect = QRect(content_x, det_top, max_width, self._height - det_top)
            painter.drawText(det_rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, self._details)

        painter.end()
