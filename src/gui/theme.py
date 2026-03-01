"""
Centralized dark theme for Cadence.
Single source of truth — applied once at startup via apply_theme().
"""

from PySide6.QtWidgets import (
    QApplication, QDialog, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox,
)
from PySide6.QtGui import (
    QPalette, QColor, QPainter, QPen, QBrush, QPainterPath,
    QGuiApplication, QRegion, QFont, QPixmap, QIcon,
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPointF

# -- Color tokens (importable by other modules) ----------------------------
BG_PRIMARY = "#1a1a2e"
BG_SURFACE = "#2d2d4e"
BG_ELEVATED = "#252545"
BORDER = "#3d3d5c"
ACCENT = "#3498db"
ACCENT_HOVER = "#2980b9"
ACCENT_PRESSED = "#1f6da1"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "rgba(255, 255, 255, 180)"
TEXT_DISABLED = "rgba(255, 255, 255, 100)"

# -- Application-wide stylesheet -------------------------------------------
STYLESHEET = f"""
/* -- Base ------------------------------------------------- */
QWidget, QDialog {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "Calibri";
}}

/* -- Group boxes ------------------------------------------ */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_SECONDARY};
}}

/* -- Buttons ---------------------------------------------- */
QPushButton {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background-color: {BG_PRIMARY};
    border-color: {BORDER};
}}

/* -- Combo boxes ------------------------------------------ */
QComboBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    background: transparent;
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_PRIMARY};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRIMARY};
    color: {TEXT_PRIMARY};
    outline: none;
}}

/* -- Spin boxes ------------------------------------------- */
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_SURFACE};
    border: none;
    width: 16px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_PRIMARY};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_PRIMARY};
}}

/* -- Line edits ------------------------------------------- */
QLineEdit {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}

/* -- Lists ------------------------------------------------ */
QListWidget {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    outline: none;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}
QListWidget::item:hover {{
    background-color: {BORDER};
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}

/* -- Tree widgets ----------------------------------------- */
QTreeWidget {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    outline: none;
}}
QTreeWidget::item {{
    padding: 3px 6px;
    border-radius: 3px;
}}
QTreeWidget::item:hover {{
    background-color: {BORDER};
}}
QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}
QTreeWidget::branch {{
    background-color: transparent;
}}

/* -- Text edits ------------------------------------------- */
QTextEdit {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
}}

/* -- Dialog button boxes ---------------------------------- */
QDialogButtonBox QPushButton {{
    min-width: 70px;
}}

/* -- Radio buttons & checkboxes --------------------------- */
QRadioButton, QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 6px;
}}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_SURFACE};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* -- Progress bars ---------------------------------------- */
QProgressBar {{
    border: 2px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
}}
QProgressBar::chunk {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #4CAF50, stop:1 #8BC34A
    );
    border-radius: 3px;
}}

/* -- Context menus ---------------------------------------- */
QMenu {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    color: {TEXT_PRIMARY};
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}
QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 4px 8px;
}}

/* -- Message boxes ---------------------------------------- */
QMessageBox {{
    background-color: {BG_PRIMARY};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}

/* -- Splitter --------------------------------------------- */
QSplitter::handle {{
    background-color: {BORDER};
    width: 2px;
}}

/* -- Scroll bars ------------------------------------------ */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 4px 1px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: rgba(255, 255, 255, 40);
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: rgba(255, 255, 255, 70);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 1px 4px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background-color: rgba(255, 255, 255, 40);
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: rgba(255, 255, 255, 70);
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* -- Tooltips --------------------------------------------- */
QToolTip {{
    background-color: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
}}

/* -- Form labels ------------------------------------------ */
QLabel {{
    background-color: transparent;
}}
"""


def apply_theme(app: QApplication):
    """Apply the Cadence dark theme to the entire application."""
    app.setFont(QFont("Calibri"))
    app.setStyleSheet(STYLESHEET)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_PRIMARY))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_ELEVATED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(255, 255, 255, 100))
    app.setPalette(palette)


# -- Painted icons ---------------------------------------------------------

def _dialog_close_icon(size=14):
    """Create a painted X icon for dialog close buttons."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(255, 255, 255, 180))
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    m = size * 0.25
    p.drawLine(QPointF(m, m), QPointF(size - m, size - m))
    p.drawLine(QPointF(size - m, m), QPointF(m, size - m))
    p.end()
    return QIcon(pix)


# -- Rounded frameless dialog -----------------------------------------------

class RoundedDialog(QDialog):
    """QDialog with rounded corners and a custom draggable title bar."""

    CORNER_RADIUS = 12
    TITLE_BAR_HEIGHT = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_start = None
        self._centered = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._close_btn = QPushButton(self)
        self._close_btn.setIcon(_dialog_close_icon())
        self._close_btn.setObjectName("_rdClose")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.clicked.connect(self.reject)
        self._close_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._close_btn.setStyleSheet(f"""
            QPushButton#_rdClose {{
                background: transparent;
                border: none;
                border-radius: 14px;
            }}
            QPushButton#_rdClose:hover {{
                background-color: #e74c3c;
            }}
        """)

    def setLayout(self, layout):
        m = layout.contentsMargins()
        layout.setContentsMargins(
            max(m.left(), 12),
            self.TITLE_BAR_HEIGHT + 6,
            max(m.right(), 12),
            max(m.bottom(), 12),
        )
        super().setLayout(layout)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        r = self.rect().adjusted(0, 0, -1, -1)
        path.addRoundedRect(r, self.CORNER_RADIUS, self.CORNER_RADIUS)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QBrush(QColor(BG_PRIMARY)))
        p.drawPath(path)
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(1, self.TITLE_BAR_HEIGHT, self.width() - 2, self.TITLE_BAR_HEIGHT)
        f = p.font()
        f.setPixelSize(13)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT_PRIMARY))
        p.drawText(
            16, 0, self.width() - 60, self.TITLE_BAR_HEIGHT,
            Qt.AlignmentFlag.AlignVCenter, self.windowTitle(),
        )
        p.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._close_btn.move(self.width() - 40, (self.TITLE_BAR_HEIGHT - 28) // 2)
        self._close_btn.raise_()
        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(),
            self.CORNER_RADIUS, self.CORNER_RADIUS,
        )
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._centered:
            screen = None
            if self.parent():
                screen = self.parent().screen()
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                self.move(
                    geo.center().x() - self.width() // 2,
                    geo.center().y() - self.height() // 2,
                )
            self._centered = True
        self._close_btn.raise_()

    def mousePressEvent(self, event):
        if (event.position().y() < self.TITLE_BAR_HEIGHT
                and event.button() == Qt.MouseButton.LeftButton):
            self._drag_start = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_start)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)


# -- Rounded message boxes -------------------------------------------------

class MessageBox(RoundedDialog):
    """Drop-in replacement for QMessageBox using the rounded theme."""

    Yes = "yes"
    No = "no"

    def __init__(self, title, message, buttons=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self._result = None

        layout = QVBoxLayout()
        layout.setSpacing(16)

        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(msg_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if buttons is None:
            buttons = [("OK", self.accept)]

        for label, callback in buttons:
            btn = QPushButton(label)
            btn.setFixedWidth(80)
            btn.clicked.connect(callback)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    @staticmethod
    def flash(parent, title, message, duration_ms=2000):
        """Show a message briefly with fade in/out, then auto-dismiss."""
        FADE_IN = 150
        FADE_OUT = 300

        dlg = MessageBox(title, message, buttons=[], parent=parent)
        dlg.setWindowOpacity(0.0)

        def run():
            fi = QPropertyAnimation(dlg, b"windowOpacity")
            fi.setDuration(FADE_IN)
            fi.setStartValue(0.0)
            fi.setEndValue(1.0)
            fi.setEasingCurve(QEasingCurve.Type.OutQuad)
            dlg._fi = fi
            fi.start()

            def fade_out():
                fo = QPropertyAnimation(dlg, b"windowOpacity")
                fo.setDuration(FADE_OUT)
                fo.setStartValue(1.0)
                fo.setEndValue(0.0)
                fo.setEasingCurve(QEasingCurve.Type.InQuad)
                fo.finished.connect(dlg.accept)
                dlg._fo = fo
                fo.start()

            QTimer.singleShot(duration_ms, fade_out)

        QTimer.singleShot(0, run)
        dlg.exec()

    @staticmethod
    def information(parent, title, message):
        dlg = MessageBox(title, message, parent=parent)
        dlg.exec()

    @staticmethod
    def warning(parent, title, message):
        dlg = MessageBox(title, message, parent=parent)
        dlg.exec()

    @staticmethod
    def critical(parent, title, message):
        dlg = MessageBox(title, message, parent=parent)
        dlg.exec()

    @staticmethod
    def question(parent, title, message):
        result = [MessageBox.No]

        def on_yes():
            result[0] = MessageBox.Yes
            dlg.accept()

        def on_no():
            result[0] = MessageBox.No
            dlg.reject()

        dlg = MessageBox(
            title, message,
            buttons=[("Yes", on_yes), ("No", on_no)],
            parent=parent,
        )
        dlg.exec()
        return result[0]


class InputBox(RoundedDialog):
    """Drop-in replacement for QInputDialog using the rounded theme."""

    def __init__(self, title, prompt, text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self._value = ""
        self._accepted = False

        layout = QVBoxLayout()
        layout.setSpacing(12)

        lbl = QLabel(prompt)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 12px;")
        layout.addWidget(lbl)

        self._edit = QLineEdit()
        self._edit.setText(text)
        self._edit.selectAll()
        layout.addWidget(self._edit)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._edit.setFocus()
        self._edit.returnPressed.connect(self._on_ok)

    def _on_ok(self):
        self._value = self._edit.text()
        self._accepted = True
        self.accept()

    @staticmethod
    def getText(parent, title, prompt, text=""):
        """Show a themed text input dialog. Returns (text, ok)."""
        dlg = InputBox(title, prompt, text, parent=parent)
        dlg.exec()
        return dlg._value, dlg._accepted

    @staticmethod
    def getItem(parent, title, prompt, items, current=0, editable=False):
        """Show a themed combo-box selection dialog. Returns (text, ok)."""
        dlg = RoundedDialog(parent)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(350)
        result = [""]
        accepted = [False]

        layout = QVBoxLayout()
        layout.setSpacing(12)

        lbl = QLabel(prompt)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 12px;")
        layout.addWidget(lbl)

        combo = QComboBox()
        combo.addItems(items)
        if 0 <= current < len(items):
            combo.setCurrentIndex(current)
        combo.setEditable(editable)
        layout.addWidget(combo)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        def on_ok():
            result[0] = combo.currentText()
            accepted[0] = True
            dlg.accept()

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(dlg.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dlg.setLayout(layout)
        dlg.exec()
        return result[0], accepted[0]
