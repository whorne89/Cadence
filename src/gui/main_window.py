"""
Main window for Cadence.
Frameless rounded window with custom title bar and 3-panel layout.
"""

import time
from math import cos, sin, pi

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizeGrip,
    QPushButton, QTextEdit, QLabel, QSplitter,
    QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem,
    QMenu,
)
from PySide6.QtCore import Signal, QTimer, Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QFont, QTextCursor, QAction, QPainter, QPen, QBrush,
    QColor, QPainterPath, QRegion, QGuiApplication, QPixmap, QIcon,
)

from version import __version__
from gui.theme import (
    BG_PRIMARY, BG_SURFACE, BORDER, ACCENT,
    TEXT_PRIMARY, TEXT_SECONDARY, MessageBox, InputBox,
)

TITLE_BAR_HEIGHT = 32
CORNER_RADIUS = 8


# ── Painted icons ────────────────────────────────────────────────

def _paint_icon(size, draw_fn, color=TEXT_PRIMARY, pen_width=1.3):
    """Create a QIcon by painting with QPainter."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(pen_width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    draw_fn(p, float(size))
    p.end()
    return QIcon(pix)


def _icon_close(sz=12):
    def draw(p, s):
        m = s * 0.28
        p.drawLine(QPointF(m, m), QPointF(s - m, s - m))
        p.drawLine(QPointF(s - m, m), QPointF(m, s - m))
    return _paint_icon(sz, draw)


def _icon_minimize(sz=12):
    def draw(p, s):
        m = s * 0.25
        y = s * 0.5
        p.drawLine(QPointF(m, y), QPointF(s - m, y))
    return _paint_icon(sz, draw)


def _icon_maximize(sz=12):
    def draw(p, s):
        m = s * 0.24
        p.drawRect(QRectF(m, m, s - 2 * m, s - 2 * m))
    return _paint_icon(sz, draw)


def _icon_restore(sz=12):
    def draw(p, s):
        m = s * 0.18
        d = s * 0.22
        w = s * 0.44
        # back rectangle (partial, upper-right)
        p.drawPolyline([
            QPointF(m + d, m + w),
            QPointF(m + d, m),
            QPointF(m + d + w, m),
            QPointF(m + d + w, m + w),
            QPointF(m + w, m + w),
        ])
        # front rectangle (complete, lower-left)
        p.drawRect(QRectF(m, m + d, w, w))
    return _paint_icon(sz, draw)


def _icon_hamburger(sz=12):
    def draw(p, s):
        m = s * 0.2
        for frac in (0.26, 0.50, 0.74):
            y = s * frac
            p.drawLine(QPointF(m, y), QPointF(s - m, y))
    return _paint_icon(sz, draw)


def _icon_chevron_right(sz=12):
    def draw(p, s):
        mx = s * 0.38
        my = s * 0.22
        mid = s * 0.5
        p.drawLine(QPointF(mx, my), QPointF(s - mx, mid))
        p.drawLine(QPointF(s - mx, mid), QPointF(mx, s - my))
    return _paint_icon(sz, draw)


def _icon_gear(sz=14):
    def draw(p, s):
        c = s / 2.0
        outer = s * 0.44
        inner = s * 0.30
        teeth = 8
        step = pi / teeth
        path = QPainterPath()
        for i in range(teeth * 2):
            angle = i * step - pi / 2
            r = outer if i % 2 == 0 else inner
            x = c + r * cos(angle)
            y = c + r * sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        p.drawPath(path)
        # center hole
        p.drawEllipse(QPointF(c, c), s * 0.10, s * 0.10)
    return _paint_icon(sz, draw)


def _icon_plus(sz=10):
    def draw(p, s):
        m = s * 0.2
        mid = s / 2.0
        p.drawLine(QPointF(mid, m), QPointF(mid, s - m))
        p.drawLine(QPointF(m, mid), QPointF(s - m, mid))
    return _paint_icon(sz, draw, pen_width=1.5)


def _icon_sort_desc(sz=12):
    """Down arrow — newest first."""
    def draw(p, s):
        m = s * 0.25
        mid = s * 0.5
        # Arrow shaft
        p.drawLine(QPointF(mid, m), QPointF(mid, s - m))
        # Arrow head
        p.drawLine(QPointF(mid, s - m), QPointF(m + s * 0.05, s * 0.55))
        p.drawLine(QPointF(mid, s - m), QPointF(s - m - s * 0.05, s * 0.55))
    return _paint_icon(sz, draw, pen_width=1.5)


def _icon_sort_asc(sz=12):
    """Up arrow — oldest first."""
    def draw(p, s):
        m = s * 0.25
        mid = s * 0.5
        # Arrow shaft
        p.drawLine(QPointF(mid, m), QPointF(mid, s - m))
        # Arrow head
        p.drawLine(QPointF(mid, m), QPointF(m + s * 0.05, s * 0.45))
        p.drawLine(QPointF(mid, m), QPointF(s - m - s * 0.05, s * 0.45))
    return _paint_icon(sz, draw, pen_width=1.5)


# ── Draggable title bar ─────────────────────────────────────────

class _TitleBar(QWidget):
    """Draggable title bar that moves its parent window."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._drag_pos = None
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        self.setObjectName("titleBar")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._window.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            if self._window._maximized:
                old_w = self._window.width()
                self._window._toggle_maximize()
                mouse = event.globalPosition().toPoint()
                new_w = self._window.width()
                ratio = min(event.position().x() / old_w, 1.0) if old_w else 0.5
                self._window.move(
                    int(mouse.x() - new_w * ratio),
                    mouse.y() - self.height() // 2,
                )
                self._drag_pos = mouse - self._window.pos()
            else:
                self._window.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self._window._toggle_maximize()


# ── Main window ──────────────────────────────────────────────────

class MainWindow(QWidget):
    """Main transcript window with 3-panel layout and custom title bar."""

    start_requested = Signal()
    stop_requested = Signal()
    settings_requested = Signal()
    folder_selected = Signal(str)
    transcript_selected = Signal(str, str)
    folder_created = Signal(str)
    folder_renamed = Signal(str, str)
    folder_deleted = Signal(str)
    transcript_renamed = Signal(str, str, str)
    transcript_deleted = Signal(str, str)
    transcript_moved = Signal(str, str, str)
    sort_order_changed = Signal(bool)  # True = descending (newest first)
    participant_changed = Signal(str, str)  # filepath, participant_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadence")
        self.setMinimumSize(800, 500)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._recording = False
        self._start_time = 0
        self._current_folder = None
        self._maximized = False
        self._normal_geometry = None
        self._centered = False
        self._sort_descending = True
        self._current_segments = []

        self._setup_ui()
        self._setup_timer()

    # ── UI setup ─────────────────────────────────────────────────

    def _make_icon_btn(self, icon, size, tooltip, css, slot):
        """Create a QPushButton with a painted icon."""
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setIconSize(QSize(size[0] - 12, size[1] - 10))
        btn.setFixedSize(*size)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(css)
        btn.clicked.connect(slot)
        return btn

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Build icons ---
        self._ico_hamburger = _icon_hamburger()
        self._ico_chevron = _icon_chevron_right()
        self._ico_maximize = _icon_maximize()
        self._ico_restore = _icon_restore()

        # --- Reusable style snippets ---
        win_btn_css = (
            f"QPushButton {{ background:transparent; border:none; border-radius:4px; }}"
            f"QPushButton:hover {{ background-color:{BG_SURFACE}; }}"
        )
        close_css = (
            f"QPushButton {{ background:transparent; border:none; border-radius:4px; }}"
            f"QPushButton:hover {{ background-color:#e74c3c; }}"
        )
        flat_css = (
            f"QPushButton {{ background:transparent; border:none; }}"
            f"QPushButton:hover {{ background-color:{BG_SURFACE}; border-radius:3px; }}"
        )

        # --- Custom title bar ---
        title_bar = _TitleBar(self)
        title_bar.setStyleSheet("QWidget#titleBar { background:transparent; }")
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(12, 0, 4, 0)
        tb.setSpacing(2)

        title_label = QLabel("Cadence")
        title_label.setFont(QFont("Calibri", 11, QFont.Weight.Bold))
        tb.addWidget(title_label)
        tb.addStretch()

        self._min_btn = self._make_icon_btn(
            _icon_minimize(), (36, 28), "Minimize", win_btn_css, self.showMinimized)
        tb.addWidget(self._min_btn)

        self._max_btn = self._make_icon_btn(
            self._ico_maximize, (36, 28), "Maximize", win_btn_css, self._toggle_maximize)
        tb.addWidget(self._max_btn)

        self._close_btn = self._make_icon_btn(
            _icon_close(), (36, 28), "Close", close_css, self.close)
        tb.addWidget(self._close_btn)

        root.addWidget(title_bar)

        # --- Top bar: [hamburger] [BADGE] 00:00:00 · 0 words … [gear] ---
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(6, 0, 6, 2)
        top_bar.setSpacing(8)

        self.sidebar_btn = QPushButton()
        self.sidebar_btn.setIcon(self._ico_hamburger)
        self.sidebar_btn.setIconSize(QSize(12, 12))
        self.sidebar_btn.setFixedSize(22, 20)
        self.sidebar_btn.setToolTip("Toggle sidebar")
        self.sidebar_btn.setStyleSheet(flat_css)
        self.sidebar_btn.clicked.connect(self._toggle_sidebar)
        top_bar.addWidget(self.sidebar_btn)

        self.status_label = QLabel("READY")
        self.status_label.setFont(QFont("Calibri", 8, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(18)
        self._set_status_badge("idle")
        top_bar.addWidget(self.status_label)

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setFont(QFont("Consolas", 9))
        top_bar.addWidget(self.timer_label)

        sep = QLabel("\u00b7")
        sep.setStyleSheet(f"color:{TEXT_SECONDARY};")
        top_bar.addWidget(sep)

        self.info_label = QLabel("0 words")
        self.info_label.setFont(QFont("Calibri", 8))
        self.info_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
        top_bar.addWidget(self.info_label)

        top_bar.addStretch()

        self.settings_btn = QPushButton()
        self.settings_btn.setIcon(_icon_gear())
        self.settings_btn.setIconSize(QSize(14, 14))
        self.settings_btn.setFixedSize(24, 22)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setStyleSheet(flat_css)
        self.settings_btn.clicked.connect(self.settings_requested.emit)
        top_bar.addWidget(self.settings_btn)

        root.addLayout(top_bar)

        # --- 3-panel splitter ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self._sidebar_visible = True

        # Left panel – Folder tree
        self.left_panel = QWidget()
        ll = QVBoxLayout(self.left_panel)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        fh = QHBoxLayout()
        fh.setContentsMargins(4, 2, 4, 2)
        fl = QLabel("Folders")
        fl.setFont(QFont("Calibri", 9, QFont.Weight.Bold))
        fh.addWidget(fl)
        fh.addStretch()

        self.add_folder_btn = QPushButton()
        self.add_folder_btn.setIcon(_icon_plus())
        self.add_folder_btn.setIconSize(QSize(10, 10))
        self.add_folder_btn.setFixedSize(20, 18)
        self.add_folder_btn.setToolTip("New Folder")
        self.add_folder_btn.setStyleSheet(flat_css)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        fh.addWidget(self.add_folder_btn)
        ll.addLayout(fh)

        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setFont(QFont("Calibri", 9))
        self.folder_tree.setIndentation(12)
        self.folder_tree.itemClicked.connect(self._on_folder_clicked)
        self.folder_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.folder_tree.customContextMenuRequested.connect(self._on_folder_context_menu)
        ll.addWidget(self.folder_tree)
        self.splitter.addWidget(self.left_panel)

        # Middle panel – Transcript list
        self.mid_panel = QWidget()
        ml = QVBoxLayout(self.mid_panel)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        th = QHBoxLayout()
        th.setContentsMargins(4, 2, 4, 2)
        tl = QLabel("Transcripts")
        tl.setFont(QFont("Calibri", 9, QFont.Weight.Bold))
        th.addWidget(tl)
        th.addStretch()

        self._ico_sort_desc = _icon_sort_desc()
        self._ico_sort_asc = _icon_sort_asc()
        self.sort_btn = QPushButton()
        self.sort_btn.setIcon(self._ico_sort_desc)
        self.sort_btn.setIconSize(QSize(10, 10))
        self.sort_btn.setFixedSize(20, 18)
        self.sort_btn.setToolTip("Sort: Newest first")
        self.sort_btn.setStyleSheet(flat_css)
        self.sort_btn.clicked.connect(self._on_sort_toggle)
        th.addWidget(self.sort_btn)
        ml.addLayout(th)

        self.transcript_list = QListWidget()
        self.transcript_list.setFont(QFont("Calibri", 9))
        self.transcript_list.itemClicked.connect(self._on_transcript_clicked)
        self.transcript_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript_list.customContextMenuRequested.connect(self._on_transcript_context_menu)
        ml.addWidget(self.transcript_list)
        self.splitter.addWidget(self.mid_panel)

        # Right panel – Transcript viewer
        rp = QWidget()
        rl = QVBoxLayout(rp)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # "Meeting with" header
        participant_row = QHBoxLayout()
        participant_row.setContentsMargins(4, 2, 4, 2)
        self.participant_btn = QPushButton("Meeting with... (click to set)")
        self.participant_btn.setFont(QFont("Calibri", 9))
        self.participant_btn.setToolTip("Click to set who this meeting is with")
        self.participant_btn.setStyleSheet(
            f"QPushButton {{ background-color:{BG_SURFACE}; border:1px solid {BORDER}; "
            f"border-radius:4px; color:{TEXT_SECONDARY}; "
            f"text-align:left; padding:3px 8px; }}"
            f"QPushButton:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}"
        )
        self.participant_btn.clicked.connect(self._on_participant_clicked)
        participant_row.addWidget(self.participant_btn)
        participant_row.addStretch()
        rl.addLayout(participant_row)

        self._current_transcript_path = None

        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Calibri", 11))
        rl.addWidget(self.transcript_area)
        self.splitter.addWidget(rp)

        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.splitter.setSizes([130, 130, 540])
        self.splitter.setHandleWidth(2)
        root.addWidget(self.splitter)

        # --- Bottom controls ---
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 2, 4, 4)
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(32)
        self.record_btn.setFont(QFont("Calibri", 10))
        self.record_btn.clicked.connect(self._on_record_clicked)
        controls.addWidget(self.record_btn)
        controls.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMinimumHeight(32)
        self.clear_btn.setFont(QFont("Calibri", 10))
        self.clear_btn.clicked.connect(self._on_clear)
        controls.addWidget(self.clear_btn)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setMinimumHeight(32)
        self.copy_btn.setFont(QFont("Calibri", 10))
        self.copy_btn.clicked.connect(self._on_copy)
        controls.addWidget(self.copy_btn)

        root.addLayout(controls)

        # Resize grip (bottom-right corner)
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(14, 14)
        self._grip.setStyleSheet("background:transparent;")

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer)

    # ── Frameless window drawing ─────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = CORNER_RADIUS if not self._maximized else 0
        path = QPainterPath()
        rect = self.rect().adjusted(0, 0, -1, -1)
        if r > 0:
            path.addRoundedRect(rect, r, r)
        else:
            path.addRect(rect)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QBrush(QColor(BG_PRIMARY)))
        p.drawPath(path)
        # title bar divider
        p.drawLine(1, TITLE_BAR_HEIGHT, self.width() - 2, TITLE_BAR_HEIGHT)
        p.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._maximized:
            path = QPainterPath()
            path.addRoundedRect(
                0, 0, self.width(), self.height(),
                CORNER_RADIUS, CORNER_RADIUS,
            )
            self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        else:
            self.clearMask()
        self._grip.move(self.width() - 14, self.height() - 14)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._centered:
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                self.move(
                    geo.center().x() - self.width() // 2,
                    geo.center().y() - self.height() // 2,
                )
            self._centered = True

    def _toggle_maximize(self):
        if self._maximized:
            self._maximized = False
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
            self._max_btn.setIcon(self._ico_maximize)
        else:
            self._normal_geometry = self.geometry()
            self._maximized = True
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen:
                self.setGeometry(screen.availableGeometry())
            self._max_btn.setIcon(self._ico_restore)

    # ── Sidebar toggle ───────────────────────────────────────────

    def _toggle_sidebar(self):
        self._sidebar_visible = not self._sidebar_visible
        self.left_panel.setVisible(self._sidebar_visible)
        self.mid_panel.setVisible(self._sidebar_visible)
        self.sidebar_btn.setIcon(
            self._ico_hamburger if self._sidebar_visible else self._ico_chevron
        )

    # ── Status badge ─────────────────────────────────────────────

    def _set_status_badge(self, state):
        styles = {
            "idle": (
                f"background-color:{BG_SURFACE}; color:{TEXT_SECONDARY};"
                f" border:1px solid {BORDER}; border-radius:4px; padding:2px 12px;"
            ),
            "recording": (
                "background-color:#dc3232; color:white;"
                " border:1px solid #dc3232; border-radius:4px; padding:2px 12px;"
            ),
            "done": (
                "background-color:#2ea043; color:white;"
                " border:1px solid #2ea043; border-radius:4px; padding:2px 12px;"
            ),
            "processing": (
                "background-color:#d29922; color:white;"
                " border:1px solid #d29922; border-radius:4px; padding:2px 12px;"
            ),
        }
        self.status_label.setStyleSheet(styles.get(state, styles["idle"]))

    # ── Recording state ──────────────────────────────────────────

    def _on_record_clicked(self):
        if self._recording:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_recording_state(self):
        self._recording = True
        self._start_time = time.time()
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet("background-color:#dc3232; color:white;")
        self.status_label.setText("RECORDING")
        self._set_status_badge("recording")
        self._timer.start(1000)

    def set_idle_state(self):
        self._recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("READY")
        self._set_status_badge("idle")
        self._timer.stop()

    def set_done_state(self):
        self._recording = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("DONE")
        self._set_status_badge("done")
        self._timer.stop()

    def set_processing_state(self, message="Processing..."):
        self._recording = False
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Processing...")
        self.status_label.setText(message)
        self._set_status_badge("processing")
        self._timer.stop()

    # ── Transcript display ───────────────────────────────────────

    def append_segment(self, text, timestamp=0.0):
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        mins = int(timestamp) // 60
        secs = int(timestamp) % 60
        ts_str = f"{mins:02d}:{secs:02d}"
        cursor.insertHtml(
            f'<span style="color:{TEXT_SECONDARY}; font-size:10px">[{ts_str}]</span> '
            f'{text}<br>'
        )
        self.transcript_area.setTextCursor(cursor)
        self.transcript_area.ensureCursorVisible()
        self._update_word_count()

    def set_transcript(self, segments):
        self._current_segments = list(segments)
        self.transcript_area.clear()
        for seg in segments:
            self.append_segment(seg["text"], seg.get("start", 0.0))

    def _update_word_count(self):
        text_content = self.transcript_area.toPlainText()
        word_count = len(text_content.split()) if text_content.strip() else 0
        self.info_label.setText(f"{word_count} words")

    def set_transcript_meta(self, filepath, participant=""):
        """Set metadata for the currently displayed transcript."""
        self._current_transcript_path = filepath
        if participant:
            self.participant_btn.setText(f"Meeting with {participant}")
        else:
            self.participant_btn.setText("Meeting with... (click to set)")

    def _on_participant_clicked(self):
        current = self.participant_btn.text()
        if current.startswith("Meeting with ") and not current.endswith("(click to set)"):
            existing = current[len("Meeting with "):]
        else:
            existing = ""
        name, ok = InputBox.getText(
            self, "Meeting Participant", "Who is this meeting with?",
            text=existing)
        if ok and self._current_transcript_path:
            display = name.strip()
            if display:
                self.participant_btn.setText(f"Meeting with {display}")
            else:
                self.participant_btn.setText("Meeting with... (click to set)")
            self.participant_changed.emit(self._current_transcript_path, display)

    # ── Clear / Copy ─────────────────────────────────────────────

    def _on_clear(self):
        if not self.transcript_area.toPlainText().strip():
            return
        reply = MessageBox.question(
            self, "Clear Transcript",
            "Are you sure you want to clear the transcript?")
        if reply == MessageBox.Yes:
            self.transcript_area.clear()
            self._update_word_count()

    def _on_copy(self):
        from PySide6.QtWidgets import QApplication
        text = self.transcript_area.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    # ── Folder panel ─────────────────────────────────────────────

    def set_folders(self, folders):
        self.folder_tree.clear()
        for name in folders:
            item = QTreeWidgetItem([name])
            self.folder_tree.addTopLevelItem(item)

    def _on_folder_clicked(self, item):
        folder_name = item.text(0)
        self._current_folder = folder_name
        self.folder_selected.emit(folder_name)

    def _on_add_folder(self):
        name, ok = InputBox.getText(self, "New Folder", "Folder name:")
        if ok and name.strip():
            self.folder_created.emit(name.strip())

    def _on_folder_context_menu(self, pos):
        item = self.folder_tree.itemAt(pos)
        if item is None:
            return
        folder_name = item.text(0)
        menu = QMenu(self)
        rename_action = QAction("Rename", self)
        delete_action = QAction("Delete", self)
        menu.addAction(rename_action)
        menu.addAction(delete_action)

        action = menu.exec(self.folder_tree.mapToGlobal(pos))
        if action == rename_action:
            new_name, ok = InputBox.getText(
                self, "Rename Folder", "New name:", text=folder_name)
            if ok and new_name.strip() and new_name.strip() != folder_name:
                self.folder_renamed.emit(folder_name, new_name.strip())
        elif action == delete_action:
            reply = MessageBox.question(
                self, "Delete Folder",
                f"Delete folder '{folder_name}' and all its transcripts?")
            if reply == MessageBox.Yes:
                self.folder_deleted.emit(folder_name)

    # ── Transcript list panel ────────────────────────────────────

    def _on_sort_toggle(self):
        self._sort_descending = not self._sort_descending
        if self._sort_descending:
            self.sort_btn.setIcon(self._ico_sort_desc)
            self.sort_btn.setToolTip("Sort: Newest first")
        else:
            self.sort_btn.setIcon(self._ico_sort_asc)
            self.sort_btn.setToolTip("Sort: Oldest first")
        self.sort_order_changed.emit(self._sort_descending)

    def set_transcripts(self, transcripts):
        self.transcript_list.clear()
        for t in transcripts:
            item = QListWidgetItem(t["name"])
            item.setData(Qt.ItemDataRole.UserRole, t["path"])
            self.transcript_list.addItem(item)

    def _on_transcript_clicked(self, item):
        if self._current_folder:
            name = item.text()
            self.transcript_selected.emit(self._current_folder, name)

    def _on_transcript_context_menu(self, pos):
        item = self.transcript_list.itemAt(pos)
        if item is None or self._current_folder is None:
            return
        name = item.text()
        menu = QMenu(self)
        rename_action = QAction("Rename", self)
        delete_action = QAction("Delete", self)
        move_action = QAction("Move to...", self)
        menu.addAction(rename_action)
        menu.addAction(move_action)
        menu.addAction(delete_action)

        action = menu.exec(self.transcript_list.mapToGlobal(pos))
        if action == rename_action:
            new_name, ok = InputBox.getText(
                self, "Rename Transcript", "New name:", text=name)
            if ok and new_name.strip() and new_name.strip() != name:
                self.transcript_renamed.emit(self._current_folder, name, new_name.strip())
        elif action == delete_action:
            reply = MessageBox.question(
                self, "Delete Transcript", f"Delete transcript '{name}'?")
            if reply == MessageBox.Yes:
                self.transcript_deleted.emit(self._current_folder, name)
        elif action == move_action:
            folders = []
            for i in range(self.folder_tree.topLevelItemCount()):
                f = self.folder_tree.topLevelItem(i).text(0)
                if f != self._current_folder:
                    folders.append(f)
            if not folders:
                MessageBox.information(self, "Move", "No other folders available.")
                return
            dest, ok = InputBox.getItem(
                self, "Move to Folder", "Destination:", folders, 0, False)
            if ok and dest:
                self.transcript_moved.emit(self._current_folder, name, dest)

    # ── Timer ────────────────────────────────────────────────────

    def _update_timer(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def closeEvent(self, event):
        """Hide instead of close (stay in tray)."""
        event.ignore()
        self.hide()
