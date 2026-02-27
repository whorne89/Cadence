"""
Main window for Cadence.
Shows live transcript with speaker labels, recording controls, and timer.
"""

import time
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame,
)
from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtGui import QFont, QTextCursor


class MainWindow(QMainWindow):
    """Main transcript window with recording controls."""

    start_requested = Signal()
    stop_requested = Signal()
    reprocess_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadence")
        self.setMinimumSize(600, 500)
        self._recording = False
        self._start_time = 0
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header with timer and status
        header = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.addWidget(self.status_label)
        header.addStretch()
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setFont(QFont("Consolas", 14))
        header.addWidget(self.timer_label)
        layout.addLayout(header)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        # Transcript area
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self.transcript_area)

        # Controls
        controls = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.setFont(QFont("Segoe UI", 11))
        self.record_btn.clicked.connect(self._on_record_clicked)
        controls.addWidget(self.record_btn)

        self.reprocess_btn = QPushButton("Reprocess")
        self.reprocess_btn.setMinimumHeight(40)
        self.reprocess_btn.setFont(QFont("Segoe UI", 11))
        self.reprocess_btn.setEnabled(False)
        self.reprocess_btn.clicked.connect(self.reprocess_requested.emit)
        controls.addWidget(self.reprocess_btn)
        layout.addLayout(controls)

        # Word count
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.info_label)

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer)

    def _on_record_clicked(self):
        if self._recording:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_recording_state(self):
        self._recording = True
        self._start_time = time.time()
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet("background-color: #dc3232; color: white;")
        self.status_label.setText("Recording...")
        self.reprocess_btn.setEnabled(False)
        self.transcript_area.clear()
        self._timer.start(1000)

    def set_idle_state(self):
        self._recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("Ready")
        self._timer.stop()

    def set_processing_state(self):
        self._recording = False
        self.record_btn.setEnabled(False)
        self.status_label.setText("Reprocessing...")
        self._timer.stop()

    def set_done_state(self):
        self.record_btn.setEnabled(True)
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("Done")
        self.reprocess_btn.setEnabled(True)
        self._timer.stop()

    def append_segment(self, speaker, text):
        """Append a transcript segment with speaker label."""
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if speaker == "you":
            label = "You"
            color = "#4a90d9"
        else:
            label = "Them"
            color = "#d94a4a"
        cursor.insertHtml(f'<b style="color:{color}">{label}:</b> {text}<br>')
        self.transcript_area.setTextCursor(cursor)
        self.transcript_area.ensureCursorVisible()
        text_content = self.transcript_area.toPlainText()
        word_count = len(text_content.split())
        self.info_label.setText(f"{word_count} words")

    def set_transcript(self, segments):
        """Replace transcript with full list of segments."""
        self.transcript_area.clear()
        for seg in segments:
            self.append_segment(seg["speaker"], seg["text"])

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
