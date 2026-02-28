"""
Main window for Cadence.
3-panel layout: folder tree | transcript list | transcript viewer.
"""

import time
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame, QSplitter,
    QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem,
    QInputDialog, QMenu, QMessageBox,
)
from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtGui import QFont, QTextCursor, QAction


class MainWindow(QMainWindow):
    """Main transcript window with 3-panel layout."""

    start_requested = Signal()
    stop_requested = Signal()
    folder_selected = Signal(str)
    transcript_selected = Signal(str, str)
    folder_created = Signal(str)
    folder_renamed = Signal(str, str)
    folder_deleted = Signal(str)
    transcript_renamed = Signal(str, str, str)
    transcript_deleted = Signal(str, str)
    transcript_moved = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadence")
        self.setMinimumSize(800, 500)
        self._recording = False
        self._start_time = 0
        self._current_folder = None
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Compact header: [BADGE] 00:00:00 · 0 words
        header = QHBoxLayout()
        header.setContentsMargins(2, 2, 2, 2)
        header.setSpacing(8)

        self.status_label = QLabel("READY")
        self.status_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(22)
        self._set_status_badge("idle")
        header.addWidget(self.status_label)

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setFont(QFont("Consolas", 11))
        header.addWidget(self.timer_label)

        sep = QLabel("\u00b7")
        sep.setStyleSheet("color: #888;")
        header.addWidget(sep)

        self.info_label = QLabel("0 words")
        self.info_label.setFont(QFont("Segoe UI", 9))
        self.info_label.setStyleSheet("color: #888;")
        header.addWidget(self.info_label)

        header.addStretch()
        layout.addLayout(header)

        # 3-panel splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Folder tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        folder_label = QLabel("Folders")
        folder_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        left_layout.addWidget(folder_label)
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setFont(QFont("Segoe UI", 9))
        self.folder_tree.itemClicked.connect(self._on_folder_clicked)
        self.folder_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.folder_tree.customContextMenuRequested.connect(self._on_folder_context_menu)
        left_layout.addWidget(self.folder_tree)
        self.add_folder_btn = QPushButton("+")
        self.add_folder_btn.setFixedHeight(24)
        self.add_folder_btn.setToolTip("New Folder")
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        left_layout.addWidget(self.add_folder_btn)
        self.splitter.addWidget(left_panel)

        # Middle panel - Transcript list
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(2)
        transcript_label = QLabel("Transcripts")
        transcript_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        mid_layout.addWidget(transcript_label)
        self.transcript_list = QListWidget()
        self.transcript_list.setFont(QFont("Segoe UI", 9))
        self.transcript_list.itemClicked.connect(self._on_transcript_clicked)
        self.transcript_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript_list.customContextMenuRequested.connect(self._on_transcript_context_menu)
        mid_layout.addWidget(self.transcript_list)
        self.splitter.addWidget(mid_panel)

        # Right panel - Transcript viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Segoe UI", 11))
        right_layout.addWidget(self.transcript_area)
        self.splitter.addWidget(right_panel)

        # Collapsible side panels, transcript viewer gets priority
        self.splitter.setCollapsible(0, True)   # folders
        self.splitter.setCollapsible(1, True)   # transcripts
        self.splitter.setCollapsible(2, False)  # viewer
        self.splitter.setSizes([140, 140, 520])
        layout.addWidget(self.splitter)

        # Bottom controls
        controls = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.setFont(QFont("Segoe UI", 11))
        self.record_btn.clicked.connect(self._on_record_clicked)
        controls.addWidget(self.record_btn)

        controls.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setFont(QFont("Segoe UI", 11))
        self.clear_btn.clicked.connect(self._on_clear)
        controls.addWidget(self.clear_btn)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setMinimumHeight(40)
        self.copy_btn.setFont(QFont("Segoe UI", 11))
        self.copy_btn.clicked.connect(self._on_copy)
        controls.addWidget(self.copy_btn)

        layout.addLayout(controls)

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer)

    # --- Status badge ---

    def _set_status_badge(self, state):
        """Set the status badge style. States: idle, recording, done."""
        styles = {
            "idle": (
                "background-color: #555; color: #ccc; border-radius: 4px; padding: 2px 12px;"
            ),
            "recording": (
                "background-color: #dc3232; color: white; border-radius: 4px; padding: 2px 12px;"
            ),
            "done": (
                "background-color: #2ea043; color: white; border-radius: 4px; padding: 2px 12px;"
            ),
        }
        self.status_label.setStyleSheet(styles.get(state, styles["idle"]))

    # --- Recording state ---

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

    # --- Transcript display ---

    def append_segment(self, speaker, text, timestamp=0.0):
        """Append a transcript segment with speaker label and timestamp."""
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if speaker == "you":
            label = "You"
            color = "#4a90d9"
        else:
            label = "Them"
            color = "#d94a4a"
        mins = int(timestamp) // 60
        secs = int(timestamp) % 60
        ts_str = f"{mins:02d}:{secs:02d}"
        cursor.insertHtml(
            f'<span style="color:#888; font-size:10px">[{ts_str}]</span> '
            f'<b style="color:{color}">{label}:</b> {text}<br>'
        )
        self.transcript_area.setTextCursor(cursor)
        self.transcript_area.ensureCursorVisible()
        self._update_word_count()

    def set_transcript(self, segments):
        """Replace transcript with full list of segments."""
        self.transcript_area.clear()
        for seg in segments:
            self.append_segment(seg["speaker"], seg["text"], seg.get("start", 0.0))

    def _update_word_count(self):
        text_content = self.transcript_area.toPlainText()
        word_count = len(text_content.split()) if text_content.strip() else 0
        self.info_label.setText(f"{word_count} words")

    # --- Clear / Copy ---

    def _on_clear(self):
        if not self.transcript_area.toPlainText().strip():
            return
        reply = QMessageBox.question(
            self, "Clear Transcript",
            "Are you sure you want to clear the transcript?")
        if reply == QMessageBox.StandardButton.Yes:
            self.transcript_area.clear()
            self._update_word_count()

    def _on_copy(self):
        from PySide6.QtWidgets import QApplication
        text = self.transcript_area.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    # --- Folder panel ---

    def set_folders(self, folders):
        """Populate the folder tree with a list of folder names."""
        self.folder_tree.clear()
        for name in folders:
            item = QTreeWidgetItem([name])
            self.folder_tree.addTopLevelItem(item)

    def _on_folder_clicked(self, item):
        folder_name = item.text(0)
        self._current_folder = folder_name
        self.folder_selected.emit(folder_name)

    def _on_add_folder(self):
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
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
            new_name, ok = QInputDialog.getText(self, "Rename Folder", "New name:", text=folder_name)
            if ok and new_name.strip() and new_name.strip() != folder_name:
                self.folder_renamed.emit(folder_name, new_name.strip())
        elif action == delete_action:
            reply = QMessageBox.question(self, "Delete Folder",
                f"Delete folder '{folder_name}' and all its transcripts?")
            if reply == QMessageBox.StandardButton.Yes:
                self.folder_deleted.emit(folder_name)

    # --- Transcript list panel ---

    def set_transcripts(self, transcripts):
        """Populate transcript list. transcripts is a list of dicts with 'name' and 'path'."""
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
            new_name, ok = QInputDialog.getText(self, "Rename Transcript", "New name:", text=name)
            if ok and new_name.strip() and new_name.strip() != name:
                self.transcript_renamed.emit(self._current_folder, name, new_name.strip())
        elif action == delete_action:
            reply = QMessageBox.question(self, "Delete Transcript",
                f"Delete transcript '{name}'?")
            if reply == QMessageBox.StandardButton.Yes:
                self.transcript_deleted.emit(self._current_folder, name)
        elif action == move_action:
            folders = []
            for i in range(self.folder_tree.topLevelItemCount()):
                f = self.folder_tree.topLevelItem(i).text(0)
                if f != self._current_folder:
                    folders.append(f)
            if not folders:
                QMessageBox.information(self, "Move", "No other folders available.")
                return
            dest, ok = QInputDialog.getItem(self, "Move to Folder", "Destination:", folders, 0, False)
            if ok and dest:
                self.transcript_moved.emit(self._current_folder, name, dest)

    # --- Timer ---

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
