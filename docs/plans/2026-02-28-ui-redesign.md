# UI Redesign & Storage Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign Cadence from a single-panel transcript viewer to a 3-panel layout with folder navigation, transcript list, and viewer; switch storage from JSON+WAV to plain .txt; remove audio reprocessing; add clear/copy buttons and configurable transcription interval.

**Architecture:** The main window becomes a QSplitter-based 3-panel layout. SessionManager is rewritten to manage folders and .txt files on disk. ReprocessWorker and all audio-saving code are removed. Config gains a transcription_interval setting. Settings dialog is simplified.

**Tech Stack:** Python 3.12, PySide6 (QSplitter, QTreeWidget, QListWidget, QTextEdit), pathlib for file ops.

---

### Task 1: Update ConfigManager — remove reprocess, add interval

**Files:**
- Modify: `src/utils/config.py`
- Test: `tests/test_config.py`

**Step 1: Update the test file to cover new config keys**

Add test for `transcription_interval` and remove reprocess model test:

```python
# In tests/test_config.py, add to existing test_defaults or add new test:

def test_transcription_interval_default(tmp_path):
    config = ConfigManager(config_file=tmp_path / "settings.json")
    assert config.get_transcription_interval() == 5.0

def test_transcription_interval_range(tmp_path):
    config = ConfigManager(config_file=tmp_path / "settings.json")
    config.set("whisper", "transcription_interval", value=2.0)
    assert config.get_transcription_interval() == 2.0
    config.set("whisper", "transcription_interval", value=8.0)
    assert config.get_transcription_interval() == 8.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `get_transcription_interval` not defined

**Step 3: Update ConfigManager**

In `src/utils/config.py`:

- In `DEFAULT_CONFIG["whisper"]`: remove `"reprocess_model_size": "small"`, add `"transcription_interval": 5.0`
- In `DEFAULT_CONFIG["session"]`: remove `"auto_reprocess": False`, remove `"save_audio": True`
- Remove `get_reprocess_model_size()` method
- Add `get_transcription_interval()` method:

```python
def get_transcription_interval(self):
    return self.get("whisper", "transcription_interval", default=5.0)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/config.py tests/test_config.py
git commit -m "feat: add transcription_interval config, remove reprocess config"
```

---

### Task 2: Rewrite SessionManager for .txt files and folder management

**Files:**
- Modify: `src/core/session_manager.py`
- Modify: `tests/test_session_manager.py`

**Step 1: Write failing tests for new SessionManager**

Replace `tests/test_session_manager.py` with tests for:

```python
import pytest
from pathlib import Path
from core.session_manager import SessionManager

@pytest.fixture
def sm(tmp_path):
    return SessionManager(sessions_dir=tmp_path)

def test_list_folders_empty(sm):
    folders = sm.list_folders()
    assert folders == []

def test_create_folder(sm):
    sm.create_folder("Project Alpha")
    folders = sm.list_folders()
    assert "Project Alpha" in folders

def test_rename_folder(sm):
    sm.create_folder("Old Name")
    sm.rename_folder("Old Name", "New Name")
    folders = sm.list_folders()
    assert "New Name" in folders
    assert "Old Name" not in folders

def test_delete_folder(sm):
    sm.create_folder("To Delete")
    sm.delete_folder("To Delete")
    folders = sm.list_folders()
    assert "To Delete" not in folders

def test_save_transcript_creates_date_folder(sm):
    segments = [
        {"speaker": "you", "text": "Hello", "start": 5.0},
        {"speaker": "them", "text": "Hi there", "start": 12.0},
    ]
    path = sm.save_transcript(segments, duration=45.0, model="base")
    assert path is not None
    assert path.endswith(".txt")
    # Should be inside a date folder
    p = Path(path)
    assert p.exists()
    content = p.read_text()
    assert "Hello" in content
    assert "[00:05] You:" in content
    assert "[00:12] Them:" in content
    assert "Duration:" in content

def test_load_transcript(sm):
    segments = [
        {"speaker": "you", "text": "Test line", "start": 3.0},
    ]
    path = sm.save_transcript(segments, duration=10.0, model="base")
    loaded = sm.load_transcript(path)
    assert loaded["segments"][0]["speaker"] == "you"
    assert loaded["segments"][0]["text"] == "Test line"
    assert loaded["duration"] == "00:00:10"
    assert loaded["model"] == "base"

def test_list_transcripts(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base")
    folder = Path(path).parent.name
    transcripts = sm.list_transcripts(folder)
    assert len(transcripts) >= 1

def test_rename_transcript(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base")
    p = Path(path)
    folder = p.parent.name
    old_name = p.stem
    sm.rename_transcript(folder, old_name, "My Meeting")
    transcripts = sm.list_transcripts(folder)
    names = [t["name"] for t in transcripts]
    assert "My Meeting" in names

def test_move_transcript(sm):
    sm.create_folder("Destination")
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base")
    p = Path(path)
    src_folder = p.parent.name
    name = p.stem
    sm.move_transcript(src_folder, name, "Destination")
    assert len(sm.list_transcripts("Destination")) == 1
    assert len(sm.list_transcripts(src_folder)) == 0

def test_delete_transcript(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base")
    p = Path(path)
    folder = p.parent.name
    name = p.stem
    sm.delete_transcript(folder, name)
    assert len(sm.list_transcripts(folder)) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_session_manager.py -v`
Expected: FAIL

**Step 3: Rewrite SessionManager**

Replace `src/core/session_manager.py` with new implementation:

```python
"""
Session manager for Cadence.
Manages transcript folders and .txt file persistence.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("Cadence")


class SessionManager:
    """Manages transcript folders and .txt file storage."""

    def __init__(self, sessions_dir=None):
        if sessions_dir is None:
            from utils.resource_path import get_app_data_path
            self.sessions_dir = Path(get_app_data_path("sessions"))
        else:
            self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # --- Folder operations ---

    def list_folders(self):
        """List all folders (subdirectories), sorted alphabetically."""
        return sorted([
            d.name for d in self.sessions_dir.iterdir()
            if d.is_dir()
        ])

    def create_folder(self, name):
        """Create a new folder."""
        folder = self.sessions_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Folder created: {name}")

    def rename_folder(self, old_name, new_name):
        """Rename a folder."""
        old = self.sessions_dir / old_name
        new = self.sessions_dir / new_name
        if old.exists():
            old.rename(new)
            logger.info(f"Folder renamed: {old_name} -> {new_name}")

    def delete_folder(self, name):
        """Delete a folder and all its contents."""
        folder = self.sessions_dir / name
        if folder.exists():
            shutil.rmtree(folder)
            logger.info(f"Folder deleted: {name}")

    # --- Transcript operations ---

    def save_transcript(self, segments, duration=0.0, model="base", folder=None, name=None):
        """
        Save transcript as a .txt file. Returns the file path.
        Auto-creates a date folder if folder is not specified.
        Auto-generates a name if not specified.
        """
        if folder is None:
            folder = datetime.now().strftime("%Y-%m-%d")
        folder_path = self.sessions_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = f"Recording {datetime.now().strftime('%H-%M')}"

        filepath = folder_path / f"{name}.txt"
        # Avoid collisions
        counter = 1
        while filepath.exists():
            filepath = folder_path / f"{name} ({counter}).txt"
            counter += 1

        # Format duration
        dur_int = int(duration)
        h = dur_int // 3600
        m = (dur_int % 3600) // 60
        s = dur_int % 60
        dur_str = f"{h:02d}:{m:02d}:{s:02d}"

        # Build file content
        lines = []
        lines.append("Cadence Transcript")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Duration: {dur_str}")
        lines.append(f"Model: {model}")
        lines.append("")
        lines.append("---")
        lines.append("")
        for seg in segments:
            ts = seg.get("start", 0.0)
            mins = int(ts) // 60
            secs = int(ts) % 60
            speaker = "You" if seg["speaker"] == "you" else "Them"
            lines.append(f"[{mins:02d}:{secs:02d}] {speaker}: {seg['text']}")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Transcript saved: {filepath}")
        return str(filepath)

    def load_transcript(self, filepath):
        """
        Load and parse a .txt transcript file.
        Returns dict with metadata and segments.
        """
        p = Path(filepath)
        content = p.read_text(encoding="utf-8")
        lines = content.split("\n")

        result = {
            "name": p.stem,
            "date": "",
            "duration": "",
            "model": "",
            "segments": [],
        }

        in_header = True
        for line in lines:
            if in_header:
                if line.startswith("Date:"):
                    result["date"] = line[5:].strip()
                elif line.startswith("Duration:"):
                    result["duration"] = line[9:].strip()
                elif line.startswith("Model:"):
                    result["model"] = line[6:].strip()
                elif line.strip() == "---":
                    in_header = False
                continue

            line = line.strip()
            if not line:
                continue

            # Parse "[MM:SS] Speaker: text"
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                ts_str = line[1:bracket_end]
                rest = line[bracket_end + 1:].strip()
                # Parse timestamp
                try:
                    parts = ts_str.split(":")
                    start = int(parts[0]) * 60 + int(parts[1])
                except (ValueError, IndexError):
                    start = 0.0
                # Parse speaker
                if rest.startswith("You:"):
                    speaker = "you"
                    text = rest[4:].strip()
                elif rest.startswith("Them:"):
                    speaker = "them"
                    text = rest[5:].strip()
                else:
                    speaker = "unknown"
                    text = rest
                result["segments"].append({
                    "speaker": speaker,
                    "text": text,
                    "start": float(start),
                })

        return result

    def list_transcripts(self, folder):
        """List all transcripts in a folder. Returns list of dicts with name and path."""
        folder_path = self.sessions_dir / folder
        if not folder_path.exists():
            return []
        transcripts = []
        for f in sorted(folder_path.glob("*.txt")):
            transcripts.append({
                "name": f.stem,
                "path": str(f),
            })
        return transcripts

    def rename_transcript(self, folder, old_name, new_name):
        """Rename a transcript file."""
        old = self.sessions_dir / folder / f"{old_name}.txt"
        new = self.sessions_dir / folder / f"{new_name}.txt"
        if old.exists():
            old.rename(new)
            logger.info(f"Transcript renamed: {old_name} -> {new_name}")

    def move_transcript(self, src_folder, name, dest_folder):
        """Move a transcript to a different folder."""
        src = self.sessions_dir / src_folder / f"{name}.txt"
        dest_dir = self.sessions_dir / dest_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.txt"
        if src.exists():
            src.rename(dest)
            logger.info(f"Transcript moved: {src_folder}/{name} -> {dest_folder}/{name}")

    def delete_transcript(self, folder, name):
        """Delete a transcript file."""
        filepath = self.sessions_dir / folder / f"{name}.txt"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Transcript deleted: {folder}/{name}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/session_manager.py tests/test_session_manager.py
git commit -m "feat: rewrite SessionManager for .txt files and folder management"
```

---

### Task 3: Redesign MainWindow with 3-panel layout

**Files:**
- Modify: `src/gui/main_window.py`

**Step 1: Rewrite MainWindow**

Replace `src/gui/main_window.py` with the 3-panel layout:

```python
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
    folder_selected = Signal(str)          # folder name
    transcript_selected = Signal(str, str) # folder, transcript name
    folder_created = Signal(str)           # new folder name
    folder_renamed = Signal(str, str)      # old name, new name
    folder_deleted = Signal(str)           # folder name
    transcript_renamed = Signal(str, str, str)  # folder, old name, new name
    transcript_deleted = Signal(str, str)  # folder, name
    transcript_moved = Signal(str, str, str)    # src folder, name, dest folder

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

        # 3-panel splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel — Folder tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        folder_label = QLabel("Folders")
        folder_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        left_layout.addWidget(folder_label)
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setFont(QFont("Segoe UI", 10))
        self.folder_tree.itemClicked.connect(self._on_folder_clicked)
        self.folder_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.folder_tree.customContextMenuRequested.connect(self._on_folder_context_menu)
        left_layout.addWidget(self.folder_tree)
        self.add_folder_btn = QPushButton("+ New Folder")
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        left_layout.addWidget(self.add_folder_btn)
        self.splitter.addWidget(left_panel)

        # Middle panel — Transcript list
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        transcript_label = QLabel("Transcripts")
        transcript_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        mid_layout.addWidget(transcript_label)
        self.transcript_list = QListWidget()
        self.transcript_list.setFont(QFont("Segoe UI", 10))
        self.transcript_list.itemClicked.connect(self._on_transcript_clicked)
        self.transcript_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript_list.customContextMenuRequested.connect(self._on_transcript_context_menu)
        mid_layout.addWidget(self.transcript_list)
        self.splitter.addWidget(mid_panel)

        # Right panel — Transcript viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Segoe UI", 11))
        right_layout.addWidget(self.transcript_area)
        self.splitter.addWidget(right_panel)

        # Set initial splitter sizes (150, 150, remaining)
        self.splitter.setSizes([150, 150, 500])
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

        # Word count
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.info_label)

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer)

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
        self.status_label.setText("Recording...")
        self.transcript_area.clear()
        self._timer.start(1000)

    def set_idle_state(self):
        self._recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("Ready")
        self._timer.stop()

    def set_done_state(self):
        self.record_btn.setEnabled(True)
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")
        self.status_label.setText("Done")
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
            # Show folder selection dialog
            folders = []
            for i in range(self.folder_tree.topLevelItemCount()):
                f = self.folder_tree.topLevelItem(i).text(0)
                if f != self._current_folder:
                    folders.append(f)
            if not folders:
                QMessageBox.information(self, "Move", "No other folders available.")
                return
            from PySide6.QtWidgets import QInputDialog
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
```

**Step 2: Run app to visually verify layout (manual)**

Run: `uv run python src/main.py`
Verify: 3 panels visible, splitter works, buttons present.

**Step 3: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: redesign MainWindow with 3-panel layout, clear/copy buttons"
```

---

### Task 4: Update SettingsDialog — remove reprocess, add interval slider

**Files:**
- Modify: `src/gui/settings_dialog.py`

**Step 1: Rewrite SettingsDialog**

```python
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
```

**Step 2: Commit**

```bash
git add src/gui/settings_dialog.py
git commit -m "feat: update SettingsDialog with interval spinner, remove reprocess model"
```

---

### Task 5: Update CadenceApp — remove reprocessing, wire new UI, use interval config

**Files:**
- Modify: `src/main.py`

**Step 1: Remove all reprocessing code and wire new signals**

Key changes to `src/main.py`:

1. **Remove imports/references:**
   - Remove `ReprocessWorker` class entirely (lines 116-157)
   - Remove `from core.streaming_transcriber import StreamingTranscriber` if still present
   - Remove `self.reprocess_transcriber` (line 192-193)
   - Remove `self._reprocess_thread`, `self._reprocess_worker` (lines 204-205)
   - Remove `self._mic_audio`, `self._system_audio` (lines 208-209)
   - Remove `reprocess()` method (lines 330-385)
   - Remove `_on_reprocess_finished()` (lines 387-407)
   - Remove `_on_reprocess_error()` (lines 409-417)
   - Remove `_cleanup_reprocess_thread()` (lines 419-428)
   - Remove audio saving block in `stop_recording()` (lines 302-313)

2. **Use interval from config:**
   - In `_start_transcription_worker()`, change `interval=5.0` to `interval=self.config.get_transcription_interval()`

3. **Update `stop_recording()` to save .txt:**
   - Replace `self.session_manager.save_session()` with:
   ```python
   segments = self.session_manager.active_session["transcript"] if self.session_manager.active_session else []
   model = self.config.get_streaming_model_size()
   self.session_manager.save_transcript(segments, duration=duration, model=model)
   ```

4. **Update `show_settings()` — remove transcriber arg:**
   ```python
   dialog = SettingsDialog(self.config, self.audio_recorder)
   ```

5. **Update `_on_settings_changed()` — remove reprocess model update, add interval handling**

6. **Wire new MainWindow signals in `main()`:**
   ```python
   # Folder/transcript navigation
   main_window.folder_selected.connect(cadence_app.on_folder_selected)
   main_window.folder_created.connect(cadence_app.on_folder_created)
   main_window.folder_renamed.connect(cadence_app.on_folder_renamed)
   main_window.folder_deleted.connect(cadence_app.on_folder_deleted)
   main_window.transcript_selected.connect(cadence_app.on_transcript_selected)
   main_window.transcript_renamed.connect(cadence_app.on_transcript_renamed)
   main_window.transcript_deleted.connect(cadence_app.on_transcript_deleted)
   main_window.transcript_moved.connect(cadence_app.on_transcript_moved)
   ```

7. **Add handler methods to CadenceApp:**
   ```python
   def on_folder_selected(self, folder_name):
       transcripts = self.session_manager.list_transcripts(folder_name)
       if self.main_window:
           self.main_window.set_transcripts(transcripts)

   def on_folder_created(self, name):
       self.session_manager.create_folder(name)
       self._refresh_folders()

   def on_folder_renamed(self, old_name, new_name):
       self.session_manager.rename_folder(old_name, new_name)
       self._refresh_folders()

   def on_folder_deleted(self, name):
       self.session_manager.delete_folder(name)
       self._refresh_folders()

   def on_transcript_selected(self, folder, name):
       transcripts = self.session_manager.list_transcripts(folder)
       for t in transcripts:
           if t["name"] == name:
               data = self.session_manager.load_transcript(t["path"])
               if self.main_window:
                   self.main_window.set_transcript(data["segments"])
               break

   def on_transcript_renamed(self, folder, old_name, new_name):
       self.session_manager.rename_transcript(folder, old_name, new_name)
       self.on_folder_selected(folder)

   def on_transcript_deleted(self, folder, name):
       self.session_manager.delete_transcript(folder, name)
       self.on_folder_selected(folder)

   def on_transcript_moved(self, src_folder, name, dest_folder):
       self.session_manager.move_transcript(src_folder, name, dest_folder)
       self.on_folder_selected(src_folder)

   def _refresh_folders(self):
       folders = self.session_manager.list_folders()
       if self.main_window:
           self.main_window.set_folders(folders)
   ```

8. **Load folders on startup in `main()`:**
   ```python
   # Load initial folder list
   cadence_app._refresh_folders()
   ```

9. **Remove the `main_window.reprocess_requested.connect(cadence_app.reprocess)` line**

**Step 2: Run app to verify (manual)**

Run: `uv run python src/main.py`
Verify: App starts, folders panel shows, can create folders, recording works, transcripts saved as .txt.

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (may need to update tests that reference old SessionManager API)

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: wire 3-panel UI, remove reprocessing, use interval config"
```

---

### Task 6: Clean up unused code

**Files:**
- Modify: `src/core/audio_recorder.py` — remove `save_audio()` method and `chunk_callback` parameter
- Delete or keep: `src/core/streaming_transcriber.py` — currently unused, remove it
- Delete or keep: `tests/test_streaming_transcriber.py` — remove if streaming_transcriber removed

**Step 1: Clean up AudioRecorder**

In `src/core/audio_recorder.py`:
- Remove `chunk_callback` from `__init__` params and all references to it
- Remove `save_audio()` method
- Remove `import soundfile` reference

**Step 2: Remove streaming_transcriber**

```bash
git rm src/core/streaming_transcriber.py tests/test_streaming_transcriber.py
```

**Step 3: Run tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unused streaming_transcriber, audio saving, chunk_callback"
```

---

### Task 7: Update tests for AudioRecorder changes

**Files:**
- Modify: `tests/test_audio_recorder.py`

**Step 1: Update tests to remove chunk_callback references**

Review `tests/test_audio_recorder.py` and remove any tests that rely on `chunk_callback` or `save_audio`. Update `AudioRecorder()` constructor calls to not pass `chunk_callback`.

**Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_audio_recorder.py
git commit -m "test: update audio recorder tests for simplified API"
```

---

### Task 8: Final integration test and cleanup

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Manual smoke test**

Run: `uv run python src/main.py`
Verify checklist:
- [ ] App starts, 3 panels visible
- [ ] Can create a folder via "+" button
- [ ] Can start/stop recording
- [ ] Transcript appears in viewer during recording
- [ ] Transcript saved as .txt in date folder after stopping
- [ ] Folder tree shows date folder
- [ ] Clicking folder shows transcript in middle panel
- [ ] Clicking transcript loads it in viewer
- [ ] Clear button clears viewer
- [ ] Copy button copies text to clipboard
- [ ] Right-click rename/delete works on folders
- [ ] Right-click rename/delete/move works on transcripts
- [ ] Settings dialog shows interval spinner (2-8s)
- [ ] Changing interval and restarting recording uses new interval

**Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final cleanup after UI redesign"
```
