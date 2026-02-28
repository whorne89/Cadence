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
