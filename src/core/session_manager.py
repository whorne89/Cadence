"""
Session manager for Cadence.
Manages transcript folders and .txt file persistence.
"""

import logging
import re
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

    def _sanitize_name(self, name):
        """Remove characters invalid in Windows filenames."""
        return re.sub(r'[<>:"/\\|?*]', '_', name.strip())

    # --- Folder operations ---

    def list_folders(self):
        """List all folders (subdirectories), sorted alphabetically."""
        return sorted([
            d.name for d in self.sessions_dir.iterdir()
            if d.is_dir()
        ])

    def create_folder(self, name):
        """Create a new folder."""
        name = self._sanitize_name(name)
        folder = self.sessions_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Folder created: {name}")

    def rename_folder(self, old_name, new_name):
        """Rename a folder."""
        new_name = self._sanitize_name(new_name)
        old = self.sessions_dir / old_name
        new = self.sessions_dir / new_name
        if old.exists():
            if new.exists():
                raise FileExistsError(f"Folder '{new_name}' already exists")
            old.rename(new)
            logger.info(f"Folder renamed: {old_name} -> {new_name}")
        else:
            logger.warning(f"Cannot rename folder: '{old_name}' does not exist")

    def delete_folder(self, name):
        """Delete a folder and all its contents."""
        folder = self.sessions_dir / name
        if folder.exists():
            shutil.rmtree(folder)
            logger.info(f"Folder deleted: {name}")
        else:
            logger.warning(f"Cannot delete folder: '{name}' does not exist")

    # --- Transcript operations ---

    def save_transcript(self, segments, duration=0.0, model="base", folder=None,
                        name=None, participant=""):
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
            now = datetime.now()
            date_part = f"{now.month}-{now.day}-{str(now.year)[2:]}"
            time_part = now.strftime("%I.%M %p").lstrip("0")
            dur_mins = int(duration) // 60
            dur_secs = int(duration) % 60
            if dur_mins > 0:
                dur_part = f"{dur_mins}m {dur_secs}s"
            else:
                dur_part = f"{dur_secs}s"
            name = f"{date_part} {time_part} ({dur_part})"
        else:
            name = self._sanitize_name(name)

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
        if participant:
            lines.append(f"Participant: {participant}")
        lines.append("")
        lines.append("---")
        lines.append("")
        for seg in segments:
            ts = seg.get("start", 0.0)
            mins = int(ts) // 60
            secs = int(ts) % 60
            text = seg["text"]
            lines.append(f"[{mins:02d}:{secs:02d}] {text}")

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
            "participant": "",
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
                elif line.startswith("Participant:"):
                    result["participant"] = line[12:].strip()
                elif line.startswith("Speaker:"):
                    # Backward compat: old "Speaker:" header → participant
                    result["participant"] = line[8:].strip()
                elif line.strip() == "---":
                    in_header = False
                continue

            line = line.strip()
            if not line:
                continue

            # Parse "[MM:SS] text" or legacy "[MM:SS] Speaker: text"
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
                # Strip legacy speaker prefixes for backward compatibility
                for prefix in ("You: ", "Them: ", "Speaker: "):
                    if rest.startswith(prefix):
                        rest = rest[len(prefix):]
                        break
                result["segments"].append({
                    "text": rest,
                    "start": float(start),
                })

        return result

    def update_participant(self, filepath, participant):
        """Update the Participant header in a transcript file."""
        p = Path(filepath)
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        lines = content.split("\n")

        new_lines = []
        found = False
        for line in lines:
            if line.startswith("Participant:") or line.startswith("Speaker:"):
                if participant:
                    new_lines.append(f"Participant: {participant}")
                found = True
            elif line.strip() == "---" and not found:
                if participant:
                    new_lines.append(f"Participant: {participant}")
                new_lines.append(line)
                found = True
            else:
                new_lines.append(line)

        # Remove empty lines before ---
        final = []
        for i, line in enumerate(new_lines):
            if line.strip() == "" and i + 1 < len(new_lines) and new_lines[i + 1].strip() == "---":
                continue
            final.append(line)
        # Re-insert blank line before ---
        result = []
        for i, line in enumerate(final):
            if line.strip() == "---" and i > 0 and final[i - 1].strip() != "":
                result.append("")
            result.append(line)
        p.write_text("\n".join(result), encoding="utf-8")
        logger.info(f"Participant updated to '{participant}' in {filepath}")

    def _parse_transcript_date(self, filepath):
        """Extract the Date header from a transcript file. Returns datetime or None."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Date:"):
                        date_str = line[5:].strip()
                        return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                    if line.strip() == "---":
                        break
        except (OSError, ValueError):
            pass
        return None

    def list_transcripts(self, folder, sort_descending=True):
        """List all transcripts in a folder, sorted by date in file content."""
        folder_path = self.sessions_dir / folder
        if not folder_path.exists():
            return []
        transcripts = []
        for f in folder_path.glob("*.txt"):
            date = self._parse_transcript_date(f)
            transcripts.append({
                "name": f.stem,
                "path": str(f),
                "date": date,
            })
        epoch = datetime.min
        transcripts.sort(
            key=lambda t: t["date"] or epoch,
            reverse=sort_descending,
        )
        return transcripts

    def rename_transcript(self, folder, old_name, new_name):
        """Rename a transcript file."""
        new_name = self._sanitize_name(new_name)
        old = self.sessions_dir / folder / f"{old_name}.txt"
        new = self.sessions_dir / folder / f"{new_name}.txt"
        if old.exists():
            if new.exists():
                raise FileExistsError(f"Transcript '{new_name}' already exists")
            old.rename(new)
            logger.info(f"Transcript renamed: {old_name} -> {new_name}")
        else:
            logger.warning(f"Cannot rename transcript: '{old_name}' does not exist in '{folder}'")

    def move_transcript(self, src_folder, name, dest_folder):
        """Move a transcript to a different folder."""
        src = self.sessions_dir / src_folder / f"{name}.txt"
        dest_dir = self.sessions_dir / dest_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.txt"
        if src.exists():
            if dest.exists():
                raise FileExistsError(f"Transcript '{name}' already exists in '{dest_folder}'")
            src.rename(dest)
            logger.info(f"Transcript moved: {src_folder}/{name} -> {dest_folder}/{name}")
        else:
            logger.warning(f"Cannot move transcript: '{name}' does not exist in '{src_folder}'")

    def get_metrics(self):
        """Compute aggregate metrics across all transcripts."""
        from datetime import timedelta
        now = datetime.now()
        week_ago = now - timedelta(days=7)

        total = 0
        total_duration = 0.0
        total_words = 0
        this_week = 0

        for folder in self.list_folders():
            for t in self.list_transcripts(folder, sort_descending=False):
                total += 1
                data = self.load_transcript(t["path"])

                # Parse duration "HH:MM:SS"
                dur = data.get("duration", "")
                if dur:
                    try:
                        parts = dur.split(":")
                        total_duration += int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    except (ValueError, IndexError):
                        pass

                # Count words
                for seg in data["segments"]:
                    total_words += len(seg["text"].split())

                # Check if recorded this week
                if t.get("date") and t["date"] >= week_ago:
                    this_week += 1

        return {
            "total_recordings": total,
            "total_duration_s": total_duration,
            "total_words": total_words,
            "avg_duration_s": total_duration / total if total else 0.0,
            "avg_words": total_words // total if total else 0,
            "recordings_this_week": this_week,
        }

    def delete_transcript(self, folder, name):
        """Delete a transcript file."""
        filepath = self.sessions_dir / folder / f"{name}.txt"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Transcript deleted: {folder}/{name}")
        else:
            logger.warning(f"Cannot delete transcript: '{name}' does not exist in '{folder}'")
