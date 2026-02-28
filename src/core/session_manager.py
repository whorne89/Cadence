"""
Session manager for Cadence.
Handles meeting session lifecycle, transcript storage, and persistence.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("Cadence")


class SessionManager:
    """Manages meeting session lifecycle and transcript persistence."""

    def __init__(self, sessions_dir=None):
        if sessions_dir is None:
            from utils.resource_path import get_app_data_path
            self.sessions_dir = Path(get_app_data_path("sessions"))
        else:
            self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.active_session = None

    def create_session(self):
        """Create a new meeting session."""
        self.active_session = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 0,
            "model_used": None,
            "reprocessed": False,
            "reprocess_model": None,
            "transcript": [],
        }
        logger.info(f"Session created: {self.active_session['id']}")
        return self.active_session

    def add_segment(self, speaker, text, start=0.0, end=0.0):
        """Add a transcript segment to the active session."""
        if self.active_session is None:
            logger.warning("No active session")
            return
        self.active_session["transcript"].append({
            "speaker": speaker,
            "text": text,
            "start": start,
            "end": end,
        })

    def set_duration(self, seconds):
        if self.active_session:
            self.active_session["duration_seconds"] = seconds

    def set_model(self, model_name):
        if self.active_session:
            self.active_session["model_used"] = model_name

    def mark_reprocessed(self, model_name):
        if self.active_session:
            self.active_session["reprocessed"] = True
            self.active_session["reprocess_model"] = model_name

    def save_session(self, filepath=None):
        """Save the active session to disk. Returns path to saved file."""
        if self.active_session is None:
            logger.warning("No active session to save")
            return None
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = self.active_session["id"][:8]
            filename = f"{ts}-{sid}.json"
            filepath = str(self.sessions_dir / filename)
        with open(filepath, 'w') as f:
            json.dump(self.active_session, f, indent=2)
        logger.info(f"Session saved: {filepath}")
        return filepath

    def load_session(self, filepath):
        """Load a session from disk."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def list_sessions(self):
        """List all saved sessions, newest first."""
        files = sorted(self.sessions_dir.glob("*.json"), reverse=True)
        sessions = []
        for f in files:
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    data["_filepath"] = str(f)
                    sessions.append(data)
            except Exception as e:
                logger.warning(f"Failed to load session {f}: {e}")
        return sessions

    def end_session(self, duration_seconds):
        if self.active_session:
            self.active_session["duration_seconds"] = duration_seconds
