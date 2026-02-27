import json
import pytest
from pathlib import Path


def test_create_session(tmp_path):
    from src.core.session_manager import SessionManager
    sm = SessionManager(sessions_dir=str(tmp_path))
    session = sm.create_session()
    assert "id" in session
    assert "created_at" in session
    assert session["transcript"] == []


def test_add_segment(tmp_path):
    from src.core.session_manager import SessionManager
    sm = SessionManager(sessions_dir=str(tmp_path))
    sm.create_session()
    sm.add_segment("you", "Hello there", start=0.0, end=1.5)
    assert len(sm.active_session["transcript"]) == 1
    assert sm.active_session["transcript"][0]["speaker"] == "you"
    assert sm.active_session["transcript"][0]["text"] == "Hello there"


def test_save_and_load_session(tmp_path):
    from src.core.session_manager import SessionManager
    sm = SessionManager(sessions_dir=str(tmp_path))
    sm.create_session()
    sm.add_segment("you", "Test line", start=0.0, end=1.0)
    filepath = sm.save_session()
    assert Path(filepath).exists()
    loaded = sm.load_session(filepath)
    assert loaded["transcript"][0]["text"] == "Test line"


def test_list_sessions(tmp_path):
    from src.core.session_manager import SessionManager
    sm = SessionManager(sessions_dir=str(tmp_path))
    sm.create_session()
    sm.save_session()
    sm.create_session()
    sm.save_session()
    sessions = sm.list_sessions()
    assert len(sessions) == 2
