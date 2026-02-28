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
