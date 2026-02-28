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


def test_list_transcripts_sorted_by_date(sm):
    """Transcripts should sort by date in file, not filename."""
    folder = "test-sort"
    sm.create_folder(folder)
    # Create files with explicit date headers (older first)
    folder_path = sm.sessions_dir / folder
    (folder_path / "zzz.txt").write_text(
        "Cadence Transcript\nDate: 2026-01-01 09:00\nDuration: 00:00:10\nModel: base\n\n---\n",
        encoding="utf-8",
    )
    (folder_path / "aaa.txt").write_text(
        "Cadence Transcript\nDate: 2026-02-15 14:30\nDuration: 00:00:10\nModel: base\n\n---\n",
        encoding="utf-8",
    )
    # Descending (newest first) — aaa (Feb) before zzz (Jan)
    desc = sm.list_transcripts(folder, sort_descending=True)
    assert desc[0]["name"] == "aaa"
    assert desc[1]["name"] == "zzz"
    # Ascending (oldest first) — zzz (Jan) before aaa (Feb)
    asc = sm.list_transcripts(folder, sort_descending=False)
    assert asc[0]["name"] == "zzz"
    assert asc[1]["name"] == "aaa"


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


# --- Sanitization tests ---

def test_sanitize_name_strips_invalid_chars(sm):
    result = sm._sanitize_name('Meeting: "Q1 <Review> | 2026"')
    assert "<" not in result
    assert ">" not in result
    assert ":" not in result
    assert '"' not in result
    assert "|" not in result
    assert result == "Meeting_ _Q1 _Review_ _ 2026_"


def test_sanitize_name_strips_whitespace(sm):
    result = sm._sanitize_name("  hello  ")
    assert result == "hello"


def test_create_folder_sanitizes_name(sm):
    sm.create_folder('Project: "Alpha"')
    folders = sm.list_folders()
    assert any("Project" in f for f in folders)
    assert not any('"' in f for f in folders)
    assert not any(':' in f for f in folders)


def test_save_transcript_sanitizes_name(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base", folder="test", name='Meeting: "Q1"')
    p = Path(path)
    assert ":" not in p.stem
    assert '"' not in p.stem


def test_rename_folder_sanitizes_new_name(sm):
    sm.create_folder("Old")
    sm.rename_folder("Old", 'New: "Name"')
    folders = sm.list_folders()
    assert "Old" not in folders
    assert any("New" in f for f in folders)
    assert not any('"' in f for f in folders)


def test_rename_transcript_sanitizes_new_name(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    path = sm.save_transcript(segments, duration=10.0, model="base", folder="test", name="original")
    sm.rename_transcript("test", "original", 'New: "Name"')
    transcripts = sm.list_transcripts("test")
    names = [t["name"] for t in transcripts]
    assert not any('"' in n for n in names)
    assert not any(':' in n for n in names)


# --- FileExistsError tests ---

def test_rename_folder_raises_if_dest_exists(sm):
    sm.create_folder("Alpha")
    sm.create_folder("Beta")
    with pytest.raises(FileExistsError, match="already exists"):
        sm.rename_folder("Alpha", "Beta")


def test_rename_transcript_raises_if_dest_exists(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    sm.save_transcript(segments, duration=10.0, model="base", folder="test", name="first")
    sm.save_transcript(segments, duration=10.0, model="base", folder="test", name="second")
    with pytest.raises(FileExistsError, match="already exists"):
        sm.rename_transcript("test", "first", "second")


def test_move_transcript_raises_if_dest_exists(sm):
    segments = [{"speaker": "you", "text": "Hello", "start": 0.0}]
    sm.save_transcript(segments, duration=10.0, model="base", folder="src", name="file")
    sm.save_transcript(segments, duration=10.0, model="base", folder="dest", name="file")
    with pytest.raises(FileExistsError, match="already exists"):
        sm.move_transcript("src", "file", "dest")


# --- Warning log tests ---

def test_rename_folder_warns_if_source_missing(sm, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="Cadence"):
        sm.rename_folder("nonexistent", "new")
    assert "does not exist" in caplog.text


def test_delete_folder_warns_if_missing(sm, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="Cadence"):
        sm.delete_folder("nonexistent")
    assert "does not exist" in caplog.text


def test_rename_transcript_warns_if_source_missing(sm, caplog):
    import logging
    sm.create_folder("test")
    with caplog.at_level(logging.WARNING, logger="Cadence"):
        sm.rename_transcript("test", "nonexistent", "new")
    assert "does not exist" in caplog.text


def test_move_transcript_warns_if_source_missing(sm, caplog):
    import logging
    sm.create_folder("dest")
    with caplog.at_level(logging.WARNING, logger="Cadence"):
        sm.move_transcript("src", "nonexistent", "dest")
    assert "does not exist" in caplog.text


def test_delete_transcript_warns_if_missing(sm, caplog):
    import logging
    sm.create_folder("test")
    with caplog.at_level(logging.WARNING, logger="Cadence"):
        sm.delete_transcript("test", "nonexistent")
    assert "does not exist" in caplog.text
