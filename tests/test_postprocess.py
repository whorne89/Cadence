"""Tests for post-processing utilities (hallucination filter, segment merging)."""

from src.main import _filter_hallucinations, merge_segments


# --- Hallucination filter tests ---

def test_filter_hallucinations_keeps_english():
    segments = [{"text": "Hello everyone, welcome to the meeting.", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 1


def test_filter_hallucinations_removes_non_english():
    # Cyrillic text is clearly non-English (100% non-ASCII letters)
    segments = [{"text": "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 0


def test_filter_hallucinations_removes_filler_only():
    segments = [{"text": "um", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 0


def test_filter_hallucinations_removes_no_letters():
    segments = [{"text": "...", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 0


def test_filter_hallucinations_removes_foreign_ascii():
    segments = [{"text": "deze vergadering begint straks", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 0


def test_filter_hallucinations_keeps_short_segments():
    segments = [{"text": "Yes", "start": 0.0}]
    result = _filter_hallucinations(segments)
    assert len(result) == 1


# --- Merge segments tests ---

def test_merge_empty():
    assert merge_segments([]) == []


def test_merge_single():
    segments = [{"text": "Hello", "start": 0.0}]
    result = merge_segments(segments)
    assert len(result) == 1


def test_merge_close_segments():
    segments = [
        {"text": "Hello", "start": 0.0},
        {"text": "world", "start": 1.0},
    ]
    result = merge_segments(segments, max_gap_s=2.0)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"


def test_merge_distant_segments():
    segments = [
        {"text": "Hello", "start": 0.0},
        {"text": "world", "start": 5.0},
    ]
    result = merge_segments(segments, max_gap_s=2.0)
    assert len(result) == 2


def test_merge_preserves_start_time():
    segments = [
        {"text": "Hello", "start": 10.0},
        {"text": "world", "start": 11.0},
    ]
    result = merge_segments(segments, max_gap_s=2.0)
    assert result[0]["start"] == 10.0
