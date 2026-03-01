"""Tests for the PostProcessWorker."""

import numpy as np
from unittest.mock import MagicMock


def test_postprocess_worker_emits_segments():
    """PostProcessWorker should keep live 'you' segments and re-transcribe system audio."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    # Only system audio is transcribed now
    transcriber.transcribe.return_value = [
        {"text": "The weather forecast calls for rain tomorrow afternoon.", "start": 0.5, "end": 1.8},
        {"text": "Please remember to submit your timesheets by Friday.", "start": 3.0, "end": 4.0},
    ]

    live_segments = [
        {"speaker": "you", "text": "I talked about the quarterly budget review yesterday.", "start": 0.0},
        {"speaker": "you", "text": "We should definitely consider expanding our marketing efforts.", "start": 2.0},
    ]

    sr = 16000
    sys_audio = np.random.randn(sr * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, live_segments, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    segments = results[0]
    # Should have 4 segments (2 live "you" + 2 re-transcribed "them")
    assert len(segments) == 4
    # transcribe called only once (system audio only)
    assert transcriber.transcribe.call_count == 1


def test_postprocess_worker_sorts_by_timestamp():
    """Segments should be sorted by start timestamp."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [
        {"text": "sys early", "start": 1.0, "end": 2.0},
    ]

    live_segments = [
        {"speaker": "you", "text": "mic late", "start": 5.0},
    ]

    sys_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, live_segments, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    segments = results[0]
    assert segments[0]["text"] == "sys early"
    assert segments[1]["text"] == "mic late"


def test_postprocess_worker_handles_empty_audio():
    """PostProcessWorker should handle empty inputs gracefully."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = []

    worker = PostProcessWorker(
        transcriber,
        live_segments=[],
        system_audio=np.array([], dtype=np.float32),
    )

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    assert results[0] == []
    # transcribe should NOT be called for empty audio
    assert not transcriber.transcribe.called


def test_postprocess_worker_handles_error():
    """PostProcessWorker should preserve live 'you' segments even if system transcription fails."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.side_effect = RuntimeError("model failed")

    sys_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(
        transcriber,
        live_segments=[{"speaker": "you", "text": "hello", "start": 0.0}],
        system_audio=sys_audio,
    )

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    # Live "you" segments are preserved even when system transcription fails
    assert len(results[0]) == 1
    assert results[0][0]["speaker"] == "you"
    assert results[0][0]["text"] == "hello"


def test_postprocess_worker_emits_progress():
    """PostProcessWorker should emit progress messages for system audio."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [{"text": "hi", "start": 0.0, "end": 1.0}]

    sys_audio = np.random.randn(16000).astype(np.float32)

    worker = PostProcessWorker(transcriber, live_segments=[], system_audio=sys_audio)

    progress_msgs = []
    worker.progress.connect(lambda msg: progress_msgs.append(msg))
    worker.run()

    assert len(progress_msgs) >= 1
    assert "system" in progress_msgs[0].lower()


def test_postprocess_preserves_you_segments_with_matching_system_text():
    """Live 'you' segments should survive even when system audio has similar text.

    This is the core fix: previously post-processing re-transcribed mic audio,
    producing bleed text identical to system segments, and dedup killed all 'you'.
    Now live 'you' segments (already echo-gate filtered) are kept as-is.
    """
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    # System audio produces text that partially overlaps with a live "you" segment
    transcriber.transcribe.return_value = [
        {"text": "Let's discuss the quarterly budget report for next month.", "start": 0.5, "end": 2.0},
    ]

    # Live "you" segment with genuinely different speech
    live_segments = [
        {"speaker": "you", "text": "I think we should increase the marketing budget.", "start": 3.0},
    ]

    sys_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, live_segments, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    segments = results[0]
    speakers = [s["speaker"] for s in segments]
    assert "you" in speakers, "Live 'you' segments must survive post-processing"
    assert "them" in speakers
    assert len(segments) == 2


def test_postprocess_only_keeps_you_from_live():
    """Only 'you' segments from live transcript are kept; 'them' are re-transcribed."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [
        {"text": "Re-transcribed system speech with better accuracy.", "start": 1.0, "end": 2.0},
    ]

    # Live has both "you" and "them" — only "you" should be kept
    live_segments = [
        {"speaker": "you", "text": "My spoken words.", "start": 0.0},
        {"speaker": "them", "text": "Old live them text.", "start": 1.0},
    ]

    sys_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, live_segments, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    segments = results[0]
    # Should have the live "you" + re-transcribed "them", NOT the old live "them"
    assert len(segments) == 2
    you_segs = [s for s in segments if s["speaker"] == "you"]
    them_segs = [s for s in segments if s["speaker"] == "them"]
    assert len(you_segs) == 1
    assert you_segs[0]["text"] == "My spoken words."
    assert len(them_segs) == 1
    assert them_segs[0]["text"] == "Re-transcribed system speech with better accuracy."
