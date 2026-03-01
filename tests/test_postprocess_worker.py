"""Tests for the PostProcessWorker."""

import numpy as np
from unittest.mock import MagicMock


def test_postprocess_worker_emits_segments():
    """PostProcessWorker should transcribe full audio and emit segments."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    # Different text per channel so echo dedup doesn't remove them
    # Mic segments need >= 6 words to avoid SequenceMatcher fallback
    transcriber.transcribe.side_effect = [
        [
            {"text": "I talked about the quarterly budget review yesterday.", "start": 0.0, "end": 1.5},
            {"text": "We should definitely consider expanding our marketing efforts.", "start": 2.0, "end": 3.2},
        ],
        [
            {"text": "The weather forecast calls for rain tomorrow afternoon.", "start": 0.5, "end": 1.8},
            {"text": "Please remember to submit your timesheets by Friday.", "start": 3.0, "end": 4.0},
        ],
    ]

    sr = 16000
    mic_audio = np.random.randn(sr * 5).astype(np.float32)
    sys_audio = np.random.randn(sr * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, mic_audio, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    segments = results[0]
    # Should have 4 segments (2 from mic + 2 from system, all different text)
    assert len(segments) == 4


def test_postprocess_worker_sorts_by_timestamp():
    """Segments should be sorted by start timestamp."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    # Mic has later timestamps, system has earlier
    transcriber.transcribe.side_effect = [
        [{"text": "mic late", "start": 5.0, "end": 6.0}],
        [{"text": "sys early", "start": 1.0, "end": 2.0}],
    ]

    mic_audio = np.random.randn(16000 * 5).astype(np.float32)
    sys_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, mic_audio, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    segments = results[0]
    assert segments[0]["text"] == "sys early"
    assert segments[1]["text"] == "mic late"


def test_postprocess_worker_handles_empty_audio():
    """PostProcessWorker should handle empty audio gracefully."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = []

    worker = PostProcessWorker(
        transcriber,
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    assert results[0] == []
    # transcribe should NOT be called for empty audio
    assert not transcriber.transcribe.called


def test_postprocess_worker_handles_error():
    """PostProcessWorker should emit empty segments on error."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.side_effect = RuntimeError("model failed")

    mic_audio = np.random.randn(16000 * 5).astype(np.float32)

    worker = PostProcessWorker(
        transcriber,
        mic_audio,
        np.array([], dtype=np.float32),
    )

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    assert results[0] == []


def test_postprocess_worker_emits_progress():
    """PostProcessWorker should emit progress messages."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [{"text": "hi", "start": 0.0, "end": 1.0}]

    mic_audio = np.random.randn(16000).astype(np.float32)
    sys_audio = np.random.randn(16000).astype(np.float32)

    worker = PostProcessWorker(transcriber, mic_audio, sys_audio)

    progress_msgs = []
    worker.progress.connect(lambda msg: progress_msgs.append(msg))
    worker.run()

    assert "microphone" in progress_msgs[0].lower()
    assert "system" in progress_msgs[1].lower()
