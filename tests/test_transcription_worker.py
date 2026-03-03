"""Tests for the energy-based TranscriptionWorker."""

import numpy as np
from unittest.mock import MagicMock


def _make_worker(**kwargs):
    """Create a TranscriptionWorker with mocked dependencies."""
    from src.main import TranscriptionWorker

    transcriber = MagicMock()
    transcriber.transcribe_text.return_value = "hello world"

    recorder = MagicMock()
    recorder.sample_rate = 16000
    recorder._mic_frames = []
    recorder._system_frames = []

    defaults = dict(
        mic_silence_threshold=0.005,
        sys_silence_threshold=0.01,
        mic_min_silence_ms=400,
        sys_min_silence_ms=500,
        mic_min_speech_s=0.3,
        sys_min_speech_s=0.5,
        max_speech_s=30.0,
    )
    defaults.update(kwargs)

    worker = TranscriptionWorker(
        transcriber=transcriber,
        audio_recorder=recorder,
        **defaults,
    )
    return worker, transcriber, recorder


def test_worker_accepts_silence_params():
    """Worker should accept per-channel silence detection parameters."""
    worker, _, _ = _make_worker()
    assert worker.mic_silence_threshold == 0.005
    assert worker.sys_silence_threshold == 0.01
    assert worker.mic_min_silence_ms == 400
    assert worker.sys_min_silence_ms == 500
    assert worker.mic_min_speech_s == 0.3
    assert worker.sys_min_speech_s == 0.5
    assert worker.max_speech_s == 30.0
    assert worker._poll_interval == 0.2


def test_worker_has_transcribe_frames_helper():
    """Worker should have a _transcribe_frames method that emits segments."""
    worker, transcriber, _ = _make_worker()

    segments = []
    worker.segment_ready.connect(lambda s, t, ts: segments.append((s, t, ts)))

    frames = [np.random.randn(16000).astype(np.float32)]
    worker._transcribe_frames(frames, "you", 1.5)

    assert len(segments) == 1
    assert segments[0] == ("you", "hello world", 1.5)
    assert transcriber.transcribe_text.called


def test_worker_transcribe_frames_skips_empty():
    """_transcribe_frames should skip empty text results."""
    worker, transcriber, _ = _make_worker()
    transcriber.transcribe_text.return_value = "   "

    segments = []
    worker.segment_ready.connect(lambda s, t, ts: segments.append((s, t, ts)))

    frames = [np.random.randn(16000).astype(np.float32)]
    worker._transcribe_frames(frames, "you", 0.0)

    assert len(segments) == 0


def test_worker_accepts_echo_gate_logging():
    """Worker should accept and store echo_gate_logging parameter."""
    worker, _, _ = _make_worker(echo_gate_logging=True)
    assert worker.echo_gate_logging is True

    worker2, _, _ = _make_worker()
    assert worker2.echo_gate_logging is False


def test_worker_transcribe_frames_handles_error():
    """_transcribe_frames should catch and log transcription errors."""
    worker, transcriber, _ = _make_worker()
    transcriber.transcribe_text.side_effect = RuntimeError("model error")

    segments = []
    worker.segment_ready.connect(lambda s, t, ts: segments.append((s, t, ts)))

    frames = [np.random.randn(16000).astype(np.float32)]
    worker._transcribe_frames(frames, "you", 0.0)

    # Should not crash, should not emit
    assert len(segments) == 0
