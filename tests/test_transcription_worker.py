"""Tests for the single-stream TranscriptionWorker."""

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

    defaults = dict(
        silence_threshold=0.005,
        min_silence_ms=400,
        min_speech_s=0.3,
        max_speech_s=30.0,
    )
    defaults.update(kwargs)

    worker = TranscriptionWorker(
        transcriber=transcriber,
        audio_recorder=recorder,
        **defaults,
    )
    return worker, transcriber, recorder


def test_worker_accepts_params():
    """Worker should accept silence detection parameters."""
    worker, _, _ = _make_worker()
    assert worker.silence_threshold == 0.005
    assert worker.min_silence_ms == 400
    assert worker.min_speech_s == 0.3
    assert worker.max_speech_s == 30.0
    assert worker._poll_interval == 0.2


def test_worker_transcribe_audio():
    """Worker should transcribe audio frames to text."""
    worker, transcriber, _ = _make_worker()

    frames = [np.random.randn(16000).astype(np.float32)]
    result = worker._transcribe_audio(frames)

    assert result == "hello world"
    assert transcriber.transcribe_text.called


def test_worker_transcribe_audio_skips_empty():
    """_transcribe_audio should return None for whitespace-only text."""
    worker, transcriber, _ = _make_worker()
    transcriber.transcribe_text.return_value = "   "

    frames = [np.random.randn(16000).astype(np.float32)]
    result = worker._transcribe_audio(frames)

    assert result is None


def test_worker_transcribe_audio_handles_error():
    """_transcribe_audio should catch and log transcription errors."""
    worker, transcriber, _ = _make_worker()
    transcriber.transcribe_text.side_effect = RuntimeError("model error")

    frames = [np.random.randn(16000).astype(np.float32)]
    result = worker._transcribe_audio(frames)

    assert result is None


def test_worker_stop():
    """Worker stop() should set _running to False."""
    worker, _, _ = _make_worker()
    worker._running = True
    worker.stop()
    assert worker._running is False
