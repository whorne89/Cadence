import pytest
import numpy as np


def test_local_agreement_confirms_on_match():
    """Text is confirmed when N consecutive predictions match."""
    from src.core.streaming_transcriber import StreamingTranscriber

    engine = StreamingTranscriber.__new__(StreamingTranscriber)
    engine.agreement_threshold = 2
    engine.sample_rate = 16000
    engine.silence_threshold = 0.01
    engine._init_state()

    result1 = engine._apply_agreement("hello world")
    assert result1["confirmed"] == ""
    assert result1["partial"] == "hello world"

    result2 = engine._apply_agreement("hello world")
    assert result2["confirmed"] == "hello world"
    assert result2["partial"] == ""


def test_local_agreement_rejects_on_mismatch():
    """Text stays partial when predictions differ."""
    from src.core.streaming_transcriber import StreamingTranscriber

    engine = StreamingTranscriber.__new__(StreamingTranscriber)
    engine.agreement_threshold = 2
    engine.sample_rate = 16000
    engine.silence_threshold = 0.01
    engine._init_state()

    result1 = engine._apply_agreement("hello world")
    result2 = engine._apply_agreement("hello word")
    assert result2["confirmed"] == ""
    assert result2["partial"] == "hello word"


def test_silence_detection():
    """Silent audio chunks should be detected."""
    from src.core.streaming_transcriber import StreamingTranscriber

    engine = StreamingTranscriber.__new__(StreamingTranscriber)
    engine.silence_threshold = 0.01

    silent = np.zeros(16000, dtype=np.float32)
    assert engine._is_silence(silent) is True

    loud = np.random.randn(16000).astype(np.float32) * 0.5
    assert engine._is_silence(loud) is False


def test_reset_clears_state():
    """reset() should clear all streaming state."""
    from src.core.streaming_transcriber import StreamingTranscriber

    engine = StreamingTranscriber.__new__(StreamingTranscriber)
    engine.agreement_threshold = 2
    engine.sample_rate = 16000
    engine.silence_threshold = 0.01
    engine._init_state()

    engine._apply_agreement("some text")
    engine.reset()

    assert engine.confirmed_text == []
    assert engine.current_partial == ""
    assert len(engine.recent_predictions) == 0
