import numpy as np
import pytest
from src.core.silence_detector import SilenceDetector


def test_silence_detected_on_quiet_audio():
    """Feeding silent frames should register as silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    # Feed 600ms of silence (3 x 200ms)
    silent = np.zeros(3200, dtype=np.float32)  # 200ms at 16kHz
    for _ in range(3):
        sd.feed(silent)
    assert sd.is_silent()


def test_speech_detected_on_loud_audio():
    """Feeding loud frames should not register as silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    loud = np.random.randn(3200).astype(np.float32) * 0.5
    sd.feed(loud)
    assert not sd.is_silent()


def test_silence_after_speech():
    """Silence should only trigger after min_silence_ms of quiet frames."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    loud = np.random.randn(3200).astype(np.float32) * 0.5
    silent = np.zeros(3200, dtype=np.float32)

    # Speech then short silence (200ms) — not enough
    sd.feed(loud)
    sd.feed(silent)
    assert not sd.is_silent()

    # More silence (total 600ms) — should trigger
    sd.feed(silent)
    sd.feed(silent)
    assert sd.is_silent()


def test_reset_clears_state():
    """reset() should clear accumulated silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    silent = np.zeros(3200, dtype=np.float32)
    for _ in range(3):
        sd.feed(silent)
    assert sd.is_silent()
    sd.reset()
    assert not sd.is_silent()


def test_speech_resets_silence_counter():
    """A loud frame in between silence should reset the silence timer."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    silent = np.zeros(3200, dtype=np.float32)
    loud = np.random.randn(3200).astype(np.float32) * 0.5

    sd.feed(silent)
    sd.feed(silent)
    # interrupt with speech
    sd.feed(loud)
    sd.feed(silent)
    sd.feed(silent)
    # only 400ms of silence after speech — not enough
    assert not sd.is_silent()


def test_empty_audio_chunk():
    """Empty audio chunks should be ignored without error."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    empty = np.array([], dtype=np.float32)
    sd.feed(empty)
    assert not sd.is_silent()

    # Empty chunks shouldn't disrupt ongoing silence tracking
    silent = np.zeros(3200, dtype=np.float32)
    for _ in range(3):
        sd.feed(silent)
    sd.feed(empty)
    assert sd.is_silent()


def test_feed_rms_silence():
    """feed_rms with low RMS should accumulate silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    # Feed 600ms of "silence" via RMS values (3 x 200ms = 3 x 3200 samples)
    for _ in range(3):
        sd.feed_rms(0.001, 3200)
    assert sd.is_silent()


def test_feed_rms_speech():
    """feed_rms with high RMS should register as speech."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    sd.feed_rms(0.5, 3200)
    assert not sd.is_silent()
    assert sd._has_had_speech


def test_feed_rms_mixed_with_feed():
    """feed_rms and feed can be interleaved."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    loud = np.random.randn(3200).astype(np.float32) * 0.5
    sd.feed(loud)  # speech via audio
    sd.feed_rms(0.001, 3200)  # silence via RMS
    sd.feed_rms(0.001, 3200)  # silence via RMS
    sd.feed_rms(0.001, 3200)  # silence via RMS
    assert sd.is_silent()
    assert sd._has_had_speech
