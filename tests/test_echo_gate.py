import numpy as np
import pytest

from core.echo_gate import is_echo, deduplicate_segments, get_audio_for_sample_range


# ── is_echo tests ────────────────────────────────────────────────


def test_echo_detected_on_correlated_audio():
    """Correlated mic and system audio should be detected as echo."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    system = np.sin(2 * np.pi * 440 * t) * 0.3
    # Mic picks up same signal with attenuation + noise
    mic = system * 0.5 + np.random.randn(sr).astype(np.float32) * 0.01
    assert is_echo(mic, system) is True


def test_no_echo_on_different_audio():
    """Unrelated mic and system audio should not be detected as echo."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    system = np.sin(2 * np.pi * 440 * t) * 0.3
    mic = np.sin(2 * np.pi * 880 * t) * 0.3  # Different frequency
    assert is_echo(mic, system) is False


def test_no_echo_when_system_silent():
    """Silent system audio should not trigger echo detection."""
    sr = 16000
    system = np.zeros(sr, dtype=np.float32)
    mic = np.random.randn(sr).astype(np.float32) * 0.05
    assert is_echo(mic, system) is False


def test_no_echo_on_short_audio():
    """Very short audio chunks should not trigger echo detection."""
    mic = np.random.randn(100).astype(np.float32)
    sys_audio = np.random.randn(100).astype(np.float32)
    assert is_echo(mic, sys_audio) is False


def test_echo_with_delayed_signal():
    """Echo should still be detected with small delay (speaker-to-mic)."""
    sr = 16000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    system = np.sin(2 * np.pi * 300 * t) * 0.3
    # Mic has same signal delayed by 5ms (~80 samples at 16kHz)
    delay = 80
    mic = np.zeros_like(system)
    mic[delay:] = system[:-delay] * 0.4
    mic += np.random.randn(len(mic)).astype(np.float32) * 0.005
    # For multi-second chunks, small delay still produces high correlation
    assert is_echo(mic, system) is True


def test_user_speaking_over_system_reduces_correlation():
    """User talking over system audio should lower correlation."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    system = np.sin(2 * np.pi * 440 * t) * 0.2
    # User's voice (different signal) dominates the mic
    user_voice = np.sin(2 * np.pi * 200 * t) * 0.5
    mic = system * 0.3 + user_voice
    # Correlation should be low enough to not trigger echo
    assert is_echo(mic, system, threshold=0.3) is False


# ── get_audio_for_sample_range tests ─────────────────────────────


def test_get_audio_full_range():
    """Should extract audio covering the full range."""
    frames = [np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32) * 2]
    result = get_audio_for_sample_range(frames, 0, 200)
    assert len(result) == 200
    assert result[0] == 1.0
    assert result[100] == 2.0


def test_get_audio_partial_range():
    """Should extract audio from middle of frames."""
    frames = [np.ones(100, dtype=np.float32) * i for i in range(5)]
    result = get_audio_for_sample_range(frames, 150, 350)
    assert len(result) == 200


def test_get_audio_empty_range():
    """Should return empty array for out-of-bounds range."""
    frames = [np.ones(100, dtype=np.float32)]
    result = get_audio_for_sample_range(frames, 200, 300)
    assert len(result) == 0


# ── deduplicate_segments tests ───────────────────────────────────


def test_deduplicate_removes_echo_segments():
    """Mic segments matching system segments should be removed."""
    segments = [
        {"speaker": "them", "text": "Hello, how are you?", "start": 1.0},
        {"speaker": "you", "text": "Hello, how are you?", "start": 1.2},
        {"speaker": "you", "text": "I'm doing great", "start": 5.0},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 2
    assert result[0]["speaker"] == "them"
    assert result[1]["text"] == "I'm doing great"


def test_deduplicate_keeps_different_text():
    """Mic segments with different text should be kept."""
    segments = [
        {"speaker": "them", "text": "What's your name?", "start": 1.0},
        {"speaker": "you", "text": "My name is Will", "start": 2.0},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 2


def test_deduplicate_ignores_distant_segments():
    """Segments too far apart in time should not be compared."""
    segments = [
        {"speaker": "them", "text": "Hello there", "start": 1.0},
        {"speaker": "you", "text": "Hello there", "start": 10.0},
    ]
    result = deduplicate_segments(segments, time_window=3.0)
    assert len(result) == 2


def test_deduplicate_partial_text_match():
    """Partial text matches above threshold should be removed."""
    segments = [
        {"speaker": "them", "text": "I think we should go ahead with the plan", "start": 1.0},
        {"speaker": "you", "text": "I think we should go ahead with the plan.", "start": 1.1},
    ]
    result = deduplicate_segments(segments, similarity_threshold=0.6)
    assert len(result) == 1
    assert result[0]["speaker"] == "them"


def test_deduplicate_empty_segments():
    """Empty segment list should return empty."""
    assert deduplicate_segments([]) == []


def test_deduplicate_no_system_segments():
    """No system segments means nothing to compare — keep all."""
    segments = [
        {"speaker": "you", "text": "Hello", "start": 1.0},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 1
