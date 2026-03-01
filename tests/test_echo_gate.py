import numpy as np
import pytest

from core.echo_gate import is_echo, deduplicate_segments, get_audio_for_sample_range, _word_overlap


# ── Helper to create speech-like test signals ────────────────────

def _make_speech_signal(duration_s, base_freq=300, sr=16000, amplitude=0.3,
                        env_freq=4.0, env_phase=0.0):
    """Create amplitude-modulated signal mimicking speech envelope."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    carrier = np.sin(2 * np.pi * base_freq * t)
    for f in [base_freq * 1.5, base_freq * 2, base_freq * 3]:
        carrier += np.sin(2 * np.pi * f * t) * np.random.uniform(0.1, 0.3)
    # Syllable-rate envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * env_freq * t + env_phase)
    return (carrier * envelope * amplitude).astype(np.float32)


# ── is_echo tests ────────────────────────────────────────────────


def test_echo_detected_on_speech_echo():
    """Mic picking up speaker audio should be detected as echo."""
    sr = 16000
    system = _make_speech_signal(2.0, base_freq=300, sr=sr)
    # Echo: delayed + attenuated (room bleed)
    delay = int(sr * 0.008)
    echo = np.zeros_like(system)
    echo[delay:] = system[:-delay] * 0.35
    mic = echo + np.random.randn(len(system)).astype(np.float32) * 0.005
    assert is_echo(mic, system) is True


def test_no_echo_when_user_speaks_over():
    """User talking over system audio should NOT trigger echo."""
    sr = 16000
    system = _make_speech_signal(2.0, base_freq=300, sr=sr, amplitude=0.3,
                                 env_freq=4.0)
    # Echo from speakers
    delay = int(sr * 0.008)
    echo = np.zeros_like(system)
    echo[delay:] = system[:-delay] * 0.3
    # User's own voice with different speech rhythm
    user = _make_speech_signal(2.0, base_freq=150, sr=sr, amplitude=0.4,
                               env_freq=3.2, env_phase=1.5)
    mic = echo + user
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


def test_echo_with_reverb():
    """Echo with multiple reflections should still be detected."""
    sr = 16000
    system = _make_speech_signal(3.0, base_freq=400, sr=sr)
    # Multiple reflections
    delay1 = int(sr * 0.005)
    delay2 = int(sr * 0.020)
    echo = np.zeros_like(system)
    echo[delay1:] = system[:-delay1] * 0.35
    echo[delay2:] += system[:-delay2] * 0.12
    mic = echo + np.random.randn(len(system)).astype(np.float32) * 0.005
    assert is_echo(mic, system) is True


def test_no_echo_independent_speech():
    """Two independent speech signals should not be detected as echo."""
    sr = 16000
    system = _make_speech_signal(2.0, base_freq=300, sr=sr,
                                 env_freq=4.0)
    # Completely independent mic signal with different rhythm
    mic = _make_speech_signal(2.0, base_freq=180, sr=sr,
                              env_freq=3.0, env_phase=2.0)
    assert is_echo(mic, system) is False


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
    result = deduplicate_segments(segments)
    assert len(result) == 1
    assert result[0]["speaker"] == "them"


def test_deduplicate_substring_echo():
    """Mic echo that's a fragment of a longer system segment."""
    segments = [
        {"speaker": "them", "text": "If you have a large army of drones that can operate without human oversight", "start": 1.0},
        {"speaker": "you", "text": "that can operate without human oversight,", "start": 1.3},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 1
    assert result[0]["speaker"] == "them"


def test_deduplicate_echo_spanning_multiple_sys_segments():
    """Mic echo that spans multiple shorter system segments."""
    segments = [
        {"speaker": "them", "text": "the Pentagon spokesman, Sean Pernell, the day before,", "start": 20.0},
        {"speaker": "you", "text": "Pernel, the day before, he reiterated their position, we only allow all waffle use.", "start": 22.0},
        {"speaker": "them", "text": "he reiterated their position,", "start": 23.0},
        {"speaker": "them", "text": "we only allow all lawful use.", "start": 25.0},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 3
    assert all(s["speaker"] == "them" for s in result)


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


def test_word_overlap_identical():
    """Identical text should have 100% overlap."""
    assert _word_overlap("hello world", "hello world") == 1.0


def test_word_overlap_subset():
    """Mic text subset of system text should have high overlap."""
    overlap = _word_overlap(
        "should not be allowed",
        "Those two use cases should not be allowed the Pentagon has told us",
    )
    assert overlap >= 0.8


def test_word_overlap_different():
    """Unrelated text should have low overlap."""
    overlap = _word_overlap(
        "I had a great day today",
        "The weather forecast calls for rain tomorrow",
    )
    assert overlap < 0.3
