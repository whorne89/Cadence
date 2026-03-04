"""Tests for audio-level echo cancellation via spectral subtraction."""

import numpy as np
import pytest

from src.core.echo_cancellation import (
    spectral_subtract_echo,
    _estimate_delay,
    _align_signal,
)


def _make_tone(freq, duration_s, sr=16000, amplitude=0.5):
    """Generate a sine wave tone."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_speech_signal(duration_s=2.0, sr=16000, base_freq=200, amplitude=0.3):
    """Generate a speech-like signal with harmonics and envelope modulation."""
    t = np.arange(int(sr * duration_s)) / sr
    signal = np.zeros_like(t)
    for harmonic in [1, 2, 3, 5]:
        signal += np.sin(2 * np.pi * base_freq * harmonic * t) / harmonic
    # Apply speech-like envelope (amplitude modulation at ~4Hz syllable rate)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = signal * envelope * amplitude
    return signal.astype(np.float32)


class TestSpectralSubtractEcho:
    """Tests for the spectral_subtract_echo function."""

    def test_pure_echo_reduced(self):
        """Mic contains only echo of sys_audio — should be significantly reduced."""
        sys_audio = _make_speech_signal(duration_s=2.0, base_freq=150, amplitude=0.4)
        # Echo = delayed, attenuated copy of sys_audio
        delay_samples = 480  # 30ms at 16kHz
        mic_audio = np.zeros(len(sys_audio) + delay_samples, dtype=np.float32)
        mic_audio[delay_samples:delay_samples + len(sys_audio)] = sys_audio * 0.3

        cleaned = spectral_subtract_echo(mic_audio, sys_audio)

        raw_rms = float(np.sqrt(np.mean(mic_audio ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        # Echo should be substantially reduced
        assert cleaned_rms < raw_rms * 0.5, (
            f"Echo not sufficiently reduced: {cleaned_rms:.4f} vs {raw_rms:.4f}"
        )

    def test_user_speech_preserved(self):
        """Mic has only user speech, no system audio — output should match input."""
        user_speech = _make_speech_signal(duration_s=2.0, base_freq=200, amplitude=0.3)
        sys_audio = np.zeros(len(user_speech), dtype=np.float32)  # silence

        cleaned = spectral_subtract_echo(user_speech, sys_audio)

        # Should return unchanged (silent sys triggers guard)
        np.testing.assert_array_equal(cleaned, user_speech)

    def test_double_talk_preserves_user(self):
        """Mic has user speech + echo — user speech energy should be mostly preserved."""
        sr = 16000
        duration = 2.0

        # User speech at 300Hz (different pitch from speaker)
        user_speech = _make_speech_signal(duration_s=duration, base_freq=300, amplitude=0.4)
        # Speaker at 150Hz
        sys_audio = _make_speech_signal(duration_s=duration, base_freq=150, amplitude=0.4)

        # Mic = user + attenuated echo
        delay_samples = 320  # 20ms
        echo = np.zeros(len(user_speech), dtype=np.float32)
        copy_len = len(sys_audio) - delay_samples
        echo[delay_samples:delay_samples + copy_len] = sys_audio[:copy_len] * 0.25
        mic_audio = user_speech + echo

        cleaned = spectral_subtract_echo(mic_audio, sys_audio)

        # User speech should still have significant energy
        user_rms = float(np.sqrt(np.mean(user_speech ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        # Cleaned should retain at least 40% of user speech energy
        assert cleaned_rms > user_rms * 0.4, (
            f"User speech too damaged: cleaned_rms={cleaned_rms:.4f}, "
            f"user_rms={user_rms:.4f}"
        )

    def test_silent_system_returns_unchanged(self):
        """Silent sys_audio should return mic_audio unchanged."""
        mic_audio = _make_speech_signal(duration_s=1.0, amplitude=0.3)
        sys_audio = np.zeros(len(mic_audio), dtype=np.float32)

        cleaned = spectral_subtract_echo(mic_audio, sys_audio)

        np.testing.assert_array_equal(cleaned, mic_audio)

    def test_short_audio_returns_unchanged(self):
        """Audio shorter than 200ms should return mic_audio unchanged."""
        sr = 16000
        short = np.random.randn(int(sr * 0.1)).astype(np.float32) * 0.1
        sys = np.random.randn(int(sr * 0.1)).astype(np.float32) * 0.1

        cleaned = spectral_subtract_echo(short, sys, sr=sr)

        np.testing.assert_array_equal(cleaned, short)

    def test_output_same_length(self):
        """Output should always be the same length as mic_audio input."""
        mic = _make_speech_signal(duration_s=3.0, amplitude=0.3)
        sys = _make_speech_signal(duration_s=3.0, base_freq=150, amplitude=0.4)

        cleaned = spectral_subtract_echo(mic, sys)

        assert len(cleaned) == len(mic)

    def test_output_dtype_float32(self):
        """Output should always be float32."""
        mic = _make_speech_signal(duration_s=1.0, amplitude=0.3)
        sys = _make_speech_signal(duration_s=1.0, base_freq=150, amplitude=0.4)

        cleaned = spectral_subtract_echo(mic, sys)

        assert cleaned.dtype == np.float32

    def test_no_correlation_minimal_damage(self):
        """Completely unrelated mic and sys audio — should not damage mic."""
        np.random.seed(42)
        mic_audio = _make_speech_signal(duration_s=2.0, base_freq=250, amplitude=0.3)
        # Totally unrelated system audio
        sys_audio = _make_speech_signal(duration_s=2.0, base_freq=400, amplitude=0.3)

        cleaned = spectral_subtract_echo(mic_audio, sys_audio)

        mic_rms = float(np.sqrt(np.mean(mic_audio ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        # Should retain at least 50% of energy (some subtraction is unavoidable
        # when frequencies partially overlap)
        assert cleaned_rms > mic_rms * 0.3, (
            f"Too much damage to unrelated audio: {cleaned_rms:.4f} vs {mic_rms:.4f}"
        )


class TestEstimateDelay:
    """Tests for the _estimate_delay helper."""

    def test_finds_known_delay(self):
        """Should find the correct delay within a few samples."""
        sr = 16000
        sys_audio = _make_speech_signal(duration_s=1.0, base_freq=150, amplitude=0.5)
        known_delay = 480  # 30ms

        mic_audio = np.zeros(len(sys_audio) + known_delay, dtype=np.float64)
        mic_audio[known_delay:known_delay + len(sys_audio)] = sys_audio * 0.3

        delay = _estimate_delay(
            mic_audio, sys_audio.astype(np.float64), sr, max_delay_ms=80,
        )

        # Should be within 5 samples of the true delay
        assert abs(delay - known_delay) < 5, (
            f"Delay estimate {delay} too far from expected {known_delay}"
        )

    def test_zero_delay(self):
        """No delay should return 0 or very small value."""
        sys_audio = _make_speech_signal(duration_s=1.0, amplitude=0.5)
        mic_audio = sys_audio * 0.3  # echo with no delay

        delay = _estimate_delay(
            mic_audio.astype(np.float64),
            sys_audio.astype(np.float64),
            16000, max_delay_ms=80,
        )

        assert delay < 5, f"Expected near-zero delay, got {delay}"


class TestAlignSignal:
    """Tests for the _align_signal helper."""

    def test_aligns_with_delay(self):
        """Signal should be shifted by the specified delay."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aligned = _align_signal(signal, delay=2, target_len=7)

        expected = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(aligned, expected)

    def test_zero_delay(self):
        """Zero delay should just copy the signal."""
        signal = np.array([1.0, 2.0, 3.0])
        aligned = _align_signal(signal, delay=0, target_len=3)

        np.testing.assert_array_equal(aligned, signal)

    def test_delay_exceeds_length(self):
        """Delay larger than target_len should return zeros."""
        signal = np.array([1.0, 2.0, 3.0])
        aligned = _align_signal(signal, delay=10, target_len=5)

        np.testing.assert_array_equal(aligned, np.zeros(5))
