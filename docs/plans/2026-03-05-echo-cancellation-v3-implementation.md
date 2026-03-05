# Echo Cancellation v3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace broken STFT spectral subtraction with NLMS adaptive filter + noisereduce spectral gating, and fix pipeline ordering so echo gate checks raw audio first.

**Architecture:** Two new dependencies (`pyroomacoustics`, `noisereduce`) replace the hand-rolled spectral subtraction. The echo gate runs on raw audio (where thresholds are calibrated), then AEC cleans survivors before Whisper. All existing text-level dedup unchanged.

**Tech Stack:** pyroomacoustics (NLMS adaptive filter), noisereduce (spectral gating), numpy, scipy

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pyroomacoustics and noisereduce to dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
dependencies = [
    "PySide6>=6.6.0",
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "faster-whisper>=1.0.0",
    "soundfile>=0.12.1",
    "PyAudioWPatch>=0.2.12",
    "pyroomacoustics>=0.8.0",
    "noisereduce>=3.0.0",
]
```

**Step 2: Install**

Run: `uv sync`
Expected: Both packages install successfully.

**Step 3: Verify imports**

Run: `uv run python -c "import pyroomacoustics; import noisereduce; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyroomacoustics and noisereduce for echo cancellation v3"
```

---

### Task 2: Rewrite echo_cancellation.py — Tests First

**Files:**
- Modify: `tests/test_echo_cancellation.py`
- Modify: `src/core/echo_cancellation.py`

**Step 1: Rewrite test file for new API**

Replace the entire contents of `tests/test_echo_cancellation.py` with:

```python
"""Tests for audio-level echo cancellation via NLMS adaptive filter + noisereduce."""

import numpy as np
import pytest

from src.core.echo_cancellation import (
    cancel_echo,
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
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = signal * envelope * amplitude
    return signal.astype(np.float32)


class TestCancelEcho:
    """Tests for the cancel_echo function (replaces spectral_subtract_echo)."""

    def test_pure_echo_reduced(self):
        """Mic contains only echo of sys_audio — should be significantly reduced."""
        sys_audio = _make_speech_signal(duration_s=2.0, base_freq=150, amplitude=0.4)
        delay_samples = 480  # 30ms at 16kHz
        mic_audio = np.zeros(len(sys_audio) + delay_samples, dtype=np.float32)
        mic_audio[delay_samples:delay_samples + len(sys_audio)] = sys_audio * 0.3

        cleaned = cancel_echo(mic_audio, sys_audio)

        raw_rms = float(np.sqrt(np.mean(mic_audio ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        assert cleaned_rms < raw_rms * 0.5, (
            f"Echo not sufficiently reduced: {cleaned_rms:.4f} vs {raw_rms:.4f}"
        )

    def test_user_speech_preserved_silent_system(self):
        """Mic has only user speech, system is silent — output unchanged."""
        user_speech = _make_speech_signal(duration_s=2.0, base_freq=200, amplitude=0.3)
        sys_audio = np.zeros(len(user_speech), dtype=np.float32)

        cleaned = cancel_echo(user_speech, sys_audio)

        np.testing.assert_array_equal(cleaned, user_speech)

    def test_double_talk_preserves_user(self):
        """Mic has user speech + echo — user speech energy mostly preserved."""
        user_speech = _make_speech_signal(duration_s=2.0, base_freq=300, amplitude=0.4)
        sys_audio = _make_speech_signal(duration_s=2.0, base_freq=150, amplitude=0.4)

        delay_samples = 320  # 20ms
        echo = np.zeros(len(user_speech), dtype=np.float32)
        copy_len = len(sys_audio) - delay_samples
        echo[delay_samples:delay_samples + copy_len] = sys_audio[:copy_len] * 0.25
        mic_audio = user_speech + echo

        cleaned = cancel_echo(mic_audio, sys_audio)

        user_rms = float(np.sqrt(np.mean(user_speech ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        assert cleaned_rms > user_rms * 0.4, (
            f"User speech too damaged: cleaned_rms={cleaned_rms:.4f}, "
            f"user_rms={user_rms:.4f}"
        )

    def test_silent_system_returns_unchanged(self):
        """Silent sys_audio triggers guard — return mic_audio unchanged."""
        mic_audio = _make_speech_signal(duration_s=1.0, amplitude=0.3)
        sys_audio = np.zeros(len(mic_audio), dtype=np.float32)

        cleaned = cancel_echo(mic_audio, sys_audio)

        np.testing.assert_array_equal(cleaned, mic_audio)

    def test_short_audio_returns_unchanged(self):
        """Audio shorter than 200ms triggers guard — return unchanged."""
        sr = 16000
        short = np.random.randn(int(sr * 0.1)).astype(np.float32) * 0.1
        sys = np.random.randn(int(sr * 0.1)).astype(np.float32) * 0.1

        cleaned = cancel_echo(short, sys, sr=sr)

        np.testing.assert_array_equal(cleaned, short)

    def test_output_same_length(self):
        """Output should always match mic_audio length."""
        mic = _make_speech_signal(duration_s=3.0, amplitude=0.3)
        sys = _make_speech_signal(duration_s=3.0, base_freq=150, amplitude=0.4)

        cleaned = cancel_echo(mic, sys)

        assert len(cleaned) == len(mic)

    def test_output_dtype_float32(self):
        """Output should always be float32."""
        mic = _make_speech_signal(duration_s=1.0, amplitude=0.3)
        sys = _make_speech_signal(duration_s=1.0, base_freq=150, amplitude=0.4)

        cleaned = cancel_echo(mic, sys)

        assert cleaned.dtype == np.float32

    def test_no_correlation_minimal_damage(self):
        """Unrelated mic and sys audio — should not damage mic."""
        np.random.seed(42)
        mic_audio = _make_speech_signal(duration_s=2.0, base_freq=250, amplitude=0.3)
        sys_audio = _make_speech_signal(duration_s=2.0, base_freq=400, amplitude=0.3)

        cleaned = cancel_echo(mic_audio, sys_audio)

        mic_rms = float(np.sqrt(np.mean(mic_audio ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        assert cleaned_rms > mic_rms * 0.3, (
            f"Too much damage to unrelated audio: {cleaned_rms:.4f} vs {mic_rms:.4f}"
        )

    def test_longer_chunk_works(self):
        """Should handle 30-second chunks (our max_speech_s)."""
        mic = _make_speech_signal(duration_s=30.0, amplitude=0.3)
        sys = _make_speech_signal(duration_s=30.0, base_freq=150, amplitude=0.4)

        delay_samples = 480
        echo = np.zeros(len(mic), dtype=np.float32)
        copy_len = min(len(sys), len(mic) - delay_samples)
        echo[delay_samples:delay_samples + copy_len] = sys[:copy_len] * 0.3
        mic_with_echo = mic + echo

        cleaned = cancel_echo(mic_with_echo, sys)

        assert len(cleaned) == len(mic_with_echo)
        assert cleaned.dtype == np.float32


class TestEstimateDelay:
    """Tests for the _estimate_delay helper (unchanged)."""

    def test_finds_known_delay(self):
        sr = 16000
        sys_audio = _make_speech_signal(duration_s=1.0, base_freq=150, amplitude=0.5)
        known_delay = 480  # 30ms

        mic_audio = np.zeros(len(sys_audio) + known_delay, dtype=np.float64)
        mic_audio[known_delay:known_delay + len(sys_audio)] = sys_audio * 0.3

        delay = _estimate_delay(
            mic_audio, sys_audio.astype(np.float64), sr, max_delay_ms=80,
        )

        assert abs(delay - known_delay) < 5, (
            f"Delay estimate {delay} too far from expected {known_delay}"
        )

    def test_zero_delay(self):
        sys_audio = _make_speech_signal(duration_s=1.0, amplitude=0.5)
        mic_audio = sys_audio * 0.3

        delay = _estimate_delay(
            mic_audio.astype(np.float64),
            sys_audio.astype(np.float64),
            16000, max_delay_ms=80,
        )

        assert delay < 5, f"Expected near-zero delay, got {delay}"


class TestAlignSignal:
    """Tests for the _align_signal helper (unchanged)."""

    def test_aligns_with_delay(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aligned = _align_signal(signal, delay=2, target_len=7)

        expected = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(aligned, expected)

    def test_zero_delay(self):
        signal = np.array([1.0, 2.0, 3.0])
        aligned = _align_signal(signal, delay=0, target_len=3)

        np.testing.assert_array_equal(aligned, signal)

    def test_delay_exceeds_length(self):
        signal = np.array([1.0, 2.0, 3.0])
        aligned = _align_signal(signal, delay=10, target_len=5)

        np.testing.assert_array_equal(aligned, np.zeros(5))
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_echo_cancellation.py -v`
Expected: FAIL — `cancel_echo` does not exist yet.

**Step 3: Rewrite echo_cancellation.py**

Replace the entire contents of `src/core/echo_cancellation.py` with:

```python
"""
Audio-level echo cancellation for Cadence.

Two-stage pipeline:
1. NLMS adaptive filter (pyroomacoustics) — learns acoustic path, subtracts predicted echo
2. Spectral gating (noisereduce) — light cleanup of residual echo

Uses system audio (WASAPI loopback) as the far-end reference signal.
"""

import numpy as np
from scipy.signal import fftconvolve

import noisereduce as nr
from pyroomacoustics.adaptive import NLMS


def cancel_echo(mic_audio, sys_audio, sr=16000,
                filter_length=1600, mu=0.1,
                noise_reduce_strength=0.6,
                max_delay_ms=80):
    """
    Remove echo of sys_audio from mic_audio.

    Stage 1: NLMS adaptive filter uses sys_audio as reference to learn and
    subtract the acoustic echo path (speaker -> room -> mic).

    Stage 2: noisereduce spectral gating catches residual echo using
    sys_audio as the noise profile reference.

    Args:
        mic_audio: Mic signal (float32, 16kHz) — user voice + echo.
        sys_audio: System/loopback signal (float32, 16kHz) — clean reference.
        sr: Sample rate in Hz.
        filter_length: NLMS filter taps. 1600 = 100ms at 16kHz.
        mu: NLMS step size. 0.1 = balanced convergence/stability.
        noise_reduce_strength: noisereduce prop_decrease (0-1). 0.6 = moderate.
        max_delay_ms: Maximum echo delay to search for.

    Returns:
        Cleaned mic audio as float32 numpy array, same length as input.
    """
    min_samples = int(sr * 0.2)  # 200ms minimum

    # Guard: too short to process
    if len(mic_audio) < min_samples or len(sys_audio) < min_samples:
        return mic_audio

    # Guard: system audio is silent (no echo possible)
    sys_rms = float(np.sqrt(np.mean(sys_audio.astype(np.float64) ** 2)))
    if sys_rms < 0.005:
        return mic_audio

    orig_len = len(mic_audio)
    mic = mic_audio.astype(np.float64)
    sys = sys_audio.astype(np.float64)

    # Step 1: Find echo delay and align signals
    delay = _estimate_delay(mic, sys, sr, max_delay_ms)
    aligned_sys = _align_signal(sys, delay, len(mic))

    # Trim to common length
    min_len = min(len(mic), len(aligned_sys))
    mic = mic[:min_len]
    ref = aligned_sys[:min_len]

    # Step 2: NLMS adaptive filter
    # The filter learns h such that: echo ≈ h * ref
    # Then: cleaned = mic - h * ref
    try:
        # NLMS expects (reference, desired_signal, filter_length, step_size)
        # desired = mic (contains echo + speech)
        # reference = ref (the far-end signal causing the echo)
        # output y = estimate of echo, error e = mic - y = cleaned signal
        _y, e, _w = NLMS(ref, mic, filter_length, mu=mu)
        cleaned = e
    except Exception:
        # Fallback: if NLMS fails for any reason, return original
        cleaned = mic

    # Step 3: Spectral gating cleanup for residual echo
    try:
        cleaned_f32 = cleaned.astype(np.float32)
        ref_f32 = ref.astype(np.float32)
        cleaned_f32 = nr.reduce_noise(
            y=cleaned_f32,
            sr=sr,
            y_noise=ref_f32,
            prop_decrease=noise_reduce_strength,
        )
        cleaned = cleaned_f32.astype(np.float64)
    except Exception:
        # If noisereduce fails, continue with NLMS output only
        pass

    # Match original length
    result = np.zeros(orig_len, dtype=np.float64)
    copy_len = min(len(cleaned), orig_len)
    result[:copy_len] = cleaned[:copy_len]

    return result.astype(np.float32)


def _estimate_delay(mic, sys, sr, max_delay_ms):
    """
    Estimate the echo delay between sys_audio and its echo in mic_audio
    using cross-correlation.

    Returns delay in samples (0 if no clear echo detected).
    """
    max_delay_samples = int(sr * max_delay_ms / 1000)

    # Use up to first 1s for delay estimation (faster than full signal)
    est_len = min(len(mic), len(sys), sr)
    mic_est = mic[:est_len]
    sys_est = sys[:est_len]

    correlation = fftconvolve(mic_est, sys_est[::-1], mode='full')

    # Only look at positive delays (echo comes after original)
    mid = len(sys_est) - 1
    end = min(mid + max_delay_samples, len(correlation))
    positive_corr = correlation[mid:end]

    if len(positive_corr) == 0:
        return 0

    delay = int(np.argmax(np.abs(positive_corr)))

    # Validate: peak must be meaningful (not just noise)
    peak_val = np.abs(positive_corr[delay])
    mean_val = np.mean(np.abs(positive_corr))
    if mean_val > 0 and peak_val / mean_val < 2.0:
        return 0

    return delay


def _align_signal(signal, delay, target_len):
    """Shift signal by delay samples and pad/trim to target_len."""
    aligned = np.zeros(target_len, dtype=np.float64)
    if delay >= target_len:
        return aligned
    copy_len = min(len(signal), target_len - delay)
    aligned[delay:delay + copy_len] = signal[:copy_len]
    return aligned
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_echo_cancellation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/echo_cancellation.py tests/test_echo_cancellation.py
git commit -m "feat: replace spectral subtraction with NLMS adaptive filter + noisereduce"
```

---

### Task 3: Fix Pipeline Ordering in main.py

**Files:**
- Modify: `src/main.py:185-225` (the AEC + echo gate block)

**Step 1: Reorder — echo gate checks raw audio FIRST, then AEC cleans survivors**

Replace `src/main.py` lines 185-225 (the block starting with `# AEC: spectral subtraction` through the diagnostics recording) with:

```python
                    # --- Echo gate on RAW audio (before any AEC) ---
                    if len(sys_audio) > 0:
                        sys_rms = float(np.sqrt(np.mean(sys_audio.astype(np.float64) ** 2)))
                        mic_rms = float(np.sqrt(np.mean(mic_audio.astype(np.float64) ** 2)))
                        if sys_rms > 0.005:
                            ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
                            echo_detected = (
                                (ratio < 1.5 and mic_rms < 0.014) or
                                (ratio < 0.65 and mic_rms < 0.020 and sys_rms > 0.030)
                            )
                            if self.echo_gate_logging:
                                logger.info(
                                    f"Echo gate at {timestamp:.1f}s: "
                                    f"mic_rms={mic_rms:.4f}, sys_rms={sys_rms:.4f}, "
                                    f"ratio={ratio:.2f}, suppressed={echo_detected}"
                                )

                    # Tier 2: audio envelope correlation (on raw audio)
                    if not echo_detected and len(sys_audio) > 0:
                        from core.echo_gate import is_echo
                        audio_is_echo, correlation = is_echo(
                            mic_audio, sys_audio, threshold=0.7, detail=True
                        )
                        if audio_is_echo and mic_rms < 0.020:
                            echo_detected = True
                            if self.echo_gate_logging:
                                logger.info(
                                    f"Echo gate (envelope) at {timestamp:.1f}s: "
                                    f"correlation={correlation:.2f}, "
                                    f"mic_rms={mic_rms:.4f}, suppressed=True"
                                )

                    # --- AEC: only clean chunks that PASSED the echo gate ---
                    aec_applied = False
                    raw_mic_audio = None
                    if not echo_detected and len(sys_audio) > int(sr * 0.2):
                        from core.echo_cancellation import cancel_echo
                        raw_mic_audio = mic_audio.copy()
                        mic_audio = cancel_echo(mic_audio, sys_audio, sr=sr)
                        aec_applied = True

                    # Record diagnostics (after gate decision, with raw audio if AEC ran)
                    if self.echo_diagnostics and len(sys_audio) > 0 and sys_rms > 0.005:
                        self.echo_diagnostics.record_chunk(
                            mic_audio, sys_audio,
                            mic_rms, sys_rms, ratio,
                            echo_detected, timestamp,
                            raw_mic_audio=raw_mic_audio,
                        )
```

Note: This replaces lines 185-242 (the old AEC block + old echo gate + old envelope correlation). The text-level echo gate at line 244+ remains unchanged.

**Step 2: Update the import reference**

The old code imports `spectral_subtract_echo`. The new code imports `cancel_echo`. Make sure the import at line 190 (now inside the `if not echo_detected` block) references the new function name.

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (no behavior change to other components)

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "fix: reorder pipeline — echo gate checks raw audio before AEC runs"
```

---

### Task 4: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Add entry under [Unreleased]**

Add to the Changed section:

```markdown
### Changed
- Replaced STFT spectral subtraction with NLMS adaptive filter (pyroomacoustics) + spectral gating (noisereduce) for echo cancellation
- Fixed pipeline ordering: echo gate now checks raw audio before AEC processes survivors
- Added pyroomacoustics and noisereduce as dependencies
```

**Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG with echo cancellation v3 changes"
```

---

### Task 5: Run Full Test Suite and Verify

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS, no regressions.

**Step 2: Quick smoke test — start the app**

Run: `uv run python src/main.py`
Expected: App launches, tray icon appears. Start/stop recording works without crash.

**Step 3: Verify echo gate logging**

Start a recording with system audio playing. Check the log for echo gate messages like:
```
Echo gate at X.Xs: mic_rms=0.XXXX, sys_rms=0.XXXX, ratio=X.XX, suppressed=True/False
```
Confirm that `suppressed=True` appears for some chunks (gate is firing again).

---

### Task 6: Live Benchmark Test

This is a manual step after implementation:

1. Record a meeting with Cadence while also on Teams
2. After the meeting, paste the Teams transcript for comparison
3. Run the benchmark comparison (same process as Daniel 3/5)
4. Compare against previous benchmarks:
   - Echo gate suppression should be >0% (was 0% on Daniel)
   - Echo bleed should be <5 instances (was 9 on Daniel)
   - Will capture should be >=70% (was 71% on Daniel)
   - Attribution should be >=80% (was 69% on Daniel)
