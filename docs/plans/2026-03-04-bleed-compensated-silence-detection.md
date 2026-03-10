# Bleed-Compensated Silence Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix mic silence detection so it ignores speaker bleed, enabling natural short "you" segments instead of 30s monolithic blocks.

**Architecture:** Add a `feed_rms()` method to `SilenceDetector` so the caller can supply a pre-computed RMS. In the `TranscriptionWorker` mic loop, compute a bleed-compensated RMS by subtracting `0.8 * sys_rms` from `mic_rms` before feeding the silence detector. Everything downstream (AEC, echo gate, text dedup) stays unchanged.

**Tech Stack:** Python 3.12, numpy, PySide6 (QObject/Signal), faster-whisper

---

## Context

**Problem:** System audio bleeds into mic at ~0.06-0.09 RMS. Mic silence threshold is 0.005. Silence detector never triggers → always hits 30s `max_speech_s` → AEC gets huge mixed blocks it can't clean → no improvement visible to user. User speech also appears delayed by up to 30s.

**Key files:**
- `src/core/silence_detector.py` — `SilenceDetector` class with `feed()`, `is_silent()`, `reset()`
- `src/main.py` — `TranscriptionWorker` class, mic processing loop at lines 128-254
- `tests/test_silence_detector.py` — existing silence detector tests (6 tests)

**Key data points from 2026-03-04 session:**
- Mic frames: 1024 samples each (64ms at 16kHz)
- System frames: 1024 samples each, same rate
- Both frame lists (`_mic_frames`, `_system_frames`) grow in parallel from same start time
- Bleed ratio when only speaker talks: mic_rms/sys_rms ≈ 1.0-1.4
- User speech mic_rms: 0.085-0.119 (well above bleed level)

---

### Task 1: Add `feed_rms()` to SilenceDetector

**Files:**
- Modify: `src/core/silence_detector.py:19-28`
- Test: `tests/test_silence_detector.py`

**Step 1: Write the failing tests**

Add to `tests/test_silence_detector.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_silence_detector.py -v -k "feed_rms"`
Expected: FAIL with `AttributeError: 'SilenceDetector' object has no attribute 'feed_rms'`

**Step 3: Implement `feed_rms`**

Add this method to `SilenceDetector` in `src/core/silence_detector.py`, after `feed()`:

```python
def feed_rms(self, rms: float, num_samples: int):
    """Feed a pre-computed RMS value instead of raw audio.

    Behaves identically to feed() but skips RMS computation.
    Used for bleed-compensated silence detection where the caller
    adjusts the RMS before passing it in.
    """
    if num_samples == 0:
        return
    if rms < self.silence_threshold:
        self._silent_samples += num_samples
    else:
        self._silent_samples = 0
        self._has_had_speech = True
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_silence_detector.py -v`
Expected: All 9 tests PASS (6 existing + 3 new)

**Step 5: Commit**

```bash
git add src/core/silence_detector.py tests/test_silence_detector.py
git commit -m "feat: add feed_rms() to SilenceDetector for bleed compensation"
```

---

### Task 2: Add bleed-compensated mic feeding in TranscriptionWorker

**Files:**
- Modify: `src/main.py:128-135` (mic processing loop)
- Test: `tests/test_bleed_compensation.py` (new file)

**Step 1: Write the failing tests**

Create `tests/test_bleed_compensation.py`:

```python
"""Tests for bleed-compensated silence detection in TranscriptionWorker."""

import numpy as np
import pytest


def compensated_rms(mic_frame, sys_frame, bleed_factor=0.8):
    """Standalone version of the bleed compensation logic for testing."""
    mic_rms = float(np.sqrt(np.mean(mic_frame.astype(np.float64) ** 2)))
    if sys_frame is None or len(sys_frame) == 0:
        return mic_rms
    sys_rms = float(np.sqrt(np.mean(sys_frame.astype(np.float64) ** 2)))
    return max(mic_rms - bleed_factor * sys_rms, 0.0)


class TestCompensatedRms:
    """Test the bleed compensation formula."""

    def test_no_system_audio(self):
        """Without system audio, mic RMS passes through unchanged."""
        mic = np.random.randn(1024).astype(np.float32) * 0.1
        expected_rms = float(np.sqrt(np.mean(mic.astype(np.float64) ** 2)))
        assert compensated_rms(mic, None) == pytest.approx(expected_rms)
        assert compensated_rms(mic, np.array([])) == pytest.approx(expected_rms)

    def test_bleed_only_compensates_to_near_zero(self):
        """When mic is just bleed (similar RMS to sys), result is near zero."""
        # Simulate bleed: mic picks up sys at similar level
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        mic_frame = sys_frame * 1.1  # bleed is ~1.1x sys
        result = compensated_rms(mic_frame, sys_frame)
        # mic_rms ≈ 0.066, sys_rms ≈ 0.06, compensated ≈ 0.066 - 0.048 ≈ 0.018
        assert result < 0.025  # well below any reasonable speech level

    def test_real_speech_preserved(self):
        """When user speaks, compensated RMS stays well above threshold."""
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        # User speech is much louder than bleed
        mic_frame = np.random.randn(1024).astype(np.float32) * 0.15
        result = compensated_rms(mic_frame, sys_frame)
        # mic_rms ≈ 0.15, sys_rms ≈ 0.06, compensated ≈ 0.15 - 0.048 ≈ 0.102
        assert result > 0.05  # still clearly speech

    def test_silent_system_no_effect(self):
        """Silent system audio doesn't affect mic RMS."""
        mic = np.random.randn(1024).astype(np.float32) * 0.1
        sys = np.zeros(1024, dtype=np.float32)
        mic_only = compensated_rms(mic, None)
        with_silent_sys = compensated_rms(mic, sys)
        assert mic_only == pytest.approx(with_silent_sys)

    def test_floor_at_zero(self):
        """Compensated RMS never goes negative."""
        mic = np.random.randn(1024).astype(np.float32) * 0.01
        sys = np.random.randn(1024).astype(np.float32) * 0.1  # sys louder than mic
        result = compensated_rms(mic, sys)
        assert result == 0.0


class TestBleedCompensatedSilenceDetection:
    """Integration test: bleed compensation enables silence detection."""

    def test_bleed_without_compensation_blocks_silence(self):
        """Without compensation, bleed prevents silence detection."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        # Simulate 3 frames of bleed (mic picks up sys audio)
        for _ in range(3):
            bleed_frame = np.random.randn(1024).astype(np.float32) * 0.06
            sd.feed(bleed_frame)
        # Bleed RMS ≈ 0.06 >> threshold 0.005, so silence never triggers
        assert not sd.is_silent()

    def test_bleed_with_compensation_enables_silence(self):
        """With compensation, bleed is neutralized and silence triggers."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        # Simulate 10 frames of bleed with compensation
        # (need enough to exceed 400ms: 10 * 1024 / 16000 = 640ms)
        for _ in range(10):
            mic_frame = np.random.randn(1024).astype(np.float32) * 0.06
            sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
            rms = compensated_rms(mic_frame, sys_frame)
            sd.feed_rms(rms, len(mic_frame))
        assert sd.is_silent()

    def test_speech_during_bleed_still_detected(self):
        """Real user speech on top of bleed is still detected as speech."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        # User speaking while system audio plays
        mic_frame = np.random.randn(1024).astype(np.float32) * 0.15
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        rms = compensated_rms(mic_frame, sys_frame)
        sd.feed_rms(rms, len(mic_frame))
        assert sd._has_had_speech
        assert not sd.is_silent()
```

**Step 2: Run tests to verify they pass (these test the formula, not the integration)**

Run: `uv run pytest tests/test_bleed_compensation.py -v`
Expected: All tests PASS (the formula function is defined in the test file; integration tests use `SilenceDetector.feed_rms` from Task 1)

**Step 3: Modify the mic processing loop in `src/main.py`**

Replace lines 128-135 (the mic frame feeding section):

```python
            # --- Process mic audio ---
            mic_frames = self.audio_recorder._mic_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for frame in new_frames:
                    mic_detector.feed(frame)
                mic_offset = mic_len
```

With:

```python
            # --- Process mic audio ---
            mic_frames = self.audio_recorder._mic_frames
            sys_frames = self.audio_recorder._system_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for i, frame in enumerate(new_frames):
                    # Bleed-compensated silence detection:
                    # Subtract estimated speaker bleed from mic RMS so
                    # silence detector isn't fooled by system audio leaking
                    # into the mic.
                    sys_idx = mic_offset + i
                    if sys_idx < len(sys_frames):
                        sys_frame = sys_frames[sys_idx]
                        mic_rms = float(np.sqrt(np.mean(
                            frame.astype(np.float64) ** 2)))
                        sys_rms = float(np.sqrt(np.mean(
                            sys_frame.astype(np.float64) ** 2)))
                        comp_rms = max(mic_rms - 0.8 * sys_rms, 0.0)
                        mic_detector.feed_rms(comp_rms, len(frame))
                    else:
                        mic_detector.feed(frame)
                mic_offset = mic_len
```

**Step 4: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/main.py tests/test_bleed_compensation.py
git commit -m "feat: bleed-compensated silence detection for mic channel

Subtract estimated speaker bleed from mic RMS before feeding the
silence detector. This prevents system audio leaking into the mic
from blocking silence detection, which previously forced all 'you'
segments to hit the 30s max_speech ceiling."
```

---

### Task 3: Validate with recorded session data

**Files:**
- Test: `tests/test_bleed_compensation.py` (add replay test)

**Step 1: Write a replay test using the real session data**

Add to `tests/test_bleed_compensation.py`:

```python
import os


class TestReplaySession:
    """Replay the 2026-03-04 session data to verify bleed compensation works."""

    SESSION_DIR = os.path.join(".cadence", "echo_debug", "20260304_080418", "chunks")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(".cadence", "echo_debug", "20260304_080418")),
        reason="Session data not available"
    )
    def test_replay_chunks_detect_silence_boundaries(self):
        """Replay raw mic+sys audio: compensated silence detector should
        trigger BEFORE the 30s max_speech limit on at least some chunks."""
        import soundfile as sf
        from src.core.silence_detector import SilenceDetector

        chunk_size = 1024
        sr = 16000
        max_speech_samples = int(30.0 * sr)

        silence_triggered_count = 0

        for chunk_idx in range(1, 7):
            mic_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_mic_raw.wav")
            sys_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_sys.wav")

            mic_data, _ = sf.read(mic_path)
            sys_data, _ = sf.read(sys_path)

            sd = SilenceDetector(
                silence_threshold=0.005, min_silence_ms=400, sample_rate=sr
            )

            # Feed frame-by-frame with bleed compensation
            samples_fed = 0
            triggered_before_max = False
            for start in range(0, len(mic_data) - chunk_size, chunk_size):
                mic_frame = mic_data[start:start + chunk_size].astype(np.float32)
                sys_start = start
                if sys_start + chunk_size <= len(sys_data):
                    sys_frame = sys_data[sys_start:sys_start + chunk_size].astype(np.float32)
                    rms = compensated_rms(mic_frame, sys_frame)
                    sd.feed_rms(rms, chunk_size)
                else:
                    sd.feed(mic_frame)
                samples_fed += chunk_size

                if sd.is_silent() and sd._has_had_speech and samples_fed < max_speech_samples:
                    triggered_before_max = True
                    break

            if triggered_before_max:
                silence_triggered_count += 1

        # With bleed compensation, at least some of the 6 chunks should
        # have silence detected before the 30s limit
        assert silence_triggered_count >= 3, (
            f"Expected silence to trigger in >=3 of 6 chunks, got {silence_triggered_count}"
        )
```

**Step 2: Run the replay test**

Run: `uv run pytest tests/test_bleed_compensation.py::TestReplaySession -v`
Expected: PASS — silence detection triggers before 30s in the majority of chunks

**Step 3: Commit**

```bash
git add tests/test_bleed_compensation.py
git commit -m "test: add session replay test for bleed compensation"
```

---

### Task 4: Run full test suite and verify

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS, including existing echo gate, silence detector, and echo cancellation tests.

**Step 2: Verify no regressions by checking test count**

The test count should be: previous count + 3 (feed_rms tests) + 5 (compensation formula tests) + 3 (integration tests) + 1 (replay test) = previous + 12.

**Step 3: Final commit if any test adjustments needed**

---

### Summary of changes

| File | Change |
|------|--------|
| `src/core/silence_detector.py` | Add `feed_rms(rms, num_samples)` method |
| `src/main.py` lines 128-135 | Replace direct `feed()` with bleed-compensated `feed_rms()` |
| `tests/test_silence_detector.py` | 3 new tests for `feed_rms` |
| `tests/test_bleed_compensation.py` | New file: 9 tests (formula + integration + replay) |
