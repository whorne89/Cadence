# Echo Cancellation v3 — Adaptive Filter + Pipeline Fix

**Date:** 2026-03-05
**Branch:** `feature/audio-echo-cancellation`
**Status:** Design approved, ready for implementation

## Problem

The mic picks up remote speaker audio from the laptop speaker. Every "You" segment potentially contains echo of the remote speaker's voice. This affects all users without headphones (laptop mic + speaker setup).

### Current State (Broken)

The v2 echo cancellation on this branch uses STFT spectral subtraction applied BEFORE the echo gate. This caused two failures:

1. **Echo gate broken:** Thresholds were calibrated for raw audio RMS values. AEC reduces energy before the gate sees it, so the gate never fires (0/54 chunks suppressed in Daniel 3/5 benchmark).
2. **Spectral subtraction wrong tool:** STFT spectral subtraction is designed for stationary noise, not echo. Echo is a delayed, reverb-modified copy of the reference signal. The magnitude spectra don't align due to room acoustics, so subtraction either kills real speech (too aggressive) or leaves echo (too gentle).

### Benchmark Trend

| Meeting | Echo Gate Suppression | Echo Bleed | Score |
|---------|----------------------|------------|-------|
| Olav 3/3 (main, v1) | 56% | 1 | 5/10 |
| Paul 3/3 (main, v2) | 2% | 5 | 7/10 |
| Daniel 3/5 (this branch) | **0%** | **9** | **5/10** |

## Solution

Replace STFT spectral subtraction with a two-stage adaptive echo cancellation pipeline, and fix the pipeline ordering so the echo gate operates on raw audio.

### New Pipeline

```
Raw mic audio + System audio (WASAPI loopback)
    │
    ├─[1] Echo Gate on RAW audio
    │     Energy-ratio check + envelope correlation
    │     Purpose: Kill chunks that are pure echo (no real speech)
    │     Uses existing thresholds (calibrated for raw RMS)
    │     → SUPPRESS (pure echo) or PASS (has real speech)
    │
    ├─[2] NLMS Adaptive Filter (pyroomacoustics)
    │     Only runs on chunks that PASSED the gate
    │     Cross-correlate to find delay → align signals
    │     NLMS learns acoustic path (speaker → room → mic)
    │     Subtracts predicted echo from mic signal
    │     → Mic audio with bulk echo removed
    │
    ├─[3] Spectral Gating (noisereduce)
    │     Light cleanup pass using system audio as noise reference
    │     Catches residual echo the adaptive filter missed
    │     prop_decrease=0.6 (moderate, avoid damaging speech)
    │     → Clean mic audio
    │
    ├─[4] Whisper Transcription
    │     Gets cleaned audio → better transcription accuracy
    │     → Text
    │
    └─[5] Text-Level Dedup (existing echo_gate.py)
         Word overlap, clause recovery, retroactive filtering
         Catches anything that survived audio-level processing
         → Final transcript
```

### Why NLMS Adaptive Filter

We have the **ideal setup** for adaptive echo cancellation: the WASAPI loopback gives us the exact signal playing through the speakers (the "far-end reference"). The NLMS filter uses this reference to learn the acoustic transfer function (how the speaker + room + mic transforms the signal) and subtracts the predicted echo.

- **Adapts per-chunk:** Learns each user's specific speaker/mic/room setup automatically
- **Handles non-stationary signals:** Unlike spectral subtraction, adapts to changing audio content
- **Proven approach:** Same principle used by WebRTC AEC, Speex, Teams, Zoom
- **Cross-platform:** pyroomacoustics has wheels for Windows + macOS, compiles on Linux

### Why noisereduce as Stage 2

The NLMS filter removes the bulk of echo but may leave residual artifacts. `noisereduce` with `y_noise=sys_audio` performs spectral gating — it identifies frequency bands where the mic signal resembles the system audio and attenuates them. This is a lighter touch than full spectral subtraction, suitable for cleanup.

## Dependencies

| Library | Purpose | Platform | Install |
|---------|---------|----------|---------|
| `pyroomacoustics` | NLMS adaptive filter | Win/Mac wheels, Linux compile | `pip install pyroomacoustics` |
| `noisereduce` | Spectral gating cleanup | Pure Python, all platforms | `pip install noisereduce` |

Both are added to `pyproject.toml`. No PyTorch needed (noisereduce's non-torch path is sufficient).

## What Changes

### `src/core/echo_cancellation.py` — Full Rewrite

Replace STFT spectral subtraction with:

```python
def cancel_echo(mic_audio, sys_audio, sr=16000):
    """
    Two-stage echo cancellation using system audio as reference.

    Stage 1: NLMS adaptive filter (pyroomacoustics)
    Stage 2: Spectral gating cleanup (noisereduce)

    Args:
        mic_audio: numpy float32 array, mic recording (speech + echo)
        sys_audio: numpy float32 array, system audio (reference signal)
        sr: sample rate (default 16000)

    Returns:
        numpy float32 array, mic audio with echo removed
    """
```

**Stage 1 — NLMS Adaptive Filter:**
- Estimate delay via cross-correlation (keep existing `_estimate_delay`)
- Align system audio to mic audio timing
- Apply `pyroomacoustics.adaptive.NLMS` with system audio as reference
- Parameters: `filter_length=1600` (100ms at 16kHz), `mu=0.1` (step size)

**Stage 2 — Spectral Gating:**
- Apply `noisereduce.reduce_noise(y=stage1_output, sr=sr, y_noise=sys_audio, prop_decrease=0.6)`
- Light cleanup of residual echo

**Guard conditions (keep existing):**
- Skip if either audio < 200ms
- Skip if `sys_rms < 0.005` (system silent, no echo possible)

### `src/main.py` — Pipeline Reorder

**Current (broken):**
```python
# Line 185-199: AEC first, then echo gate checks cleaned audio
raw_mic_audio = mic_audio.copy()
mic_audio = spectral_subtract_echo(mic_audio, sys_audio, sr=sr)  # ← cleans first
# ... later, echo gate checks mic_rms of cleaned audio ← never fires
```

**New:**
```python
# Echo gate checks RAW audio first
mic_rms = np.sqrt(np.mean(mic_audio ** 2))
sys_rms = np.sqrt(np.mean(sys_audio ** 2))
# ... energy ratio + envelope correlation on raw audio ...

# Only if chunk passes gate: apply AEC for better transcription
if not echo_detected:
    raw_mic_audio = mic_audio.copy()
    mic_audio = cancel_echo(mic_audio, sys_audio, sr=sr)
    aec_applied = True
```

### `pyproject.toml` — Add Dependencies

```toml
dependencies = [
    # ... existing ...
    "pyroomacoustics>=0.8.0",
    "noisereduce>=3.0.0",
]
```

## What Stays the Same

- **Bleed-compensated silence detection** — `BLEED_FACTOR = 0.8`, `feed_rms()`. Working correctly.
- **Echo gate thresholds** — `ratio < 1.5 and mic_rms < 0.014`, envelope correlation at 0.7. Calibrated for raw audio, will work again once they see raw audio.
- **Text-level echo detection** — `_is_text_echo()`, `deduplicate_segments()`, `_extract_unique_clauses()`, `_retract_echo_you()`. All unchanged.
- **Audio capture** — sounddevice (mic) + PyAudioWPatch (WASAPI loopback). Unchanged.
- **Echo diagnostics** — `record_chunk()` with raw_mic_audio. Keep saving pre/post AEC for analysis.

## NLMS Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `filter_length` | 1600 samples (100ms) | Covers typical laptop speaker-to-mic acoustic path. Speaker is inches from mic, so impulse response is short. 100ms is standard for near-field. |
| `mu` (step size) | 0.1 | Balance between convergence speed and stability. 0.05 = slower but stable, 0.5 = fast but may diverge. Start at 0.1, tune if needed. |

## noisereduce Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `prop_decrease` | 0.6 | How much to reduce noise frequencies. 1.0 = full removal (aggressive), 0.0 = no removal. 0.6 is moderate — catches residual echo without damaging speech. |
| `y_noise` | sys_audio | The reference signal. noisereduce computes a noise profile from this and gates matching frequencies in the mic signal. |

## Cross-Platform Notes

The echo cancellation pipeline is **100% cross-platform**. Both `pyroomacoustics` and `noisereduce` work on Windows, macOS, and Linux. The NLMS filter operates on numpy arrays — it doesn't care where the audio came from.

The only platform-specific code remains system audio capture (PyAudioWPatch on Windows). When macOS/Linux capture backends are added later, the echo cancellation pipeline works identically.

## Testing Strategy

1. **Unit tests:** Test `cancel_echo()` with synthetic signals (known echo + speech)
2. **Session replay:** Replay raw audio from `.cadence/echo_debug/20260305_100106/chunks/` through new pipeline, compare output
3. **Live benchmark:** Record a new meeting, compare against Teams transcript using existing benchmark process
4. **Regression check:** Verify echo gate suppression rate returns to >0% (was 56% on Olav, 2% on Paul, 0% on Daniel)

## Success Criteria

- Echo gate suppression >0% (gate fires again on pure-echo chunks)
- Echo bleed instances < 5 (down from 9 on Daniel benchmark)
- Will capture rate >= 70% (maintained or improved)
- Speaker attribution >= 80% (up from 69%)
- No new hallucinations
- App stays responsive (AEC processing < 2s per 30s chunk on typical laptop)
