"""
Audio-level echo cancellation for Cadence.

Two-stage pipeline:
1. Wiener filter — estimates echo path from system audio, subtracts predicted echo
2. Spectral gating (noisereduce) — light cleanup of residual echo

Uses system audio (WASAPI loopback) as the far-end reference signal.
"""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve

import noisereduce as nr


def cancel_echo(mic_audio, sys_audio, sr=16000,
                filter_length=1600,
                noise_reduce_strength=0.3,
                max_delay_ms=80, reg=0.01):
    """
    Remove echo of sys_audio from mic_audio.

    Stage 1: Constrained Wiener filter estimates the acoustic echo path
    (speaker -> room -> mic) from cross-correlation, then subtracts the
    predicted echo. Filter is limited to filter_length taps to prevent
    over-subtraction of user speech during double-talk.

    Stage 2: noisereduce spectral gating catches residual echo using
    sys_audio as the noise profile reference.

    Args:
        mic_audio: Mic signal (float32, 16kHz) — user voice + echo.
        sys_audio: System/loopback signal (float32, 16kHz) — clean reference.
        sr: Sample rate in Hz.
        filter_length: Wiener filter taps. 1600 = 100ms at 16kHz.
        noise_reduce_strength: noisereduce prop_decrease (0-1). 0.6 = moderate.
        max_delay_ms: Maximum echo delay to search for.
        reg: Tikhonov regularization factor for Wiener filter stability.

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

    # Step 2: Constrained Wiener filter
    # Solve for echo path h: mic ≈ h * ref + user_speech
    # h = R_xx^{-1} * r_xy (Wiener-Hopf solution), truncated to filter_length
    try:
        # Autocorrelation of reference signal
        corr_xx = fftconvolve(ref, ref[::-1], mode='full')
        mid = len(ref) - 1
        r_xx = corr_xx[mid:mid + filter_length]
        r_xx[0] += reg * r_xx[0]  # Tikhonov regularization

        # Cross-correlation of mic and reference
        corr_xy = fftconvolve(mic, ref[::-1], mode='full')
        r_xy = corr_xy[mid:mid + filter_length]

        # Solve Toeplitz system for echo path filter
        h = solve_toeplitz(r_xx, r_xy)

        # Subtract estimated echo
        echo_est = fftconvolve(h, ref)[:min_len]
        cleaned = mic - echo_est
    except Exception:
        # Fallback: if Wiener filter fails, return original
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
        # If noisereduce fails, continue with Wiener output only
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
