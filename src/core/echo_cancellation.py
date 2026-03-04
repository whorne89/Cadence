"""
Audio-level echo cancellation for Cadence.

Removes speaker echo from the microphone signal using spectral subtraction
BEFORE passing audio to Whisper for transcription.
"""

import numpy as np
from scipy.signal import stft, istft, fftconvolve


def spectral_subtract_echo(mic_audio, sys_audio, sr=16000,
                           n_fft=1024, hop=256,
                           alpha_low=2.0, alpha_high=1.0,
                           beta=0.02, max_delay_ms=80):
    """
    Remove echo of sys_audio from mic_audio using spectral subtraction.

    Works in the frequency domain: subtracts the system audio's magnitude
    spectrum from the mic's, preserving the mic's phase. Uses per-band
    alpha (more aggressive below 2kHz where speaker echo concentrates).

    Args:
        mic_audio: Mic signal (float32, 16kHz) — contains user voice + echo.
        sys_audio: System/reference signal (float32, 16kHz) — clean speaker audio.
        sr: Sample rate in Hz.
        n_fft: STFT window size in samples (64ms at 16kHz).
        hop: STFT hop size in samples (16ms at 16kHz).
        alpha_low: Oversubtraction factor for frequencies below 2kHz.
        alpha_high: Oversubtraction factor for frequencies above 2kHz.
        beta: Spectral floor (fraction of original magnitude). Prevents
              musical noise artifacts.
        max_delay_ms: Maximum echo delay to search for in ms.

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

    # Step 1: Find echo delay via cross-correlation
    delay = _estimate_delay(mic, sys, sr, max_delay_ms)

    # Step 2: Align sys_audio to match echo timing
    aligned_sys = _align_signal(sys, delay, len(mic))

    # Trim to common length
    min_len = min(len(mic), len(aligned_sys))
    mic = mic[:min_len]
    ref = aligned_sys[:min_len]

    # Step 3: STFT both signals
    _, _, Z_mic = stft(mic, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z_ref = stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    mag_mic = np.abs(Z_mic)
    mag_ref = np.abs(Z_ref)
    phase_mic = np.angle(Z_mic)

    # Step 4: Per-band alpha — aggressive below 2kHz, gentle above
    n_freq_bins = mag_ref.shape[0]
    freq_bins = np.linspace(0, sr / 2, n_freq_bins)
    alpha = np.where(freq_bins < 2000, alpha_low, alpha_high)
    alpha = alpha[:, np.newaxis]  # broadcast over time frames

    # Subtract with spectral floor
    mag_clean = mag_mic - alpha * mag_ref
    mag_clean = np.maximum(mag_clean, beta * mag_mic)

    # Step 5: Reconstruct with original mic phase
    Z_clean = mag_clean * np.exp(1j * phase_mic)
    _, cleaned = istft(Z_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    # Match original length
    if len(cleaned) >= orig_len:
        cleaned = cleaned[:orig_len]
    else:
        cleaned = np.pad(cleaned, (0, orig_len - len(cleaned)))

    return cleaned.astype(np.float32)


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
        # No clear echo peak — correlation is flat/noisy
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
