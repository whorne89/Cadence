"""
Echo detection for cross-talk suppression.

When using speakers, the mic picks up system audio (echo). This module
provides tools to detect and suppress this cross-talk:

1. Audio-level: Energy ratio between mic and system audio to detect
   echo during live transcription (speaker bleed is quieter than
   direct speech).
2. Text-level: Duplicate segment removal during post-processing using
   word overlap + sequence matching fallback.
"""

import numpy as np
from difflib import SequenceMatcher


def _energy_envelope(audio, window_ms=20, sample_rate=16000):
    """Compute RMS energy envelope using fixed-size windows."""
    window = int(sample_rate * window_ms / 1000)
    n_windows = len(audio) // window
    if n_windows == 0:
        return np.array([], dtype=np.float64)
    # Reshape into windows and compute RMS per window
    trimmed = audio[:n_windows * window].astype(np.float64)
    blocks = trimmed.reshape(n_windows, window)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


def is_echo(mic_audio, sys_audio, threshold=0.6, sample_rate=16000, detail=False):
    """
    Check if mic audio is echo of system audio using energy envelope
    correlation.

    Raw waveform correlation fails because room acoustics (delay, reverb,
    frequency response) distort the signal. Energy envelopes are robust
    to these effects — the loudness pattern of the echo still tracks
    the original.

    Benchmarked results:
      - Pure echo:         ~0.96 (high)
      - User over echo:    ~0.32 (low — user's voice breaks the pattern)
      - User only:         ~0.03 (near zero)
      - Processing time:   ~4.7ms for 5s of audio

    Args:
        mic_audio: Mic audio chunk (float32 numpy array)
        sys_audio: System audio chunk from same time window
        threshold: Envelope correlation above which echo is detected
        sample_rate: Audio sample rate in Hz
        detail: If True, return (bool, correlation) tuple

    Returns:
        bool (or tuple if detail=True): True if mic audio appears to be echo
    """
    def _result(detected, corr=0.0):
        return (detected, corr) if detail else detected

    # Need at least 200ms for meaningful envelope comparison
    min_samples = int(sample_rate * 0.2)
    if len(mic_audio) < min_samples or len(sys_audio) < min_samples:
        return _result(False)

    # If system audio is too quiet, it can't cause audible echo
    sys_rms = np.sqrt(np.mean(sys_audio.astype(np.float64) ** 2))
    if sys_rms < 0.005:
        return _result(False)

    # Compute energy envelopes
    mic_env = _energy_envelope(mic_audio, sample_rate=sample_rate)
    sys_env = _energy_envelope(sys_audio, sample_rate=sample_rate)

    min_len = min(len(mic_env), len(sys_env))
    if min_len < 5:
        return _result(False)

    m = mic_env[:min_len]
    s = sys_env[:min_len]

    # Pearson correlation of envelopes
    m_c = m - m.mean()
    s_c = s - s.mean()
    m_std = m_c.std()
    s_std = s_c.std()

    if m_std < 1e-8 or s_std < 1e-8:
        return _result(False)

    correlation = float(np.mean(m_c * s_c) / (m_std * s_std))
    return _result(bool(abs(correlation) > threshold), correlation)


def get_audio_for_sample_range(frames, start_sample, end_sample):
    """
    Extract audio from a frame list covering a given sample range.

    Both mic and system channels accumulate at the same sample rate,
    so sample indices correspond to the same wall-clock time.

    Args:
        frames: List of numpy arrays (audio frames)
        start_sample: Start of range (inclusive)
        end_sample: End of range (exclusive)

    Returns:
        numpy array of audio data, or empty array if no overlap
    """
    cumulative = 0
    chunks = []
    for frame in frames:
        frame_end = cumulative + len(frame)
        if frame_end > start_sample and cumulative < end_sample:
            s = max(0, start_sample - cumulative)
            e = min(len(frame), end_sample - cumulative)
            chunks.append(frame[s:e])
        elif cumulative >= end_sample:
            break
        cumulative += len(frame)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _word_overlap(mic_text, sys_text):
    """Fraction of mic words that appear in system text."""
    # Strip punctuation for comparison
    strip = str.maketrans("", "", ".,!?;:'\"()-")
    mic_words = set(mic_text.lower().translate(strip).split())
    sys_words = set(sys_text.lower().translate(strip).split())
    mic_words.discard("")
    if not mic_words:
        return 0.0
    return len(mic_words & sys_words) / len(mic_words)


def deduplicate_segments(segments, time_window=15.0, word_overlap_threshold=0.5,
                         seq_match_threshold=0.4):
    """
    Remove mic segments that are echo of system audio segments.

    Two-pass detection:
    1. Word overlap: fraction of mic words found in combined nearby system
       text. Catches most echo even when Whisper chunks differently.
    2. Sequence matching fallback: character-level similarity against each
       individual system segment. Catches echo where Whisper produces
       completely different words (e.g. "prepared to access" vs
       "prepare to exist").

    Args:
        segments: List of {"speaker", "text", "start"} dicts, sorted by start time
        time_window: Max time difference (seconds) to consider as potential echo
        word_overlap_threshold: Fraction of mic words found in system text (0-1)
        seq_match_threshold: SequenceMatcher ratio above which echo is detected

    Returns:
        Filtered list with echo duplicates removed
    """
    if not segments:
        return segments

    sys_segments = [s for s in segments if s["speaker"] == "them"]
    if not sys_segments:
        return segments

    echo_indices = set()

    for i, seg in enumerate(segments):
        if seg["speaker"] != "you":
            continue

        mic_text = seg["text"]
        mic_start = seg["start"]

        # Collect nearby system segments
        nearby = [s for s in sys_segments
                  if abs(mic_start - s["start"]) <= time_window]
        if not nearby:
            continue

        # Pass 1: Word overlap against combined nearby system text
        combined_sys = " ".join(s["text"] for s in nearby)
        overlap = _word_overlap(mic_text, combined_sys)
        if overlap >= word_overlap_threshold:
            echo_indices.add(i)
            continue

        # Pass 2: Sequence matching for short mic segments.
        # Catches echo where Whisper transcribes completely different words
        # from mic vs system (same audio, different text). Only applied to
        # short segments (< 6 words) where word overlap is unreliable due
        # to the small word count.
        mic_words_count = len(mic_text.split())
        if mic_words_count < 6:
            mic_lower = mic_text.lower().strip()
            for sys_seg in nearby:
                sys_lower = sys_seg["text"].lower().strip()
                ratio = SequenceMatcher(None, mic_lower, sys_lower).ratio()
                if ratio >= seq_match_threshold:
                    echo_indices.add(i)
                    break

    return [s for i, s in enumerate(segments) if i not in echo_indices]
