"""
Echo detection for cross-talk suppression.

When using speakers, the mic picks up system audio (echo). This module
provides tools to detect and suppress this cross-talk:

1. Audio-level: Cross-correlation between mic and system audio chunks
   to detect echo during live transcription.
2. Text-level: Duplicate segment removal during post-processing.
"""

import numpy as np
from difflib import SequenceMatcher


def is_echo(mic_audio, sys_audio, threshold=0.3, sample_rate=16000):
    """
    Check if mic audio is echo of system audio using Pearson correlation.

    Args:
        mic_audio: Mic audio chunk (float32 numpy array)
        sys_audio: System audio chunk from same time window (float32 numpy array)
        threshold: Correlation above which echo is detected (0.0-1.0)
        sample_rate: Audio sample rate in Hz

    Returns:
        bool: True if mic audio appears to be echo of system audio
    """
    # Need at least 100ms of audio for a meaningful comparison
    min_samples = int(sample_rate * 0.1)
    if len(mic_audio) < min_samples or len(sys_audio) < min_samples:
        return False

    # Align lengths
    min_len = min(len(mic_audio), len(sys_audio))
    mic = mic_audio[:min_len].astype(np.float64)
    sys_arr = sys_audio[:min_len].astype(np.float64)

    # If system audio is too quiet, it can't cause audible echo
    sys_rms = np.sqrt(np.mean(sys_arr ** 2))
    if sys_rms < 0.005:
        return False

    # Center both signals
    mic_centered = mic - mic.mean()
    sys_centered = sys_arr - sys_arr.mean()

    mic_std = mic_centered.std()
    sys_std = sys_centered.std()

    if mic_std < 1e-8 or sys_std < 1e-8:
        return False

    # Pearson correlation (zero-lag is sufficient for multi-second chunks
    # where speaker-to-mic delay of ~1-10ms is negligible)
    correlation = np.mean(mic_centered * sys_centered) / (mic_std * sys_std)
    return bool(abs(correlation) > threshold)


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


def deduplicate_segments(segments, time_window=3.0, similarity_threshold=0.6):
    """
    Remove mic segments that are echo of system audio segments.

    Compares overlapping mic ("you") and system ("them") segments by
    text similarity. If a mic segment closely matches a system segment
    within the time window, it's echo and gets removed.

    Args:
        segments: List of {"speaker", "text", "start"} dicts, sorted by start time
        time_window: Max time difference (seconds) to consider as potential echo
        similarity_threshold: Text similarity (0-1) above which echo is detected

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

        mic_text = seg["text"].lower().strip()
        mic_start = seg["start"]

        for sys_seg in sys_segments:
            # Check time proximity
            if abs(mic_start - sys_seg["start"]) > time_window:
                continue

            # Compare text similarity
            sys_text = sys_seg["text"].lower().strip()
            similarity = SequenceMatcher(None, mic_text, sys_text).ratio()
            if similarity > similarity_threshold:
                echo_indices.add(i)
                break

    return [s for i, s in enumerate(segments) if i not in echo_indices]
