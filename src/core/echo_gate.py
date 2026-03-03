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

import re

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


ACKNOWLEDGMENT_WORDS = {"okay", "ok", "sure", "yes", "yeah", "right",
                        "no", "well", "so", "alright", "yep", "absolutely",
                        "definitely", "exactly"}


def _extract_unique_clauses(mic_text, sys_text, overlap_threshold=0.6, min_words=3):
    """
    Split mic_text into clauses and keep only those that don't overlap with sys_text.

    When a mic segment contains both genuine speech and echoed speech (e.g.,
    "So what do you mean? No, this is me being hyperbolic."), this function
    recovers the genuine clauses by checking each independently.

    If clause splitting can't isolate the genuine part (single clause, no
    punctuation), falls back to word-boundary prefix/suffix extraction:
    scans short prefixes/suffixes and recovers them if the remaining text
    is clearly echo.

    Args:
        mic_text: Text from the mic channel (may contain mixed speech + echo)
        sys_text: Text from the system channel (the echo source)
        overlap_threshold: Per-clause word overlap above which a clause is echo
        min_words: Minimum surviving words to return a result

    Returns:
        Recovered text string with echo clauses removed, or None if
        nothing meaningful survives
    """
    if not mic_text or not sys_text:
        return mic_text if mic_text else None

    # Split on sentence-ending punctuation
    clauses = re.split(r'(?<=[.?!])\s+', mic_text.strip())

    # For long single-sentence segments (12+ words, no split), fall back to commas
    if len(clauses) == 1:
        words = clauses[0].split()
        if len(words) >= 12:
            clauses = re.split(r',\s+', clauses[0])

    # Check each clause independently
    surviving = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        overlap = _word_overlap(clause, sys_text)
        if overlap < overlap_threshold:
            surviving.append(clause)

    if not surviving:
        # Clause splitting couldn't recover anything — try prefix/suffix extraction.
        # This handles cases like "Okay sign a five room deal by March 31st"
        # where a 1-word genuine prefix is glued to echo without punctuation.
        recovered = _extract_prefix_suffix(mic_text, sys_text, overlap_threshold)
        return recovered

    recovered = " ".join(surviving)
    # Check minimum word count
    strip = str.maketrans("", "", ".,!?;:'\"()-")
    clean_words = recovered.translate(strip).split()
    if len(clean_words) < min_words:
        # Below min_words from clause splitting — try prefix/suffix as fallback
        prefix_suffix = _extract_prefix_suffix(mic_text, sys_text, overlap_threshold)
        return prefix_suffix

    return recovered


def _extract_prefix_suffix(mic_text, sys_text, overlap_threshold=0.6):
    """
    Extract genuine prefix or suffix from a segment where the rest is echo.

    Scans 1-4 word prefixes: if the suffix has high overlap with sys_text
    (clearly echo) but the prefix does not, recover the prefix. Same logic
    applied in reverse for trailing suffixes.

    Only recovers short prefixes (1-2 words) if they contain a recognized
    acknowledgment word. Prefixes of 3+ words are recovered unconditionally.

    Returns:
        Recovered prefix/suffix text, or None if nothing can be extracted.
    """
    words = mic_text.strip().split()
    if len(words) < 2:
        return None

    strip_punct = str.maketrans("", "", ".,!?;:'\"()-")

    # Try prefix extraction (genuine prefix + echo suffix)
    for prefix_len in range(1, min(5, len(words))):
        prefix = " ".join(words[:prefix_len])
        suffix = " ".join(words[prefix_len:])
        if not suffix:
            continue
        suffix_overlap = _word_overlap(suffix, sys_text)
        prefix_overlap = _word_overlap(prefix, sys_text)
        if suffix_overlap >= 0.8 and prefix_overlap < overlap_threshold:
            prefix_clean = [w.lower().translate(strip_punct) for w in words[:prefix_len]]
            if prefix_len >= 3 or any(w in ACKNOWLEDGMENT_WORDS for w in prefix_clean):
                return prefix

    # Try suffix extraction (echo prefix + genuine suffix)
    for suffix_len in range(1, min(5, len(words))):
        suffix = " ".join(words[len(words) - suffix_len:])
        prefix = " ".join(words[:len(words) - suffix_len])
        if not prefix:
            continue
        prefix_overlap = _word_overlap(prefix, sys_text)
        suffix_overlap = _word_overlap(suffix, sys_text)
        if prefix_overlap >= 0.8 and suffix_overlap < overlap_threshold:
            suffix_clean = [w.lower().translate(strip_punct) for w in words[len(words) - suffix_len:]]
            if suffix_len >= 3 or any(w in ACKNOWLEDGMENT_WORDS for w in suffix_clean):
                return suffix

    return None


def deduplicate_segments(segments, time_window=30.0, word_overlap_threshold=0.5,
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
    replacements = {}  # index -> recovered text (clause-level recovery)

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

        combined_sys = " ".join(s["text"] for s in nearby)

        # Pass 1: Word overlap against combined nearby system text
        overlap = _word_overlap(mic_text, combined_sys)
        if overlap >= word_overlap_threshold:
            # Try clause-level recovery before full removal
            recovered = _extract_unique_clauses(mic_text, combined_sys)
            if recovered is not None:
                replacements[i] = recovered
            else:
                echo_indices.add(i)
            continue

        # Pass 2: Reverse overlap — check if a "them" segment's words are
        # largely contained inside this "you" segment. Catches echo that
        # got concatenated with real speech (e.g., Will's words + Paul's
        # echoed words in one "you" segment).
        # Only check "them" segments with 5+ words to avoid false positives
        # from short common-word segments like "We should do that".
        for sys_seg in nearby:
            sys_words = sys_seg["text"].split()
            if len(sys_words) < 5:
                continue
            reverse_overlap = _word_overlap(sys_seg["text"], mic_text)
            if reverse_overlap >= 0.6:
                # Try clause-level recovery before full removal
                recovered = _extract_unique_clauses(mic_text, combined_sys)
                if recovered is not None:
                    replacements[i] = recovered
                else:
                    echo_indices.add(i)
                break
        if i in echo_indices or i in replacements:
            continue

        # Pass 3: Sequence matching for short mic segments.
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

        if i in echo_indices or i in replacements:
            continue

        # Pass 3b: SequenceMatcher for medium segments (6-15 words).
        # Extends Pass 3 to catch echo where Whisper wording differences
        # between channels reduce word overlap below threshold. Uses a
        # higher ratio (0.55) than Pass 3 to be conservative with longer text.
        if 6 <= mic_words_count <= 15:
            mic_lower = mic_text.lower().strip()
            for sys_seg in nearby:
                ratio = SequenceMatcher(None, mic_lower, sys_seg["text"].lower().strip()).ratio()
                if ratio >= 0.55:
                    recovered = _extract_unique_clauses(mic_text, combined_sys)
                    if recovered is not None:
                        replacements[i] = recovered
                    else:
                        echo_indices.add(i)
                    break

    # Build result: remove echo, apply replacements
    result = []
    for i, s in enumerate(segments):
        if i in echo_indices:
            continue
        if i in replacements:
            result.append(dict(s, text=replacements[i]))
        else:
            result.append(s)
    return result


def merge_segments(segments, max_gap_s=2.0):
    """
    Merge consecutive same-speaker segments that are close in time.

    This improves readability by combining fragments that were split by
    the batch polling window into natural utterances.

    Args:
        segments: List of {"speaker", "text", "start"} dicts, sorted by start time
        max_gap_s: Maximum gap between segments to merge (seconds)

    Returns:
        List of merged segments
    """
    if not segments:
        return segments

    merged = [dict(segments[0])]  # copy first segment
    last_merge_start = segments[0]["start"]  # track last constituent's start

    for seg in segments[1:]:
        prev = merged[-1]
        # Merge if same speaker and within time gap of the last merged piece
        if (seg["speaker"] == prev["speaker"]
                and seg["start"] - last_merge_start <= max_gap_s):
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            last_merge_start = seg["start"]
        else:
            merged.append(dict(seg))
            last_merge_start = seg["start"]

    return merged
