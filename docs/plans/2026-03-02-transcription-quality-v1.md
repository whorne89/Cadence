# Transcription Quality Improvements v1

**Date:** 2026-03-02
**Baseline:** Bio-rad 3/2 session — 6/10 overall, compared against Teams ground truth
**Goal:** Measurable improvement on next benchmark test

## Fixes in This Iteration

### Fix 1: Short "You" utterances dropped (BUG-004)
**Approach:** Separate silence detection thresholds per channel

**Problem:** `speech_duration > 0.5` and `silence_threshold=0.01` applied uniformly to both mic and system. Short mic utterances ("Yeah", "OK") are quiet and brief — they either fall below the energy threshold or are under 0.5s and get rejected.

**Changes:**
- `TranscriptionWorker.__init__` accepts separate `mic_silence_threshold` and `sys_silence_threshold`
- Mic gets lower threshold (0.005) vs system (0.01) — mic is inherently quieter
- Mic gets shorter `min_speech_duration` (0.3s) vs system (0.5s) — allows brief acknowledgments
- System keeps current thresholds — its audio is consistent volume from WASAPI loopback

**Files modified:** `src/main.py` (TranscriptionWorker init + run method)

**Risk:** Lower mic threshold may capture more noise/echo. Mitigated by existing echo gate + text dedup downstream.

### Fix 2: Echo bleed dedup (BUG-002)
**Approach:** Substring/reverse overlap check in deduplicate_segments

**Problem:** At 11:04, Will's real speech + Paul's echoed speech got concatenated into one "you" segment. The overall word overlap was below 50% because Will's real words diluted it, but the tail was a near-perfect match of Paul's words.

**Changes:**
- In `deduplicate_segments`, add a reverse overlap check: if a significant portion (60%+) of a "them" segment's words appear inside a "you" segment, flag the "you" segment
- This catches the case where echo gets concatenated with real speech

**Files modified:** `src/core/echo_gate.py` (deduplicate_segments function)

### Fix 3: Whisper hallucination filtering (BUG-003)
**Approach:** Two-layer defense — language enforcement + post-processing filter

**Problem:** Whisper produced Dutch gibberish ("Dus het mij heeft lukt het app gingen") on low-energy audio despite `language="en"` being in the config. The language parameter was never passed to the transcribe calls.

**Changes:**
1. Pass `language="en"` from config to all `transcribe()` and `transcribe_text()` calls
2. Add a post-processing filter in `PostProcessWorker` that strips segments with non-ASCII-letter-heavy content or known hallucination patterns

**Files modified:** `src/main.py` (TranscriptionWorker, PostProcessWorker), `src/core/transcriber.py`

### Fix 4: Segment merging in post-processing (BUG-005)
**Approach:** Time-based merging of consecutive same-speaker segments

**Problem:** Post-processing produces 1.75x more segments than Teams. Continuous speech broken into 3-5 fragments. Standalone fillers ("um", "the", "so") clutter the transcript.

**Changes:**
- After dedup in `PostProcessWorker`, merge consecutive same-speaker segments within 2s gap
- Strip standalone filler-only segments (under 3 words, all filler)

**Files modified:** `src/core/echo_gate.py` (new merge_segments function), `src/main.py` (PostProcessWorker)

## Expected Impact on Benchmark Metrics
- Short utterance capture: 23% → target 60%+
- Echo bleed instances: 1 → target 0
- Hallucination instances: 1 → target 0
- Segment fragmentation ratio: 1.75:1 → target 1.2:1 or better
- Overall quality: 6/10 → target 7-8/10
