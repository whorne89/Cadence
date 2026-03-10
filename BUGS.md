# Cadence Bug Tracker

## Open

### BUG-004: Short utterances systematically dropped
- **Reported:** 2026-03-02
- **Severity:** Medium
- **Status:** Open
- **Description:** Short mic utterances (1-3 words like "Yeah", "OK") from the local speaker are missed. Low-energy mic signals for brief acknowledgments fall below detection thresholds.
- **Fix approach:** Lower silence threshold for mic, ensure sub-2-second speech isn't filtered out.

### BUG-006: Word-level transcription errors
- **Reported:** 2026-03-02
- **Severity:** Low (tackle later)
- **Status:** Open — cataloged for future improvement
- **Description:** ~5-7% word error rate compared to Teams. Most errors are minor substitutions. Mostly a model-size limitation (using `base`). Proper nouns will always be hard without custom vocabulary.

### BUG-007: Timestamp drift
- **Reported:** 2026-03-02
- **Severity:** Low
- **Status:** Open — cataloged
- **Description:** Cadence timestamps drift 0-19 seconds relative to Teams throughout the meeting. Not a fixed offset — variable and cumulative due to batch transcription with polling windows.

## Closed (v3.0.0)

### BUG-001: Speaker rename fails after transcript file rename
- **Closed:** 2026-03-09 — Speaker name button replaced by "Meeting with" participant button in v3.0.0.

### BUG-002: Echo bleed — other speaker's audio duplicated across channels
- **Closed:** 2026-03-09 — Resolved by removing dual-channel capture in v3.0.0.

### BUG-003: Whisper hallucination — foreign language output
- **Closed:** 2026-03-09 — Mitigated by hallucination filter (non-ASCII detection + English vocabulary check + filler-only removal).

### BUG-005: Segment fragmentation in post-processed transcript
- **Closed:** 2026-03-09 — Mitigated by merge_segments (combines consecutive segments within 2s gap).

### BUG-008: Echo bleed in "you" segments
- **Closed:** 2026-03-09 — Resolved by removing dual-channel capture in v3.0.0.
