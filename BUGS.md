# Cadence Bug Tracker

## Open

### BUG-001: Speaker rename fails after transcript file rename
- **Reported:** 2026-03-02
- **Severity:** Medium
- **Status:** Open — needs investigation
- **Description:** After renaming a transcript file in the UI, the "rename speaker" feature stops working for that transcript. The speaker label cannot be changed after the file has been renamed.
- **Steps to reproduce:** TBD — need to investigate the rename flow
- **Suspected cause:** Likely a stale file path reference; the rename_transcript updates the filename but something downstream may still hold the old path when trying to update the Speaker header.

### BUG-002: Echo bleed — other speaker's audio duplicated across channels
- **Reported:** 2026-03-02
- **Severity:** High
- **Status:** Investigated — fix needed
- **Reference session:** Bio-rad 3/2, `.cadence/echo_debug/20260302_115945/`
- **Reference ground truth:** Teams transcript provided by user (see conversation 2026-03-02)
- **Description:** The other speaker's words get captured on the mic channel and attributed to "You", causing duplication. The echo gate is too lenient — only suppressed 2 of 17 chunks.
- **Specific findings from Teams comparison:**
  - At 11:04, Paul's sentence "So the big thing with this with the Bio Rad..." appears BOTH as Speaker (correct) AND mashed into a You segment (incorrect). Near word-for-word duplicate that dedup should catch.
  - Most chunks have mic/sys ratio well below 1.0 (mic quieter than system) yet passed the echo gate. Ratios like 0.23-0.35 are clearly system-dominant but not suppressed.
  - The echo gate threshold (0.65) is too lenient for this scenario.
  - Post-processing cleaned up most echo bleed successfully (live had extensive duplication, final transcript was mostly correct), but the 11:04 case slipped through.
- **Fix approach:** Strengthen fuzzy text dedup in post-processing to catch near-identical segments across speakers. Consider lowering the echo gate ratio threshold.

### BUG-003: Whisper hallucination — foreign language output
- **Reported:** 2026-03-02
- **Severity:** Medium
- **Status:** Open
- **Description:** Whisper produces foreign-language hallucinations on low-energy or noise-only audio segments. Example: "Dus het mij heeft lukt het app gingen" at 195.8s in Bio-rad session (Dutch nonsense). This appeared in the final saved transcript at [03:15].
- **Notes:** Known whisper behavior. `language="en"` is set in config but doesn't fully prevent this. Mitigation options:
  - Post-processing filter to detect and strip non-English output
  - Minimum energy threshold before sending audio to whisper
  - Confidence score filtering (whisper returns log_prob)

### BUG-004: Short "You" utterances systematically dropped
- **Reported:** 2026-03-02
- **Severity:** High
- **Status:** Open
- **Description:** Short mic utterances (1-3 words like "Yeah", "OK", "Yeah, that's definitely a minimum") from the local speaker are systematically missed. Teams captured ~15-20 such utterances from Will that Cadence dropped entirely. Meanwhile, the other speaker's short utterances ("Yeah", "OK") are captured because system audio comes through at consistent volume.
- **Root cause:** Low-energy mic signals for brief acknowledgments fall below detection thresholds or get lost between the 5-second polling windows. The mic channel is inherently quieter/more variable than system audio.
- **Impact:** Makes it look like "You" barely participated in the conversation. Loses important contextual acknowledgments and brief questions.
- **Specific examples from Bio-rad session (all missing in Cadence but present in Teams):**
  - "Yeah. How are you?" (0:03)
  - "Yeah, yeah, I am." (0:08)
  - "I know." (0:13)
  - "Yeah, that's definitely a minimum." (7:30)
  - "Yeah." (7:45, 8:01, 8:25)
  - "Who's going to be a part of the group?" (8:18)
  - "Yeah, I would take that." (10:34)
  - "So all right, sweet." (10:57)
- **Fix approach:** Lower VAD threshold for mic channel, ensure sub-2-second speech isn't filtered out, or adjust the polling/buffering to not lose short utterances at window boundaries.

### BUG-005: Segment fragmentation in post-processed transcript
- **Reported:** 2026-03-02
- **Severity:** Medium
- **Status:** Open
- **Note:** Live fragmentation is acceptable — this only needs fixing in post-processing.
- **Description:** Post-processing produces too many small fragments. What should be one continuous thought gets split into 3-5 segments, including standalone filler words ("um", "the", "so"). Teams captures the same speech as 1 segment.
- **Examples from Bio-rad session:**
  - Paul's sentence about pre-sales language → 5 Cadence segments including standalone "um"
  - Paul's opening about Bio-Rad plans → 4 Cadence segments where Teams has 1
  - Single-word segments: [01:41] "the", [02:19] "so", [02:51] "to see", [06:48] "um", [07:03] "um"
- **Root cause:** The 5-second batch polling window creates artificial breaks in continuous speech.
- **Fix approach:** Post-processing should merge consecutive same-speaker segments that are close in time. Strip isolated filler-only segments.

### BUG-006: Word-level transcription errors
- **Reported:** 2026-03-02
- **Severity:** Low (tackle later)
- **Status:** Open — cataloged for future improvement
- **Description:** ~5-7% word error rate compared to Teams. Most errors are minor substitutions but some change meaning.
- **Specific errors from Bio-rad session (Cadence → Teams ground truth):**
  - "you on" → "Yann" (proper noun lost)
  - "of knowing" → "annoying" (meaning changed)
  - "assigned PO" → "a signed PO" (word boundary error)
  - "old gold company" → "old, old company" (substitution)
  - "I obey a" → "iObeya/Ayoveda" (product name garbled)
  - "did this shirt on" → "threw this shirt on" (verb substitution)
  - "$26,000 off" → "$26,000 op" (substitution)
  - "old and language" → "olden language" (substitution)
  - "turned it" → "termed it" (substitution)
  - "I know deal" → "find out" (garbled)
  - "the or demo board" → "your demo boards" (substitution)
  - "of knowing that" → "annoying that" (meaning changed)
  - "I'll benefit" → "All beneficial" (substitution)
  - "with it starts now" → "that starts now" (substitution)
- **Notes:** Mostly a model-size limitation (using `base`). The `small` reprocess model should improve this. Proper nouns will always be hard without custom vocabulary.

### BUG-007: Timestamp drift
- **Reported:** 2026-03-02
- **Severity:** Low
- **Status:** Open — cataloged
- **Description:** Cadence timestamps drift 0-19 seconds relative to Teams throughout the meeting. Not a fixed offset — it's variable and cumulative. Early segments show ~3-8s offset, later segments up to 19s.
- **Root cause:** Batch transcription with 5s polling windows introduces variable latency. Timestamps are approximate rather than precise.

## Closed

(none yet)
