"""Tests for merge_segments and _filter_hallucinations."""

import pytest


class TestMergeSegments:
    """Tests for echo_gate.merge_segments."""

    def test_empty_list(self):
        from src.core.echo_gate import merge_segments
        assert merge_segments([]) == []

    def test_single_segment(self):
        from src.core.echo_gate import merge_segments
        segs = [{"speaker": "you", "text": "hello", "start": 0.0}]
        result = merge_segments(segs)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_same_speaker_within_gap(self):
        from src.core.echo_gate import merge_segments
        segs = [
            {"speaker": "them", "text": "first part", "start": 0.0},
            {"speaker": "them", "text": "second part", "start": 1.5},
        ]
        result = merge_segments(segs, max_gap_s=2.0)
        assert len(result) == 1
        assert result[0]["text"] == "first part second part"
        assert result[0]["start"] == 0.0

    def test_same_speaker_beyond_gap(self):
        from src.core.echo_gate import merge_segments
        segs = [
            {"speaker": "them", "text": "first part", "start": 0.0},
            {"speaker": "them", "text": "second part", "start": 5.0},
        ]
        result = merge_segments(segs, max_gap_s=2.0)
        assert len(result) == 2

    def test_different_speakers_not_merged(self):
        from src.core.echo_gate import merge_segments
        segs = [
            {"speaker": "you", "text": "hello", "start": 0.0},
            {"speaker": "them", "text": "hi there", "start": 0.5},
        ]
        result = merge_segments(segs, max_gap_s=2.0)
        assert len(result) == 2

    def test_chain_merging_uses_last_start(self):
        """Three segments each 1.5s apart should all merge (gap from last, not first)."""
        from src.core.echo_gate import merge_segments
        segs = [
            {"speaker": "them", "text": "one", "start": 0.0},
            {"speaker": "them", "text": "two", "start": 1.5},
            {"speaker": "them", "text": "three", "start": 3.0},
        ]
        result = merge_segments(segs, max_gap_s=2.0)
        assert len(result) == 1
        assert result[0]["text"] == "one two three"

    def test_mixed_speakers_interleaved(self):
        from src.core.echo_gate import merge_segments
        segs = [
            {"speaker": "you", "text": "question", "start": 0.0},
            {"speaker": "them", "text": "answer part 1", "start": 1.0},
            {"speaker": "them", "text": "answer part 2", "start": 2.0},
            {"speaker": "you", "text": "follow up", "start": 4.0},
        ]
        result = merge_segments(segs, max_gap_s=2.0)
        assert len(result) == 3
        assert result[0]["text"] == "question"
        assert result[1]["text"] == "answer part 1 answer part 2"
        assert result[2]["text"] == "follow up"


class TestFilterHallucinations:
    """Tests for main._filter_hallucinations."""

    def _filter(self, segments):
        from src.main import _filter_hallucinations
        return _filter_hallucinations(segments)

    def test_normal_english_passes(self):
        segs = [{"speaker": "you", "text": "Hello world", "start": 0.0}]
        assert len(self._filter(segs)) == 1

    def test_dutch_hallucination_accented_filtered(self):
        """Non-ASCII accented Dutch text should be caught by character ratio filter."""
        segs = [{"speaker": "you", "text": "Düs hët mïj hëëft lükt hët äpp gïngën", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_dutch_hallucination_ascii_filtered(self):
        """ASCII Dutch text should be caught by low English vocabulary coverage."""
        segs = [{"speaker": "you", "text": "Dus het mij heeft lukt het app gingen", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_empty_text_filtered(self):
        segs = [{"speaker": "you", "text": "", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_whitespace_only_filtered(self):
        segs = [{"speaker": "you", "text": "   ", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_punctuation_only_filtered(self):
        segs = [{"speaker": "you", "text": "...", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_filler_um_filtered(self):
        segs = [{"speaker": "you", "text": "um", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_filler_the_filtered(self):
        segs = [{"speaker": "you", "text": "the", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_filler_so_filtered(self):
        segs = [{"speaker": "you", "text": "So.", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_short_real_speech_kept(self):
        """Short but meaningful text like 'Yeah' should NOT be filtered."""
        segs = [{"speaker": "you", "text": "Yeah", "start": 0.0}]
        result = self._filter(segs)
        assert len(result) == 1

    def test_two_filler_words_filtered(self):
        segs = [{"speaker": "you", "text": "um, so", "start": 0.0}]
        assert len(self._filter(segs)) == 0

    def test_three_word_sentence_kept(self):
        """Three real words should not be filtered even if short."""
        segs = [{"speaker": "you", "text": "That sounds good", "start": 0.0}]
        assert len(self._filter(segs)) == 1

    def test_long_english_sentence_kept(self):
        """Normal English sentence with 4+ words should pass the vocabulary check."""
        segs = [{"speaker": "you", "text": "I think we should push the timeline back a bit", "start": 0.0}]
        assert len(self._filter(segs)) == 1

    def test_english_with_uncommon_words_kept(self):
        """English with domain-specific words should still pass (enough common words)."""
        segs = [{"speaker": "you", "text": "The biorad assay protocol needs updating for compliance", "start": 0.0}]
        assert len(self._filter(segs)) == 1

    def test_mixed_list_filters_only_bad(self):
        segs = [
            {"speaker": "you", "text": "Hello world", "start": 0.0},
            {"speaker": "you", "text": "...", "start": 1.0},
            {"speaker": "them", "text": "um", "start": 2.0},
            {"speaker": "them", "text": "Good morning", "start": 3.0},
        ]
        result = self._filter(segs)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert result[1]["text"] == "Good morning"


class TestReverseOverlapGuard:
    """Test that reverse overlap doesn't remove genuine speech with common words."""

    def test_short_them_segment_not_used_for_reverse_check(self):
        """A short 'them' segment with common words should NOT cause 'you' removal."""
        from src.core.echo_gate import deduplicate_segments
        segments = [
            {"speaker": "them", "text": "We should do that", "start": 0.0},
            {"speaker": "you", "text": "We should definitely revisit that proposal before we decide", "start": 1.0},
        ]
        result = deduplicate_segments(segments)
        # "you" segment should survive — "them" has only 4 words, below the 5-word guard
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 1

    def test_long_them_segment_catches_echo_with_recovery(self):
        """A long 'them' segment whose words appear in 'you' should recover unique clauses."""
        from src.core.echo_gate import deduplicate_segments
        segments = [
            {"speaker": "them", "text": "So the big thing with bio rad is that timing is important and we need to push them", "start": 10.0},
            {"speaker": "you", "text": "I built some boards. So the big thing with bio rad is that timing is important and we need to push them", "start": 10.5},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        # Should recover "I built some boards." instead of removing entirely
        assert len(you_segs) == 1
        assert "I built some boards" in you_segs[0]["text"]


class TestExtractUniqueClauses:
    """Tests for echo_gate._extract_unique_clauses."""

    def test_recovers_unique_clauses(self):
        """Mixed segment should keep non-echo clauses."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "So what do you mean? They're calling your baby ugly. No, this is me being hyperbolic."
        sys = "No, this is me being hyperbolic about the situation."
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "what do you mean" in result.lower()
        assert "hyperbolic" not in result.lower()

    def test_all_echo_returns_none(self):
        """Pure echo should return None."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "The timing is important and we need to push them."
        sys = "The timing is important and we need to push them forward."
        result = _extract_unique_clauses(mic, sys)
        assert result is None

    def test_no_echo_returns_full_text(self):
        """No overlap should keep everything."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "I built some boards yesterday. They look pretty good."
        sys = "The timeline needs to be adjusted for compliance."
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "boards" in result
        assert "good" in result

    def test_minimum_word_threshold(self):
        """Single non-acknowledgment word below min_words should return None."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "But. The timing is important and we need to push them."
        sys = "The timing is important and we need to push them."
        result = _extract_unique_clauses(mic, sys)
        # "But" alone is 1 word and not an acknowledgment — should return None
        assert result is None

    def test_acknowledgment_word_prefix_recovered(self):
        """Single acknowledgment word like 'Ok' should be recovered by prefix extraction."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "Ok. The timing is important and we need to push them."
        sys = "The timing is important and we need to push them."
        result = _extract_unique_clauses(mic, sys)
        # v3: "Ok" is an acknowledgment word, prefix extraction recovers it
        assert result is not None
        assert "ok" in result.lower()

    def test_comma_fallback_for_long_sentence(self):
        """12+ word sentence without periods should split on commas."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "I think we should revisit the proposal, because the timing is important and we need to push them forward now"
        sys = "because the timing is important and we need to push them forward now"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "revisit the proposal" in result

    def test_empty_mic_text(self):
        from src.core.echo_gate import _extract_unique_clauses
        assert _extract_unique_clauses("", "some text") is None

    def test_empty_sys_text(self):
        from src.core.echo_gate import _extract_unique_clauses
        result = _extract_unique_clauses("Hello world today", "")
        assert result == "Hello world today"


class TestDeduplicateWithClauseRecovery:
    """Tests for deduplicate_segments with clause-level recovery."""

    def test_mixed_segment_partially_recovered(self):
        """Integration test: mixed segment should have echo clauses removed."""
        from src.core.echo_gate import deduplicate_segments
        segments = [
            {"speaker": "them", "text": "No this is me being hyperbolic about the situation", "start": 5.0},
            {"speaker": "you", "text": "So what do you mean? They are calling your baby ugly. No this is me being hyperbolic about the situation.", "start": 5.5},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 1
        assert "what do you mean" in you_segs[0]["text"].lower()
        assert "hyperbolic" not in you_segs[0]["text"].lower()

    def test_pure_echo_still_removed(self):
        """Pure echo should still be fully removed (no regression)."""
        from src.core.echo_gate import deduplicate_segments
        segments = [
            {"speaker": "them", "text": "The project timeline needs adjustment before we proceed", "start": 5.0},
            {"speaker": "you", "text": "The project timeline needs adjustment before we proceed", "start": 5.5},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0


# ═══════════════════════════════════════════════════════════════════
# v3 Phase 3: Prefix/Suffix Extraction Tests
# ═══════════════════════════════════════════════════════════════════


class TestPrefixSuffixExtraction:
    """Tests for _extract_prefix_suffix and its integration into _extract_unique_clauses."""

    def test_acknowledgment_prefix_recovered(self):
        """'Okay' prefix followed by echo should recover 'Okay'."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "Okay sign a five room deal by March 31st"
        sys = "sign a five room deal by March 31st"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert result.lower().startswith("okay")

    def test_sure_yes_prefix_recovered(self):
        """'sure Yes' prefix followed by echo should recover prefix."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "sure Yes the timeline needs to be adjusted for compliance"
        sys = "the timeline needs to be adjusted for compliance requirements"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "sure" in result.lower()

    def test_pure_echo_no_prefix_returns_none(self):
        """Pure echo with no genuine prefix should return None."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "sign a five room deal by March 31st"
        sys = "sign a five room deal by March 31st"
        result = _extract_unique_clauses(mic, sys)
        assert result is None

    def test_genuine_speech_untouched(self):
        """Completely genuine speech should be returned as-is."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "I built some boards yesterday and they look great"
        sys = "the timeline needs to be adjusted for compliance"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "boards" in result

    def test_trailing_suffix_recovered(self):
        """Echo prefix + genuine suffix should recover the suffix."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "the timeline needs adjustment okay I understand"
        sys = "the timeline needs adjustment for compliance"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "okay" in result.lower() or "understand" in result.lower()

    def test_three_word_prefix_recovered_without_acknowledgment(self):
        """A 3+ word prefix should be recovered even without acknowledgment words."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "I built boards sign a five room deal by March 31st"
        sys = "sign a five room deal by March 31st"
        result = _extract_unique_clauses(mic, sys)
        assert result is not None
        assert "built" in result.lower()

    def test_single_non_acknowledgment_word_not_recovered(self):
        """A single word that's NOT an acknowledgment should NOT be recovered."""
        from src.core.echo_gate import _extract_unique_clauses
        mic = "The sign a five room deal by March 31st"
        sys = "sign a five room deal by March 31st"
        result = _extract_unique_clauses(mic, sys)
        # "The" is not an acknowledgment word and is only 1 word
        assert result is None

    def test_dedup_integration_prefix_recovery(self):
        """Integration: deduplicate_segments should recover prefix via clause recovery."""
        from src.core.echo_gate import deduplicate_segments
        segments = [
            {"speaker": "them", "text": "sign a five room deal by March 31st", "start": 2.0},
            {"speaker": "you", "text": "Okay sign a five room deal by March 31st", "start": 2.5},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 1
        assert "okay" in you_segs[0]["text"].lower()
