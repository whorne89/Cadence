"""
Comprehensive echo gate verification tests.

Tests the two-tier echo gate across every scenario: real bleed data,
user speech, user-over-system, edge cases, regression against old
approaches, post-processing dedup, and full pipeline integration.

Echo gate thresholds (v4, calibrated from 249 chunks / 6 meetings):
  Tier 1: ratio < 1.5 and mic_rms < 0.08
  Tier 2: ratio < 0.65 and mic_rms < 0.15 and sys_rms > 0.015
"""

import math
import pytest

from core.echo_gate import deduplicate_segments


# ── Helper: simulate the live echo gate decision ──────────────────


def echo_gate_decision(mic_rms, sys_rms):
    """Simulate the live echo gate decision from TranscriptionWorker.

    Two-tier check (v4):
    1) Bleed: ratio < 1.5 and mic_rms < 0.08
    2) Loud-system bleed: ratio < 0.65 and mic_rms < 0.15 and sys_rms > 0.015

    Returns True if the audio should be SUPPRESSED (echo detected).
    """
    if sys_rms <= 0.005:
        return False  # System too quiet, no echo possible
    ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
    return (
        (ratio < 1.5 and mic_rms < 0.08) or
        (ratio < 0.65 and mic_rms < 0.15 and sys_rms > 0.015)
    )


def old_echo_gate_decision(mic_rms, sys_rms, ratio_threshold=1.5):
    """Simulate the OLD echo gate (ratio only, no mic_rms floor).

    Returns True if the audio would have been suppressed by the old approach.
    """
    if sys_rms <= 0.005:
        return False
    ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
    return ratio < ratio_threshold


def v3_echo_gate_decision(mic_rms, sys_rms):
    """Simulate the v3 echo gate (mic_rms < 0.040 / 0.055).

    Returns True if the v3 gate would have suppressed.
    """
    if sys_rms <= 0.005:
        return False
    ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
    return (
        (ratio < 1.5 and mic_rms < 0.040) or
        (ratio < 0.75 and mic_rms < 0.055 and sys_rms > 0.020)
    )


# ═══════════════════════════════════════════════════════════════════
# A. LIVE ECHO GATE TESTS
# ═══════════════════════════════════════════════════════════════════


# ── Test 1: Pure bleed — real data points ─────────────────────────


# Real log data from recording sessions
REAL_BLEED_DATA = [
    # Early recordings (mic_rms 0.006-0.019, quiet bleed)
    (0.0099, 0.0522),
    (0.0079, 0.0328),
    (0.0084, 0.0247),
    (0.0109, 0.0141),
    (0.0080, 0.0066),
    (0.0083, 0.0309),
    (0.0090, 0.0573),
    (0.0095, 0.0138),
    (0.0127, 0.0395),
    (0.0085, 0.0325),
    (0.0098, 0.0336),
    (0.0085, 0.0273),
    (0.0084, 0.0214),
    (0.0104, 0.0468),
    (0.0102, 0.0358),
    (0.0089, 0.0207),
    (0.0104, 0.0474),
    (0.0188, 0.0305),
    (0.0089, 0.0341),
    (0.0087, 0.0354),
    (0.0087, 0.0150),
    (0.0082, 0.0137),
    (0.0086, 0.0278),
    (0.0063, 0.0182),
    (0.0083, 0.0394),
    (0.0079, 0.0486),
    (0.0083, 0.0394),
    (0.0061, 0.0108),
    (0.0096, 0.0355),
    (0.0089, 0.0467),
    (0.0108, 0.0301),
    (0.0098, 0.0402),
    (0.0079, 0.0534),
    (0.0089, 0.0442),
    (0.0081, 0.0295),
    (0.0071, 0.0404),
    (0.0083, 0.0356),
    (0.0079, 0.0311),
    (0.0076, 0.0299),
    (0.0067, 0.0231),
    (0.0082, 0.0107),
    (0.0079, 0.0134),
    (0.0101, 0.0324),
    (0.0122, 0.0328),
    (0.0137, 0.0368),
    (0.0097, 0.0230),
    (0.0100, 0.0101),
    (0.0088, 0.0226),
    (0.0158, 0.0449),
    (0.0165, 0.0304),
    (0.0191, 0.0320),
    # iObeya 3/6 — louder bleed (mic_rms 0.029-0.061, the v3 gate missed these)
    (0.0286, 0.0608),  # chunk 1: ratio=0.47 (the only one v3 caught)
    (0.0610, 0.1226),  # chunk 2: ratio=0.50 — ESCAPED v3 (mic > 0.055)
    (0.0579, 0.0972),  # chunk 3: ratio=0.60 — ESCAPED v3
    (0.0612, 0.0954),  # chunk 4: ratio=0.64 — ESCAPED v3
    # Olav 3/3 — typical one-sided meeting bleed
    (0.0165, 0.0308),  # ratio=0.54
    (0.0190, 0.0310),  # ratio=0.61
    (0.0275, 0.0529),  # ratio=0.52
    (0.0350, 0.0650),  # ratio=0.54
    (0.0450, 0.0800),  # ratio=0.56
    (0.0730, 0.1020),  # ratio=0.72 — highest mic bleed observed
]


class TestPureBleedRealData:
    """Test 1: Verify all real bleed data points are suppressed."""

    @pytest.mark.parametrize("mic_rms,sys_rms", REAL_BLEED_DATA)
    def test_individual_bleed_entry(self, mic_rms, sys_rms):
        """Each real bleed entry should be suppressed by the two-tier gate."""
        result = echo_gate_decision(mic_rms, sys_rms)
        assert result is True, (
            f"Bleed NOT suppressed: mic_rms={mic_rms}, sys_rms={sys_rms}, "
            f"ratio={mic_rms/sys_rms:.2f}"
        )

    def test_suppression_rate_all(self):
        """All real bleed entries should be suppressed (100%)."""
        suppressed = sum(1 for m, s in REAL_BLEED_DATA if echo_gate_decision(m, s))
        assert suppressed == len(REAL_BLEED_DATA), (
            f"Expected {len(REAL_BLEED_DATA)} suppressed, got {suppressed}"
        )

    def test_v3_missed_iobea_bleed(self):
        """v3 gate (mic < 0.040/0.055) missed iObeya chunks 2-4."""
        iobea_escaped = [
            (0.0610, 0.1226),  # chunk 2
            (0.0579, 0.0972),  # chunk 3
            (0.0612, 0.0954),  # chunk 4
        ]
        for mic_rms, sys_rms in iobea_escaped:
            assert v3_echo_gate_decision(mic_rms, sys_rms) is False, (
                "v3 should have MISSED this (that's the bug we're fixing)"
            )
            assert echo_gate_decision(mic_rms, sys_rms) is True, (
                f"v4 should CATCH this: mic={mic_rms}, sys={sys_rms}"
            )


# ── Test 2: User speaking alone ──────────────────────────────────


class TestUserSpeakingAlone:
    """Test 2: User speech should NEVER be suppressed.

    Real user speech mic_rms observed: 0.085-0.119 (from bleed plan doc).
    All values well above the 0.08 tier 1 ceiling.
    """

    def test_quiet_user_speaking(self):
        """Quiet user speech: mic_rms=0.085 → NOT suppressed."""
        assert echo_gate_decision(0.085, 0.010) is False

    def test_normal_user_speaking(self):
        """Normal user speech: mic_rms=0.110 → NOT suppressed."""
        assert echo_gate_decision(0.110, 0.010) is False

    def test_quiet_user_system_silent(self):
        """Quiet user, system silent → NOT suppressed (sys below threshold)."""
        assert echo_gate_decision(0.085, 0.002) is False

    def test_loud_user_system_silent(self):
        """Loud user, system silent → NOT suppressed."""
        assert echo_gate_decision(0.150, 0.003) is False

    def test_whisper_no_system(self):
        """Very quiet whisper, no system audio → NOT suppressed."""
        assert echo_gate_decision(0.050, 0.000) is False

    def test_normal_user_no_system(self):
        """Normal user volume, no system audio → NOT suppressed."""
        assert echo_gate_decision(0.120, 0.000) is False

    def test_loud_user_no_system(self):
        """Loud user, no system → NOT suppressed."""
        assert echo_gate_decision(0.200, 0.000) is False

    def test_user_with_very_quiet_background(self):
        """User speaking with very quiet system background."""
        assert echo_gate_decision(0.090, 0.004) is False


# ── Test 3: User speaking OVER system audio (critical case) ──────


class TestUserOverSystemAudio:
    """Test 3: THE primary failure mode we must avoid.

    When user speaks over system audio, mic picks up both voice and bleed.
    mic_rms ~ sqrt(voice^2 + bleed^2), where bleed ~ sys_rms * 0.25.
    ALL of these MUST pass (not be suppressed).

    Voice levels from real data: quiet=0.085, normal=0.12, loud=0.20
    System levels from real data: quiet=0.015, normal=0.06, loud=0.12
    """

    # Voice x System test matrix (realistic levels)
    VOICE_SYSTEM_MATRIX = [
        # (voice_rms, sys_rms, label)
        (0.085, 0.015, "Quiet voice x Quiet system"),
        (0.085, 0.060, "Quiet voice x Normal system"),
        (0.085, 0.120, "Quiet voice x Loud system"),
        (0.120, 0.015, "Normal voice x Quiet system"),
        (0.120, 0.060, "Normal voice x Normal system"),
        (0.120, 0.120, "Normal voice x Loud system"),
        (0.200, 0.015, "Loud voice x Quiet system"),
        (0.200, 0.060, "Loud voice x Normal system"),
        (0.200, 0.120, "Loud voice x Loud system"),
    ]

    @pytest.mark.parametrize("voice_rms,sys_rms,label", VOICE_SYSTEM_MATRIX)
    def test_user_over_system_not_suppressed(self, voice_rms, sys_rms, label):
        """User speaking over system audio should NEVER be suppressed."""
        bleed = sys_rms * 0.25
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
        result = echo_gate_decision(mic_rms, sys_rms)
        assert result is False, (
            f"{label}: SUPPRESSED! mic_rms={mic_rms:.4f}, sys_rms={sys_rms}, "
            f"ratio={mic_rms/sys_rms:.2f}. User speech would be lost!"
        )

    def test_all_9_matrix_entries_pass(self):
        """Verify all 9 voice x system combinations pass."""
        failures = []
        for voice_rms, sys_rms, label in self.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            if echo_gate_decision(mic_rms, sys_rms):
                failures.append(f"{label}: mic_rms={mic_rms:.4f}")
        assert not failures, f"User speech suppressed in: {failures}"

    def test_computed_mic_rms_values_above_floor(self):
        """All user-over-system mic_rms values should be above 0.08."""
        for voice_rms, sys_rms, label in self.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            assert mic_rms > 0.08, (
                f"{label}: mic_rms={mic_rms:.4f} is below floor 0.08"
            )


# ── Test 4: Edge cases ───────────────────────────────────────────


class TestEdgeCases:
    """Test 4: Boundary conditions and edge cases."""

    def test_sys_at_threshold_boundary(self):
        """sys_rms=0.005 (not > 0.005) → NOT suppressed."""
        assert echo_gate_decision(0.008, 0.005) is False

    def test_sys_just_above_threshold(self):
        """sys_rms=0.006 (> 0.005), mic<floor, ratio<1.5 → suppressed."""
        result = echo_gate_decision(0.008, 0.006)
        ratio = 0.008 / 0.006  # 1.333
        assert ratio < 1.5
        assert 0.008 < 0.08
        assert result is True

    def test_mic_exactly_at_floor(self):
        """mic_rms=0.08 exactly → NOT suppressed (not < 0.08)."""
        assert echo_gate_decision(0.08, 0.060) is False

    def test_mic_just_below_floor(self):
        """mic_rms=0.079, ratio<1.5 → suppressed."""
        result = echo_gate_decision(0.079, 0.060)
        ratio = 0.079 / 0.060  # 1.317
        assert ratio < 1.5
        assert 0.079 < 0.08
        assert result is True

    def test_very_loud_system_with_proportional_bleed(self):
        """sys_rms=0.12, mic_rms=0.080 (bleed only), ratio=0.67.
        Tier 2: ratio < 0.65? No (0.67 > 0.65). Tier 1: mic < 0.08? No.
        Leaks — but would be caught by post-AEC check or text dedup."""
        assert echo_gate_decision(0.080, 0.12) is False

    def test_ratio_above_threshold_mic_below_floor(self):
        """ratio>1.5 even though mic<floor → NOT suppressed (ratio check fails)."""
        result = echo_gate_decision(0.010, 0.006)
        ratio = 0.010 / 0.006  # 1.667
        assert ratio > 1.5
        assert result is False

    def test_zero_sys_rms(self):
        """sys_rms=0 → NOT suppressed (below 0.005 threshold)."""
        assert echo_gate_decision(0.010, 0.0) is False

    def test_both_zero(self):
        """Both mic and sys zero → NOT suppressed."""
        assert echo_gate_decision(0.0, 0.0) is False

    def test_very_high_ratio_low_mic(self):
        """High ratio (mic >> sys) with low absolute mic → NOT suppressed."""
        assert echo_gate_decision(0.012, 0.006) is False  # ratio=2.0

    def test_extremely_quiet_system(self):
        """System barely audible, mic has faint bleed → NOT suppressed."""
        assert echo_gate_decision(0.003, 0.004) is False  # sys <= 0.005

    def test_mic_at_maximum_observed_bleed(self):
        """mic_rms at maximum observed bleed (0.073, Olav) → suppressed."""
        assert echo_gate_decision(0.073, 0.102) is True

    def test_iobea_bleed_level(self):
        """iObeya chunk 2 bleed: mic=0.061, sys=0.123, ratio=0.50."""
        assert echo_gate_decision(0.061, 0.123) is True

    def test_tier2_catches_loud_bleed(self):
        """Tier 2 catches loud bleed with very low ratio.
        mic_rms=0.10, sys_rms=0.20, ratio=0.50 → Tier 2 catches."""
        assert echo_gate_decision(0.10, 0.20) is True  # ratio=0.50 < 0.65


# ── Test 5: Regression — v3 approach failed on iObeya ────────────


class TestRegressionV3Approach:
    """Test 5: Verify the v3 approach would miss bleed that v4 catches."""

    def test_v3_missed_moderate_bleed(self):
        """v3 gate (mic < 0.040/0.055) missed iObeya-level bleed."""
        # iObeya chunk 2: mic=0.061, sys=0.123
        assert v3_echo_gate_decision(0.061, 0.123) is False  # v3 missed
        assert echo_gate_decision(0.061, 0.123) is True  # v4 catches

    def test_v3_missed_olav_level_bleed(self):
        """v3 gate missed higher Olav-type bleed."""
        # Olav: mic=0.073, sys=0.102
        assert v3_echo_gate_decision(0.073, 0.102) is False  # v3 missed
        assert echo_gate_decision(0.073, 0.102) is True  # v4 catches

    @pytest.mark.parametrize("voice_rms,sys_rms,label",
                             TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX)
    def test_v4_does_not_suppress_user_speech(self, voice_rms, sys_rms, label):
        """v4 thresholds still protect all user-over-system scenarios."""
        bleed = sys_rms * 0.25
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
        assert echo_gate_decision(mic_rms, sys_rms) is False, (
            f"{label}: v4 incorrectly suppressed! mic_rms={mic_rms:.4f}"
        )

    def test_count_v3_vs_v4_on_bleed(self):
        """v4 catches more bleed than v3 without new false positives."""
        v3_caught = sum(1 for m, s in REAL_BLEED_DATA
                        if v3_echo_gate_decision(m, s))
        v4_caught = sum(1 for m, s in REAL_BLEED_DATA
                        if echo_gate_decision(m, s))
        assert v4_caught > v3_caught, (
            f"v4 should catch more bleed: v3={v3_caught}, v4={v4_caught}"
        )
        assert v4_caught == len(REAL_BLEED_DATA), (
            f"v4 should catch ALL bleed: {v4_caught}/{len(REAL_BLEED_DATA)}"
        )


# ═══════════════════════════════════════════════════════════════════
# B. POST-PROCESSING DEDUP TESTS
# ═══════════════════════════════════════════════════════════════════


# ── Test 6: Leaked bleed caught by dedup ─────────────────────────


class TestLeakedBleedCaughtByDedup:
    """Test 6: The outlier bleed that leaks through live gate gets removed by dedup."""

    def test_leaked_bleed_removed_by_dedup(self):
        """Outlier bleed entry leaks through live gate, dedup catches it."""
        segments = [
            {"speaker": "them", "text": "We have said that we were willing to support the Department.", "start": 10.0},
            {"speaker": "you", "text": "We have said that we were willing to support the Department.", "start": 11.0},  # Echo of above
            {"speaker": "them", "text": "The timeline has been driven by the Department of War.", "start": 20.0},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0, "Leaked bleed segment should be removed by dedup"

    def test_leaked_bleed_partial_text_removed(self):
        """Partial text match from leaked bleed also caught by dedup."""
        segments = [
            {"speaker": "them", "text": "We are willing to support the Department of War for as long as needed.", "start": 10.0},
            {"speaker": "you", "text": "willing to support the Department.", "start": 11.0},  # Partial echo
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0


# ── Test 7: Real transcript data tests still pass ────────────────


class TestRealTranscriptDataReference:
    """Test 7: Verify dedup works on representative samples."""

    def test_dedup_identical_text_removed(self):
        """Identical text from you/them within time window → removed."""
        segments = [
            {"speaker": "them", "text": "This is the quarterly report.", "start": 5.0},
            {"speaker": "you", "text": "This is the quarterly report.", "start": 5.5},
        ]
        result = deduplicate_segments(segments)
        assert len(result) == 1
        assert result[0]["speaker"] == "them"

    def test_dedup_keeps_unique_user_speech(self):
        """Unique user text near system text → kept."""
        segments = [
            {"speaker": "them", "text": "The market is up today.", "start": 5.0},
            {"speaker": "you", "text": "I need to check on the servers.", "start": 6.0},
        ]
        result = deduplicate_segments(segments)
        assert len(result) == 2

    def test_dedup_catches_echo_at_wider_window(self):
        """Echo at 40s gap (within new 45s window) is caught."""
        segments = [
            {"speaker": "them", "text": "The deployment timeline is aggressive and needs review.", "start": 10.0},
            {"speaker": "you", "text": "The deployment timeline is aggressive and needs review.", "start": 48.0},  # 38s gap
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0, "Echo at 38s gap should be caught by 45s window"


# ── Test 8: User-over-system NOT removed by dedup ────────────────


class TestUserOverSystemNotRemovedByDedup:
    """Test 8: When user speaks over system, their words are DIFFERENT."""

    def test_different_content_kept(self):
        """User speech with completely different content is kept."""
        segments = [
            {"speaker": "them", "text": "The quarterly earnings report shows significant growth in revenue.", "start": 10.0},
            {"speaker": "you", "text": "I disagree with that assessment, the numbers don't add up.", "start": 11.0},
            {"speaker": "them", "text": "We need to focus on expanding our market share.", "start": 15.0},
            {"speaker": "you", "text": "What about the customer retention metrics we discussed?", "start": 16.0},
        ]
        result = deduplicate_segments(segments)
        assert len(result) == 4, "All segments should be kept (different text content)"

    def test_user_response_to_system_kept(self):
        """User directly responding to system audio is kept."""
        segments = [
            {"speaker": "them", "text": "What do you think about the proposal?", "start": 5.0},
            {"speaker": "you", "text": "I think it needs more work on the budget section.", "start": 6.0},
        ]
        result = deduplicate_segments(segments)
        assert len(result) == 2

    def test_user_interjection_during_system(self):
        """User interjecting during system playback is kept."""
        segments = [
            {"speaker": "them", "text": "We should consider the implications of this policy change.", "start": 10.0},
            {"speaker": "you", "text": "Wait, can we go back to the previous slide?", "start": 11.5},
            {"speaker": "them", "text": "The financial projections indicate steady growth.", "start": 15.0},
        ]
        result = deduplicate_segments(segments)
        assert len(result) == 3


# ── Test 9: Mixed echo and genuine overlapping speech ────────────


class TestMixedEchoAndGenuineSpeech:
    """Test 9: Some segments are echo, some are genuine speech over system."""

    def test_mixed_scenario(self):
        """Echo removed, genuine overlapping speech kept."""
        segments = [
            {"speaker": "them", "text": "We have said that we were willing to support the Department.", "start": 10.0},
            {"speaker": "you", "text": "We have said that we were willing to support the Department.", "start": 11.0},  # Pure echo
            {"speaker": "them", "text": "The timeline has been driven by the Department of War.", "start": 20.0},
            {"speaker": "you", "text": "But that contradicts what they said last week about the budget.", "start": 21.0},  # Genuine speech
            {"speaker": "them", "text": "We are trying to reach a deal here.", "start": 30.0},
            {"speaker": "you", "text": "trying to reach a deal here.", "start": 31.0},  # Partial echo
        ]
        result = deduplicate_segments(segments)

        you_segs = [s for s in result if s["speaker"] == "you"]
        you_texts = [s["text"] for s in you_segs]

        assert "We have said that we were willing to support the Department." not in you_texts
        assert "But that contradicts what they said last week about the budget." in you_texts
        assert "trying to reach a deal here." not in you_texts

    def test_alternating_echo_and_speech(self):
        """Alternating pattern of echo and real speech."""
        segments = [
            {"speaker": "them", "text": "First we need to address the security concerns.", "start": 1.0},
            {"speaker": "you", "text": "First we need to address the security concerns.", "start": 1.5},  # Echo
            {"speaker": "you", "text": "Actually I have a question about that.", "start": 5.0},  # Real
            {"speaker": "them", "text": "The deployment timeline is aggressive.", "start": 8.0},
            {"speaker": "you", "text": "The deployment timeline is aggressive.", "start": 8.3},  # Echo
            {"speaker": "you", "text": "Can we push it back two weeks?", "start": 12.0},  # Real
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        you_texts = [s["text"] for s in you_segs]

        assert len(you_segs) == 2
        assert "Actually I have a question about that." in you_texts
        assert "Can we push it back two weeks?" in you_texts


# ═══════════════════════════════════════════════════════════════════
# C. FULL PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestFullPipelineIntegration:
    """Test 10: Simulate a complete recording session end-to-end."""

    def test_complete_session_timeline(self):
        """Full timeline: system plays, user silent, user speaks over, etc."""
        events = []

        # 0-10s: System playing, user silent (pure bleed — low mic)
        bleed_entries_0_10 = [
            (0.0085, 0.0350, 2.0),
            (0.0092, 0.0420, 5.0),
            (0.0078, 0.0310, 8.0),
        ]
        for mic_rms, sys_rms, ts in bleed_entries_0_10:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))
            assert suppressed is True, f"Bleed at {ts}s not suppressed"

        # 10-15s: System playing, user speaks over it (realistic levels)
        voice_rms = 0.12
        sys_rms = 0.060
        bleed = sys_rms * 0.25
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
        suppressed = echo_gate_decision(mic_rms, sys_rms)
        events.append(("user_over_system", suppressed, 12.0))
        assert suppressed is False, "User speaking over system was suppressed!"

        # 15-25s: System playing, user silent (moderate bleed — iObeya-like)
        bleed_entries_15_25 = [
            (0.055, 0.100, 17.0),  # ratio=0.55, like iObeya
            (0.060, 0.095, 20.0),  # ratio=0.63, like iObeya
            (0.045, 0.080, 23.0),  # ratio=0.56
        ]
        for mic_rms, sys_rms, ts in bleed_entries_15_25:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))
            assert suppressed is True, f"iObeya-level bleed at {ts}s not suppressed"

        # 25-30s: System quiet, user speaks
        suppressed = echo_gate_decision(0.110, 0.003)
        events.append(("user_alone", suppressed, 27.0))
        assert suppressed is False, "User speaking alone was suppressed!"

        # 30-40s: System playing, user silent (all caught)
        bleed_entries_30_40 = [
            (0.0088, 0.0350, 32.0),
            (0.0185, 0.0310, 35.0),
            (0.070, 0.100, 38.0),  # loud bleed, ratio=0.70
        ]
        for mic_rms, sys_rms, ts in bleed_entries_30_40:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))
            assert suppressed is True, f"Bleed at {ts}s not suppressed"

        # Collect segments that passed the live gate
        live_segments = [
            {"speaker": "you", "text": "I disagree with that point about the timeline.", "start": 12.0},
            {"speaker": "you", "text": "Let me check the notes from last week.", "start": 27.0},
        ]
        system_segments = [
            {"speaker": "them", "text": "The market analysis shows declining trends.", "start": 3.0},
            {"speaker": "them", "text": "We need to focus on the quarterly targets.", "start": 8.0},
            {"speaker": "them", "text": "Revenue projections are updated in the report.", "start": 17.0},
            {"speaker": "them", "text": "We need to focus on the quarterly targets.", "start": 33.0},
        ]

        all_segments = live_segments + system_segments
        all_segments.sort(key=lambda s: s["start"])
        final = deduplicate_segments(all_segments)

        you_segs = [s for s in final if s["speaker"] == "you"]
        you_texts = [s["text"] for s in you_segs]

        assert "I disagree with that point about the timeline." in you_texts
        assert "Let me check the notes from last week." in you_texts
        assert len(you_segs) == 2

        them_segs = [s for s in final if s["speaker"] == "them"]
        assert len(them_segs) == 4


# ── Test 11: Worst case scenarios ─────────────────────────────────


class TestWorstCaseScenarios:
    """Test 11: Boundary cases for the echo gate."""

    def test_user_at_speech_floor(self):
        """User speech at the mic_rms floor (0.085) over loud system."""
        sys_rms = 0.12
        voice_rms = 0.085
        bleed = sys_rms * 0.25  # 0.030
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)  # ~0.090
        assert mic_rms > 0.08, "Combined mic should be above floor"
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_bleed_at_upper_range(self):
        """Bleed at the highest observed level (0.073) still suppressed."""
        assert echo_gate_decision(0.073, 0.102) is True  # ratio=0.72

    def test_tier2_catches_loud_speaker_bleed(self):
        """Tier 2 catches bleed when speakers are very loud."""
        # sys_rms=0.15, bleed=0.10 (high bleed factor), ratio=0.67
        # Tier 1 fails (mic > 0.08), Tier 2 fails (ratio > 0.65)
        # This edge case leaks — acceptable, caught by post-AEC or dedup
        assert echo_gate_decision(0.10, 0.15) is False

        # sys_rms=0.20, bleed=0.12, ratio=0.60 → Tier 2 catches
        assert echo_gate_decision(0.12, 0.20) is True  # ratio=0.60 < 0.65

    def test_loud_bleed_leak_caught_by_dedup(self):
        """Verify dedup catches bleed that leaks through the gate."""
        segments = [
            {"speaker": "them", "text": "We need to address the infrastructure concerns immediately.", "start": 5.0},
            {"speaker": "you", "text": "We need to address the infrastructure concerns immediately.", "start": 5.5},
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0


# ── Test 12: Threshold sensitivity ────────────────────────────────


class TestThresholdSensitivity:
    """Test 12: Verify the two-tier formula's parameters are correct."""

    def test_tier1_catches_all_low_mic_bleed(self):
        """Tier 1 (mic < 0.08, ratio < 1.5) catches all bleed below 0.08."""
        tier1_count = sum(
            1 for mic_rms, sys_rms in REAL_BLEED_DATA
            if sys_rms > 0.005 and mic_rms / sys_rms < 1.5 and mic_rms < 0.08
        )
        # All entries with mic < 0.08 and ratio < 1.5 should be caught
        total_below_08 = sum(1 for m, s in REAL_BLEED_DATA if m < 0.08)
        assert tier1_count == total_below_08

    def test_two_tier_catches_all(self):
        """Combined two-tier formula catches 100% of real bleed."""
        suppressed = sum(
            1 for m, s in REAL_BLEED_DATA if echo_gate_decision(m, s)
        )
        assert suppressed == len(REAL_BLEED_DATA)

    def test_all_user_speech_safe(self):
        """Two-tier formula never suppresses any user speech scenario."""
        # Real user speech values
        assert echo_gate_decision(0.085, 0.010) is False
        assert echo_gate_decision(0.110, 0.040) is False
        assert echo_gate_decision(0.120, 0.060) is False

        # Simulated user-over-system
        for voice_rms, sys_rms, label in TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            assert echo_gate_decision(mic_rms, sys_rms) is False, (
                f"{label}: user speech suppressed! mic_rms={mic_rms:.4f}"
            )

    def test_separation_gap(self):
        """Verify clear gap between max bleed mic_rms and min user speech."""
        max_bleed_mic = max(m for m, s in REAL_BLEED_DATA)
        min_user_mic = min(
            math.sqrt(v ** 2 + (s * 0.25) ** 2)
            for v, s, _ in TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX
        )
        # There should be a gap between bleed and speech
        assert min_user_mic > max_bleed_mic, (
            f"No gap: max_bleed={max_bleed_mic:.4f}, min_user={min_user_mic:.4f}"
        )
        # And both should be on the correct side of the 0.08 floor
        assert max_bleed_mic < 0.08, f"Max bleed {max_bleed_mic:.4f} >= 0.08"
        assert min_user_mic > 0.08, f"Min user {min_user_mic:.4f} <= 0.08"


# ═══════════════════════════════════════════════════════════════════
# D. AUDIO ENVELOPE CORRELATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestAudioEnvelopeCorrelationIntegration:
    """Test the is_echo() function itself (still used for reference)."""

    def _make_speech_signal(self, duration_s, base_freq=300, sr=16000,
                            amplitude=0.3, env_freq=4.0, env_phase=0.0):
        """Create amplitude-modulated signal mimicking speech envelope."""
        import numpy as np
        t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
        carrier = np.sin(2 * np.pi * base_freq * t)
        for f in [base_freq * 1.5, base_freq * 2, base_freq * 3]:
            carrier += np.sin(2 * np.pi * f * t) * 0.2
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * env_freq * t + env_phase)
        return (carrier * envelope * amplitude).astype(np.float32)

    def test_pure_echo_blocked_by_correlation(self):
        """Pure echo (high correlation) should be detected by is_echo()."""
        import numpy as np
        from core.echo_gate import is_echo
        sr = 16000
        system = self._make_speech_signal(2.0, base_freq=300, sr=sr)
        delay = int(sr * 0.008)
        echo = np.zeros_like(system)
        echo[delay:] = system[:-delay] * 0.35
        mic = echo + np.random.randn(len(system)).astype(np.float32) * 0.005

        detected, corr = is_echo(mic, system, threshold=0.7, detail=True)
        assert detected is True, f"Pure echo should be detected (corr={corr:.2f})"
        assert corr > 0.7, f"Correlation should exceed threshold (corr={corr:.2f})"

    def test_user_over_system_not_blocked(self):
        """User speaking over system audio should NOT be detected as echo."""
        import numpy as np
        from core.echo_gate import is_echo
        sr = 16000
        system = self._make_speech_signal(2.0, base_freq=300, sr=sr, amplitude=0.3)
        delay = int(sr * 0.008)
        echo = np.zeros_like(system)
        echo[delay:] = system[:-delay] * 0.3
        user = self._make_speech_signal(2.0, base_freq=150, sr=sr, amplitude=0.4,
                                         env_freq=3.2, env_phase=1.5)
        mic = echo + user

        detected, corr = is_echo(mic, system, threshold=0.7, detail=True)
        assert detected is False, f"User over system should NOT be echo (corr={corr:.2f})"

    def test_user_alone_not_blocked(self):
        """User speaking alone (no system audio) should NOT be detected."""
        import numpy as np
        from core.echo_gate import is_echo
        sr = 16000
        system = np.zeros(sr * 2, dtype=np.float32)
        mic = self._make_speech_signal(2.0, base_freq=150, sr=sr, amplitude=0.3)

        detected = is_echo(mic, system, threshold=0.7)
        assert detected is False, "User alone should not trigger echo"
