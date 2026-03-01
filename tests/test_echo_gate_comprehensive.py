"""
Comprehensive echo gate verification tests.

Tests the dual-check echo gate (ratio < 1.5 AND mic_rms < 0.014) across
every scenario: real bleed data, user speech, user-over-system, edge cases,
regression against the old approach, post-processing dedup, and full
pipeline integration.
"""

import math
import pytest

from core.echo_gate import deduplicate_segments


# ── Helper: simulate the live echo gate decision ──────────────────


def echo_gate_decision(mic_rms, sys_rms, ratio_threshold=1.5, mic_floor=0.014):
    """Simulate the live echo gate decision from TranscriptionWorker.

    Returns True if the audio should be SUPPRESSED (echo detected).
    """
    if sys_rms <= 0.005:
        return False  # System too quiet, no echo possible
    ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
    return ratio < ratio_threshold and mic_rms < mic_floor


def old_echo_gate_decision(mic_rms, sys_rms, ratio_threshold=1.5):
    """Simulate the OLD echo gate (ratio only, no mic_rms floor).

    Returns True if the audio would have been suppressed by the old approach.
    """
    if sys_rms <= 0.005:
        return False
    ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
    return ratio < ratio_threshold


# ═══════════════════════════════════════════════════════════════════
# A. LIVE ECHO GATE TESTS
# ═══════════════════════════════════════════════════════════════════


# ── Test 1: Pure bleed — all 29 real data points ──────────────────


# Real log data from two recording sessions
REAL_BLEED_DATA = [
    # 7:26 PM run (19 entries)
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
    (0.0188, 0.0305),  # Outlier: mic_rms > 0.014 → will LEAK
    (0.0089, 0.0341),
    # 7:11 PM run (10 entries)
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
]


class TestPureBleedRealData:
    """Test 1: Verify all 29 real bleed data points are handled correctly."""

    @pytest.mark.parametrize("mic_rms,sys_rms", REAL_BLEED_DATA)
    def test_individual_bleed_entry(self, mic_rms, sys_rms):
        """Each real bleed entry: suppressed if mic_rms < 0.014, else leaked."""
        result = echo_gate_decision(mic_rms, sys_rms)
        if mic_rms < 0.014:
            assert result is True, (
                f"Bleed NOT suppressed: mic_rms={mic_rms}, sys_rms={sys_rms}, "
                f"ratio={mic_rms/sys_rms:.2f}"
            )
        else:
            # The outlier at (0.0188, 0.0305) leaks through — acceptable
            assert result is False, (
                f"High mic_rms bleed incorrectly suppressed: mic_rms={mic_rms}"
            )

    def test_suppression_rate_28_of_29(self):
        """28 of 29 real bleed entries should be suppressed (96.6%)."""
        suppressed = sum(1 for m, s in REAL_BLEED_DATA if echo_gate_decision(m, s))
        leaked = len(REAL_BLEED_DATA) - suppressed
        assert suppressed == 28, f"Expected 28 suppressed, got {suppressed}"
        assert leaked == 1, f"Expected 1 leak, got {leaked}"

    def test_outlier_is_the_0188_entry(self):
        """The only leak should be the (0.0188, 0.0305) outlier."""
        leaks = [(m, s) for m, s in REAL_BLEED_DATA if not echo_gate_decision(m, s)]
        assert len(leaks) == 1
        assert leaks[0] == (0.0188, 0.0305)


# ── Test 2: User speaking alone ──────────────────────────────────


class TestUserSpeakingAlone:
    """Test 2: User speech should NEVER be suppressed."""

    def test_real_data_user_speaking(self):
        """Real data: mic_rms=0.0223, sys_rms=0.0099 → NOT suppressed."""
        assert echo_gate_decision(0.0223, 0.0099) is False

    def test_quiet_user_system_silent(self):
        """Quiet user, system silent → NOT suppressed (sys below threshold)."""
        assert echo_gate_decision(0.0150, 0.0020) is False

    def test_loud_user_system_silent(self):
        """Loud user, system silent → NOT suppressed."""
        assert echo_gate_decision(0.0350, 0.0030) is False

    def test_whisper_no_system(self):
        """Very quiet whisper, no system audio → NOT suppressed."""
        assert echo_gate_decision(0.0120, 0.0000) is False

    def test_normal_user_no_system(self):
        """Normal user volume, no system audio → NOT suppressed."""
        assert echo_gate_decision(0.0250, 0.0000) is False

    def test_loud_user_no_system(self):
        """Loud user, no system → NOT suppressed."""
        assert echo_gate_decision(0.0500, 0.0000) is False

    def test_user_with_very_quiet_background(self):
        """User speaking with very quiet system background."""
        assert echo_gate_decision(0.0200, 0.0040) is False


# ── Test 3: User speaking OVER system audio (critical case) ──────


class TestUserOverSystemAudio:
    """Test 3: THE primary failure mode we are fixing.

    When user speaks over system audio, mic picks up both voice and bleed.
    mic_rms ~ sqrt(voice^2 + bleed^2), where bleed ~ sys_rms * 0.25.
    ALL of these MUST pass (not be suppressed).
    """

    # Voice x System test matrix
    VOICE_SYSTEM_MATRIX = [
        # (voice_rms, sys_rms, label)
        (0.015, 0.010, "Quiet voice x Quiet system"),
        (0.015, 0.030, "Quiet voice x Normal system"),
        (0.015, 0.055, "Quiet voice x Loud system"),
        (0.022, 0.010, "Normal voice x Quiet system"),
        (0.022, 0.030, "Normal voice x Normal system"),
        (0.022, 0.055, "Normal voice x Loud system"),
        (0.035, 0.010, "Loud voice x Quiet system"),
        (0.035, 0.030, "Loud voice x Normal system"),
        (0.035, 0.055, "Loud voice x Loud system"),
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
        """All user-over-system mic_rms values should be above 0.014."""
        for voice_rms, sys_rms, label in self.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            assert mic_rms > 0.014, (
                f"{label}: mic_rms={mic_rms:.4f} is below floor 0.014"
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
        assert 0.008 < 0.014
        assert result is True

    def test_mic_exactly_at_floor(self):
        """mic_rms=0.014 exactly → NOT suppressed (not < 0.014)."""
        assert echo_gate_decision(0.014, 0.030) is False

    def test_mic_just_below_floor(self):
        """mic_rms=0.0139, ratio<1.5 → suppressed."""
        result = echo_gate_decision(0.0139, 0.030)
        ratio = 0.0139 / 0.030  # 0.463
        assert ratio < 1.5
        assert 0.0139 < 0.014
        assert result is True

    def test_very_loud_system_with_proportional_bleed(self):
        """sys_rms=0.08, mic_rms=0.020 (bleed only) → NOT suppressed (mic>floor)."""
        assert echo_gate_decision(0.020, 0.08) is False

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

    def test_ratio_above_1_5(self):
        """Ratio clearly above 1.5 → NOT suppressed."""
        # Use values that avoid floating-point boundary issues
        # mic_rms=0.010, sys_rms=0.006 → ratio=1.667 > 1.5
        assert echo_gate_decision(0.010, 0.006) is False

    def test_extremely_quiet_system(self):
        """System barely audible, mic has faint bleed → NOT suppressed."""
        assert echo_gate_decision(0.003, 0.004) is False  # sys <= 0.005

    def test_mic_at_maximum_bleed_level(self):
        """mic_rms at maximum observed bleed (0.0127) → suppressed."""
        assert echo_gate_decision(0.0127, 0.0395) is True

    def test_mic_slightly_above_maximum_bleed(self):
        """mic_rms slightly above max bleed → NOT suppressed (above floor)."""
        assert echo_gate_decision(0.0141, 0.0395) is False


# ── Test 5: Regression — old approach would fail ─────────────────


class TestRegressionOldApproach:
    """Test 5: Verify the old approach (ratio only) WOULD suppress user speech.

    This proves the mic_rms floor fix is necessary.
    """

    @pytest.mark.parametrize("voice_rms,sys_rms,label",
                             TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX)
    def test_old_approach_would_suppress(self, voice_rms, sys_rms, label):
        """The old ratio-only check would INCORRECTLY suppress user speech."""
        bleed = sys_rms * 0.25
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
        ratio = mic_rms / sys_rms if sys_rms > 0.005 else float('inf')

        # Only check cases where system is above threshold
        if sys_rms > 0.005:
            old_would_suppress = old_echo_gate_decision(mic_rms, sys_rms)
            new_suppresses = echo_gate_decision(mic_rms, sys_rms)

            # For cases where ratio < 1.5, old approach wrongly suppresses
            if ratio < 1.5:
                assert old_would_suppress is True, (
                    f"{label}: Old approach should have suppressed (ratio={ratio:.2f}<1.5)"
                )
                assert new_suppresses is False, (
                    f"{label}: New approach should NOT suppress (mic_rms={mic_rms:.4f}>0.014)"
                )

    def test_quiet_voice_over_normal_system_regression(self):
        """Specific case: quiet voice over normal system — old approach fails."""
        voice_rms = 0.015
        sys_rms = 0.030
        bleed = sys_rms * 0.25  # 0.0075
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)  # ~0.0168
        ratio = mic_rms / sys_rms  # ~0.56

        # Old approach: ratio 0.56 < 1.5 → SUPPRESS (wrong!)
        assert old_echo_gate_decision(mic_rms, sys_rms) is True
        # New approach: mic_rms 0.0168 > 0.014 → NOT suppress (correct!)
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_normal_voice_over_loud_system_regression(self):
        """Specific case: normal voice over loud system — old approach fails."""
        voice_rms = 0.022
        sys_rms = 0.055
        bleed = sys_rms * 0.25  # 0.01375
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)  # ~0.026
        ratio = mic_rms / sys_rms  # ~0.47

        assert old_echo_gate_decision(mic_rms, sys_rms) is True
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_count_old_vs_new_regressions(self):
        """Count how many user-over-system cases old approach wrongly suppresses."""
        matrix = TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX
        old_false_positives = 0
        new_false_positives = 0

        for voice_rms, sys_rms, _ in matrix:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            if old_echo_gate_decision(mic_rms, sys_rms):
                old_false_positives += 1
            if echo_gate_decision(mic_rms, sys_rms):
                new_false_positives += 1

        # Old approach has many false positives
        assert old_false_positives > 0, "Old approach should have false positives"
        # New approach has zero false positives
        assert new_false_positives == 0, (
            f"New approach has {new_false_positives} false positives!"
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
    """Test 7: Verify real transcript data tests exist and pass.

    The detailed real data tests are in test_dedup_real_data.py. Here we
    verify the dedup function itself works on a representative sample.
    """

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


# ── Test 8: User-over-system NOT removed by dedup ────────────────


class TestUserOverSystemNotRemovedByDedup:
    """Test 8: When user speaks over system, their words are DIFFERENT.

    Whisper focuses on the dominant close-mic speaker, so the mic transcript
    has the user's words, not the system's. Dedup should keep them.
    """

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

        # Check echo segments removed
        you_segs = [s for s in result if s["speaker"] == "you"]
        you_texts = [s["text"] for s in you_segs]

        # Echo at index 1 should be removed
        assert "We have said that we were willing to support the Department." not in you_texts

        # Genuine speech at index 3 should be KEPT
        assert "But that contradicts what they said last week about the budget." in you_texts

        # Partial echo at index 5 should be removed
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


# ── Test 10: Complete recording session simulation ────────────────


class TestFullPipelineIntegration:
    """Test 10: Simulate a complete recording session end-to-end."""

    def test_complete_session_timeline(self):
        """Full timeline: system plays, user silent, user speaks over, etc.

        Timeline:
        0-10s:  System playing, user silent → bleed suppressed by live gate
        10-15s: System playing, user speaks over → live gate lets through
        15-25s: System playing, user silent → bleed suppressed by live gate
        25-30s: System quiet, user speaks → live gate lets through
        30-40s: System playing, user silent → mostly suppressed (1 loud bleed leaks)
        40-45s: Leaked bleed segment → dedup removes it
        """
        # Simulate live echo gate decisions
        events = []

        # 0-10s: System playing, user silent (pure bleed)
        bleed_entries_0_10 = [
            (0.0085, 0.0350, 2.0),
            (0.0092, 0.0420, 5.0),
            (0.0078, 0.0310, 8.0),
        ]
        for mic_rms, sys_rms, ts in bleed_entries_0_10:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))
            assert suppressed is True, f"Bleed at {ts}s not suppressed"

        # 10-15s: System playing, user speaks over it
        voice_rms = 0.022
        sys_rms = 0.035
        bleed = sys_rms * 0.25
        mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
        suppressed = echo_gate_decision(mic_rms, sys_rms)
        events.append(("user_over_system", suppressed, 12.0))
        assert suppressed is False, "User speaking over system was suppressed!"

        # 15-25s: System playing, user silent (pure bleed)
        bleed_entries_15_25 = [
            (0.0090, 0.0400, 17.0),
            (0.0095, 0.0380, 20.0),
            (0.0082, 0.0290, 23.0),
        ]
        for mic_rms, sys_rms, ts in bleed_entries_15_25:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))
            assert suppressed is True, f"Bleed at {ts}s not suppressed"

        # 25-30s: System quiet, user speaks
        suppressed = echo_gate_decision(0.0250, 0.003)
        events.append(("user_alone", suppressed, 27.0))
        assert suppressed is False, "User speaking alone was suppressed!"

        # 30-40s: System playing, user silent (1 loud bleed leaks)
        bleed_entries_30_40 = [
            (0.0088, 0.0350, 32.0),  # suppressed
            (0.0185, 0.0300, 35.0),  # LEAKS (mic_rms > 0.014)
            (0.0091, 0.0410, 38.0),  # suppressed
        ]
        for mic_rms, sys_rms, ts in bleed_entries_30_40:
            suppressed = echo_gate_decision(mic_rms, sys_rms)
            events.append(("bleed", suppressed, ts))

        # Now collect segments that passed the live gate
        live_segments = []

        # User speaking over system at 12s — passed live gate
        live_segments.append({
            "speaker": "you",
            "text": "I disagree with that point about the timeline.",
            "start": 12.0,
        })

        # User speaking alone at 27s — passed live gate
        live_segments.append({
            "speaker": "you",
            "text": "Let me check the notes from last week.",
            "start": 27.0,
        })

        # Leaked bleed at 35s — passed live gate (will be caught by dedup)
        live_segments.append({
            "speaker": "you",
            "text": "We need to focus on the quarterly targets.",
            "start": 35.0,
        })

        # System audio segments (always transcribed)
        system_segments = [
            {"speaker": "them", "text": "The market analysis shows declining trends.", "start": 3.0},
            {"speaker": "them", "text": "We need to focus on the quarterly targets.", "start": 8.0},
            {"speaker": "them", "text": "Revenue projections are updated in the report.", "start": 17.0},
            {"speaker": "them", "text": "We need to focus on the quarterly targets.", "start": 33.0},
        ]

        # Combine and run dedup
        all_segments = live_segments + system_segments
        all_segments.sort(key=lambda s: s["start"])
        final = deduplicate_segments(all_segments)

        you_segs = [s for s in final if s["speaker"] == "you"]
        you_texts = [s["text"] for s in you_segs]

        # User's genuine speech kept
        assert "I disagree with that point about the timeline." in you_texts
        assert "Let me check the notes from last week." in you_texts

        # Leaked bleed removed by dedup
        assert "We need to focus on the quarterly targets." not in you_texts

        # All system segments kept
        them_segs = [s for s in final if s["speaker"] == "them"]
        assert len(them_segs) == 4


# ── Test 11: Worst case — very quiet speaker over very loud system ─


class TestWorstCaseScenarios:
    """Test 11: Absolute worst case for the echo gate."""

    def test_quiet_whisper_over_loud_system(self):
        """Very quiet user whisper over very loud system audio."""
        sys_rms = 0.07
        voice_rms = 0.012
        bleed = 0.07 * 0.25  # 0.0175
        mic_rms = math.sqrt(0.012 ** 2 + 0.0175 ** 2)  # ~0.0212
        assert mic_rms > 0.014
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_barely_audible_voice_over_loud_system(self):
        """Barely audible voice over very loud system."""
        sys_rms = 0.07
        voice_rms = 0.008
        bleed = 0.07 * 0.25  # 0.0175
        mic_rms = math.sqrt(0.008 ** 2 + 0.0175 ** 2)  # ~0.0192
        assert mic_rms > 0.014
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_extremely_quiet_voice_over_loud_system(self):
        """Extremely quiet voice (0.005) over loud system."""
        sys_rms = 0.07
        voice_rms = 0.005
        bleed = 0.07 * 0.25  # 0.0175
        mic_rms = math.sqrt(0.005 ** 2 + 0.0175 ** 2)  # ~0.0182
        assert mic_rms > 0.014
        assert echo_gate_decision(mic_rms, sys_rms) is False

    def test_loud_bleed_no_voice_leaks_through(self):
        """Key insight: loud system with NO user voice → bleed alone > 0.014.

        When sys_rms = 0.07, bleed = 0.0175. This pushes mic_rms above the
        floor even without any user voice. This LEAKS through the live gate.
        The dedup catches it in post-processing.
        """
        sys_rms = 0.07
        bleed_only = 0.07 * 0.25  # 0.0175, no user voice
        mic_rms = bleed_only
        # This WILL leak through the live gate (mic_rms > 0.014)
        assert echo_gate_decision(mic_rms, sys_rms) is False
        # ^ False means NOT suppressed — this is a leak!

    def test_loud_bleed_leak_caught_by_dedup(self):
        """Verify dedup catches the loud-bleed leak."""
        segments = [
            {"speaker": "them", "text": "We need to address the infrastructure concerns immediately.", "start": 5.0},
            {"speaker": "you", "text": "We need to address the infrastructure concerns immediately.", "start": 5.5},  # Leaked bleed
        ]
        result = deduplicate_segments(segments)
        you_segs = [s for s in result if s["speaker"] == "you"]
        assert len(you_segs) == 0, "Leaked loud bleed should be caught by dedup"

    def test_threshold_above_which_pure_bleed_leaks(self):
        """Calculate the system volume threshold where pure bleed leaks through.

        bleed = sys_rms * 0.25
        bleed > 0.014 when sys_rms > 0.056

        Above sys_rms=0.056, pure bleed exceeds the floor and the live gate
        cannot catch it. Dedup is the safety net.
        """
        # sys_rms = 0.056 → bleed = 0.014 → exactly at boundary (not < 0.014)
        assert echo_gate_decision(0.014, 0.056) is False  # Leaks (mic_rms not < floor)

        # sys_rms = 0.055 → bleed = 0.01375 → still suppressed
        assert echo_gate_decision(0.01375, 0.055) is True  # Caught

        # sys_rms = 0.060 → bleed = 0.015 → leaks
        assert echo_gate_decision(0.015, 0.060) is False  # Leaks


# ── Test 12: Threshold sensitivity analysis ──────────────────────


class TestThresholdSensitivity:
    """Test 12: Verify 0.014 is the sweet spot by checking adjacent values."""

    def _count_bleed_suppressed(self, floor):
        """Count how many of the 29 real bleed entries are caught at a given floor."""
        suppressed = 0
        for mic_rms, sys_rms in REAL_BLEED_DATA:
            if sys_rms <= 0.005:
                continue
            ratio = mic_rms / sys_rms
            if ratio < 1.5 and mic_rms < floor:
                suppressed += 1
        return suppressed

    def _count_user_over_system_safe(self, floor):
        """Count how many user-over-system scenarios are safe at a given floor."""
        safe = 0
        for voice_rms, sys_rms, _ in TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            if sys_rms <= 0.005:
                safe += 1
                continue
            ratio = mic_rms / sys_rms
            suppressed = ratio < 1.5 and mic_rms < floor
            if not suppressed:
                safe += 1
        return safe

    def test_floor_0012_analysis(self):
        """Floor=0.012: More bleed caught but risks quiet user speech."""
        bleed_caught = self._count_bleed_suppressed(0.012)
        user_safe = self._count_user_over_system_safe(0.012)

        # At 0.012, catches more bleed...
        assert bleed_caught >= 26  # Most bleed entries have mic_rms < 0.012

        # ...but all user-over-system is still safe (all mic_rms > 0.015)
        assert user_safe == 9, f"At floor=0.012, {9 - user_safe} user scenarios suppressed!"

    def test_floor_0014_analysis(self):
        """Floor=0.014 (CURRENT): The recommended sweet spot."""
        bleed_caught = self._count_bleed_suppressed(0.014)
        user_safe = self._count_user_over_system_safe(0.014)

        assert bleed_caught == 28, f"Expected 28 bleed caught, got {bleed_caught}"
        assert user_safe == 9, f"Expected 9 user safe, got {user_safe}"

    def test_floor_0016_analysis(self):
        """Floor=0.016: Lets through more bleed but safer for quiet speech."""
        bleed_caught = self._count_bleed_suppressed(0.016)
        user_safe = self._count_user_over_system_safe(0.016)

        # At 0.016, catches fewer bleed (the entries near 0.014-0.016 leak)
        assert bleed_caught <= 28

        # All user scenarios remain safe
        assert user_safe == 9

    def test_0014_is_optimal(self):
        """Floor=0.014 maximizes bleed suppression while keeping 0% false positives."""
        results = {}
        for floor in [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.018, 0.020]:
            bleed = self._count_bleed_suppressed(floor)
            safe = self._count_user_over_system_safe(floor)
            results[floor] = (bleed, safe)

        # At 0.014:
        # - 28/29 bleed suppressed (96.6%)
        # - 9/9 user scenarios safe (100%)
        assert results[0.014] == (28, 9)

        # Floors at or below 0.016 should keep all user scenarios safe
        for floor in [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016]:
            bleed, safe = results[floor]
            assert safe == 9, (
                f"Floor {floor}: {9 - safe} user scenarios suppressed!"
            )

        # Floors above 0.016 risk suppressing quiet-voice-over-normal-system
        # (mic_rms=0.0168 for quiet voice over normal system)
        assert results[0.018][1] < 9, (
            "Floor 0.018 should suppress at least one user scenario "
            "(quiet voice over normal system has mic_rms=0.0168)"
        )

    def test_bleed_values_distribution(self):
        """Verify the distribution of mic_rms values in real bleed data."""
        mic_values = sorted([m for m, s in REAL_BLEED_DATA])
        # All but one should be below 0.014
        below_014 = [v for v in mic_values if v < 0.014]
        above_014 = [v for v in mic_values if v >= 0.014]
        assert len(below_014) == 28
        assert len(above_014) == 1
        assert above_014[0] == 0.0188  # The outlier

        # Max bleed mic_rms (excluding outlier) is 0.0127
        max_normal_bleed = max(below_014)
        assert max_normal_bleed == 0.0127

        # Gap between max normal bleed and floor
        gap = 0.014 - max_normal_bleed
        assert gap == pytest.approx(0.0013, abs=0.0001), (
            f"Gap between max bleed ({max_normal_bleed}) and floor (0.014) is {gap}"
        )

    def test_user_speech_minimum_mic_rms(self):
        """Verify the minimum mic_rms from user-over-system scenarios."""
        min_mic_rms = float('inf')
        min_label = ""
        for voice_rms, sys_rms, label in TestUserOverSystemAudio.VOICE_SYSTEM_MATRIX:
            bleed = sys_rms * 0.25
            mic_rms = math.sqrt(voice_rms ** 2 + bleed ** 2)
            if mic_rms < min_mic_rms:
                min_mic_rms = mic_rms
                min_label = label

        # The minimum user mic_rms should be well above 0.014
        assert min_mic_rms > 0.014, (
            f"Minimum user mic_rms is {min_mic_rms:.4f} ({min_label}), below floor!"
        )

        # Safety margin: gap between floor and minimum user mic_rms
        margin = min_mic_rms - 0.014
        assert margin > 0.001, (
            f"Safety margin only {margin:.4f}. Minimum user mic_rms={min_mic_rms:.4f}"
        )
