"""Test dedup against real transcript data from all test runs."""
from core.echo_gate import deduplicate_segments


def test_711pm_prepared_to_access():
    """7:11 PM - 'Prepared to access' vs 'Prepare to exist' (different words)."""
    segments = [
        {"speaker": "you", "text": "All right, we're giving in another shot.", "start": 1.0},
        {"speaker": "you", "text": "We're seeing how it is.", "start": 4.0},
        {"speaker": "you", "text": "Let's go.", "start": 8.0},
        {"speaker": "them", "text": "What do you think what's your response? So you know in in the statement we issued yesterday", "start": 9.0},
        {"speaker": "them", "text": "We said that we were willing even if the Department of War", "start": 16.0},
        {"speaker": "them", "text": "Takes these unprecedented measures against us", "start": 22.0},
        {"speaker": "them", "text": "We have said that you know even if they take these extreme actions", "start": 30.0},
        {"speaker": "them", "text": "We'll do everything we can to support the Department of War", "start": 33.0},
        {"speaker": "them", "text": "For as long as it takes to offboard us", "start": 39.0},
        {"speaker": "them", "text": "Prepare to exist. Yeah, so so we have offered", "start": 47.0},
        {"speaker": "you", "text": "Prepared to access.", "start": 48.0},
        {"speaker": "them", "text": "Continuity we're actually deeply concerned about this", "start": 51.0},
    ]
    result = deduplicate_segments(segments)
    texts = [s["text"] for s in result]
    # Echo should be removed
    assert "Prepared to access." not in texts
    # Genuine speech should be kept
    assert "All right, we're giving in another shot." in texts
    assert "We're seeing how it is." in texts
    assert "Let's go." in texts


def test_705pm_multi_segment_echo():
    """7:05 PM - Echo spanning multiple system segments."""
    segments = [
        {"speaker": "you", "text": "All right, we're going to try again, see how it works.", "start": 0.0},
        {"speaker": "you", "text": "Let's see with a video.", "start": 9.0},
        {"speaker": "them", "text": "did not concede in any meaningful way.", "start": 12.0},
        {"speaker": "them", "text": "the Pentagon spokesman, Sean Pernell, the day before,", "start": 20.0},
        {"speaker": "you", "text": "Pernel, the day before, he reiterated their position, we only allow all waffle use.", "start": 22.0},
        {"speaker": "them", "text": "he reiterated their position,", "start": 23.0},
        {"speaker": "them", "text": "we only allow all lawful use.", "start": 25.0},
        {"speaker": "them", "text": "So they have not exceeded, and they have not,", "start": 31.0},
        {"speaker": "you", "text": "So they have not exceeded in any way agreed to our exceptions in any meaningful way.", "start": 32.0},
        {"speaker": "them", "text": "in any way, agree to our exceptions", "start": 36.0},
        {"speaker": "them", "text": "in any meaningful way.", "start": 39.0},
        {"speaker": "them", "text": "their selfishness, referring to anthropic,", "start": 45.0},
        {"speaker": "you", "text": "to anthropic is putting American lives at risk, our troops in danger, and our national", "start": 47.0},
        {"speaker": "them", "text": "is putting American lives at risk,", "start": 47.0},
        {"speaker": "them", "text": "our troops in danger, and our national security in jeopardy.", "start": 49.0},
    ]
    result = deduplicate_segments(segments)
    you_segs = [s for s in result if s["speaker"] == "you"]
    # Only genuine user speech should remain
    assert len(you_segs) == 2
    assert you_segs[0]["text"].startswith("All right")
    assert you_segs[1]["text"].startswith("Let's see")


def test_testrun2_full():
    """Test Run 2 - Full transcript with many echo segments."""
    segments = [
        {"speaker": "you", "text": "All right, we're testing Cadence again.", "start": 8.0},
        {"speaker": "you", "text": "We're testing to see how it operates with the echo,", "start": 13.0},
        {"speaker": "you", "text": "and we did a lot of work to try to fix for the echo,", "start": 20.0},
        {"speaker": "you", "text": "and we're going to see the results of that work.", "start": 24.0},
        {"speaker": "you", "text": "Let's go ahead and start up a video.", "start": 28.0},
        {"speaker": "you", "text": "You know, our adversaries made some point have them,", "start": 47.0},
        {"speaker": "them", "text": "You know they you know our adversaries made some point have them so perhaps", "start": 54.0},
        {"speaker": "them", "text": "They may at some point be needed for the defense of democracy", "start": 58.0},
        {"speaker": "you", "text": "for the defense of democracy, but we have some concerns about them.", "start": 61.0},
        {"speaker": "them", "text": "But we have some concerns about them first the AI systems of today are nowhere near reliable enough", "start": 62.0},
        {"speaker": "you", "text": "First, the AI systems of today are nowhere near reliable enough", "start": 65.0},
        {"speaker": "them", "text": "You know anyone who's worked with AI models understands that there's a basic unpredictability to them", "start": 72.0},
        {"speaker": "you", "text": "that there's a basic unpredictability to them", "start": 76.0},
        {"speaker": "them", "text": "Then in a purely technical way we have not solved and there's an oversight question too", "start": 77.0},
        {"speaker": "you", "text": "And there's an oversight question too.", "start": 81.0},
        {"speaker": "them", "text": "If you have a large army of drones or robots that can operate without any human oversight", "start": 82.0},
        {"speaker": "you", "text": "that can operate without any human oversight,", "start": 87.0},
        {"speaker": "them", "text": "Whether there aren't human soldiers to make the decisions about who to target who to shoot at", "start": 88.0},
        {"speaker": "you", "text": "about who to target, who to shoot at, that presents concerns.", "start": 92.0},
        {"speaker": "them", "text": "Presence concerns and we need to have a conversation about how that's overseen and we haven't had that conversation yet", "start": 95.0},
        {"speaker": "you", "text": "and we haven't had that conversation yet.", "start": 100.0},
        {"speaker": "them", "text": "Those two use cases should should not be allowed the Pentagon has told us", "start": 105.0},
        {"speaker": "you", "text": "should not be allowed.", "start": 107.0},
        {"speaker": "you", "text": "The Pentagon has told us.", "start": 110.0},
    ]
    result = deduplicate_segments(segments)
    you_segs = [s for s in result if s["speaker"] == "you"]
    # Only the first 5 genuine segments (before video) + "You know..." should remain
    genuine_starts = [
        "All right",
        "We're testing",
        "and we did",
        "and we're going",
        "Let's go",
    ]
    for start in genuine_starts:
        assert any(s["text"].startswith(start) for s in you_segs), \
            f"Genuine segment starting with '{start}' was incorrectly removed"
    # Echo segments should all be removed
    echo_texts = [
        "for the defense of democracy",
        "First, the AI systems",
        "that there's a basic unpredictability",
        "And there's an oversight question",
        "that can operate without any human oversight",
        "about who to target",
        "and we haven't had that conversation",
        "should not be allowed.",
        "The Pentagon has told us.",
    ]
    for echo in echo_texts:
        assert not any(s["text"].startswith(echo) for s in you_segs), \
            f"Echo segment '{echo}' was not removed"


def test_genuine_speech_not_removed():
    """Ensure genuine user speech is never removed as echo."""
    # User and system talking about completely different things
    segments = [
        {"speaker": "them", "text": "The weather forecast calls for rain tomorrow afternoon.", "start": 1.0},
        {"speaker": "you", "text": "I need to finish the report by Friday.", "start": 2.0},
        {"speaker": "them", "text": "We should plan for the quarterly review next week.", "start": 5.0},
        {"speaker": "you", "text": "Can we schedule a meeting for Thursday?", "start": 6.0},
    ]
    result = deduplicate_segments(segments)
    assert len(result) == 4  # Nothing should be removed


def test_923pm_delayed_echo():
    """9:23 PM - Echo appearing 8+ seconds after system segment."""
    segments = [
        {"speaker": "them", "text": "We appreciate you taking the time you are Dario on a day the CEO of Anthropics, all right?", "start": 1.0},
        {"speaker": "them", "text": "That's correct. Yeah. Well, my first question to you is why won't you release Anthropics AI without restrictions to the US government?", "start": 7.0},
        {"speaker": "them", "text": "Yeah, so, you know, we should maybe back up a bit for a little bit of context.", "start": 15.0},
        {"speaker": "you", "text": "to US government?", "start": 15.5},  # Echo of system at 7.0s
        {"speaker": "them", "text": "So, um, you know, Anthropic actually has been the most lean forward of all the AI companies in", "start": 19.0},
        {"speaker": "you", "text": "Wow. This is all very interesting.", "start": 38.0},  # Genuine speech
        {"speaker": "you", "text": "What do you say?", "start": 39.0},  # Genuine speech
        {"speaker": "you", "text": "It's, wow. It's so interesting.", "start": 48.0},  # Genuine speech
    ]
    result = deduplicate_segments(segments)
    texts = [s["text"] for s in result]
    # Echo should be removed (even with 8.5s gap)
    assert "to US government?" not in texts
    # Genuine speech should be kept
    assert "Wow. This is all very interesting." in texts
    assert "What do you say?" in texts
    assert "It's, wow. It's so interesting." in texts


def test_648pm_clean_result():
    """6:48 PM - Already-clean transcript should not lose segments."""
    segments = [
        {"speaker": "you", "text": "All right, we're testing again.", "start": 3.0},
        {"speaker": "you", "text": "Let's give it a shot with some video.", "start": 13.0},
        {"speaker": "them", "text": "that they have agreed, in principle, to these two restrictions", "start": 22.0},
        {"speaker": "them", "text": "why couldn't an agreement be reached?", "start": 29.0},
        {"speaker": "them", "text": "determined by the kind of three-day window that they gave us", "start": 34.0},
        {"speaker": "them", "text": "ultimatum to agree to their terms,", "start": 44.0},
    ]
    result = deduplicate_segments(segments)
    you_segs = [s for s in result if s["speaker"] == "you"]
    assert len(you_segs) == 2  # Both genuine segments kept
