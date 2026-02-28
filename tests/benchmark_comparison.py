"""
Benchmark: Old fixed-timer chunking vs New silence-detection chunking.

Simulates realistic speech patterns and compares how each approach
segments the audio for transcription. Does NOT require a microphone
or the actual Whisper model — uses synthetic audio and mock transcription
to test the chunking logic itself.
"""

import sys
import time
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Setup path
sys.path.insert(0, "src")

from core.silence_detector import SilenceDetector


# -- Synthetic audio generation ------------------------------------

def generate_speech_block(duration_s, sr=16000, amplitude=0.3):
    """Generate loud audio simulating speech (random noise at speech-like amplitude)."""
    return np.random.randn(int(sr * duration_s)).astype(np.float32) * amplitude


def generate_silence(duration_s, sr=16000):
    """Generate near-silent audio."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def generate_background_noise(duration_s, sr=16000, amplitude=0.003):
    """Generate very quiet background noise (below silence threshold)."""
    return np.random.randn(int(sr * duration_s)).astype(np.float32) * amplitude


@dataclass
class SpeechEvent:
    """Represents a speech segment in our test scenario."""
    start: float   # seconds
    end: float     # seconds
    speaker: str
    text: str


# -- Test Scenarios ------------------------------------------------

def scenario_natural_conversation():
    """
    Realistic meeting conversation with natural pauses.
    Two speakers, varying sentence lengths, natural pause points.
    """
    events = [
        SpeechEvent(0.0, 3.2, "you", "So I think we should start by looking at the Q4 numbers"),
        # 0.8s pause
        SpeechEvent(4.0, 6.5, "them", "Yeah the revenue looks solid but I'm worried about margins"),
        # 1.2s pause
        SpeechEvent(7.7, 12.1, "you", "Right the margins dropped about three percent compared to last quarter and I think that's mainly due to the new hires we brought on in September and October"),
        # 0.6s pause
        SpeechEvent(12.7, 14.2, "them", "Makes sense that's a big team expansion"),
        # 1.5s pause
        SpeechEvent(15.7, 21.3, "you", "Exactly and if you look at the per-employee revenue it's actually trending up which means the investment is paying off we just need to give it another quarter"),
        # 0.9s pause
        SpeechEvent(22.2, 24.8, "them", "OK let's revisit margins in the March review then"),
        # 2.0s longer pause (topic change)
        SpeechEvent(26.8, 30.5, "you", "Sounds good now on the product side we shipped three major features last month"),
        # 0.7s pause
        SpeechEvent(31.2, 33.0, "them", "Which ones specifically"),
        # 0.5s brief pause
        SpeechEvent(33.5, 38.2, "you", "The new dashboard the API v2 endpoints and the mobile push notifications all went live in the last two weeks of December"),
    ]
    return events, 40.0  # total duration


def scenario_long_monologue():
    """
    One person speaking for a long time with only brief micro-pauses.
    Tests the max_speech_s safety valve.
    """
    events = [
        SpeechEvent(0.0, 35.0, "you", "This is a very long explanation about system architecture that goes on and on covering databases caching layers API design and deployment strategies without any significant pause because the speaker is really passionate about the topic"),
        # 2.0s pause
        SpeechEvent(37.0, 39.0, "them", "That's a lot to take in"),
    ]
    return events, 41.0


def scenario_rapid_back_and_forth():
    """
    Quick exchanges with short pauses (< 500ms between some).
    Tests whether short pauses correctly DON'T trigger transcription.
    """
    events = [
        SpeechEvent(0.0, 1.5, "you", "Did you see the email"),
        # 0.3s pause (too short for silence detection)
        SpeechEvent(1.8, 3.0, "them", "Yeah I saw it"),
        # 0.3s pause
        SpeechEvent(3.3, 4.8, "you", "What do you think"),
        # 0.8s pause (long enough)
        SpeechEvent(5.6, 7.5, "them", "I think we should go ahead with option B"),
        # 1.0s pause
        SpeechEvent(8.5, 9.5, "you", "Agreed let's do it"),
    ]
    return events, 11.0


def scenario_short_recording():
    """Very short recording — just a couple seconds."""
    events = [
        SpeechEvent(0.0, 2.5, "you", "Quick test one two three"),
    ]
    return events, 4.0


# -- Audio builder ------------------------------------------------

def build_audio_from_events(events, total_duration, sr=16000):
    """Build separate mic and system audio arrays from speech events."""
    total_samples = int(sr * total_duration)
    mic_audio = generate_background_noise(total_duration, sr)
    sys_audio = generate_background_noise(total_duration, sr)

    for event in events:
        start_sample = int(event.start * sr)
        end_sample = int(event.end * sr)
        length = end_sample - start_sample
        speech = np.random.randn(length).astype(np.float32) * 0.3

        if event.speaker == "you":
            mic_audio[start_sample:end_sample] = speech
        else:
            sys_audio[start_sample:end_sample] = speech

    return mic_audio, sys_audio


def build_frames_from_audio(audio, frame_size=1024):
    """Split continuous audio into frames (mimicking AudioRecorder)."""
    frames = []
    for i in range(0, len(audio), frame_size):
        frames.append(audio[i:i + frame_size].copy())
    return frames


# -- Old approach: Fixed timer -------------------------------------

def simulate_old_approach(mic_frames, sys_frames, events, interval=5.0, sr=16000):
    """
    Simulate the OLD fixed-timer TranscriptionWorker.
    Every `interval` seconds, transcribe whatever accumulated.
    """
    total_samples = sum(len(f) for f in mic_frames)
    total_duration = total_samples / sr

    mic_offset = 0
    sys_offset = 0
    chunks = []

    t = 0
    while t < total_duration:
        t += interval
        # How many frames have accumulated by time t?
        target_samples = int(t * sr)

        # Mic
        mic_target_frames = 0
        acc = 0
        for i, f in enumerate(mic_frames):
            acc += len(f)
            if acc >= target_samples:
                mic_target_frames = i + 1
                break
        else:
            mic_target_frames = len(mic_frames)

        if mic_target_frames > mic_offset:
            chunk_frames = mic_frames[mic_offset:mic_target_frames]
            chunk_audio = np.concatenate(chunk_frames)
            chunk_start = sum(len(f) for f in mic_frames[:mic_offset]) / sr
            chunk_end = sum(len(f) for f in mic_frames[:mic_target_frames]) / sr
            rms = np.sqrt(np.mean(chunk_audio ** 2))

            # Find which speech events overlap this chunk
            overlapping = [e for e in events if e.speaker == "you"
                          and e.start < chunk_end and e.end > chunk_start]

            chunks.append({
                "speaker": "you",
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "rms": rms,
                "has_speech": rms > 0.01,
                "overlapping_events": len(overlapping),
                "cuts_mid_sentence": any(
                    chunk_start > e.start and chunk_start < e.end
                    for e in overlapping
                ),
            })
            mic_offset = mic_target_frames

        # System (same logic)
        sys_target_frames = 0
        acc = 0
        for i, f in enumerate(sys_frames):
            acc += len(f)
            if acc >= target_samples:
                sys_target_frames = i + 1
                break
        else:
            sys_target_frames = len(sys_frames)

        if sys_target_frames > sys_offset:
            chunk_frames = sys_frames[sys_offset:sys_target_frames]
            chunk_audio = np.concatenate(chunk_frames)
            chunk_start = sum(len(f) for f in sys_frames[:sys_offset]) / sr
            chunk_end = sum(len(f) for f in sys_frames[:sys_target_frames]) / sr
            rms = np.sqrt(np.mean(chunk_audio ** 2))

            overlapping = [e for e in events if e.speaker == "them"
                          and e.start < chunk_end and e.end > chunk_start]

            chunks.append({
                "speaker": "them",
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "rms": rms,
                "has_speech": rms > 0.01,
                "overlapping_events": len(overlapping),
                "cuts_mid_sentence": any(
                    chunk_start > e.start and chunk_start < e.end
                    for e in overlapping
                ),
            })
            sys_offset = sys_target_frames

    return chunks


# -- New approach: Silence detection -------------------------------

def simulate_new_approach(mic_frames, sys_frames, events,
                          silence_threshold=0.01, min_silence_ms=500,
                          max_speech_s=30.0, poll_interval=0.2, sr=16000):
    """
    Simulate the NEW energy-based TranscriptionWorker.
    Polls every poll_interval seconds, transcribes on silence detection.
    """
    mic_detector = SilenceDetector(silence_threshold, min_silence_ms, sr)
    sys_detector = SilenceDetector(silence_threshold, min_silence_ms, sr)

    total_samples = sum(len(f) for f in mic_frames)
    total_duration = total_samples / sr

    mic_offset = 0
    sys_offset = 0
    mic_speech_start = 0
    sys_speech_start = 0
    chunks = []

    t = 0
    while t < total_duration:
        t += poll_interval
        target_samples = int(t * sr)

        # --- Mic ---
        mic_target = 0
        acc = 0
        for i, f in enumerate(mic_frames):
            acc += len(f)
            if acc >= target_samples:
                mic_target = i + 1
                break
        else:
            mic_target = len(mic_frames)

        if mic_target > mic_offset:
            new_frames = mic_frames[mic_offset:mic_target]
            for frame in new_frames:
                mic_detector.feed(frame)
            mic_offset = mic_target

            speech_frames = mic_frames[mic_speech_start:mic_offset]
            speech_samples = sum(len(f) for f in speech_frames)
            speech_duration = speech_samples / sr

            should_transcribe = (
                mic_detector.is_silent() and mic_detector._has_had_speech
                and speech_duration > 0.5
            ) or (
                speech_duration >= max_speech_s
            )

            if should_transcribe:
                chunk_start = sum(len(f) for f in mic_frames[:mic_speech_start]) / sr
                chunk_end = sum(len(f) for f in mic_frames[:mic_offset]) / sr
                chunk_audio = np.concatenate(speech_frames)
                rms = np.sqrt(np.mean(chunk_audio ** 2))

                overlapping = [e for e in events if e.speaker == "you"
                              and e.start < chunk_end and e.end > chunk_start]

                chunks.append({
                    "speaker": "you",
                    "start": chunk_start,
                    "end": chunk_end,
                    "duration": chunk_end - chunk_start,
                    "rms": rms,
                    "has_speech": rms > 0.01,
                    "overlapping_events": len(overlapping),
                    "cuts_mid_sentence": any(
                        chunk_start > e.start and chunk_start < e.end
                        for e in overlapping
                    ),
                    "trigger": "silence" if mic_detector.is_silent() else "max_speech",
                })
                mic_speech_start = mic_offset
                mic_detector.reset()
            elif mic_detector.is_silent() and not mic_detector._has_had_speech:
                mic_speech_start = mic_offset
                mic_detector.reset()

        # --- System ---
        sys_target = 0
        acc = 0
        for i, f in enumerate(sys_frames):
            acc += len(f)
            if acc >= target_samples:
                sys_target = i + 1
                break
        else:
            sys_target = len(sys_frames)

        if sys_target > sys_offset:
            new_frames = sys_frames[sys_offset:sys_target]
            for frame in new_frames:
                sys_detector.feed(frame)
            sys_offset = sys_target

            speech_frames = sys_frames[sys_speech_start:sys_offset]
            speech_samples = sum(len(f) for f in speech_frames)
            speech_duration = speech_samples / sr

            should_transcribe = (
                sys_detector.is_silent() and sys_detector._has_had_speech
                and speech_duration > 0.5
            ) or (
                speech_duration >= max_speech_s
            )

            if should_transcribe:
                chunk_start = sum(len(f) for f in sys_frames[:sys_speech_start]) / sr
                chunk_end = sum(len(f) for f in sys_frames[:sys_offset]) / sr
                chunk_audio = np.concatenate(speech_frames)
                rms = np.sqrt(np.mean(chunk_audio ** 2))

                overlapping = [e for e in events if e.speaker == "them"
                              and e.start < chunk_end and e.end > chunk_start]

                chunks.append({
                    "speaker": "them",
                    "start": chunk_start,
                    "end": chunk_end,
                    "duration": chunk_end - chunk_start,
                    "rms": rms,
                    "has_speech": rms > 0.01,
                    "overlapping_events": len(overlapping),
                    "cuts_mid_sentence": any(
                        chunk_start > e.start and chunk_start < e.end
                        for e in overlapping
                    ),
                    "trigger": "silence" if sys_detector.is_silent() else "max_speech",
                })
                sys_speech_start = sys_offset
                sys_detector.reset()
            elif sys_detector.is_silent() and not sys_detector._has_had_speech:
                sys_speech_start = sys_offset
                sys_detector.reset()

    # Flush remaining
    for speaker, frames, speech_start, offset, detector in [
        ("you", mic_frames, mic_speech_start, mic_offset, mic_detector),
        ("them", sys_frames, sys_speech_start, sys_offset, sys_detector),
    ]:
        remaining = frames[speech_start:offset]
        if remaining:
            chunk_audio = np.concatenate(remaining)
            if len(chunk_audio) > sr * 0.3:  # skip tiny remnants
                chunk_start = sum(len(f) for f in frames[:speech_start]) / sr
                chunk_end = sum(len(f) for f in frames[:offset]) / sr
                rms = np.sqrt(np.mean(chunk_audio ** 2))

                overlapping = [e for e in events if e.speaker == speaker
                              and e.start < chunk_end and e.end > chunk_start]

                chunks.append({
                    "speaker": speaker,
                    "start": chunk_start,
                    "end": chunk_end,
                    "duration": chunk_end - chunk_start,
                    "rms": rms,
                    "has_speech": rms > 0.01,
                    "overlapping_events": len(overlapping),
                    "cuts_mid_sentence": False,
                    "trigger": "flush",
                })

    chunks.sort(key=lambda c: c["start"])
    return chunks


# -- Reporting -----------------------------------------------------

def print_chunks(label, chunks):
    print(f"\n  {label}:")
    if not chunks:
        print("    (no chunks)")
        return

    for i, c in enumerate(chunks):
        trigger = f" [{c.get('trigger', 'timer')}]" if 'trigger' in c else ""
        cut = " ** MID-SENTENCE CUT **" if c["cuts_mid_sentence"] else ""
        speech = "speech" if c["has_speech"] else "silent"
        print(f"    #{i+1}: [{c['speaker']:>4}] {c['start']:5.1f}s - {c['end']:5.1f}s "
              f"({c['duration']:4.1f}s) rms={c['rms']:.4f} {speech}{trigger}{cut}")


def analyze_chunks(chunks):
    """Compute quality metrics for a set of chunks."""
    total = len(chunks)
    with_speech = sum(1 for c in chunks if c["has_speech"])
    mid_cuts = sum(1 for c in chunks if c["cuts_mid_sentence"])
    silent_chunks = sum(1 for c in chunks if not c["has_speech"])
    avg_duration = np.mean([c["duration"] for c in chunks]) if chunks else 0

    return {
        "total_chunks": total,
        "with_speech": with_speech,
        "silent_chunks": silent_chunks,
        "mid_sentence_cuts": mid_cuts,
        "avg_chunk_duration": avg_duration,
    }


def print_comparison(old_metrics, new_metrics):
    print(f"\n  {'Metric':<25} {'Old (5s timer)':>15} {'New (silence)':>15} {'Improvement':>15}")
    print(f"  {'-'*70}")

    for key in ["total_chunks", "with_speech", "silent_chunks", "mid_sentence_cuts", "avg_chunk_duration"]:
        old_val = old_metrics[key]
        new_val = new_metrics[key]
        if isinstance(old_val, float):
            old_str = f"{old_val:.1f}s"
            new_str = f"{new_val:.1f}s"
        else:
            old_str = str(old_val)
            new_str = str(new_val)

        if key == "mid_sentence_cuts":
            if old_val > 0 and new_val < old_val:
                imp = f"-{old_val - new_val} cuts"
            elif new_val == 0:
                imp = "ELIMINATED" if old_val > 0 else "none"
            else:
                imp = f"{new_val - old_val:+d}"
        elif key == "silent_chunks":
            imp = f"-{old_val - new_val}" if new_val < old_val else str(new_val - old_val)
        else:
            imp = ""

        label = key.replace("_", " ").title()
        print(f"  {label:<25} {old_str:>15} {new_str:>15} {imp:>15}")


# -- SilenceDetector unit stress tests -----------------------------

def test_silence_detector_edge_cases():
    print("\n" + "=" * 70)
    print("SILENCE DETECTOR EDGE CASE TESTS")
    print("=" * 70)

    sr = 16000
    frame_ms = 200
    frame_samples = int(sr * frame_ms / 1000)

    # Test 1: Threshold sensitivity
    print("\n  Test 1: RMS values at various amplitudes")
    for amp in [0.001, 0.005, 0.008, 0.01, 0.012, 0.02, 0.05, 0.1, 0.3]:
        audio = np.random.randn(frame_samples).astype(np.float32) * amp
        rms = np.sqrt(np.mean(audio ** 2))
        above = "SPEECH" if rms >= 0.01 else "SILENT"
        print(f"    amplitude={amp:.3f} -> rms={rms:.5f} -> {above}")

    # Test 2: Transition timing
    print("\n  Test 2: Silence detection timing (how quickly is silence detected?)")
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=sr)

    # Feed speech
    speech = generate_speech_block(2.0, sr)
    for i in range(0, len(speech), frame_samples):
        sd.feed(speech[i:i + frame_samples])

    # Now feed silence frame by frame and track when is_silent() triggers
    silence = generate_silence(2.0, sr)
    triggered_at = None
    for i in range(0, len(silence), frame_samples):
        sd.feed(silence[i:i + frame_samples])
        elapsed_ms = (i + frame_samples) / sr * 1000
        if sd.is_silent() and triggered_at is None:
            triggered_at = elapsed_ms
    print(f"    Silence detected after {triggered_at:.0f}ms of silence (threshold: 500ms)")

    # Test 3: Noise floor resilience
    print("\n  Test 3: Background noise levels")
    for noise_amp in [0.001, 0.003, 0.005, 0.008, 0.01, 0.015]:
        sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=sr)
        noise = np.random.randn(sr).astype(np.float32) * noise_amp
        for i in range(0, len(noise), frame_samples):
            sd.feed(noise[i:i + frame_samples])
        status = "SILENT" if sd.is_silent() else "not silent"
        rms = np.sqrt(np.mean(noise ** 2))
        print(f"    noise amp={noise_amp:.3f} (rms={rms:.5f}) -> {status}")

    # Test 4: Performance
    print("\n  Test 4: SilenceDetector performance (1 hour of audio)")
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=sr)
    audio_1hr = np.random.randn(sr * 3600).astype(np.float32) * 0.1
    t0 = time.perf_counter()
    for i in range(0, len(audio_1hr), frame_samples):
        sd.feed(audio_1hr[i:i + frame_samples])
    elapsed = time.perf_counter() - t0
    feeds = len(audio_1hr) // frame_samples
    print(f"    {feeds} feed() calls in {elapsed:.3f}s ({elapsed/feeds*1000000:.1f}us per call)")
    print(f"    Processing rate: {3600/elapsed:.0f}x realtime")


# -- Main ----------------------------------------------------------

def main():
    sr = 16000
    np.random.seed(42)  # Reproducible results

    print("=" * 70)
    print("CHUNKING COMPARISON: Old (5s Timer) vs New (Silence Detection)")
    print("=" * 70)

    scenarios = [
        ("Natural Conversation (40s)", scenario_natural_conversation),
        ("Long Monologue (41s)", scenario_long_monologue),
        ("Rapid Back-and-Forth (11s)", scenario_rapid_back_and_forth),
        ("Short Recording (4s)", scenario_short_recording),
    ]

    for name, scenario_fn in scenarios:
        events, total_duration = scenario_fn()

        print(f"\n{'-' * 70}")
        print(f"SCENARIO: {name}")
        print(f"{'-' * 70}")

        # Print ground truth
        print(f"\n  Ground truth events ({len(events)} speech segments):")
        for e in events:
            print(f"    [{e.speaker:>4}] {e.start:5.1f}s - {e.end:5.1f}s: \"{e.text[:60]}{'...' if len(e.text) > 60 else ''}\"")

        # Build audio
        mic_audio, sys_audio = build_audio_from_events(events, total_duration, sr)
        mic_frames = build_frames_from_audio(mic_audio)
        sys_frames = build_frames_from_audio(sys_audio)

        # Simulate old approach
        old_chunks = simulate_old_approach(mic_frames, sys_frames, events, interval=5.0, sr=sr)
        print_chunks("OLD (5s fixed timer)", old_chunks)
        old_metrics = analyze_chunks(old_chunks)

        # Simulate new approach
        new_chunks = simulate_new_approach(mic_frames, sys_frames, events,
                                            silence_threshold=0.01, min_silence_ms=500,
                                            max_speech_s=30.0, poll_interval=0.2, sr=sr)
        print_chunks("NEW (silence detection)", new_chunks)
        new_metrics = analyze_chunks(new_chunks)

        # Compare
        print_comparison(old_metrics, new_metrics)

    # Run edge case tests
    test_silence_detector_edge_cases()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
