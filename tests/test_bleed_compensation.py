"""Tests for bleed-compensated silence detection in TranscriptionWorker."""

import os

import numpy as np
import pytest


def compensated_rms(mic_frame, sys_frame, bleed_factor=0.8):
    """Standalone version of the bleed compensation logic for testing."""
    mic_rms = float(np.sqrt(np.mean(mic_frame.astype(np.float64) ** 2)))
    if sys_frame is None or len(sys_frame) == 0:
        return mic_rms
    sys_rms = float(np.sqrt(np.mean(sys_frame.astype(np.float64) ** 2)))
    return max(mic_rms - bleed_factor * sys_rms, 0.0)


class TestCompensatedRms:
    """Test the bleed compensation formula."""

    def test_no_system_audio(self):
        """Without system audio, mic RMS passes through unchanged."""
        mic = np.random.randn(1024).astype(np.float32) * 0.1
        expected_rms = float(np.sqrt(np.mean(mic.astype(np.float64) ** 2)))
        assert compensated_rms(mic, None) == pytest.approx(expected_rms)
        assert compensated_rms(mic, np.array([])) == pytest.approx(expected_rms)

    def test_bleed_only_compensates_to_near_zero(self):
        """When mic is just bleed (similar RMS to sys), result is near zero."""
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        mic_frame = sys_frame * 1.1  # bleed is ~1.1x sys
        result = compensated_rms(mic_frame, sys_frame)
        assert result < 0.025

    def test_real_speech_preserved(self):
        """When user speaks, compensated RMS stays well above threshold."""
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        mic_frame = np.random.randn(1024).astype(np.float32) * 0.15
        result = compensated_rms(mic_frame, sys_frame)
        assert result > 0.05

    def test_silent_system_no_effect(self):
        """Silent system audio doesn't affect mic RMS."""
        mic = np.random.randn(1024).astype(np.float32) * 0.1
        sys = np.zeros(1024, dtype=np.float32)
        mic_only = compensated_rms(mic, None)
        with_silent_sys = compensated_rms(mic, sys)
        assert mic_only == pytest.approx(with_silent_sys)

    def test_floor_at_zero(self):
        """Compensated RMS never goes negative."""
        mic = np.random.randn(1024).astype(np.float32) * 0.01
        sys = np.random.randn(1024).astype(np.float32) * 0.1
        result = compensated_rms(mic, sys)
        assert result == 0.0


class TestBleedCompensatedSilenceDetection:
    """Integration test: bleed compensation enables silence detection."""

    def test_bleed_without_compensation_blocks_silence(self):
        """Without compensation, bleed prevents silence detection."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        for _ in range(3):
            bleed_frame = np.random.randn(1024).astype(np.float32) * 0.06
            sd.feed(bleed_frame)
        assert not sd.is_silent()

    def test_bleed_with_compensation_enables_silence(self):
        """With compensation, bleed is neutralized and silence triggers."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        for _ in range(10):
            # Simulate real bleed: mic picks up attenuated system audio.
            # Typical bleed is ~60-70% of system level; 0.8 factor
            # over-compensates, driving compensated RMS to zero.
            sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
            mic_frame = sys_frame * 0.7  # bleed is ~70% of system level
            rms = compensated_rms(mic_frame, sys_frame)
            sd.feed_rms(rms, len(mic_frame))
        assert sd.is_silent()

    def test_speech_during_bleed_still_detected(self):
        """Real user speech on top of bleed is still detected as speech."""
        from src.core.silence_detector import SilenceDetector

        sd = SilenceDetector(silence_threshold=0.005, min_silence_ms=400, sample_rate=16000)
        mic_frame = np.random.randn(1024).astype(np.float32) * 0.15
        sys_frame = np.random.randn(1024).astype(np.float32) * 0.06
        rms = compensated_rms(mic_frame, sys_frame)
        sd.feed_rms(rms, len(mic_frame))
        assert sd._has_had_speech
        assert not sd.is_silent()


class TestReplaySession:
    """Replay the 2026-03-04 session data to verify bleed compensation works."""

    SESSION_DIR = os.path.join(".cadence", "echo_debug", "20260304_080418", "chunks")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(".cadence", "echo_debug", "20260304_080418")),
        reason="Session data not available"
    )
    def test_replay_chunks_detect_silence_boundaries(self):
        """Replay raw mic+sys audio: compensated silence detector should
        trigger BEFORE the 30s max_speech limit on at least some chunks."""
        import soundfile as sf
        from src.core.silence_detector import SilenceDetector

        chunk_size = 1024
        sr = 16000
        max_speech_samples = int(30.0 * sr)
        # Use 200ms min_silence — the session audio has short inter-speaker
        # pauses (150-300ms). The key validation is that bleed compensation
        # makes these pauses detectable at all; without it, max gaps are <20ms.
        min_silence_ms = 200

        silence_triggered_count = 0

        for chunk_idx in range(1, 7):
            mic_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_mic_raw.wav")
            sys_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_sys.wav")

            mic_data, _ = sf.read(mic_path)
            sys_data, _ = sf.read(sys_path)

            sd = SilenceDetector(
                silence_threshold=0.005, min_silence_ms=min_silence_ms, sample_rate=sr
            )

            # Feed frame-by-frame with bleed compensation
            samples_fed = 0
            triggered_before_max = False
            for start in range(0, len(mic_data) - chunk_size, chunk_size):
                mic_frame = mic_data[start:start + chunk_size].astype(np.float32)
                sys_start = start
                if sys_start + chunk_size <= len(sys_data):
                    sys_frame = sys_data[sys_start:sys_start + chunk_size].astype(np.float32)
                    rms = compensated_rms(mic_frame, sys_frame)
                    sd.feed_rms(rms, chunk_size)
                else:
                    sd.feed(mic_frame)
                samples_fed += chunk_size

                if sd.is_silent() and sd._has_had_speech and samples_fed < max_speech_samples:
                    triggered_before_max = True
                    break

            if triggered_before_max:
                silence_triggered_count += 1

        # With bleed compensation, at least some of the 6 chunks should
        # have silence detected before the 30s limit
        assert silence_triggered_count >= 3, (
            f"Expected silence to trigger in >=3 of 6 chunks, got {silence_triggered_count}"
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(".cadence", "echo_debug", "20260304_080418")),
        reason="Session data not available"
    )
    def test_compensation_dramatically_increases_silence_gaps(self):
        """Bleed compensation should increase detectable silence gaps by >=5x
        compared to raw mic audio, proving bleed was masking real pauses."""
        import soundfile as sf

        chunk_size = 128  # fine granularity for accurate measurement
        sr = 16000
        threshold = 0.005

        for chunk_idx in range(1, 7):
            mic_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_mic_raw.wav")
            sys_path = os.path.join(self.SESSION_DIR, f"chunk_{chunk_idx:04d}_sys.wav")

            mic_data, _ = sf.read(mic_path)
            sys_data, _ = sf.read(sys_path)

            # Measure longest silence gap WITHOUT compensation (raw mic)
            max_raw = 0
            cur_raw = 0
            for start in range(0, len(mic_data) - chunk_size, chunk_size):
                mic_frame = mic_data[start:start + chunk_size].astype(np.float32)
                raw_rms = float(np.sqrt(np.mean(mic_frame.astype(np.float64) ** 2)))
                if raw_rms < threshold:
                    cur_raw += chunk_size
                else:
                    max_raw = max(max_raw, cur_raw)
                    cur_raw = 0
            max_raw = max(max_raw, cur_raw)

            # Measure longest silence gap WITH compensation
            max_comp = 0
            cur_comp = 0
            for start in range(0, len(mic_data) - chunk_size, chunk_size):
                mic_frame = mic_data[start:start + chunk_size].astype(np.float32)
                if start + chunk_size <= len(sys_data):
                    sys_frame = sys_data[start:start + chunk_size].astype(np.float32)
                    comp_rms = compensated_rms(mic_frame, sys_frame)
                else:
                    comp_rms = float(np.sqrt(np.mean(mic_frame.astype(np.float64) ** 2)))
                if comp_rms < threshold:
                    cur_comp += chunk_size
                else:
                    max_comp = max(max_comp, cur_comp)
                    cur_comp = 0
            max_comp = max(max_comp, cur_comp)

            # Compensation should dramatically increase detectable gaps.
            # Raw mic has ~8-16ms gaps (bleed keeps RMS above threshold).
            # Compensated has 150-300ms gaps (real inter-speaker pauses).
            assert max_comp > max_raw * 5, (
                f"Chunk {chunk_idx}: compensated gap ({max_comp / sr * 1000:.0f}ms) "
                f"should be >5x raw gap ({max_raw / sr * 1000:.0f}ms)"
            )
