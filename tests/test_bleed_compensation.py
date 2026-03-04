"""Tests for bleed-compensated silence detection in TranscriptionWorker."""

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
