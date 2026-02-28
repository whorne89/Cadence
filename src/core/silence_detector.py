"""
Energy-based silence detection for Cadence.
Tracks RMS energy of audio frames to detect speech pauses.
"""

import numpy as np


class SilenceDetector:
    """Detects silence gaps in a stream of audio frames using RMS energy."""

    def __init__(self, silence_threshold=0.01, min_silence_ms=500, sample_rate=16000):
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate
        self._silent_samples = 0
        self._has_had_speech = False

    def feed(self, audio_chunk: np.ndarray):
        """Feed an audio chunk and update silence tracking."""
        if len(audio_chunk) == 0:
            return
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms < self.silence_threshold:
            self._silent_samples += len(audio_chunk)
        else:
            self._silent_samples = 0
            self._has_had_speech = True

    def is_silent(self) -> bool:
        """True if silence has lasted at least min_silence_ms."""
        min_samples = int(self.sample_rate * self.min_silence_ms / 1000)
        return self._silent_samples >= min_samples

    def reset(self):
        """Reset all state."""
        self._silent_samples = 0
        self._has_had_speech = False
