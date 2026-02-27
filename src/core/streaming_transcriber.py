"""
Real-time streaming transcription engine for Cadence.
Uses chunked processing with Local Agreement-N confirmation.
"""

import logging
import time
import numpy as np
from collections import deque

logger = logging.getLogger("Cadence")


class StreamingTranscriber:
    """
    Real-time Whisper transcription with Local Agreement.

    Processes audio in chunks. Text is only confirmed when N consecutive
    predictions match, reducing false transcriptions.
    """

    def __init__(
        self,
        transcriber,
        chunk_duration=1.0,
        sample_rate=16000,
        agreement_threshold=2,
        silence_threshold=0.01,
    ):
        """
        Args:
            transcriber: A Transcriber instance for inference
            chunk_duration: Seconds of audio per processing chunk
            sample_rate: Audio sample rate (Hz)
            agreement_threshold: Consecutive matching predictions needed to confirm
            silence_threshold: RMS energy below which audio is silence
        """
        self.transcriber = transcriber
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.agreement_threshold = agreement_threshold
        self.silence_threshold = silence_threshold
        self._init_state()

    def _init_state(self):
        """Initialize/reset streaming state."""
        self.audio_buffer = []
        self.recent_predictions = deque(maxlen=self.agreement_threshold)
        self.confirmed_text = []
        self.current_partial = ""

    def _is_silence(self, audio_chunk):
        """Check if audio chunk is silence based on RMS energy."""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return bool(energy < self.silence_threshold)

    def _apply_agreement(self, prediction):
        """
        Apply Local Agreement-N policy to a prediction.
        Returns dict with 'partial' and 'confirmed' keys.
        """
        self.recent_predictions.append(prediction)

        if len(self.recent_predictions) >= self.agreement_threshold:
            if len(set(self.recent_predictions)) == 1:
                # All predictions match — confirm
                self.confirmed_text.append(prediction)
                self.current_partial = ""
                self.recent_predictions.clear()
                return {"partial": "", "confirmed": prediction}

        # No agreement yet
        self.current_partial = prediction
        return {"partial": prediction, "confirmed": ""}

    def process_chunk(self, audio_chunk):
        """
        Process a single audio chunk.

        Args:
            audio_chunk: numpy float32 array of audio samples

        Returns:
            dict with 'partial', 'confirmed', 'latency' keys
        """
        start_time = time.time()

        if self._is_silence(audio_chunk):
            return {
                "partial": self.current_partial,
                "confirmed": "",
                "latency": time.time() - start_time,
            }

        # Accumulate in buffer
        self.audio_buffer.append(audio_chunk)
        buffer_samples = sum(len(c) for c in self.audio_buffer)
        needed_samples = int(self.chunk_duration * self.sample_rate)

        if buffer_samples < needed_samples:
            return {
                "partial": self.current_partial,
                "confirmed": "",
                "latency": time.time() - start_time,
            }

        # Enough audio — combine and transcribe
        combined = np.concatenate(self.audio_buffer)
        self.audio_buffer = []

        try:
            prediction = self.transcriber.transcribe_text(combined)
        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            return {
                "partial": self.current_partial,
                "confirmed": "",
                "latency": time.time() - start_time,
                "error": str(e),
            }

        result = self._apply_agreement(prediction)
        result["latency"] = time.time() - start_time
        return result

    def finalize(self):
        """Flush remaining buffer and return final transcript."""
        if self.audio_buffer:
            combined = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            try:
                final_text = self.transcriber.transcribe_text(combined)
                if final_text:
                    self.confirmed_text.append(final_text)
            except Exception as e:
                logger.error(f"Error finalizing stream: {e}")

        return " ".join(self.confirmed_text)

    def get_full_transcript(self):
        """Get all confirmed text so far."""
        return " ".join(self.confirmed_text)

    def reset(self):
        """Reset state for a new session."""
        self._init_state()
