"""
Whisper transcription engine for Cadence.
CPU-only batch transcription using faster-whisper.
"""

import logging
import threading
import tempfile
import os
import numpy as np
import soundfile as sf

logger = logging.getLogger("Cadence")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


class Transcriber:
    """Batch-mode Whisper transcription engine (CPU, int8)."""

    VALID_MODELS = ["tiny", "base", "small", "medium"]

    def __init__(self, model_size="base", model_dir=None):
        self.model_size = model_size
        self.model_dir = model_dir
        self.model = None
        self._lock = threading.Lock()

    def _load_model(self):
        """Lazy load Whisper model on first use."""
        if self.model is not None:
            return
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed")
        logger.info(f"Loading Whisper model '{self.model_size}' (CPU, int8)...")
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            download_root=self.model_dir,
        )
        logger.info("Model loaded successfully")

    def transcribe(self, audio_data, language=None, progress_callback=None):
        """
        Transcribe audio data to text.

        Args:
            audio_data: numpy array of float32 audio samples at 16kHz
            language: Language code or None for auto-detect
            progress_callback: Optional callable(float) receiving progress 0.0-1.0

        Returns:
            List of dicts with 'text', 'start', 'end' keys
        """
        with self._lock:
            self._load_model()
            total_duration = len(audio_data) / 16000.0
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_data, 16000)
            try:
                segments, info = self.model.transcribe(
                    temp_path,
                    language=language,
                    vad_filter=True,
                    beam_size=5,
                )
                results = []
                for segment in segments:
                    results.append({
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                    })
                    if progress_callback and total_duration > 0:
                        progress = min(segment.end / total_duration, 1.0)
                        progress_callback(progress)
                return results
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def transcribe_text(self, audio_data, language=None):
        """Convenience: transcribe and return plain text."""
        segments = self.transcribe(audio_data, language=language)
        return " ".join(s["text"] for s in segments)

    def change_model(self, model_size):
        """Switch to a different model size. Unloads current model."""
        if model_size == self.model_size and self.model is not None:
            return
        with self._lock:
            self.model = None
            self.model_size = model_size
            logger.info(f"Model changed to '{model_size}' (will load on next use)")
