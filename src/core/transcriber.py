"""
Whisper transcription engine for Cadence.
CPU-only batch transcription using faster-whisper.
"""

import logging
import shutil
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

    # Expected download sizes (MB) for progress estimation
    MODEL_SIZES_MB = {
        "tiny": 70,
        "base": 140,
        "small": 500,
        "medium": 1500,
    }

    def __init__(self, model_size="base", model_dir=None):
        self.model_size = model_size
        self.model_dir = model_dir
        self.model = None
        self._lock = threading.Lock()

        # Resolve the models directory (used for download checks)
        if self.model_dir:
            self.models_dir = self.model_dir
        else:
            from utils.resource_path import get_app_data_path
            self.models_dir = get_app_data_path("models")

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
            download_root=self.models_dir,
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

    # -- Model download helpers -----------------------------------------------

    def _cache_dir_name(self, model_size):
        """Return the huggingface_hub cache directory name for a model."""
        if "/" in model_size:
            return "models--" + model_size.replace("/", "--")
        return f"models--Systran--faster-whisper-{model_size}"

    def is_model_downloaded(self, model_size):
        """
        Check if a model is fully downloaded.

        Returns True only when the cache directory exists, contains no
        ``.incomplete`` blobs, and at least one snapshot has a ``model.bin``.
        """
        model_path = os.path.join(self.models_dir, self._cache_dir_name(model_size))

        if not os.path.isdir(model_path):
            return False

        # .incomplete blobs mean a partial download
        blobs_dir = os.path.join(model_path, "blobs")
        if os.path.isdir(blobs_dir):
            for fname in os.listdir(blobs_dir):
                if fname.endswith(".incomplete"):
                    return False

        # At least one snapshot must contain model.bin
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snap in os.listdir(snapshots_dir):
                if os.path.isfile(os.path.join(snapshots_dir, snap, "model.bin")):
                    return True

        return False

    def clean_partial_download(self, model_size):
        """
        Remove partially downloaded model files so the next download starts fresh.

        Returns True if a partial download was cleaned up.
        """
        model_path = os.path.join(self.models_dir, self._cache_dir_name(model_size))

        if not os.path.isdir(model_path):
            return False

        # Detect partial state
        blobs_dir = os.path.join(model_path, "blobs")
        has_incomplete = False
        if os.path.isdir(blobs_dir):
            for fname in os.listdir(blobs_dir):
                if fname.endswith(".incomplete"):
                    has_incomplete = True
                    break

        has_model_bin = False
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snap in os.listdir(snapshots_dir):
                if os.path.isfile(os.path.join(snapshots_dir, snap, "model.bin")):
                    has_model_bin = True
                    break

        if has_incomplete or (os.path.isdir(snapshots_dir) and not has_model_bin):
            logger.info(
                f"Cleaning partial download for {model_size}: "
                f"incomplete={has_incomplete}, model_bin={has_model_bin}"
            )
            try:
                shutil.rmtree(model_path)
                logger.info(f"Removed partial download directory: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to clean partial download: {e}")
                return False

        return False

    def clean_all_partial_downloads(self):
        """Scan models_dir for any partial downloads and clean them up."""
        cleaned = []
        for model_size in self.VALID_MODELS:
            if self.clean_partial_download(model_size):
                cleaned.append(model_size)
        return cleaned
