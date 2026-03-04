"""
Tests for settings dialog quality-of-life features:
- Config statistics reset
- Transcriber model download helpers
- Partial download cleanup
"""

import json
import os
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Config: reset_statistics
# ---------------------------------------------------------------------------

class TestConfigResetStatistics:
    """Tests for ConfigManager.reset_statistics()."""

    def test_reset_clears_all_stats(self, tmp_path):
        from src.utils.config import ConfigManager

        config = ConfigManager(config_file=tmp_path / "settings.json")
        config.set("stats", "total_recordings", value=42)
        config.set("stats", "total_words", value=9999)
        config.set("stats", "total_duration_s", value=3600)
        config.set("stats", "you_words", value=500)
        config.set("stats", "speaker_words", value=800)
        config.save()

        config.reset_statistics()

        assert config.get("stats", "total_recordings") == 0
        assert config.get("stats", "total_words") == 0
        assert config.get("stats", "total_duration_s") == 0
        assert config.get("stats", "you_words") == 0
        assert config.get("stats", "speaker_words") == 0

    def test_reset_persists_to_disk(self, tmp_path):
        from src.utils.config import ConfigManager

        config_file = tmp_path / "settings.json"
        config = ConfigManager(config_file=config_file)
        config.set("stats", "total_recordings", value=10)
        config.save()

        config.reset_statistics()

        # Reload from disk
        config2 = ConfigManager(config_file=config_file)
        assert config2.get("stats", "total_recordings") == 0

    def test_reset_does_not_affect_other_settings(self, tmp_path):
        from src.utils.config import ConfigManager

        config = ConfigManager(config_file=tmp_path / "settings.json")
        config.set("whisper", "model_size", value="small")
        config.set("user", "first_name", value="Alice")
        config.set("stats", "total_recordings", value=5)
        config.save()

        config.reset_statistics()

        assert config.get("whisper", "model_size") == "small"
        assert config.get("user", "first_name") == "Alice"
        assert config.get("stats", "total_recordings") == 0


class TestConfigIncrementStat:
    """Tests for ConfigManager.increment_stat()."""

    def test_increment_from_zero(self, tmp_path):
        from src.utils.config import ConfigManager

        config = ConfigManager(config_file=tmp_path / "settings.json")
        config.increment_stat("total_recordings")
        assert config.get("stats", "total_recordings") == 1

    def test_increment_by_amount(self, tmp_path):
        from src.utils.config import ConfigManager

        config = ConfigManager(config_file=tmp_path / "settings.json")
        config.increment_stat("total_words", amount=150)
        config.increment_stat("total_words", amount=50)
        assert config.get("stats", "total_words") == 200

    def test_get_stats_returns_dict(self, tmp_path):
        from src.utils.config import ConfigManager

        config = ConfigManager(config_file=tmp_path / "settings.json")
        config.increment_stat("total_recordings", amount=3)
        stats = config.get_stats()
        assert isinstance(stats, dict)
        assert stats["total_recordings"] == 3


# ---------------------------------------------------------------------------
# Transcriber: model download detection + cleanup
# ---------------------------------------------------------------------------

class TestTranscriberModelHelpers:
    """Tests for Transcriber.is_model_downloaded / clean_partial_download."""

    def _make_transcriber(self, tmp_path):
        """Create a Transcriber with a temporary models directory."""
        from src.core.transcriber import Transcriber

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)
        t = Transcriber(model_size="base", model_dir=models_dir)
        t.models_dir = models_dir
        return t

    def test_model_not_downloaded_when_dir_missing(self, tmp_path):
        t = self._make_transcriber(tmp_path)
        assert t.is_model_downloaded("tiny") is False

    def test_model_downloaded_when_complete(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        # Simulate a complete download structure
        model_dir = os.path.join(
            t.models_dir, "models--Systran--faster-whisper-base"
        )
        snap_dir = os.path.join(model_dir, "snapshots", "abc123")
        os.makedirs(snap_dir, exist_ok=True)
        # Create model.bin
        with open(os.path.join(snap_dir, "model.bin"), "wb") as f:
            f.write(b"fake model data")

        assert t.is_model_downloaded("base") is True

    def test_model_not_downloaded_when_incomplete(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        model_dir = os.path.join(
            t.models_dir, "models--Systran--faster-whisper-small"
        )
        blobs_dir = os.path.join(model_dir, "blobs")
        os.makedirs(blobs_dir, exist_ok=True)
        # Create .incomplete file
        with open(os.path.join(blobs_dir, "sha256-abc.incomplete"), "wb") as f:
            f.write(b"partial")

        assert t.is_model_downloaded("small") is False

    def test_clean_partial_download_removes_incomplete(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        model_dir = os.path.join(
            t.models_dir, "models--Systran--faster-whisper-small"
        )
        blobs_dir = os.path.join(model_dir, "blobs")
        os.makedirs(blobs_dir, exist_ok=True)
        with open(os.path.join(blobs_dir, "sha256-abc.incomplete"), "wb") as f:
            f.write(b"partial")

        result = t.clean_partial_download("small")
        assert result is True
        assert not os.path.exists(model_dir)

    def test_clean_partial_download_noop_when_complete(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        model_dir = os.path.join(
            t.models_dir, "models--Systran--faster-whisper-base"
        )
        snap_dir = os.path.join(model_dir, "snapshots", "abc123")
        os.makedirs(snap_dir, exist_ok=True)
        with open(os.path.join(snap_dir, "model.bin"), "wb") as f:
            f.write(b"model data")

        result = t.clean_partial_download("base")
        assert result is False
        assert os.path.exists(model_dir)

    def test_clean_partial_when_missing_model_bin(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        model_dir = os.path.join(
            t.models_dir, "models--Systran--faster-whisper-medium"
        )
        snap_dir = os.path.join(model_dir, "snapshots", "abc123")
        os.makedirs(snap_dir, exist_ok=True)
        # Directory exists but no model.bin

        result = t.clean_partial_download("medium")
        assert result is True
        assert not os.path.exists(model_dir)

    def test_clean_all_partial_downloads(self, tmp_path):
        t = self._make_transcriber(tmp_path)

        # Create two partial downloads
        for name in ["tiny", "small"]:
            model_dir = os.path.join(
                t.models_dir, f"models--Systran--faster-whisper-{name}"
            )
            blobs_dir = os.path.join(model_dir, "blobs")
            os.makedirs(blobs_dir, exist_ok=True)
            with open(os.path.join(blobs_dir, "sha256.incomplete"), "wb") as f:
                f.write(b"partial")

        cleaned = t.clean_all_partial_downloads()
        assert "tiny" in cleaned
        assert "small" in cleaned

    def test_cache_dir_name(self, tmp_path):
        t = self._make_transcriber(tmp_path)
        assert t._cache_dir_name("base") == "models--Systran--faster-whisper-base"
        assert t._cache_dir_name("Systran/faster-whisper-large-v3") == \
            "models--Systran--faster-whisper-large-v3"
