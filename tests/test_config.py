import json
import pytest
from pathlib import Path


def test_config_loads_defaults(tmp_path):
    """ConfigManager should use defaults when no file exists."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    assert config.get("whisper", "model_size") == "base"
    assert config.get("whisper", "compute_type") == "int8"
    assert config.get("audio", "sample_rate") == 16000


def test_config_save_and_load(tmp_path):
    """ConfigManager should persist changes to disk."""
    from src.utils.config import ConfigManager
    config_file = tmp_path / "settings.json"
    config = ConfigManager(config_file=config_file)
    config.set("whisper", "model_size", value="small")
    config.save()

    config2 = ConfigManager(config_file=config_file)
    assert config2.get("whisper", "model_size") == "small"


def test_config_merge_preserves_new_defaults(tmp_path):
    """Loading old config should pick up new default keys."""
    from src.utils.config import ConfigManager
    config_file = tmp_path / "settings.json"
    config_file.write_text(json.dumps({"whisper": {"model_size": "tiny"}}))

    config = ConfigManager(config_file=config_file)
    assert config.get("whisper", "model_size") == "tiny"
    assert config.get("audio", "sample_rate") == 16000


def test_config_get_with_default(tmp_path):
    """get() should return default for missing keys."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    assert config.get("nonexistent", "key", default="fallback") == "fallback"


def test_transcription_interval_default(tmp_path):
    """get_transcription_interval() should return 5.0 by default."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    assert config.get_transcription_interval() == 5.0


def test_transcription_interval_range(tmp_path):
    """transcription_interval should accept values in the valid range."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    config.set("whisper", "transcription_interval", value=2.0)
    assert config.get_transcription_interval() == 2.0
    config.set("whisper", "transcription_interval", value=8.0)
    assert config.get_transcription_interval() == 8.0
