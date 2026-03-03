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


def test_debug_defaults(tmp_path):
    """Debug settings should default to False."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    assert config.is_debug_enabled() is False
    assert config.is_echo_debug_enabled() is False
    assert config.is_echo_gate_logging_enabled() is False


def test_debug_persistence(tmp_path):
    """Debug settings should persist across save/load."""
    from src.utils.config import ConfigManager
    config_file = tmp_path / "settings.json"
    config = ConfigManager(config_file=config_file)
    config.set("debug", "enabled", value=True)
    config.set("debug", "echo_diagnostics", value=True)
    config.set("debug", "echo_gate_logging", value=True)
    config.save()

    config2 = ConfigManager(config_file=config_file)
    assert config2.is_debug_enabled() is True
    assert config2.is_echo_debug_enabled() is True
    assert config2.is_echo_gate_logging_enabled() is True


def test_debug_accessors(tmp_path):
    """Debug accessor methods should reflect set values."""
    from src.utils.config import ConfigManager
    config = ConfigManager(config_file=tmp_path / "settings.json")
    config.set("debug", "enabled", value=True)
    assert config.is_debug_enabled() is True
    config.set("debug", "echo_gate_logging", value=True)
    assert config.is_echo_gate_logging_enabled() is True


