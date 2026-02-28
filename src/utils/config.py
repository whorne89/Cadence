"""
Configuration management for Cadence.
"""

import copy
import json
from pathlib import Path

from utils.resource_path import get_app_data_path


class ConfigManager:
    """Manages application configuration and settings."""

    DEFAULT_CONFIG = {
        "version": "0.1.0",
        "whisper": {
            "model_size": "base",
            "language": "en",
            "compute_type": "int8",
            "streaming_model_size": "base",
            "reprocess_model_size": "small",
        },
        "audio": {
            "sample_rate": 16000,
            "mic_device_index": None,
            "system_device_index": None,
            "channels": 1,
        },
        "session": {
            "auto_reprocess": False,
            "save_audio": True,
        },
        "ui": {
            "show_notifications": True,
            "minimize_to_tray": True,
        },
    }

    def __init__(self, config_file=None):
        if config_file is None:
            self.config_file = Path(get_app_data_path()) / "settings.json"
        else:
            self.config_file = Path(config_file)
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.load()

    def load(self):
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self.config = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            else:
                self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        except Exception:
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)

    def save(self):
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception:
            return False

    def get(self, *keys, default=None):
        """Get a nested config value. Example: config.get("whisper", "model_size")"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value):
        """Set a nested config value. Example: config.set("whisper", "model_size", value="small")"""
        if not keys:
            return
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def get_model_size(self):
        return self.get("whisper", "model_size", default="base")

    def get_streaming_model_size(self):
        return self.get("whisper", "streaming_model_size", default="base")

    def get_reprocess_model_size(self):
        return self.get("whisper", "reprocess_model_size", default="small")

    def get_mic_device(self):
        return self.get("audio", "mic_device_index", default=None)

    def get_system_device(self):
        return self.get("audio", "system_device_index", default=None)

    def _merge_configs(self, default, loaded):
        """Deep merge loaded config with defaults."""
        merged = default.copy()
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
