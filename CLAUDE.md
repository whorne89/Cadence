# Cadence

Meeting transcription desktop app with real-time speaker attribution.

## Tech Stack
- Python 3.12, uv, PySide6
- faster-whisper (CPU, int8) for transcription
- sounddevice (mic) + PyAudioWPatch (WASAPI loopback) for audio
- PyInstaller for building exe

## Architecture
Follows Resonance pattern: `src/core/` (business logic), `src/gui/` (PySide6 UI), `src/utils/` (config, logging, paths).
System tray app with full window on demand. QThread for async transcription.

## Key Patterns
- ConfigManager: JSON config at `.cadence/settings.json`
- resource_path: `get_resource_path()` / `get_app_data_path()` for dev vs bundled
- QThread + QObject worker pattern for background transcription
- Signal/slot for cross-thread communication
- Dual audio: mic (sounddevice) + system (WASAPI loopback via PyAudioWPatch)

## Running
```
uv sync
uv run python src/main.py
```

## Testing
```
uv run pytest tests/ -v
```

## Building
```
scripts\build.bat
```
