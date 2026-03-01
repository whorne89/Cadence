# Cadence

Real-time meeting transcription with speaker attribution. Records both your microphone and system audio, transcribes in real-time, and labels who said what.

## Features

- **Real-time transcription** — Live speech-to-text during meetings using faster-whisper (CPU, int8)
- **Dual audio capture** — Simultaneous mic + system audio (WASAPI loopback) recording
- **Speaker attribution** — Automatic "You" vs "Speaker" labels with customizable per-transcript speaker names
- **Post-processing** — Full-audio re-transcription after recording for higher accuracy
- **Echo cancellation** — Hybrid energy + text-based echo gate suppresses speaker bleed between channels
- **Activity dashboard** — Session metrics: recordings, duration, word counts, weekly stats
- **Toast notifications** — In-app dark pill overlays for recording status, startup, and transcription results
- **Session management** — Organize transcripts in folders with date-sorted history
- **Dark theme UI** — Frameless rounded window with collapsible sidebar and themed dialogs
- **Bug reporting** — One-click GitHub issue creation with system info and recent logs
- **System tray** — Minimal footprint with tray icon, runs in background

## Quick Start

```
uv sync
uv run python src/main.py
```

## Requirements

- Python 3.12
- Windows 10/11 (WASAPI loopback requires Windows)

## Tech Stack

- **UI**: PySide6 (Qt for Python)
- **Transcription**: faster-whisper (Whisper models: tiny, base, small, medium)
- **Mic capture**: sounddevice
- **System audio**: PyAudioWPatch (WASAPI loopback)
- **Build**: PyInstaller

## Building

```
scripts\build.bat
```

## Testing

```
uv run pytest tests/ -v
```

## License

MIT
