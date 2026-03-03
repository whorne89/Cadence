# Cadence

Real-time meeting transcription with speaker attribution. Records both your microphone and system audio, transcribes in real-time, and labels who said what.

## Features

- **Real-time transcription** — Live speech-to-text during meetings using faster-whisper (CPU, int8)
- **Dual audio capture** — Simultaneous mic + system audio (WASAPI loopback) recording
- **Speaker attribution** — Automatic "You" vs "Speaker" labels with customizable per-transcript speaker names
- **Multi-layer echo cancellation** — Three-tier live echo gate (energy, envelope correlation, text matching) plus post-processing dedup with clause-level recovery
- **Post-processing** — Full-audio re-transcription after recording for higher accuracy
- **Model selection** — Four Whisper model tiers: Fastest (tiny), Balanced (base), Accurate (small), Precision (medium)
- **Session management** — Organize transcripts in folders with date-sorted history
- **Activity dashboard** — Session metrics: recordings, duration, word counts, weekly stats
- **Dark theme UI** — Frameless rounded window with collapsible sidebar and themed dialogs
- **System tray** — Minimal footprint with tray icon, runs in background

## Install

Download the latest release from the [Releases](https://github.com/whorne89/Cadence/releases) page and run `Cadence.exe`.

## Development

### Requirements

- Python 3.12
- Windows 10/11 (WASAPI loopback requires Windows)
- [uv](https://docs.astral.sh/uv/) package manager

### Run from source

```
uv sync
uv run python src/main.py
```

### Test

```
uv run pytest tests/ -v
```

### Build EXE

```
scripts\build.bat
```

Output goes to `dist/Cadence/`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | PySide6 (Qt for Python) |
| Transcription | faster-whisper (Whisper, CPU, int8) |
| Mic capture | sounddevice |
| System audio | PyAudioWPatch (WASAPI loopback) |
| Build | PyInstaller |

## License

MIT
