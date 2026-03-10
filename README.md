# Cadence

Real-time meeting transcription. Records your microphone, transcribes speech in real-time using Whisper, and saves organized transcripts.

## Features

- **Real-time transcription** — Live speech-to-text during meetings using faster-whisper (CPU, int8)
- **Post-processing** — Full-audio re-transcription after recording for higher accuracy
- **Model selection** — Four Whisper model tiers: Fastest (tiny), Balanced (base), Accurate (small), Precision (medium)
- **Meeting participant** — Tag transcripts with who the meeting was with
- **Session management** — Organize transcripts in folders with date-sorted history
- **Activity dashboard** — Session metrics: recordings, duration, word counts, weekly stats
- **Dark theme UI** — Frameless rounded window with collapsible sidebar and themed dialogs
- **System tray** — Minimal footprint with tray icon, runs in background

## Install

Download the latest release from the [Releases](https://github.com/whorne89/Cadence/releases) page and run `Cadence.exe`.

## Development

### Requirements

- Python 3.12
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
| Audio capture | sounddevice |
| Build | PyInstaller |

## License

MIT
