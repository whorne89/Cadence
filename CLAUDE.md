# Cadence

Meeting transcription desktop app with real-time speech-to-text.

## Tech Stack
- Python 3.12, uv, PySide6
- faster-whisper (CPU, int8) for transcription
- sounddevice for mic capture
- PyInstaller for building exe

## Architecture
Follows Resonance pattern: `src/core/` (business logic), `src/gui/` (PySide6 UI), `src/utils/` (config, logging, paths).
System tray app with full window on demand. QThread for async transcription.

## Key Patterns
- ConfigManager: JSON config at `.cadence/settings.json`
- resource_path: `get_resource_path()` / `get_app_data_path()` for dev vs bundled
- QThread + QObject worker pattern for background transcription
- Signal/slot for cross-thread communication
- Single mic stream via sounddevice

## Changelog & Versioning
- **Always update `CHANGELOG.md`** when making any functional change (features, fixes, refactors).
- Follow [Keep a Changelog](https://keepachangelog.com/) format: Added, Changed, Fixed, Removed sections.
- Follow [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH (breaking.feature.fix).
- Version must be updated in **three places** when bumping: `src/version.py`, `pyproject.toml`, and the new `CHANGELOG.md` entry header.
- Add changelog entries under an `## [Unreleased]` section during development. When committing/releasing, replace with the version number and date.
- **Never include benchmark data, test meeting names, participant names, or internal testing details in the changelog.** The changelog is public-facing.

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

## Releasing a New Version
When tagging a new version, follow this full release process:

1. **Update version** in all three places: `src/version.py`, `pyproject.toml`, `CHANGELOG.md` header.
2. **Update `CHANGELOG.md`**: Move `[Unreleased]` entries under the new version header with today's date.
3. **Commit** the version bump and changelog update.
4. **Push to main**: `git push origin main`.
5. **Build the EXE**: Run `scripts\build.bat`. Output goes to `dist/Cadence/`.
6. **Zip the build**: Create `Cadence-vX.Y.Z-windows.zip` from `dist/Cadence/`.
7. **Create GitHub release with tag**: Use `gh release create vX.Y.Z` with the changelog notes for that version and attach the zip as a release asset.
8. **Only tag milestone versions** — not every commit. Tags should correspond to meaningful releases users would download.
