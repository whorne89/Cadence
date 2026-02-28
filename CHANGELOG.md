# Changelog

All notable changes to Cadence are documented here.

## [0.9.0] - 2026-02-28

### Added
- Echo gate for mic/system audio cross-talk suppression
  - Energy envelope correlation detects echo during live transcription
  - Word overlap dedup removes echo segments during post-processing
- Post-processing progress indicator (percentage shown in status bar)
- Transcript sorting by date (newest first by default)
  - Sort toggle button in transcript panel (ascending/descending)
  - Sorts by date stored in file header, not filename
- UI polish: thin scrollbars, settings quality presets, painted close icon
- Settings dialog redesign with quality presets (Fastest/Balanced/Accurate/Precision)
- Auto-detect option for mic and system audio devices

### Fixed
- Live echo gate was never running due to silence detector reset bug
- Echo dedup switched from SequenceMatcher to word overlap for better accuracy
- System audio now uses default WASAPI loopback instead of first device found
- Silent channel flooding in TranscriptionWorker
- Live transcript preserved when post-processing fails

## [0.8.0] - 2026-02-27

### Added
- Post-processing: full-audio re-transcription after recording stops
- Energy-based SilenceDetector for speech pause detection
- VAD-based chunking for smarter transcription timing

### Fixed
- Frame refs initialized before loop to prevent NameError on quick stop
- Guard SilenceDetector against empty arrays

### Removed
- Obsolete transcription_interval config and UI

## [0.7.0] - 2026-02-27

### Added
- Dark theme visual overhaul with frameless rounded window
- Hamburger toggle for collapsible sidebar
- Status badge and top stats bar (recording time, word count)
- Compact header layout

### Fixed
- Transcript not appearing after recording, improved naming
- Clear confirmation dialog, start/stop recording state bug

## [0.6.0] - 2026-02-26

### Added
- 3-panel layout redesign (folders, transcripts, viewer)
- Clear and copy buttons for transcript
- Settings dialog with interval spinner
- Folder management with context menus (rename, delete, move)

### Fixed
- Windows path safety and filename sanitization in SessionManager

### Changed
- SessionManager rewritten for .txt files and folder management

## [0.5.0] - 2026-02-26

### Added
- Main application controller wiring all components
- PyInstaller build configuration
- Launcher scripts (with and without console)

## [0.4.0] - 2026-02-26

### Added
- GUI components: system tray icon, main window, settings dialog

## [0.3.0] - 2026-02-26

### Added
- Session manager for transcript persistence
- Dual-channel audio recorder (mic + WASAPI loopback)

## [0.2.0] - 2026-02-26

### Added
- Batch transcription engine (faster-whisper, CPU, int8)
- Streaming transcriber with Local Agreement algorithm

## [0.1.0] - 2026-02-26

### Added
- Project scaffold with directory structure and dependencies
- Utils layer (config, logger, resource_path)
