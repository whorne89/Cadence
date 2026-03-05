# Changelog

All notable changes to Cadence are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Audio-level echo cancellation:** Spectral subtraction removes speaker echo from mic signal before Whisper transcription
- **Bleed-compensated silence detection:** Subtracts estimated speaker bleed from mic RMS before feeding silence detector, enabling natural speech segmentation instead of 30s monolithic blocks
- **Sound effects:** Piano chimes on recording start/stop (toggleable in Settings)
- **Auto-updater:** Checks for new releases on startup and shows toast notification with download link
- **Microphone level meter:** Test Microphone button in Settings opens real-time audio level monitor
- **Model download management:** Automatic prompt to download missing Whisper models with progress dialog; partial download cleanup on retry
- **Check for Updates button** in Settings for manual update checks
- **Reset Statistics button** in Settings to clear usage metrics
- **Usage statistics tracking:** Records total recordings, duration, word counts per speaker
- **Echo diagnostics:** Pre-AEC raw mic audio now saved alongside post-AEC audio for comparison

### Changed
- Replaced STFT spectral subtraction with Wiener filter (scipy) + spectral gating (noisereduce) for echo cancellation — significantly better echo removal while preserving user speech during double-talk
- Fixed pipeline ordering: echo gate now checks raw audio before AEC processes survivors, restoring gate threshold effectiveness
- Recalibrated echo gate thresholds based on real meeting data — mic_rms limits raised from 0.014/0.020 to 0.040/0.055, ratio widened to catch more bleed patterns
- Added post-AEC echo detection: suppresses chunks where AEC removed >70% of signal energy (confirms chunk was mostly echo)
- Raised envelope correlation mic_rms guard from 0.020 to 0.055 to match recalibrated energy gate
- Added pyroomacoustics and noisereduce as dependencies
- Mic minimum silence duration lowered from 400ms to 200ms (bleed-compensated silence gaps are shorter than 400ms in fast-paced meetings)
- Settings dialog now scrollable with fixed-width layout to accommodate new sections

## [2.2.0] - 2026-03-03

### Added
- **Phase 1 — Widened time window:** Echo detection window expanded from 15s to 30s across all three detection layers (deduplicate_segments, _is_text_echo, _retract_echo_you). Catches delayed echo up to 30s apart. Recent segment retention increased from 60s to 90s.
- **Phase 2 — Audio envelope correlation in live pipeline:** `is_echo()` now runs as tier 3 in the live transcription pipeline (after energy gate, before Whisper). Uses envelope correlation threshold of 0.7 with mic_rms < 0.020 guard to catch echo that Whisper transcribes as different words per channel.
- **Phase 3 — Prefix/suffix extraction:** New `_extract_prefix_suffix()` function with `ACKNOWLEDGMENT_WORDS` set. When clause splitting can't isolate genuine speech (no punctuation delimiter), scans 1-4 word prefixes/suffixes. Recovers acknowledgments like "Okay", "Sure", "Yes" glued to echo without punctuation.
- **Phase 4 — SequenceMatcher for medium segments:** Pass 3b in `deduplicate_segments()` applies character-level SequenceMatcher (ratio >= 0.55) to 6-15 word segments. Catches echo where Whisper wording differences reduce word overlap below threshold. Routes through clause recovery before full removal.
- 20 new tests covering all four phases (261 total, up from 241)

### Changed
- `deduplicate_segments()` default `time_window`: 15.0 → 30.0
- `_is_text_echo()` default `time_window`: 15.0 → 30.0
- `_retract_echo_you()` default `time_window`: 15.0 → 30.0
- `_recent_them` / `_recent_you` retention cutoff: 60s → 90s
- `_extract_unique_clauses()` now falls back to prefix/suffix extraction when clause splitting fails

## [2.1.0] - 2026-03-03

### Added
- **Clause-level echo recovery:** `_extract_unique_clauses()` splits mixed segments on sentence boundaries, checks per-clause word overlap, keeps genuine clauses while removing echoed ones
- Clause recovery integrated into three detection layers: `deduplicate_segments` (post-processing), `_is_text_echo` (live), `_retract_echo_you` (retroactive)
- **Debug Settings UI:** Master toggle with sub-options for echo diagnostics and echo gate logging
- **Echo diagnostics system:** `echo_diagnostics.py` saves WAV files + metadata per audio chunk for offline analysis
- Conditional echo gate logging behind settings toggle
- Live settings refresh: `_on_settings_changed` updates diagnostics and logging flags without restart
- Per-channel silence thresholds — mic: 0.005/400ms/0.3s, sys: 0.01/500ms/0.5s
- Reverse overlap dedup — catches echo concatenated with real speech (5+ word "them" segments)
- Language enforcement — `language="en"` on all transcribe calls
- Hallucination filter — non-English text (character ratio + vocabulary coverage) and filler-only segments
- Segment merging — `merge_segments()` combines same-speaker segments within 2s gap
- Comma-based clause splitting fallback for 12+ word single-sentence segments

### Changed
- Transcription quality overhaul for improved accuracy and echo detection

## [2.0.0] - 2026-03-01

### Added
- Speaker labels: "Speaker" replaces "Them", customizable per-transcript speaker name
- Your Profile: name field in settings replaces "You" label in transcripts
- Activity dashboard: 8 metric cards in Settings (recordings, duration, words, weekly stats)
- Toast notifications: in-app dark pill overlays replace Windows balloon notifications
- About dialog: rounded themed About page matching app design
- Themed input dialogs: all input prompts use rounded dark theme (folders, speaker names, moves)
- Bug report section in settings: pre-filled GitHub issue with system info and recent logs
- Recording started/completed toast notifications with details (time, duration, word count)
- Startup toast notification with model info

### Fixed
- Duplicate microphone devices filtered by name in device list
- Microphone list now shows only enabled devices (default host API only)
- Combo box dropdown visual artifact removed
- Speaker name no longer leaks across transcripts when switching
- Settings save confirmation only appears when changes are actually made

### Changed
- Window title simplified to "Cadence" (version removed)
- Profile field changed from "First name" to "Name" (supports full name)

## [1.0.0] - 2026-02-28

### Added
- Hybrid post-processing with text-based echo cancellation
- Two-tier echo gate: energy + text overlap detection for speaker bleed suppression
- Retroactive echo retraction: removes "you" segments matching later "them" transcriptions
- Text echo gate with 65% word overlap threshold

### Fixed
- Widened dedup time window from 8s to 15s for delayed echo
- Echo gate no longer suppresses user speech over system audio

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
- Live echo gate replaced with energy ratio approach (envelope correlation
  produced -0.4 to +0.4 in practice, never detecting echo)
- Live echo gate was never running due to silence detector reset bug
- Echo dedup switched from SequenceMatcher to word overlap for better accuracy
- Echo dedup now combines nearby system segments before overlap check
  (mic echo spanning multiple system segments was being missed)
- Echo dedup SequenceMatcher fallback for short segments where Whisper
  transcribes completely different words from mic vs system audio
- Live echo gate two-tier suppression: tier 1 catches normal bleed
  (mic_rms < 0.014), tier 2 catches loud-system bleed (mic_rms < 0.020
  when sys_rms > 0.030 and ratio < 0.65). 100% bleed suppression on
  all real test data while never suppressing user speech
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
