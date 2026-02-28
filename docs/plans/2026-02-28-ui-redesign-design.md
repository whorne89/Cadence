# Cadence UI Redesign & Storage Overhaul

## Summary

Redesign the main window from a single-panel transcript viewer to a 3-panel layout with folder navigation, transcript list, and transcript viewer. Switch storage from JSON + WAV to plain `.txt` files. Remove audio reprocessing. Add clear/copy buttons and configurable transcription interval.

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Cadence                                    00:00:00  Ready │
├────────┬──────────┬─────────────────────────────────────────┤
│Folders │Transcripts│                                        │
│        │          │                                         │
│📁 Today │ Meeting1 │  [00:05] You: Hello everyone           │
│📁 2/27  │ Meeting2 │  [00:12] Them: Hey, let's get started  │
│📁 2/26  │          │  [00:18] You: So the agenda today...   │
│📁 Team  │          │                                         │
│         │          │                                         │
│ [+ New] │          │                                         │
│         │          │                                         │
├────────┴──────────┴─────────────────────────────────────────┤
│ [Start Recording]              [Clear] [Copy]   124 words   │
└─────────────────────────────────────────────────────────────┘
```

### Left Panel — Folder Tree (~150px)
- QTreeWidget or QListWidget showing folders
- Auto-created date folders (YYYY-MM-DD) when a recording is saved
- User can create custom folders via "+" button
- Right-click context menu: Rename, Delete
- Selecting a folder populates the middle panel

### Middle Panel — Transcript List (~150px)
- QListWidget showing transcripts in the selected folder
- Displays transcript name and time
- Click to load transcript in the viewer
- Right-click context menu: Rename, Delete, Move to Folder
- New transcripts auto-named "Recording HH-MM" (can be renamed)

### Right Panel — Transcript Viewer (fills remaining space)
- QTextEdit (read-only), same as current
- Shows speaker-attributed transcript with timestamps
- [MM:SS] You: text / [MM:SS] Them: text

### Bottom Bar
- Start/Stop Recording button (existing)
- Clear Transcript button (new) — clears viewer
- Copy Transcript button (new) — copies plain text to clipboard
- Word count label (existing)

### Header
- Status label + timer (existing, unchanged)

## Storage

### Format: Plain Text
```
Cadence Transcript
Date: 2026-02-28 14:30
Duration: 00:12:45
Model: base

---

[00:05] You: Hello everyone
[00:12] Them: Hey, let's get started
```

### Folder Structure on Disk
```
.cadence/sessions/
├── 2026-02-28/
│   ├── Recording 14-30.txt
│   └── Team standup.txt
├── 2026-02-27/
│   └── Sprint planning.txt
└── Project Alpha/          ← user-created folder
    └── Kickoff call.txt
```

- Mirrors UI folder tree exactly
- Date folders auto-created on recording save
- Custom folders created via UI

### Removals
- Remove WAV audio saving (no more save_audio, _mic_audio, _system_audio)
- Remove ReprocessWorker class
- Remove reprocess_transcriber instance
- Remove reprocess button and related UI states
- Remove reprocess-related session fields (reprocessed, reprocess_model)

## Settings Changes

### New
- **Transcription interval**: Slider/SpinBox, range 2-8 seconds, default 5
  - Controls TranscriptionWorker polling interval
  - Shorter = more responsive but higher CPU
  - Longer = less CPU, better per-chunk accuracy

### Remove
- Reprocess model size selector

### Keep
- Streaming model size selector
- Microphone device selector
- System audio device selector

## Session Manager Changes

- Save as `.txt` instead of `.json`
- Parse `.txt` files back into segments for display
- Support folder creation/deletion/renaming on disk
- Support transcript renaming/moving between folders
- `list_folders()` — enumerate subdirectories
- `list_transcripts(folder)` — list .txt files in a folder
- `create_folder(name)` — mkdir
- `rename_folder(old, new)` — rename directory
- `delete_folder(name)` — rmdir (with confirmation)
- `rename_transcript(folder, old_name, new_name)` — rename file
- `move_transcript(src_folder, name, dest_folder)` — move file
- `delete_transcript(folder, name)` — delete file

## What Stays the Same

- Dual-channel audio capture (mic + WASAPI loopback)
- You/Them speaker attribution
- TranscriptionWorker polling architecture (interval now configurable)
- System tray behavior
- Streaming model and transcription pipeline
- Config system (just different keys)

## Future (Not in this iteration)
- LLM-based post-processing (summarize, clean up, custom prompts)
- Multi-speaker diarization (distinguish multiple remote speakers)
