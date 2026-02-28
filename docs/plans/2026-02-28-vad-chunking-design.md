# VAD-Based Chunking + Post-Processing Design

## Problem

TranscriptionWorker transcribes audio on a fixed 5-second timer, cutting mid-sentence at arbitrary boundaries. This produces fragmented transcript entries that split a single thought across two lines.

## Solution

Two-layer approach:
1. **Live**: Energy-based silence detection triggers transcription at natural speech pauses
2. **Post-processing**: Full Whisper pass on complete audio after recording stops, replacing live fragments with clean segments

## Architecture

### Live Chunking (replaces fixed timer)

```
Audio frames → RMS energy check (every 200ms)
                    ↓
              Track speech state (speaking / silence)
                    ↓
              Silence ≥ 500ms? → Transcribe accumulated speech buffer
                    ↓
              No silence for 30s? → Force-transcribe (safety valve)
```

**Why RMS over silero-VAD for live?**
- RMS: 0.01ms per check, works incrementally, trivially simple
- Silero-VAD: batch-only, needs full buffer, overkill for "is it quiet?"
- Silero-VAD runs inside Whisper's `vad_filter=True` anyway during transcription

**Parameters (configurable with sensible defaults):**
- `silence_threshold`: RMS level below which audio counts as silence (default: 0.01)
- `min_silence_ms`: Silence duration before triggering transcription (default: 500ms)
- `max_speech_s`: Safety valve — force transcription after this long without silence (default: 30s)
- `energy_check_interval_ms`: How often to check RMS (default: 200ms)

### Post-Processing (new step on recording stop)

```
Recording stops → Concatenate all mic frames
                → Concatenate all system frames
                → Run transcriber.transcribe() on each (vad_filter=True, full audio)
                → Replace live transcript entries with cleaned segments
                → Save cleaned version
```

**Timing benchmarks (base model, CPU, int8):**
- Whisper processes at ~8-20x realtime
- 30-min meeting (~50% speech) → ~1-2 min reprocessing
- VAD skips silence automatically, so only speech gets transcribed

**UX during reprocessing:**
- Show a brief status indicator ("Cleaning up transcript...")
- Live transcript stays visible until reprocessing completes
- Swap in the cleaned version when done

## Files Changed

| File | Change |
|------|--------|
| `src/main.py` | Rewrite TranscriptionWorker: replace sleep-timer with energy-based polling loop. Add post-processing step in CadenceApp.stop_recording flow. |
| `src/gui/main_window.py` | Add signal/slot to replace transcript content after reprocessing. Add "cleaning up" status indicator. |
| `src/core/transcriber.py` | No changes — existing `transcribe()` with `vad_filter=True` handles post-processing. |
| `src/core/audio_recorder.py` | No changes — frame accumulation works as-is. |

## What Does NOT Change

- Audio recording pipeline (mic + WASAPI loopback)
- Transcriber model/API
- Session manager / save format
- UI layout / theme
- No new dependencies

## Edge Cases

- **Noisy environment**: RMS threshold may need tuning. Expose as a setting. The post-processing pass uses silero-VAD (via Whisper) which handles this well.
- **Very long speech without pause**: Safety valve at 30s prevents indefinite buffering. The chunk will be large but complete.
- **Short recordings (<5s)**: Energy detection still works; post-processing is near-instant.
- **Recording stopped mid-speech**: Final audio gets captured and included in post-processing.

## Testing Strategy

- Unit test: energy-based silence detector with synthetic audio
- Unit test: TranscriptionWorker state machine (mock audio + transcriber)
- Integration: verify post-processing replaces live segments correctly
