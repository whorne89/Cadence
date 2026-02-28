# VAD-Based Chunking + Post-Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed 5-second transcription timer with energy-based silence detection for live chunking, and add a full-audio post-processing pass after recording stops.

**Architecture:** TranscriptionWorker gets a new polling loop that checks RMS energy every 200ms instead of sleeping for N seconds. When silence is detected (500ms+), accumulated speech frames are transcribed. After recording stops, a PostProcessWorker re-transcribes the full audio with Whisper's VAD filtering to produce clean, properly-bounded segments that replace the live output.

**Tech Stack:** Python 3.12, PySide6 (QThread/signals), numpy (RMS), faster-whisper (transcription + built-in silero-VAD)

---

### Task 1: Add silence detection utility

**Files:**
- Create: `src/core/silence_detector.py`
- Create: `tests/test_silence_detector.py`

**Step 1: Write the failing tests**

```python
# tests/test_silence_detector.py
import numpy as np
import pytest
from src.core.silence_detector import SilenceDetector


def test_silence_detected_on_quiet_audio():
    """Feeding silent frames should register as silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    # Feed 600ms of silence (3 x 200ms)
    silent = np.zeros(3200, dtype=np.float32)  # 200ms at 16kHz
    for _ in range(3):
        sd.feed(silent)
    assert sd.is_silent()


def test_speech_detected_on_loud_audio():
    """Feeding loud frames should not register as silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    loud = np.random.randn(3200).astype(np.float32) * 0.5
    sd.feed(loud)
    assert not sd.is_silent()


def test_silence_after_speech():
    """Silence should only trigger after min_silence_ms of quiet frames."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    loud = np.random.randn(3200).astype(np.float32) * 0.5
    silent = np.zeros(3200, dtype=np.float32)

    # Speech then short silence (200ms) — not enough
    sd.feed(loud)
    sd.feed(silent)
    assert not sd.is_silent()

    # More silence (total 600ms) — should trigger
    sd.feed(silent)
    sd.feed(silent)
    assert sd.is_silent()


def test_reset_clears_state():
    """reset() should clear accumulated silence."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    silent = np.zeros(3200, dtype=np.float32)
    for _ in range(3):
        sd.feed(silent)
    assert sd.is_silent()
    sd.reset()
    assert not sd.is_silent()


def test_speech_resets_silence_counter():
    """A loud frame in between silence should reset the silence timer."""
    sd = SilenceDetector(silence_threshold=0.01, min_silence_ms=500, sample_rate=16000)
    silent = np.zeros(3200, dtype=np.float32)
    loud = np.random.randn(3200).astype(np.float32) * 0.5

    sd.feed(silent)
    sd.feed(silent)
    # interrupt with speech
    sd.feed(loud)
    sd.feed(silent)
    sd.feed(silent)
    # only 400ms of silence after speech — not enough
    assert not sd.is_silent()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_silence_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.silence_detector'`

**Step 3: Write minimal implementation**

```python
# src/core/silence_detector.py
"""
Energy-based silence detection for Cadence.
Tracks RMS energy of audio frames to detect speech pauses.
"""

import numpy as np


class SilenceDetector:
    """Detects silence gaps in a stream of audio frames using RMS energy."""

    def __init__(self, silence_threshold=0.01, min_silence_ms=500, sample_rate=16000):
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate
        self._silent_samples = 0
        self._has_had_speech = False

    def feed(self, audio_chunk: np.ndarray):
        """Feed an audio chunk and update silence tracking."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms < self.silence_threshold:
            self._silent_samples += len(audio_chunk)
        else:
            self._silent_samples = 0
            self._has_had_speech = True

    def is_silent(self) -> bool:
        """True if silence has lasted at least min_silence_ms."""
        min_samples = int(self.sample_rate * self.min_silence_ms / 1000)
        return self._silent_samples >= min_samples

    def reset(self):
        """Reset all state."""
        self._silent_samples = 0
        self._has_had_speech = False
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_silence_detector.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/core/silence_detector.py tests/test_silence_detector.py
git commit -m "feat: add energy-based SilenceDetector for speech pause detection"
```

---

### Task 2: Rewrite TranscriptionWorker with energy-based polling

**Files:**
- Modify: `src/main.py` (TranscriptionWorker class, lines 39-112)
- Create: `tests/test_transcription_worker.py`

**Step 1: Write the failing tests**

```python
# tests/test_transcription_worker.py
"""Tests for the energy-based TranscriptionWorker."""

import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock


def _make_worker(speech_frames=None, silent_frames=None):
    """Create a TranscriptionWorker with mocked dependencies."""
    from src.main import TranscriptionWorker

    transcriber = MagicMock()
    transcriber.transcribe_text.return_value = "hello world"

    recorder = MagicMock()
    recorder.sample_rate = 16000
    recorder._mic_frames = speech_frames or []
    recorder._system_frames = []

    worker = TranscriptionWorker(
        transcriber=transcriber,
        audio_recorder=recorder,
        silence_threshold=0.01,
        min_silence_ms=500,
        max_speech_s=30.0,
    )
    return worker, transcriber, recorder


def test_worker_has_silence_detection_params():
    """Worker should accept silence detection parameters."""
    worker, _, _ = _make_worker()
    assert worker.silence_threshold == 0.01
    assert worker.min_silence_ms == 500
    assert worker.max_speech_s == 30.0


def test_worker_emits_on_silence_after_speech():
    """Worker should transcribe when silence is detected after speech."""
    sr = 16000
    # 1 second of loud audio (speech) + 0.6s of silence
    speech = [np.random.randn(sr).astype(np.float32) * 0.5]
    silence = [np.zeros(int(sr * 0.6), dtype=np.float32)]

    worker, transcriber, recorder = _make_worker()
    recorder._mic_frames = speech + silence

    # Manually process frames (don't start the thread loop)
    segments = []
    worker.segment_ready.connect(lambda s, t, ts: segments.append((s, t, ts)))

    worker._process_source(
        recorder._mic_frames, "you", 0
    )

    # Should have transcribed the speech before the silence
    assert transcriber.transcribe_text.called
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcription_worker.py -v`
Expected: FAIL — TranscriptionWorker doesn't accept new params / `_process_source` doesn't exist

**Step 3: Rewrite TranscriptionWorker in `src/main.py`**

Replace lines 39-112 of `src/main.py` (the entire TranscriptionWorker class) with:

```python
class TranscriptionWorker(QObject):
    """
    Worker that transcribes audio using energy-based silence detection.

    Instead of a fixed timer, polls for new audio every 200ms and uses
    RMS energy to detect speech pauses. Transcribes accumulated speech
    when silence is detected (or when max_speech_s safety valve triggers).
    """

    segment_ready = Signal(str, str, float)  # speaker, text, timestamp_seconds
    finished = Signal()

    def __init__(self, transcriber, audio_recorder,
                 silence_threshold=0.01, min_silence_ms=500,
                 max_speech_s=30.0):
        super().__init__()
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.max_speech_s = max_speech_s
        self._running = False
        self._poll_interval = 0.2  # 200ms

    def run(self):
        """Main loop — polls audio and transcribes on silence boundaries."""
        self._running = True
        sr = self.audio_recorder.sample_rate

        # Track offsets into frame lists
        mic_offset = 0
        sys_offset = 0

        # Per-source silence detectors
        from core.silence_detector import SilenceDetector
        mic_detector = SilenceDetector(self.silence_threshold, self.min_silence_ms, sr)
        sys_detector = SilenceDetector(self.silence_threshold, self.min_silence_ms, sr)

        # Per-source speech buffers (frames since last transcription)
        mic_speech_start = 0
        sys_speech_start = 0

        logger.info(
            f"Transcription worker started (silence_threshold={self.silence_threshold}, "
            f"min_silence={self.min_silence_ms}ms, max_speech={self.max_speech_s}s)"
        )

        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break

            # --- Process mic audio ---
            mic_frames = self.audio_recorder._mic_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for frame in new_frames:
                    mic_detector.feed(frame)
                mic_offset = mic_len

                # Check if we should transcribe
                speech_frames = mic_frames[mic_speech_start:mic_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    mic_detector.is_silent() and speech_duration > 0.5
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    prev_samples = sum(len(f) for f in mic_frames[:mic_speech_start])
                    timestamp = prev_samples / sr
                    self._transcribe_frames(speech_frames, "you", timestamp)
                    mic_speech_start = mic_offset
                    mic_detector.reset()

            # --- Process system audio ---
            sys_frames = self.audio_recorder._system_frames
            sys_len = len(sys_frames)
            if sys_len > sys_offset:
                new_frames = sys_frames[sys_offset:sys_len]
                for frame in new_frames:
                    sys_detector.feed(frame)
                sys_offset = sys_len

                speech_frames = sys_frames[sys_speech_start:sys_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    sys_detector.is_silent() and speech_duration > 0.5
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
                    timestamp = prev_samples / sr
                    self._transcribe_frames(speech_frames, "them", timestamp)
                    sys_speech_start = sys_offset
                    sys_detector.reset()

            # --- Flush remaining audio when stopping ---
        # Transcribe any leftover speech
        mic_remaining = mic_frames[mic_speech_start:mic_offset] if mic_offset > mic_speech_start else []
        if mic_remaining:
            prev_samples = sum(len(f) for f in mic_frames[:mic_speech_start])
            self._transcribe_frames(mic_remaining, "you", prev_samples / sr)

        sys_remaining = sys_frames[sys_speech_start:sys_offset] if sys_offset > sys_speech_start else []
        if sys_remaining:
            prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
            self._transcribe_frames(sys_remaining, "them", prev_samples / sr)

        logger.info("Transcription worker stopped")
        self.finished.emit()

    def _transcribe_frames(self, frames, speaker, timestamp):
        """Concatenate frames and transcribe."""
        try:
            audio = np.concatenate(frames)
            if len(audio) > 0:
                text = self.transcriber.transcribe_text(audio)
                if text and text.strip():
                    self.segment_ready.emit(speaker, text.strip(), timestamp)
        except Exception as e:
            logger.error(f"Transcription error ({speaker}): {e}")

    def stop(self):
        self._running = False
```

**Step 4: Update `_start_transcription_worker` in CadenceApp (line ~196)**

Replace the `TranscriptionWorker(...)` instantiation to pass new params instead of `interval`:

```python
    self._transcription_worker = TranscriptionWorker(
        self.streaming_transcriber,
        self.audio_recorder,
        silence_threshold=0.01,
        min_silence_ms=500,
        max_speech_s=30.0,
    )
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_transcription_worker.py tests/test_silence_detector.py -v`
Expected: All pass

**Step 6: Run full test suite to check nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: All 35+ tests pass

**Step 7: Commit**

```bash
git add src/main.py tests/test_transcription_worker.py
git commit -m "feat: rewrite TranscriptionWorker with energy-based silence detection"
```

---

### Task 3: Add PostProcessWorker

**Files:**
- Modify: `src/main.py` (add PostProcessWorker class, modify CadenceApp.stop_recording)

**Step 1: Write the failing tests**

```python
# tests/test_postprocess_worker.py
"""Tests for the PostProcessWorker."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from PySide6.QtCore import QCoreApplication
import sys


@pytest.fixture(scope="module")
def qapp():
    """Ensure QCoreApplication exists for signal/slot tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)
    return app


def test_postprocess_worker_emits_segments(qapp):
    """PostProcessWorker should transcribe full audio and emit segments."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [
        {"text": "Hello there.", "start": 0.0, "end": 1.5},
        {"text": "How are you?", "start": 2.0, "end": 3.2},
    ]

    sr = 16000
    mic_audio = np.random.randn(sr * 5).astype(np.float32)
    sys_audio = np.random.randn(sr * 5).astype(np.float32)

    worker = PostProcessWorker(transcriber, mic_audio, sys_audio)

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    segments = results[0]
    # Should have segments from both mic and system
    assert len(segments) >= 2


def test_postprocess_worker_handles_empty_audio(qapp):
    """PostProcessWorker should handle empty audio gracefully."""
    from src.main import PostProcessWorker

    transcriber = MagicMock()
    transcriber.transcribe.return_value = []

    worker = PostProcessWorker(
        transcriber,
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )

    results = []
    worker.segments_ready.connect(lambda segs: results.append(segs))
    worker.run()

    assert len(results) == 1
    assert results[0] == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_postprocess_worker.py -v`
Expected: FAIL — `ImportError: cannot import name 'PostProcessWorker'`

**Step 3: Add PostProcessWorker to `src/main.py`**

Add this class after TranscriptionWorker (before CadenceApp):

```python
class PostProcessWorker(QObject):
    """
    Re-transcribes full audio after recording stops.

    Runs in a QThread. Takes complete mic and system audio arrays,
    transcribes each with vad_filter=True for clean segment boundaries,
    and emits the merged result.
    """

    segments_ready = Signal(list)  # list of {speaker, text, start} dicts
    progress = Signal(str)  # status message
    finished = Signal()

    def __init__(self, transcriber, mic_audio, system_audio):
        super().__init__()
        self.transcriber = transcriber
        self.mic_audio = mic_audio
        self.system_audio = system_audio

    def run(self):
        """Transcribe full audio and emit cleaned segments."""
        segments = []

        try:
            # Transcribe mic audio
            if len(self.mic_audio) > 0:
                self.progress.emit("Processing microphone audio...")
                mic_segments = self.transcriber.transcribe(self.mic_audio)
                for seg in mic_segments:
                    segments.append({
                        "speaker": "you",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Transcribe system audio
            if len(self.system_audio) > 0:
                self.progress.emit("Processing system audio...")
                sys_segments = self.transcriber.transcribe(self.system_audio)
                for seg in sys_segments:
                    segments.append({
                        "speaker": "them",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Sort by timestamp
            segments.sort(key=lambda s: s["start"])

        except Exception as e:
            logger.error(f"Post-processing error: {e}")

        self.segments_ready.emit(segments)
        self.finished.emit()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_postprocess_worker.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/main.py tests/test_postprocess_worker.py
git commit -m "feat: add PostProcessWorker for full-audio re-transcription"
```

---

### Task 4: Wire post-processing into CadenceApp.stop_recording

**Files:**
- Modify: `src/main.py` (CadenceApp class — stop_recording, new helpers)
- Modify: `src/gui/main_window.py` (add set_processing_state, replace transcript)

**Step 1: Add `set_processing_state` to MainWindow**

In `src/gui/main_window.py`, add after `set_done_state` (around line 552):

```python
    def set_processing_state(self, message="Processing..."):
        self._recording = False
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Processing...")
        self.status_label.setText(message)
        self._set_status_badge("processing")
        self._timer.stop()
```

**Step 2: Modify CadenceApp.stop_recording to launch post-processing**

Replace `stop_recording` in `src/main.py` (lines 231-265) with:

```python
    def stop_recording(self):
        """Stop recording and launch post-processing."""
        self.logger.info("Stopping recording...")

        # Stop transcription worker first
        self._stop_transcription_worker()

        # Stop audio capture — returns full audio arrays
        mic_audio, system_audio = self.audio_recorder.stop_recording()
        self._recording_duration = self.audio_recorder.get_duration()

        # Update UI to processing state
        if self.main_window is not None:
            self.main_window.set_processing_state("Cleaning up transcript...")

        # Launch post-processing in background thread
        self._start_postprocess(mic_audio, system_audio)

    def _start_postprocess(self, mic_audio, system_audio):
        """Launch PostProcessWorker in a background QThread."""
        self._postprocess_thread = QThread()
        self._postprocess_worker = PostProcessWorker(
            self.streaming_transcriber, mic_audio, system_audio
        )
        self._postprocess_worker.moveToThread(self._postprocess_thread)

        self._postprocess_thread.started.connect(self._postprocess_worker.run)
        self._postprocess_worker.progress.connect(self._on_postprocess_progress)
        self._postprocess_worker.segments_ready.connect(self._on_postprocess_done)
        self._postprocess_worker.finished.connect(self._cleanup_postprocess_thread)

        self._postprocess_thread.start()

    def _on_postprocess_progress(self, message):
        """Update UI with post-processing progress."""
        if self.main_window is not None:
            self.main_window.status_label.setText(message)

    def _on_postprocess_done(self, segments):
        """Replace live transcript with post-processed segments and save."""
        self.logger.info(f"Post-processing complete: {len(segments)} segments")

        # Replace live segments with cleaned version
        self._current_segments = segments

        # Update transcript display
        if self.main_window is not None:
            self.main_window.set_transcript(segments)
            self.main_window.set_done_state()

        # Save transcript
        model = self.config.get_streaming_model_size()
        self.session_manager.save_transcript(
            segments, duration=self._recording_duration, model=model,
            folder=self._selected_folder,
        )

        # Refresh folder/transcript lists
        self._refresh_folders()
        if self._selected_folder:
            self.on_folder_selected(self._selected_folder)

        if self.tray_icon is not None:
            self.tray_icon.set_idle_state()
            self.tray_icon.show_notification(
                "Recording Complete",
                f"Duration: {int(self._recording_duration)}s — Transcript cleaned up"
            )

    def _cleanup_postprocess_thread(self):
        """Clean up post-processing thread resources."""
        if self._postprocess_thread is not None:
            self._postprocess_thread.quit()
            self._postprocess_thread.wait(2000)
            self._postprocess_thread.deleteLater()
            self._postprocess_thread = None
        if self._postprocess_worker is not None:
            self._postprocess_worker.deleteLater()
            self._postprocess_worker = None
```

**Step 3: Initialize new instance vars in CadenceApp.__init__**

Add after line 155 (`self._selected_folder = None`):

```python
        self._postprocess_thread = None
        self._postprocess_worker = None
        self._recording_duration = 0.0
```

**Step 4: Update quit() to clean up post-processing thread**

Add before `QApplication.quit()` in the `quit` method:

```python
        # Stop post-processing if running
        if self._postprocess_thread is not None and self._postprocess_thread.isRunning():
            self._postprocess_thread.quit()
            self._postprocess_thread.wait(3000)
```

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/main.py src/gui/main_window.py
git commit -m "feat: wire post-processing into stop_recording flow"
```

---

### Task 5: Remove obsolete transcription_interval config

**Files:**
- Modify: `src/utils/config.py` (remove `transcription_interval` default and getter)
- Modify: `src/gui/settings_dialog.py` (remove interval spinner if present)
- Modify: `tests/test_config.py` (remove/update interval tests)

**Step 1: Check if settings_dialog has interval UI**

Read `src/gui/settings_dialog.py` and look for `transcription_interval` or interval spinner references. Remove any UI for the fixed interval setting since it's no longer used.

**Step 2: Update config defaults**

In `src/utils/config.py`, remove `"transcription_interval": 5.0` from DEFAULT_CONFIG and remove `get_transcription_interval` method. The silence detection params can be added later if we want them user-configurable.

**Step 3: Update tests**

Remove `test_transcription_interval_default` and `test_transcription_interval_range` from `tests/test_config.py`.

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/utils/config.py src/gui/settings_dialog.py tests/test_config.py
git commit -m "chore: remove obsolete transcription_interval config"
```

---

### Task 6: Manual smoke test

**No files changed — manual verification only.**

**Step 1: Run the app**

```bash
uv run python src/main.py
```

**Step 2: Verify live transcription**

- Start recording
- Speak a sentence, pause for ~1 second, speak another
- Verify: transcript entries appear at natural pause boundaries, not on a fixed timer
- Verify: long continuous speech gets force-transcribed at ~30s

**Step 3: Verify post-processing**

- Stop recording
- Verify: status shows "Cleaning up transcript..."
- Verify: transcript gets replaced with cleaned version
- Verify: final transcript has proper sentence boundaries

**Step 4: Verify edge cases**

- Very short recording (<3s): should work fine
- Recording with no speech (silence only): should produce empty/minimal transcript
- Quit during post-processing: should shut down cleanly

**Step 5: Commit any fixes found during smoke testing**
