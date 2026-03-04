"""
Cadence - Meeting Transcription
Main entry point that orchestrates all components.
"""

import os
import sys

# PyInstaller windowed mode (console=False) sets sys.stdout/stderr to None.
# Libraries like huggingface_hub use tqdm which calls sys.stderr.write(),
# crashing with "NoneType has no attribute 'write'". Redirect to devnull.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import ctypes
import logging
import time

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, QThread, QTimer, Qt
import numpy as np


def set_windows_app_id():
    """Set Windows AppUserModelID for proper taskbar/tray display."""
    try:
        app_id = "Cadence.MeetingTranscription.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Not on Windows or API not available


from core.audio_recorder import AudioRecorder
from core.transcriber import Transcriber
from core.session_manager import SessionManager
from gui.system_tray import SystemTrayIcon
from gui.main_window import MainWindow
from gui.settings_dialog import SettingsDialog
from gui.theme import apply_theme
from core.echo_diagnostics import EchoDiagnostics
from core.sound_effects import SoundEffects
from core.updater import UpdateChecker, UpdateWorker
from gui.update_toast import UpdateToast
from utils.config import ConfigManager
from utils.logger import setup_logger


logger = logging.getLogger("Cadence")


class TranscriptionWorker(QObject):
    """
    Worker that transcribes audio using energy-based silence detection.

    Instead of a fixed timer, polls for new audio every 200ms and uses
    RMS energy to detect speech pauses. Transcribes accumulated speech
    when silence is detected (or when max_speech_s safety valve triggers).
    """

    # Fraction of sys_rms subtracted from mic_rms to compensate for speaker
    # bleed into the microphone. Derived from session data: bleed ratio
    # (mic_rms/sys_rms when only remote speaker talks) is 1.0-1.4.
    # 0.8 under-compensates to avoid suppressing real user speech.
    BLEED_FACTOR = 0.8

    segment_ready = Signal(str, str, float)  # speaker, text, timestamp_seconds
    segment_retracted = Signal(float)  # timestamp of "you" segment to remove
    finished = Signal()

    def __init__(self, transcriber, audio_recorder,
                 mic_silence_threshold=0.005, sys_silence_threshold=0.01,
                 mic_min_silence_ms=200, sys_min_silence_ms=500,
                 mic_min_speech_s=0.3, sys_min_speech_s=0.5,
                 max_speech_s=30.0, echo_diagnostics=None,
                 language=None, echo_gate_logging=False):
        super().__init__()
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.mic_silence_threshold = mic_silence_threshold
        self.sys_silence_threshold = sys_silence_threshold
        self.mic_min_silence_ms = mic_min_silence_ms
        self.sys_min_silence_ms = sys_min_silence_ms
        self.mic_min_speech_s = mic_min_speech_s
        self.sys_min_speech_s = sys_min_speech_s
        self.max_speech_s = max_speech_s
        self.echo_diagnostics = echo_diagnostics
        self.language = language
        self.echo_gate_logging = echo_gate_logging
        self._running = False
        self._poll_interval = 0.2  # 200ms
        # Buffers for text-based echo filtering (both directions)
        self._recent_them = []  # list of (timestamp, text)
        self._recent_you = []   # list of (timestamp, text) — for retroactive filtering

    def run(self):
        """Main loop — polls audio and transcribes on silence boundaries."""
        self._running = True
        sr = self.audio_recorder.sample_rate

        # Track offsets into frame lists
        mic_offset = 0
        sys_offset = 0

        # Per-source silence detectors (mic is more sensitive to capture short utterances)
        from core.silence_detector import SilenceDetector
        from core.echo_gate import get_audio_for_sample_range
        mic_detector = SilenceDetector(self.mic_silence_threshold, self.mic_min_silence_ms, sr)
        sys_detector = SilenceDetector(self.sys_silence_threshold, self.sys_min_silence_ms, sr)

        # Per-source speech buffer start indices
        mic_speech_start = 0
        sys_speech_start = 0

        # Initialize frame references before loop (avoids NameError if loop never runs)
        mic_frames = self.audio_recorder._mic_frames
        sys_frames = self.audio_recorder._system_frames

        logger.info(
            f"Transcription worker started ("
            f"mic: threshold={self.mic_silence_threshold}, silence={self.mic_min_silence_ms}ms, "
            f"min_speech={self.mic_min_speech_s}s | "
            f"sys: threshold={self.sys_silence_threshold}, silence={self.sys_min_silence_ms}ms, "
            f"min_speech={self.sys_min_speech_s}s | "
            f"max_speech={self.max_speech_s}s, language={self.language})"
        )

        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break

            # --- Process mic audio ---
            mic_frames = self.audio_recorder._mic_frames
            sys_frames = self.audio_recorder._system_frames
            mic_len = len(mic_frames)
            if mic_len > mic_offset:
                new_frames = mic_frames[mic_offset:mic_len]
                for i, frame in enumerate(new_frames):
                    # Bleed-compensated silence detection:
                    # Subtract estimated speaker bleed from mic RMS so
                    # silence detector isn't fooled by system audio leaking
                    # into the mic.
                    sys_idx = mic_offset + i
                    if sys_idx < len(sys_frames):
                        sys_frame = sys_frames[sys_idx]
                        mic_rms = float(np.sqrt(np.mean(
                            frame.astype(np.float64) ** 2)))
                        sys_rms = float(np.sqrt(np.mean(
                            sys_frame.astype(np.float64) ** 2)))
                        comp_rms = max(mic_rms - self.BLEED_FACTOR * sys_rms, 0.0)
                        mic_detector.feed_rms(comp_rms, len(frame))
                    else:
                        mic_detector.feed(frame)
                mic_offset = mic_len

                speech_frames = mic_frames[mic_speech_start:mic_offset]
                speech_samples = sum(len(f) for f in speech_frames)
                speech_duration = speech_samples / sr

                should_transcribe = (
                    mic_detector.is_silent() and mic_detector._has_had_speech
                    and speech_duration > self.mic_min_speech_s
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    mic_start_sample = sum(len(f) for f in mic_frames[:mic_speech_start])
                    timestamp = mic_start_sample / sr

                    # Echo gate: suppress mic if it's just speaker bleed
                    # Speaker bleed is much quieter than direct speech into mic.
                    # If system audio is active and mic isn't significantly louder,
                    # it's just bleed — suppress it.
                    echo_detected = False
                    mic_audio = np.concatenate(speech_frames)
                    mic_end_sample = mic_start_sample + len(mic_audio)
                    sys_audio = get_audio_for_sample_range(
                        self.audio_recorder._system_frames,
                        mic_start_sample, mic_end_sample,
                    )

                    # AEC: spectral subtraction removes echo from mic audio
                    # before any further processing or transcription.
                    aec_applied = False
                    raw_mic_audio = None
                    if len(sys_audio) > int(sr * 0.2):
                        from core.echo_cancellation import spectral_subtract_echo
                        raw_mic_audio = mic_audio.copy()
                        mic_audio = spectral_subtract_echo(
                            mic_audio, sys_audio, sr=sr,
                        )
                        aec_applied = True

                    if len(sys_audio) > 0:
                        sys_rms = float(np.sqrt(np.mean(sys_audio.astype(np.float64) ** 2)))
                        mic_rms = float(np.sqrt(np.mean(mic_audio.astype(np.float64) ** 2)))
                        if sys_rms > 0.005:
                            # Two-tier echo gate:
                            # 1) Normal bleed: ratio < 1.5 and mic_rms < 0.014
                            #    catches ~95% of bleed (mic_rms stays below 0.013)
                            # 2) Loud-system bleed: when sys_rms > 0.030, louder
                            #    system audio produces proportionally louder bleed
                            #    (mic_rms 0.014-0.020). Safe to raise the floor
                            #    when ratio is low (< 0.65), confirming bleed pattern.
                            ratio = mic_rms / sys_rms if sys_rms > 0 else float('inf')
                            echo_detected = (
                                (ratio < 1.5 and mic_rms < 0.014) or
                                (ratio < 0.65 and mic_rms < 0.020 and sys_rms > 0.030)
                            )
                            if self.echo_gate_logging:
                                logger.info(
                                    f"Echo gate at {timestamp:.1f}s: "
                                    f"mic_rms={mic_rms:.4f}, sys_rms={sys_rms:.4f}, "
                                    f"ratio={ratio:.2f}, suppressed={echo_detected}"
                                )
                            if self.echo_diagnostics:
                                self.echo_diagnostics.record_chunk(
                                    mic_audio, sys_audio,
                                    mic_rms, sys_rms, ratio,
                                    echo_detected, timestamp,
                                    raw_mic_audio=raw_mic_audio,
                                )

                    # Tier 3: audio envelope correlation
                    # Catches echo that energy gate misses when Whisper
                    # transcribes different words per channel (text match fails).
                    if not echo_detected and len(sys_audio) > 0:
                        from core.echo_gate import is_echo
                        audio_is_echo, correlation = is_echo(
                            mic_audio, sys_audio, threshold=0.7, detail=True
                        )
                        if audio_is_echo and mic_rms < 0.020:
                            echo_detected = True
                            if self.echo_gate_logging:
                                logger.info(
                                    f"Echo gate (envelope) at {timestamp:.1f}s: "
                                    f"correlation={correlation:.2f}, "
                                    f"mic_rms={mic_rms:.4f}, suppressed=True"
                                )

                    if echo_detected:
                        logger.info(f"Echo suppressed at {timestamp:.1f}s (energy gate)")
                    else:
                        # Energy gate passed — transcribe, then text-compare
                        # Use AEC-cleaned audio if available
                        text = self._transcribe_audio(
                            speech_frames,
                            audio=mic_audio if aec_applied else None,
                        )
                        if text:
                            is_echo, recovered = self._is_text_echo(text, timestamp)
                            if is_echo and recovered is None:
                                logger.info(
                                    f"Echo suppressed at {timestamp:.1f}s "
                                    f"(text match with 'them')"
                                )
                            else:
                                emit_text = recovered if is_echo else text
                                self._recent_you.append((timestamp, emit_text))
                                # Keep only last 90 seconds of "you" text
                                cutoff = timestamp - 90.0
                                self._recent_you = [
                                    (t, tx) for t, tx in self._recent_you if t > cutoff
                                ]
                                self.segment_ready.emit("you", emit_text, timestamp)

                    mic_speech_start = mic_offset
                    mic_detector.reset()
                elif mic_detector.is_silent() and not mic_detector._has_had_speech:
                    # Pure silence — skip ahead without transcribing
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
                    sys_detector.is_silent() and sys_detector._has_had_speech
                    and speech_duration > self.sys_min_speech_s
                ) or (
                    speech_duration >= self.max_speech_s
                )

                if should_transcribe:
                    prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
                    timestamp = prev_samples / sr
                    text = self._transcribe_audio(speech_frames)
                    if text:
                        self._recent_them.append((timestamp, text))
                        # Keep only last 90 seconds of "them" text
                        cutoff = timestamp - 90.0
                        self._recent_them = [
                            (t, tx) for t, tx in self._recent_them if t > cutoff
                        ]
                        self.segment_ready.emit("them", text, timestamp)
                        # Retroactive check: retract "you" segments that match
                        self._retract_echo_you(text, timestamp)
                    sys_speech_start = sys_offset
                    sys_detector.reset()
                elif sys_detector.is_silent() and not sys_detector._has_had_speech:
                    # Pure silence — skip ahead without transcribing
                    sys_speech_start = sys_offset
                    sys_detector.reset()

        # --- Flush remaining audio when stopping ---
        # Flush system FIRST so _recent_them is populated for mic text gate
        sys_remaining = sys_frames[sys_speech_start:sys_offset] if sys_offset > sys_speech_start else []
        if sys_remaining:
            prev_samples = sum(len(f) for f in sys_frames[:sys_speech_start])
            timestamp = prev_samples / sr
            text = self._transcribe_audio(sys_remaining)
            if text:
                self._recent_them.append((timestamp, text))
                self.segment_ready.emit("them", text, timestamp)
                self._retract_echo_you(text, timestamp)

        # Then flush mic with text echo gate
        mic_remaining = mic_frames[mic_speech_start:mic_offset] if mic_offset > mic_speech_start else []
        if mic_remaining:
            prev_samples = sum(len(f) for f in mic_frames[:mic_speech_start])
            timestamp = prev_samples / sr
            text = self._transcribe_audio(mic_remaining)
            if text:
                is_echo, recovered = self._is_text_echo(text, timestamp)
                if is_echo and recovered is None:
                    logger.info(f"Echo suppressed at {timestamp:.1f}s (text match, flush)")
                else:
                    emit_text = recovered if is_echo else text
                    self.segment_ready.emit("you", emit_text, timestamp)

        logger.info("Transcription worker stopped")
        self.finished.emit()

    def _transcribe_audio(self, frames, audio=None):
        """Concatenate frames and transcribe to text. Returns None on failure.

        If *audio* is provided (pre-processed numpy array), use it directly
        instead of concatenating frames.
        """
        try:
            if audio is None:
                audio = np.concatenate(frames)
            if len(audio) > 0:
                text = self.transcriber.transcribe_text(audio, language=self.language)
                if text and text.strip():
                    return text.strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return None

    def _is_text_echo(self, mic_text, timestamp, time_window=30.0, overlap_threshold=0.65):
        """
        Check if mic text matches recent 'them' transcriptions (text-level echo gate).

        Returns:
            (is_echo, recovered_text_or_None): If echo detected but clause recovery
            succeeds, returns (True, recovered_text). If full echo, returns (True, None).
            If not echo, returns (False, None).
        """
        from core.echo_gate import _word_overlap, _extract_unique_clauses
        for them_time, them_text in self._recent_them:
            if abs(timestamp - them_time) <= time_window:
                overlap = _word_overlap(mic_text, them_text)
                if overlap >= overlap_threshold:
                    # Try clause-level recovery
                    combined_them = " ".join(
                        tx for t, tx in self._recent_them
                        if abs(timestamp - t) <= time_window
                    )
                    recovered = _extract_unique_clauses(mic_text, combined_them)
                    if recovered is not None:
                        logger.info(
                            f"Text echo partial: {overlap:.0%} overlap with 'them' "
                            f"at {them_time:.1f}s, recovered: '{recovered}'"
                        )
                        return (True, recovered)
                    logger.info(
                        f"Text echo: {overlap:.0%} word overlap with 'them' at {them_time:.1f}s"
                    )
                    return (True, None)
        return (False, None)

    def _retract_echo_you(self, them_text, them_timestamp, time_window=30.0, overlap_threshold=0.65):
        """Retroactively retract 'you' segments that match newly transcribed 'them' text."""
        from core.echo_gate import _word_overlap, _extract_unique_clauses
        surviving = []
        for you_time, you_text in self._recent_you:
            if abs(you_time - them_timestamp) <= time_window:
                overlap = _word_overlap(you_text, them_text)
                if overlap >= overlap_threshold:
                    # Try clause-level recovery before full retraction
                    recovered = _extract_unique_clauses(you_text, them_text)
                    if recovered is not None:
                        logger.info(
                            f"Retroactive echo partial: 'you' at {you_time:.1f}s "
                            f"recovered: '{recovered}'"
                        )
                        # Retract old segment, emit recovered text
                        self.segment_retracted.emit(you_time)
                        self.segment_ready.emit("you", recovered, you_time)
                        surviving.append((you_time, recovered))
                    else:
                        logger.info(
                            f"Retroactive echo retraction: 'you' at {you_time:.1f}s "
                            f"matches 'them' at {them_timestamp:.1f}s ({overlap:.0%} overlap)"
                        )
                        self.segment_retracted.emit(you_time)
                    continue
            surviving.append((you_time, you_text))
        self._recent_you = surviving

    def _transcribe_frames(self, frames, speaker, timestamp):
        """Concatenate frames and transcribe (used for flush on stop)."""
        text = self._transcribe_audio(frames)
        if text:
            if speaker == "them":
                self._recent_them.append((timestamp, text))
            self.segment_ready.emit(speaker, text, timestamp)

    def stop(self):
        self._running = False


def _filter_hallucinations(segments):
    """
    Remove segments that are likely Whisper hallucinations.

    Detects:
    - Non-English text (high ratio of non-ASCII or accented characters)
    - ASCII foreign language text (low English vocabulary coverage)
    - Filler-only segments (just "um", "the", "so", etc.)
    """
    import re
    FILLER_WORDS = {"um", "uh", "the", "a", "so", "and", "but", "or", "like"}

    # Top ~150 most common English words — covers ~80% of typical speech.
    # Used to detect ASCII-alphabet foreign languages (Dutch, German, etc.)
    # that bypass the non-ASCII character filter.
    COMMON_ENGLISH = {
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
        "the", "a", "an", "this", "that", "these", "those",
        "is", "am", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must",
        "not", "no", "yes", "yeah", "ok", "okay",
        "and", "or", "but", "if", "so", "because", "when", "while", "then",
        "than", "that", "which", "who", "what", "where", "how", "why",
        "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
        "up", "out", "off", "over", "into", "through", "between", "after",
        "before", "during", "around", "down",
        "all", "each", "every", "both", "some", "any", "many", "much", "more",
        "most", "other", "another", "such", "own",
        "just", "also", "very", "really", "actually", "right", "well", "now",
        "here", "there", "still", "already", "even", "only", "too",
        "go", "going", "get", "got", "make", "take", "come", "see", "know",
        "think", "say", "said", "tell", "give", "want", "need", "use", "try",
        "look", "like", "good", "new", "first", "last", "long", "great",
        "little", "big", "old", "next", "back", "way", "time", "thing",
        "people", "work", "day", "part", "let", "put",
    }

    filtered = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        # Check for text with no alphabetic content (e.g., "...", "!?!?")
        letters = re.findall(r'[a-zA-Z\u00C0-\u024F]', text)
        if not letters:
            logger.info(f"Hallucination filtered (no letters): '{text}'")
            continue

        # Check for non-English: if >30% of letters are non-ASCII, likely hallucination
        ascii_letters = sum(1 for c in letters if ord(c) < 128)
        if ascii_letters / len(letters) < 0.7:
            logger.info(f"Hallucination filtered: '{text}'")
            continue

        # Check for filler-only: strip punctuation, check if all words are fillers
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        if clean and len(clean) <= 2 and all(w in FILLER_WORDS for w in clean):
            logger.info(f"Filler filtered: '{text}'")
            continue

        # Check for ASCII foreign language: if segment has 4+ words and zero
        # match common English vocabulary, it's almost certainly not English.
        # Real English speech always has at least some function words (the,
        # is, for, with, etc.) even in domain-heavy sentences. Foreign
        # language text (Dutch, German, etc.) scores 0 matches.
        if len(clean) >= 4:
            english_count = sum(1 for w in clean if w in COMMON_ENGLISH)
            if english_count == 0:
                logger.info(f"Hallucination filtered (no English words in "
                            f"{len(clean)} words): '{text}'")
                continue

        filtered.append(seg)
    return filtered


class PostProcessWorker(QObject):
    """
    Post-processes audio after recording stops.

    Keeps live "you" segments (already echo-gate filtered) and only
    re-transcribes system audio for cleaner "them" segments. This avoids
    re-introducing speaker bleed that the echo gate already rejected.
    """

    segments_ready = Signal(list)  # list of {speaker, text, start} dicts
    progress = Signal(str)  # status message
    finished = Signal()

    def __init__(self, transcriber, live_segments, system_audio, language=None):
        super().__init__()
        self.transcriber = transcriber
        self.live_segments = live_segments
        self.system_audio = system_audio
        self.language = language

    def run(self):
        """Merge live 'you' segments with re-transcribed system audio."""
        segments = []

        try:
            # Keep live "you" segments (already echo-gate filtered)
            for seg in self.live_segments:
                if seg.get("speaker") == "you":
                    segments.append({
                        "speaker": "you",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Re-transcribe system audio for cleaner "them" segments
            if len(self.system_audio) > 0:
                self.progress.emit("Processing system audio...")
                def sys_progress(p):
                    pct = int(p * 100)
                    self.progress.emit(f"Processing system audio... {pct}%")
                sys_segments = self.transcriber.transcribe(
                    self.system_audio, language=self.language,
                    progress_callback=sys_progress,
                )
                for seg in sys_segments:
                    segments.append({
                        "speaker": "them",
                        "text": seg["text"],
                        "start": seg["start"],
                    })

            # Sort by timestamp, then remove echo duplicates (safety net)
            segments.sort(key=lambda s: s["start"])
            from core.echo_gate import deduplicate_segments, merge_segments
            segments = deduplicate_segments(segments)

            # Filter hallucinations (non-English output, filler-only segments)
            segments = _filter_hallucinations(segments)

            # Merge consecutive same-speaker segments for readability
            segments = merge_segments(segments)

        except Exception as e:
            logger.error(f"Post-processing error: {e}")

        self.segments_ready.emit(segments)
        self.finished.emit()


class CadenceApp(QObject):
    """Main application controller that wires all components together."""

    def __init__(self):
        super().__init__()

        # Initialize logger
        self.logger = setup_logger()
        self.logger.info("Starting Cadence...")

        # Configuration
        self.config = ConfigManager()

        # Session management
        self.session_manager = SessionManager()

        # Audio recorder — no callback, just records
        self.audio_recorder = AudioRecorder()

        # Apply saved audio device settings
        mic_device = self.config.get_mic_device()
        if mic_device is not None:
            self.audio_recorder.set_mic_device(mic_device)
        system_device = self.config.get_system_device()
        if system_device is not None:
            self.audio_recorder.set_system_device(system_device)

        # Streaming transcriber (lightweight model for real-time)
        streaming_model_size = self.config.get_streaming_model_size()
        self.streaming_transcriber = Transcriber(model_size=streaming_model_size)

        # Clean up any partial model downloads from previous crashes
        cleaned = self.streaming_transcriber.clean_all_partial_downloads()
        if cleaned:
            self.logger.info(f"Cleaned partial model downloads: {cleaned}")

        # GUI components (set after creation in main())
        self.tray_icon = None
        self.main_window = None

        # Transcription worker thread state
        self._transcription_thread = None
        self._transcription_worker = None

        # Current recording segments
        self._current_segments = []
        self._selected_folder = None

        self._postprocess_thread = None
        self._postprocess_worker = None
        self._recording_duration = 0.0
        self._pending_system_audio = None

        # Echo diagnostics (developer debug tool)
        self.echo_diagnostics = EchoDiagnostics(
            enabled=self.config.is_echo_debug_enabled()
        )

        # Sound effects
        self.sound_effects = SoundEffects()

        # Auto-update checker (runs 8s after startup to avoid blocking)
        self._update_thread = None
        self._update_worker = None
        self._update_toast = None
        self._update_info = None
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._check_for_updates)
        self._update_timer.start(8000)

        self.logger.info("Application initialized")

    def _on_segment(self, speaker, text, timestamp):
        """
        Handle a transcript segment from the transcription worker.
        Called on the main thread via Qt signal (thread-safe).
        """
        segment = {"speaker": speaker, "text": text, "start": timestamp}
        self._current_segments.append(segment)
        if self.main_window is not None:
            self.main_window.append_segment(speaker, text, timestamp)

    def _on_segment_retracted(self, timestamp):
        """
        Remove a 'you' segment that was retroactively identified as echo.
        Called when system audio transcription reveals a matching 'you' segment.
        """
        self.logger.info(f"Retracting 'you' segment at {timestamp:.1f}s (echo)")
        self._current_segments = [
            s for s in self._current_segments
            if not (s["speaker"] == "you" and s["start"] == timestamp)
        ]
        if self.main_window is not None:
            self.main_window.set_transcript(self._current_segments)

    def start_recording(self):
        """Start a new recording session."""
        self.logger.info("Starting recording...")

        # Reset segments for new recording
        self._current_segments = []

        # Start audio capture
        self.audio_recorder.start_recording()

        # Start echo diagnostics session if enabled
        self.echo_diagnostics.start_session()

        # Start transcription worker in a QThread
        self._start_transcription_worker()

        # Update UI
        if self.main_window is not None:
            self.main_window.set_recording_state()
        if self.tray_icon is not None:
            self.tray_icon.set_recording_state()
            from datetime import datetime
            start_time = datetime.now().strftime("%I:%M %p").lstrip("0")
            self.tray_icon.show_notification(
                "Recording", "Recording started",
                details=f"Started at {start_time}",
            )

        # Play start chime
        if self.config.is_sound_effects_enabled():
            self.sound_effects.play_start_tone()

    def _start_transcription_worker(self):
        """Launch the transcription worker in a background QThread."""
        # Clean up any previous worker
        self._stop_transcription_worker()

        self._transcription_thread = QThread()
        language = self.config.get("whisper", "language", default="en")
        echo_gate_logging = self.config.is_echo_gate_logging_enabled()
        self._transcription_worker = TranscriptionWorker(
            self.streaming_transcriber,
            self.audio_recorder,
            mic_silence_threshold=0.005,
            sys_silence_threshold=0.01,
            mic_min_silence_ms=200,
            sys_min_silence_ms=500,
            mic_min_speech_s=0.3,
            sys_min_speech_s=0.5,
            max_speech_s=30.0,
            echo_diagnostics=self.echo_diagnostics,
            language=language,
            echo_gate_logging=echo_gate_logging,
        )
        self._transcription_worker.moveToThread(self._transcription_thread)

        # Connect signals
        self._transcription_thread.started.connect(self._transcription_worker.run)
        self._transcription_worker.segment_ready.connect(self._on_segment)
        self._transcription_worker.segment_retracted.connect(self._on_segment_retracted)
        self._transcription_worker.finished.connect(
            self._on_transcription_finished, Qt.ConnectionType.QueuedConnection
        )

        self._transcription_thread.start()

    def _stop_transcription_worker(self):
        """Signal the transcription worker to stop."""
        if self._transcription_worker is not None:
            self._transcription_worker.stop()
        if self._transcription_thread is not None and self._transcription_thread.isRunning():
            self._transcription_thread.quit()
            self._transcription_thread.wait(3000)

    def _cleanup_transcription_thread(self):
        """Clean up transcription thread resources."""
        if self._transcription_thread is not None:
            self._transcription_thread.quit()
            self._transcription_thread.wait(2000)
            self._transcription_thread.deleteLater()
            self._transcription_thread = None
        if self._transcription_worker is not None:
            self._transcription_worker.deleteLater()
            self._transcription_worker = None

    def _on_transcription_finished(self):
        """Called when the transcription worker finishes (including flushing remaining audio)."""
        self._cleanup_transcription_thread()
        # If we're stopping a recording, start post-processing now
        if self._pending_system_audio is not None:
            self._begin_postprocess()

    def _begin_postprocess(self):
        """Start post-processing after the transcription worker has fully stopped."""
        system_audio = self._pending_system_audio
        self._pending_system_audio = None

        self.echo_diagnostics.save_live_transcript(self._current_segments)

        if self.main_window is not None:
            self.main_window.set_processing_state("Cleaning up transcript...")

        self._start_postprocess(None, system_audio)

    def stop_recording(self):
        """Stop recording and launch post-processing after worker finishes."""
        self.logger.info("Stopping recording...")

        # Play stop chime
        if self.config.is_sound_effects_enabled():
            self.sound_effects.play_stop_tone()

        # Signal transcription worker to stop (non-blocking — let it flush)
        if self._transcription_worker is not None:
            self._transcription_worker.stop()

        # Stop audio capture — returns full audio arrays
        _, system_audio = self.audio_recorder.stop_recording()
        self._recording_duration = self.audio_recorder.get_duration()

        # Store system audio; post-processing starts when worker emits finished
        self._pending_system_audio = system_audio

        # Update UI
        if self.main_window is not None:
            self.main_window.set_processing_state("Finishing transcription...")

        # If no worker running, start post-processing immediately
        if self._transcription_thread is None or not self._transcription_thread.isRunning():
            self._begin_postprocess()

    def _start_postprocess(self, mic_audio, system_audio):
        """Launch PostProcessWorker in a background QThread."""
        self._postprocess_thread = QThread()
        language = self.config.get("whisper", "language", default="en")
        self._postprocess_worker = PostProcessWorker(
            self.streaming_transcriber, list(self._current_segments), system_audio,
            language=language,
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

        # If post-processing failed (empty result but we had live segments), keep live
        if not segments and self._current_segments:
            self.logger.warning("Post-processing returned empty; keeping live transcript")
            segments = self._current_segments

        # Save post-processed transcript and finish diagnostics session
        self.echo_diagnostics.save_postprocessed_transcript(segments)
        self.echo_diagnostics.finish_session()

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
            dur = int(self._recording_duration)
            dur_m, dur_s = dur // 60, dur % 60
            dur_str = f"{dur_m}m {dur_s}s" if dur_m > 0 else f"{dur_s}s"
            word_count = sum(len(s["text"].split()) for s in segments)
            seg_count = len(segments)
            from datetime import datetime
            end_time = datetime.now().strftime("%I:%M %p").lstrip("0")
            self.tray_icon.show_notification(
                "Transcription Complete",
                "Recording saved",
                details=(
                    f"Duration: {dur_str}\n"
                    f"Segments: {seg_count} | Words: {word_count:,}\n"
                    f"Finished at {end_time}"
                ),
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

    def show_settings(self):
        """Open the settings dialog."""
        self.logger.info("Opening settings dialog")
        try:
            dialog = SettingsDialog(
                self.config, self.audio_recorder,
                session_manager=self.session_manager,
                transcriber=self.streaming_transcriber,
                parent=self.main_window,
            )
            dialog.settings_changed.connect(self._on_settings_changed)
            dialog.check_for_updates.connect(self.check_for_updates_manual)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"Failed to show settings dialog: {e}")
            if self.tray_icon is not None:
                self.tray_icon.show_error(f"Failed to open settings: {e}")

    def _on_settings_changed(self):
        """Apply updated settings from the dialog."""
        self.logger.info("Settings changed - applying updates")

        # Update audio devices
        mic_device = self.config.get_mic_device()
        self.audio_recorder.set_mic_device(mic_device)
        system_device = self.config.get_system_device()
        self.audio_recorder.set_system_device(system_device)

        # Update streaming model
        streaming_model = self.config.get_streaming_model_size()
        if streaming_model != self.streaming_transcriber.model_size:
            self.streaming_transcriber.change_model(streaming_model)

        # Update echo diagnostics and logging
        self.echo_diagnostics.enabled = self.config.is_echo_debug_enabled()
        if self._transcription_worker is not None:
            self._transcription_worker.echo_gate_logging = self.config.is_echo_gate_logging_enabled()

        # Update speaker labels
        self._apply_speaker_labels()

    def _apply_speaker_labels(self):
        """Update main window speaker labels from config."""
        if self.main_window is not None:
            first_name = self.config.get_first_name()
            you_label = first_name if first_name else "You"
            self.main_window.set_speaker_labels(you_label=you_label)

    def _on_speaker_name_changed(self, filepath, name):
        """Save updated speaker name to transcript file."""
        self.session_manager.update_speaker_name(filepath, name)

    def on_folder_selected(self, folder_name):
        """Load and display transcripts for the selected folder."""
        self._selected_folder = folder_name
        sort_desc = self.main_window._sort_descending if self.main_window else True
        transcripts = self.session_manager.list_transcripts(folder_name, sort_descending=sort_desc)
        if self.main_window:
            self.main_window.set_transcripts(transcripts)

    def on_folder_created(self, name):
        """Create a new folder and refresh the folder list."""
        self.session_manager.create_folder(name)
        self._refresh_folders()

    def on_folder_renamed(self, old_name, new_name):
        """Rename a folder and refresh the folder list."""
        self.session_manager.rename_folder(old_name, new_name)
        self._refresh_folders()

    def on_folder_deleted(self, name):
        """Delete a folder and refresh the folder list."""
        self.session_manager.delete_folder(name)
        self._refresh_folders()

    def on_transcript_selected(self, folder, name):
        """Load and display the selected transcript."""
        transcripts = self.session_manager.list_transcripts(folder)
        for t in transcripts:
            if t["name"] == name:
                data = self.session_manager.load_transcript(t["path"])
                if self.main_window:
                    self.main_window.set_transcript(data["segments"])
                    self.main_window.set_transcript_meta(
                        t["path"], data.get("speaker_name", ""))
                break

    def on_transcript_renamed(self, folder, old_name, new_name):
        """Rename a transcript and refresh the transcript list."""
        self.session_manager.rename_transcript(folder, old_name, new_name)
        self.on_folder_selected(folder)

    def on_transcript_deleted(self, folder, name):
        """Delete a transcript and refresh the transcript list."""
        self.session_manager.delete_transcript(folder, name)
        self.on_folder_selected(folder)

    def on_transcript_moved(self, src_folder, name, dest_folder):
        """Move a transcript to another folder and refresh the transcript list."""
        self.session_manager.move_transcript(src_folder, name, dest_folder)
        self.on_folder_selected(src_folder)

    def on_sort_order_changed(self, descending):
        """Re-sort the transcript list when sort order changes."""
        if self._selected_folder:
            self.on_folder_selected(self._selected_folder)

    def _refresh_folders(self):
        """Refresh the folder list in the main window."""
        folders = self.session_manager.list_folders()
        if self.main_window:
            self.main_window.set_folders(folders)

    # -- Auto-update --------------------------------------------------------

    def _check_for_updates(self):
        """Launch update check in a background QThread."""
        self.logger.info("Checking for updates...")
        self._update_thread = QThread()
        self._update_worker = UpdateWorker()
        self._update_worker.moveToThread(self._update_thread)

        self._update_thread.started.connect(self._update_worker.run)
        self._update_worker.update_available.connect(self._on_update_available)
        self._update_worker.finished.connect(self._cleanup_update_thread)

        self._update_thread.start()

    def _cleanup_update_thread(self):
        """Clean up the update checker thread."""
        if self._update_thread is not None:
            self._update_thread.quit()
            self._update_thread.wait(2000)
            self._update_thread.deleteLater()
            self._update_thread = None
        if self._update_worker is not None:
            self._update_worker.deleteLater()
            self._update_worker = None

    def _on_update_available(self, update_info):
        """Show the update toast when a new version is found."""
        self._update_info = update_info
        self.logger.info(f"Update available: {update_info.version_str}")

        from utils.resource_path import is_bundled
        if not is_bundled():
            source_msg = UpdateChecker.get_source_update_message(update_info)
            body = source_msg + "\n\n" + (update_info.release_body or "")
        else:
            body = update_info.release_body

        self._update_toast = UpdateToast(
            update_info.version_str,
            release_body=body,
        )
        self._update_toast.accepted.connect(self._apply_update)
        self._update_toast.dismissed.connect(self._dismiss_update)
        self._update_toast.show_toast()

    def _apply_update(self):
        """Download and apply the update (bundled) or just dismiss (source)."""
        from utils.resource_path import is_bundled

        if self._update_info is None:
            return

        if not is_bundled():
            self.logger.info("Source install — user should run: git pull && uv sync")
            self._update_toast = None
            return

        self.logger.info(f"Downloading update {self._update_info.version_str}...")
        if self.tray_icon is not None:
            self.tray_icon.show_notification(
                "Downloading Update",
                f"Downloading Cadence {self._update_info.version_str}...",
            )

        checker = UpdateChecker()
        downloaded = checker.download_update(self._update_info)
        if downloaded is None:
            self.logger.error("Update download failed")
            if self.tray_icon is not None:
                self.tray_icon.show_notification(
                    "Update Failed",
                    "Could not download the update. Try again later.",
                )
            return

        success = checker.apply_update(downloaded)
        if success:
            self.logger.info("Update applied, shutting down for restart")
            self.quit()
        else:
            self.logger.error("Failed to apply update")
            if self.tray_icon is not None:
                self.tray_icon.show_notification(
                    "Update Failed",
                    "Could not apply the update. Try again later.",
                )

    def _dismiss_update(self):
        """User dismissed the update toast."""
        self.logger.info("Update dismissed by user")
        self._update_toast = None

    def check_for_updates_manual(self):
        """Manually trigger an update check (e.g. from settings button)."""
        self.logger.info("Manual update check requested")
        if self._update_thread is not None and self._update_thread.isRunning():
            self.logger.info("Update check already in progress")
            return
        self._check_for_updates()

    def show_window(self):
        """Show and raise the main window."""
        if self.main_window is not None:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()

    def quit(self):
        """Clean shutdown of the application."""
        self.logger.info("Shutting down Cadence...")

        # Stop update timer and thread
        self._update_timer.stop()
        if self._update_thread is not None and self._update_thread.isRunning():
            self._update_thread.quit()
            self._update_thread.wait(2000)

        # Stop transcription worker
        self._stop_transcription_worker()

        # Stop post-processing if running
        if self._postprocess_thread is not None and self._postprocess_thread.isRunning():
            self._postprocess_thread.quit()
            self._postprocess_thread.wait(3000)

        # Stop recording if active
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()

        QApplication.quit()


def main():
    """Application entry point."""
    # Set Windows app ID before creating QApplication
    set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName("Cadence")
    app.setApplicationDisplayName("Cadence")
    app.setQuitOnLastWindowClosed(False)  # Keep running with system tray

    # Apply dark theme
    apply_theme(app)

    # Create application controller
    cadence_app = CadenceApp()

    # Create GUI components
    tray_icon = SystemTrayIcon()
    main_window = MainWindow()

    # Attach GUI to app controller
    cadence_app.tray_icon = tray_icon
    cadence_app.main_window = main_window

    # Wire tray icon signals
    tray_icon.start_requested.connect(cadence_app.start_recording)
    tray_icon.stop_requested.connect(cadence_app.stop_recording)
    tray_icon.window_requested.connect(cadence_app.show_window)
    tray_icon.settings_requested.connect(cadence_app.show_settings)
    tray_icon.quit_requested.connect(cadence_app.quit)

    # Wire main window signals
    main_window.start_requested.connect(cadence_app.start_recording)
    main_window.stop_requested.connect(cadence_app.stop_recording)
    main_window.folder_selected.connect(cadence_app.on_folder_selected)
    main_window.folder_created.connect(cadence_app.on_folder_created)
    main_window.folder_renamed.connect(cadence_app.on_folder_renamed)
    main_window.folder_deleted.connect(cadence_app.on_folder_deleted)
    main_window.transcript_selected.connect(cadence_app.on_transcript_selected)
    main_window.transcript_renamed.connect(cadence_app.on_transcript_renamed)
    main_window.transcript_deleted.connect(cadence_app.on_transcript_deleted)
    main_window.transcript_moved.connect(cadence_app.on_transcript_moved)
    main_window.sort_order_changed.connect(cadence_app.on_sort_order_changed)
    main_window.settings_requested.connect(cadence_app.show_settings)
    main_window.speaker_name_changed.connect(cadence_app._on_speaker_name_changed)

    # Apply speaker labels from config
    cadence_app._apply_speaker_labels()

    # Load folders on startup
    cadence_app._refresh_folders()

    # Show the main window
    main_window.show()

    # Startup toast
    model_id = cadence_app.config.get_streaming_model_size()
    model_names = {"tiny": "Fastest", "base": "Balanced", "small": "Accurate", "medium": "Precision"}
    model_label = model_names.get(model_id, model_id)
    tray_icon.show_notification(
        "Cadence Ready",
        "Ready to record",
        details=f"Model: {model_label}",
    )

    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
