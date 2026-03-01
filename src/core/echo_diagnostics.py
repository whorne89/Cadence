"""
Echo gate diagnostic system.

When enabled, saves audio chunks the echo gate processes as WAV files
alongside metadata (RMS levels, ratios, decisions). Also snapshots
live vs post-processed transcripts and generates a diff report.

Toggle via settings.json: {"debug": {"echo_diagnostics": true}}
"""

import json
import struct
import threading
import queue
import logging
import time
from pathlib import Path

import numpy as np

from utils.resource_path import get_app_data_path

logger = logging.getLogger("Cadence")

SAMPLE_RATE = 16000


class EchoDiagnostics:
    """Records echo gate decisions and audio for offline analysis."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self._session_dir = None
        self._chunk_index = 0
        self._chunks_passed = 0
        self._chunks_suppressed = 0
        self._queue = None
        self._writer_thread = None
        self._stop_event = None

    def start_session(self):
        """Create a timestamped session directory and start the writer thread."""
        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(get_app_data_path("echo_debug"))
        self._session_dir = base / timestamp
        (self._session_dir / "chunks").mkdir(parents=True, exist_ok=True)

        self._chunk_index = 0
        self._chunks_passed = 0
        self._chunks_suppressed = 0

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self._writer_thread.start()
        logger.info(f"Echo diagnostics session started: {self._session_dir}")

    def record_chunk(self, mic_audio, sys_audio, mic_rms, sys_rms, ratio,
                     echo_detected, timestamp):
        """Enqueue a chunk for background writing. No-op when disabled."""
        if not self.enabled or self._queue is None:
            return

        self._chunk_index += 1
        if echo_detected:
            self._chunks_suppressed += 1
        else:
            self._chunks_passed += 1

        # Copy arrays so the caller can reuse buffers
        item = {
            "index": self._chunk_index,
            "mic_audio": np.array(mic_audio, dtype=np.float32),
            "sys_audio": np.array(sys_audio, dtype=np.float32),
            "mic_rms": mic_rms,
            "sys_rms": sys_rms,
            "ratio": ratio,
            "echo_detected": echo_detected,
            "timestamp": timestamp,
        }
        self._queue.put(item)

    def save_live_transcript(self, segments):
        """Save the live transcript segments before post-processing."""
        if not self.enabled or self._session_dir is None:
            return
        path = self._session_dir / "live_transcript.json"
        path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
        logger.info(f"Echo diagnostics: saved live transcript ({len(segments)} segments)")

    def save_postprocessed_transcript(self, segments):
        """Save post-processed segments and generate diff report."""
        if not self.enabled or self._session_dir is None:
            return
        path = self._session_dir / "postprocessed_transcript.json"
        path.write_text(json.dumps(segments, indent=2), encoding="utf-8")

        # Generate diff report
        self._generate_diff_report(segments)
        logger.info(f"Echo diagnostics: saved postprocessed transcript ({len(segments)} segments)")

    def finish_session(self):
        """Flush the writer queue and write the session manifest."""
        if not self.enabled or self._queue is None:
            return

        # Signal writer to stop and wait for it to drain
        self._stop_event.set()
        self._writer_thread.join(timeout=10)

        # Write manifest
        manifest = {
            "chunks_total": self._chunk_index,
            "chunks_passed": self._chunks_passed,
            "chunks_suppressed": self._chunks_suppressed,
            "session_dir": str(self._session_dir),
        }
        manifest_path = self._session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(
            f"Echo diagnostics session finished: "
            f"{self._chunk_index} chunks ({self._chunks_passed} passed, "
            f"{self._chunks_suppressed} suppressed)"
        )

    # --- internal ---

    def _writer_loop(self):
        """Background thread that writes WAV and metadata files."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._write_chunk(item)

    def _write_chunk(self, item):
        """Write a single chunk's WAV files and metadata JSON."""
        idx = item["index"]
        label = "suppressed" if item["echo_detected"] else "pass"
        chunks_dir = self._session_dir / "chunks"

        # Write mic WAV
        mic_path = chunks_dir / f"chunk_{idx:04d}_{label}.wav"
        _write_wav(mic_path, item["mic_audio"], SAMPLE_RATE)

        # Write system WAV
        sys_path = chunks_dir / f"chunk_{idx:04d}_sys.wav"
        _write_wav(sys_path, item["sys_audio"], SAMPLE_RATE)

        # Write metadata
        meta_path = chunks_dir / f"chunk_{idx:04d}.json"
        meta = {
            "index": idx,
            "mic_rms": item["mic_rms"],
            "sys_rms": item["sys_rms"],
            "ratio": item["ratio"],
            "echo_detected": item["echo_detected"],
            "timestamp": item["timestamp"],
            "mic_samples": len(item["mic_audio"]),
            "sys_samples": len(item["sys_audio"]),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _generate_diff_report(self, postprocessed_segments):
        """Generate a human-readable diff between live and post-processed transcripts."""
        live_path = self._session_dir / "live_transcript.json"
        if not live_path.exists():
            return

        live_segments = json.loads(live_path.read_text(encoding="utf-8"))
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Echo Gate Diagnostic Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        report_lines.append(f"Live segments:           {len(live_segments)}")
        report_lines.append(f"Post-processed segments: {len(postprocessed_segments)}")
        removed = len(live_segments) - len(postprocessed_segments)
        report_lines.append(f"Segments removed:        {removed}")
        report_lines.append("")

        # Show live-only segments (removed by post-processing)
        post_texts = {s["text"] for s in postprocessed_segments}
        removed_segs = [s for s in live_segments if s["text"] not in post_texts]
        if removed_segs:
            report_lines.append("-" * 40)
            report_lines.append("REMOVED by post-processing:")
            report_lines.append("-" * 40)
            for s in removed_segs:
                t = s.get("start", 0)
                report_lines.append(f"  [{t:6.1f}s] [{s['speaker']}] {s['text']}")
            report_lines.append("")

        # Show post-process-only segments (new or changed)
        live_texts = {s["text"] for s in live_segments}
        new_segs = [s for s in postprocessed_segments if s["text"] not in live_texts]
        if new_segs:
            report_lines.append("-" * 40)
            report_lines.append("NEW in post-processing:")
            report_lines.append("-" * 40)
            for s in new_segs:
                t = s.get("start", 0)
                report_lines.append(f"  [{t:6.1f}s] [{s['speaker']}] {s['text']}")
            report_lines.append("")

        report_path = self._session_dir / "diff_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")


def _write_wav(path, audio, sample_rate):
    """Write float32 audio as 16-bit PCM WAV using stdlib wave-compatible struct packing."""
    # Convert float32 [-1, 1] to int16
    int16_audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    data = int16_audio.tobytes()

    num_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(data)
    fmt_chunk_size = 16

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", fmt_chunk_size))
        f.write(struct.pack("<HHIIHH",
                            1,  # PCM format
                            num_channels,
                            sample_rate,
                            sample_rate * num_channels * sample_width,  # byte rate
                            num_channels * sample_width,  # block align
                            sample_width * 8))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)
