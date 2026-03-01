"""Tests for the echo diagnostics system."""

import json
import struct

import numpy as np
import pytest

from core.echo_diagnostics import EchoDiagnostics, _write_wav


class TestDisabledNoOp:
    """When disabled, all methods should be safe no-ops."""

    def test_disabled_start_session(self, tmp_path):
        diag = EchoDiagnostics(enabled=False)
        diag.start_session()
        assert diag._session_dir is None

    def test_disabled_record_chunk(self):
        diag = EchoDiagnostics(enabled=False)
        # Should not raise
        diag.record_chunk(
            np.zeros(100), np.zeros(100),
            0.01, 0.02, 0.5, False, 1.0,
        )

    def test_disabled_save_transcripts(self):
        diag = EchoDiagnostics(enabled=False)
        diag.save_live_transcript([{"speaker": "you", "text": "hi", "start": 0}])
        diag.save_postprocessed_transcript([])

    def test_disabled_finish_session(self):
        diag = EchoDiagnostics(enabled=False)
        diag.finish_session()


class TestChunkWriting:
    """Test that audio chunks and metadata are written correctly."""

    def _make_diagnostics(self, tmp_path, monkeypatch):
        """Create an enabled diagnostics instance that writes to tmp_path."""
        diag = EchoDiagnostics(enabled=True)
        # Patch get_app_data_path to use tmp_path
        monkeypatch.setattr(
            "core.echo_diagnostics.get_app_data_path",
            lambda subdir="": str(tmp_path / subdir) if subdir else str(tmp_path),
        )
        return diag

    def test_chunk_wav_and_metadata(self, tmp_path, monkeypatch):
        diag = self._make_diagnostics(tmp_path, monkeypatch)
        diag.start_session()

        mic = np.random.randn(1600).astype(np.float32) * 0.1
        sys = np.random.randn(1600).astype(np.float32) * 0.05

        diag.record_chunk(mic, sys, 0.01, 0.02, 0.5, False, 1.0)
        diag.record_chunk(mic, sys, 0.008, 0.03, 0.27, True, 2.0)

        # Finish to flush the queue
        diag.finish_session()

        chunks_dir = diag._session_dir / "chunks"

        # First chunk: passed
        assert (chunks_dir / "chunk_0001_pass.wav").exists()
        assert (chunks_dir / "chunk_0001_sys.wav").exists()
        meta1 = json.loads((chunks_dir / "chunk_0001.json").read_text())
        assert meta1["echo_detected"] is False
        assert meta1["mic_rms"] == 0.01
        assert meta1["ratio"] == 0.5

        # Second chunk: suppressed
        assert (chunks_dir / "chunk_0002_suppressed.wav").exists()
        assert (chunks_dir / "chunk_0002_sys.wav").exists()
        meta2 = json.loads((chunks_dir / "chunk_0002.json").read_text())
        assert meta2["echo_detected"] is True

    def test_wav_file_is_valid(self, tmp_path):
        """Verify the WAV file has correct RIFF/WAVE headers and data."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, audio, 16000)

        with open(wav_path, "rb") as f:
            # RIFF header
            assert f.read(4) == b"RIFF"
            file_size = struct.unpack("<I", f.read(4))[0]
            assert f.read(4) == b"WAVE"

            # fmt chunk
            assert f.read(4) == b"fmt "
            fmt_size = struct.unpack("<I", f.read(4))[0]
            assert fmt_size == 16
            fmt_data = struct.unpack("<HHIIHH", f.read(16))
            assert fmt_data[0] == 1  # PCM
            assert fmt_data[1] == 1  # mono
            assert fmt_data[2] == 16000  # sample rate

            # data chunk
            assert f.read(4) == b"data"
            data_size = struct.unpack("<I", f.read(4))[0]
            assert data_size == len(audio) * 2  # 16-bit = 2 bytes per sample

            # Verify RIFF size
            assert file_size == 36 + data_size


class TestTranscriptSaving:
    """Test transcript saving and diff report generation."""

    def _make_diagnostics(self, tmp_path, monkeypatch):
        diag = EchoDiagnostics(enabled=True)
        monkeypatch.setattr(
            "core.echo_diagnostics.get_app_data_path",
            lambda subdir="": str(tmp_path / subdir) if subdir else str(tmp_path),
        )
        return diag

    def test_save_live_transcript(self, tmp_path, monkeypatch):
        diag = self._make_diagnostics(tmp_path, monkeypatch)
        diag.start_session()

        segments = [
            {"speaker": "you", "text": "hello", "start": 0.0},
            {"speaker": "them", "text": "world", "start": 1.0},
        ]
        diag.save_live_transcript(segments)

        live_path = diag._session_dir / "live_transcript.json"
        assert live_path.exists()
        saved = json.loads(live_path.read_text())
        assert len(saved) == 2
        assert saved[0]["text"] == "hello"

    def test_diff_report(self, tmp_path, monkeypatch):
        diag = self._make_diagnostics(tmp_path, monkeypatch)
        diag.start_session()

        live = [
            {"speaker": "you", "text": "hello", "start": 0.0},
            {"speaker": "you", "text": "echo bleed", "start": 1.0},
            {"speaker": "them", "text": "world", "start": 1.5},
        ]
        post = [
            {"speaker": "you", "text": "hello", "start": 0.0},
            {"speaker": "them", "text": "world", "start": 1.5},
        ]

        diag.save_live_transcript(live)
        diag.save_postprocessed_transcript(post)

        report_path = diag._session_dir / "diff_report.txt"
        assert report_path.exists()
        report = report_path.read_text()
        assert "Live segments:           3" in report
        assert "Post-processed segments: 2" in report
        assert "Segments removed:        1" in report
        assert "echo bleed" in report
        assert "REMOVED by post-processing" in report


class TestManifest:
    """Test manifest generation."""

    def test_manifest_written(self, tmp_path, monkeypatch):
        diag = EchoDiagnostics(enabled=True)
        monkeypatch.setattr(
            "core.echo_diagnostics.get_app_data_path",
            lambda subdir="": str(tmp_path / subdir) if subdir else str(tmp_path),
        )
        diag.start_session()

        mic = np.zeros(800, dtype=np.float32)
        sys = np.zeros(800, dtype=np.float32)
        diag.record_chunk(mic, sys, 0.01, 0.02, 0.5, False, 0.0)
        diag.record_chunk(mic, sys, 0.005, 0.03, 0.17, True, 1.0)
        diag.record_chunk(mic, sys, 0.02, 0.01, 2.0, False, 2.0)

        diag.finish_session()

        manifest_path = diag._session_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["chunks_total"] == 3
        assert manifest["chunks_passed"] == 2
        assert manifest["chunks_suppressed"] == 1
