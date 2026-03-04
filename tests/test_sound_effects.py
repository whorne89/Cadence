import os
import struct
import wave
import pytest
import numpy as np


def test_sound_effects_init_generates_wav_files(tmp_path, monkeypatch):
    """SoundEffects.__init__ should generate start.wav and stop.wav if missing."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    # Patch get_app_data_path to return our tmp sounds dir
    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects()

    assert os.path.exists(os.path.join(sounds_dir, "start.wav"))
    assert os.path.exists(os.path.join(sounds_dir, "stop.wav"))


def test_generated_wav_is_valid(tmp_path, monkeypatch):
    """Generated WAV files should be valid 16-bit mono PCM."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects(sample_rate=44100)

    start_path = os.path.join(sounds_dir, "start.wav")
    with wave.open(start_path, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 44100
        assert wf.getnframes() > 0


def test_user_override_respected(tmp_path, monkeypatch):
    """If user places a custom start.wav, SoundEffects should use it instead of generating."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    # Write a tiny valid WAV as the user override
    custom_path = os.path.join(sounds_dir, "start.wav")
    _write_tiny_wav(custom_path)

    original_size = os.path.getsize(custom_path)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects()

    # The user file should not have been overwritten
    assert os.path.getsize(custom_path) == original_size
    assert sfx._start_path == custom_path


def test_generate_piano_tone_shape(tmp_path, monkeypatch):
    """_generate_piano_tone should return an array of the expected length."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects(sample_rate=44100, volume=0.3)

    tone = sfx._generate_piano_tone(freq=440, duration=0.5)
    expected_samples = int(44100 * 0.5)
    assert len(tone) == expected_samples
    assert tone.dtype == np.float64


def test_tone_amplitude_respects_volume(tmp_path, monkeypatch):
    """Generated tone peak amplitude should not exceed the configured volume."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects(sample_rate=44100, volume=0.3)

    tone = sfx._generate_piano_tone(freq=523)
    peak = np.max(np.abs(tone))
    # Peak should be approximately equal to volume (with reverb it normalizes to volume)
    assert peak <= 0.3 + 0.01  # small tolerance


def test_start_tone_higher_pitch_than_stop(tmp_path, monkeypatch):
    """Start tone (523 Hz) should have higher frequency content than stop tone (392 Hz)."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects(sample_rate=44100)

    start_tone = sfx._generate_piano_tone(freq=523, duration=0.5)
    stop_tone = sfx._generate_piano_tone(freq=392, duration=0.5)

    # Use FFT to find dominant frequency
    start_fft = np.abs(np.fft.rfft(start_tone))
    stop_fft = np.abs(np.fft.rfft(stop_tone))

    freqs = np.fft.rfftfreq(len(start_tone), 1.0 / 44100)

    start_dominant = freqs[np.argmax(start_fft)]
    stop_dominant = freqs[np.argmax(stop_fft)]

    assert start_dominant > stop_dominant


def test_write_wav_file_structure(tmp_path, monkeypatch):
    """_write_wav should produce a file with correct RIFF/WAV headers."""
    sounds_dir = str(tmp_path / "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    monkeypatch.setattr(
        "src.core.sound_effects.get_app_data_path",
        lambda subdir="": sounds_dir if subdir == "sounds" else str(tmp_path),
    )

    from src.core.sound_effects import SoundEffects
    sfx = SoundEffects(sample_rate=22050)

    test_path = os.path.join(tmp_path, "test_output.wav")
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)) * 0.3
    sfx._write_wav(test_path, tone)

    with open(test_path, 'rb') as f:
        # RIFF header
        assert f.read(4) == b'RIFF'
        riff_size = struct.unpack('<I', f.read(4))[0]
        assert f.read(4) == b'WAVE'

        # fmt chunk
        assert f.read(4) == b'fmt '
        fmt_size = struct.unpack('<I', f.read(4))[0]
        assert fmt_size == 16
        audio_format = struct.unpack('<H', f.read(2))[0]
        assert audio_format == 1  # PCM
        channels = struct.unpack('<H', f.read(2))[0]
        assert channels == 1
        sample_rate = struct.unpack('<I', f.read(4))[0]
        assert sample_rate == 22050

        # data chunk — skip byte rate (4) + block align (2) + bits per sample (2)
        f.read(8)
        assert f.read(4) == b'data'


def _write_tiny_wav(path, sr=44100, duration=0.01):
    """Helper: write a minimal valid WAV file."""
    n = int(sr * duration)
    pcm = np.zeros(n, dtype=np.int16)
    data_size = len(pcm) * 2
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm.tobytes())
