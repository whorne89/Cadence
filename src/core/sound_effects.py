"""
Sound effects for Cadence.
Generates short piano-like tones as WAV files and plays via Windows native audio API.
Uses SND_FILENAME | SND_ASYNC for reliable non-blocking playback that
doesn't conflict with sounddevice recording.

Custom sounds: Drop your own start.wav / stop.wav into .cadence/sounds/
to override the generated defaults.
"""

import os
import struct
import numpy as np
import winsound

from utils.logger import get_logger
from utils.resource_path import get_app_data_path


class SoundEffects:
    """Generates and plays short notification tones."""

    def __init__(self, sample_rate=44100, volume=0.3):
        self.sample_rate = sample_rate
        self.volume = volume
        self.logger = get_logger()

        # Sound files live in .cadence/sounds/
        sounds_dir = get_app_data_path("sounds")

        self._start_path = self._resolve_sound(sounds_dir, 'start.wav')
        self._stop_path = self._resolve_sound(sounds_dir, 'stop.wav')

    def _resolve_sound(self, sounds_dir, filename):
        """Find sound file: user override > generate default.

        Priority:
        1. User-provided file in .cadence/sounds/
        2. Auto-generated default (written to .cadence/sounds/)
        """
        user_path = os.path.join(sounds_dir, filename)
        if os.path.exists(user_path):
            return user_path

        # Generate default — higher pitch for start, lower for stop
        freq = 523 if 'start' in filename else 392
        self._write_wav(user_path, self._generate_piano_tone(freq=freq))
        return user_path

    def _generate_piano_tone(self, freq, duration=0.6):
        """Generate a piano-like tone with room reverb.

        Models piano string physics: sharp hammer attack, harmonics with
        inharmonicity (upper partials slightly sharp from string stiffness),
        and per-partial decay rates. Reverb via simulated early reflections.
        """
        sr = self.sample_rate
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Piano partials: (harmonic_number, amplitude, decay_rate)
        # Higher harmonics are quieter and decay faster
        inharmonicity = 0.0004
        partials = [
            (1, 1.00, 5),
            (2, 0.65, 7),
            (3, 0.35, 9),
            (4, 0.20, 12),
            (5, 0.12, 15),
            (6, 0.06, 18),
        ]

        tone = np.zeros(n_samples)
        for n, amp, decay in partials:
            # Piano inharmonicity: f_n = n * f * sqrt(1 + B*n^2)
            partial_freq = n * freq * np.sqrt(1 + inharmonicity * n * n)
            tone += amp * np.sin(2 * np.pi * partial_freq * t) * np.exp(-t * decay)

        # Sharp attack (3ms) -- hammer strike
        attack_n = int(sr * 0.003)
        if attack_n > 0:
            tone[:attack_n] *= np.linspace(0, 1, attack_n)

        # Normalize before reverb
        peak = np.max(np.abs(tone))
        if peak > 0:
            tone = tone / peak

        # Room reverb: early reflections at decreasing amplitude
        output = tone.copy()
        reflections = [
            (0.035, 0.25),   # 35ms -- first wall
            (0.070, 0.18),   # 70ms -- opposite wall
            (0.120, 0.10),   # 120ms -- room fill
            (0.180, 0.06),   # 180ms -- reverb tail
        ]
        for delay_s, amp in reflections:
            d = int(sr * delay_s)
            if d < n_samples:
                output[d:] += tone[:n_samples - d] * amp

        # Final normalize and apply volume
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * self.volume

        return output

    def _write_wav(self, path, tone):
        """Write tone data to a 16-bit mono WAV file."""
        pcm = np.clip(tone * 32767, -32768, 32767).astype(np.int16)
        data_size = len(pcm) * 2

        with open(path, 'wb') as f:
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))                    # PCM format
            f.write(struct.pack('<H', 1))                    # mono
            f.write(struct.pack('<I', self.sample_rate))     # sample rate
            f.write(struct.pack('<I', self.sample_rate * 2)) # byte rate
            f.write(struct.pack('<H', 2))                    # block align
            f.write(struct.pack('<H', 16))                   # bits per sample
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(pcm.tobytes())

    def play_start_tone(self):
        """Play chime (recording started). Non-blocking."""
        try:
            winsound.PlaySound(
                self._start_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            )
        except Exception as e:
            self.logger.warning(f"Sound playback failed: {e}")

    def play_stop_tone(self):
        """Play chime (recording stopped). Non-blocking."""
        try:
            winsound.PlaySound(
                self._stop_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            )
        except Exception as e:
            self.logger.warning(f"Sound playback failed: {e}")
