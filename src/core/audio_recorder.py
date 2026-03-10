"""
Single-stream audio recorder for Cadence.
Captures microphone audio via sounddevice.
"""

import logging
import threading
import queue
import time
import numpy as np

import sounddevice as sd

logger = logging.getLogger("Cadence")


class AudioRecorder:
    """
    Records microphone audio as a single stream.

    Accumulated frames are available via _mic_frames
    for the TranscriptionWorker to process incrementally.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        self.is_recording = False
        self._mic_thread = None
        self._mic_frames = []
        self._start_time = 0

        # Device index
        self._mic_device = None

    def list_mic_devices(self):
        """List available microphone input devices from the default host API only."""
        devices = []
        seen_names = set()
        try:
            default_api = sd.query_hostapis(0)
            default_api_index = 0
        except Exception:
            default_api_index = 0
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and dev['hostapi'] == default_api_index:
                name = dev['name']
                if name in seen_names:
                    continue
                seen_names.add(name)
                devices.append({
                    'index': i,
                    'name': name,
                    'channels': dev['max_input_channels'],
                    'sample_rate': int(dev['default_samplerate']),
                })
        return devices

    def set_mic_device(self, device_index):
        """Set microphone device index."""
        self._mic_device = device_index

    def start_recording(self):
        """Start recording from microphone."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        self.is_recording = True
        self._mic_frames = []
        self._start_time = time.time()

        self._mic_thread = threading.Thread(target=self._record_mic, daemon=True)
        self._mic_thread.start()

        logger.info("Recording started (microphone)")

    def _record_mic(self):
        """Record from microphone via sounddevice."""
        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic status: {status}")
            audio_queue.put(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self._mic_device,
                callback=callback,
                blocksize=1024,
            ):
                while self.is_recording:
                    try:
                        audio = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self._mic_frames.append(audio)
        except Exception as e:
            logger.error(f"Mic recording error: {e}")

    def stop_recording(self):
        """Stop recording and return audio data."""
        if not self.is_recording:
            return np.array([], dtype=np.float32)

        self.is_recording = False

        if self._mic_thread:
            self._mic_thread.join(timeout=2.0)

        mic_audio = np.concatenate(self._mic_frames) if self._mic_frames else np.array([], dtype=np.float32)

        logger.info(f"Recording stopped. Duration: {self.get_duration():.1f}s")
        return mic_audio

    def get_duration(self):
        """Get current recording duration in seconds."""
        if not hasattr(self, '_start_time') or self._start_time == 0:
            return 0.0
        return time.time() - self._start_time
