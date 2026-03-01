"""
Dual-channel audio recorder for Cadence.
Captures microphone and system audio (WASAPI loopback) on separate channels.
"""

import logging
import threading
import queue
import time
import numpy as np

import sounddevice as sd

logger = logging.getLogger("Cadence")

# Try importing pyaudiowpatch for WASAPI loopback
try:
    import pyaudiowpatch as pyaudio
    WASAPI_AVAILABLE = True
except ImportError:
    WASAPI_AVAILABLE = False
    logger.warning("PyAudioWPatch not available — system audio capture disabled")


class AudioRecorder:
    """
    Records mic and system audio on separate channels.

    Mic: via sounddevice (InputStream)
    System: via PyAudioWPatch (WASAPI loopback)

    Accumulated frames are available via _mic_frames / _system_frames
    for the TranscriptionWorker to process.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        self.is_recording = False
        self._mic_thread = None
        self._system_thread = None
        self._mic_frames = []
        self._system_frames = []
        self._start_time = 0

        # Device indices
        self._mic_device = None
        self._system_device = None

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

    def list_system_devices(self):
        """List available WASAPI loopback devices for system audio."""
        if not WASAPI_AVAILABLE:
            return []
        devices = []
        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev['hostApi'] == wasapi_info['index'] and dev.get('isLoopbackDevice', False):
                    devices.append({
                        'index': i,
                        'name': dev['name'],
                        'channels': dev['maxInputChannels'],
                        'sample_rate': int(dev['defaultSampleRate']),
                    })
        finally:
            p.terminate()
        return devices

    def set_mic_device(self, device_index):
        """Set microphone device index."""
        self._mic_device = device_index

    def set_system_device(self, device_index):
        """Set system audio (WASAPI loopback) device index."""
        self._system_device = device_index

    def start_recording(self):
        """Start recording from mic and system audio."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        self.is_recording = True
        self._mic_frames = []
        self._system_frames = []
        self._start_time = time.time()

        self._mic_thread = threading.Thread(target=self._record_mic, daemon=True)
        self._mic_thread.start()

        if WASAPI_AVAILABLE:
            self._system_thread = threading.Thread(target=self._record_system, daemon=True)
            self._system_thread.start()

        logger.info("Recording started (mic + system)")

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

    def _record_system(self):
        """Record system audio via WASAPI loopback."""
        if not WASAPI_AVAILABLE:
            return

        p = pyaudio.PyAudio()
        pyaudio_chunk = 1024

        try:
            device_index = self._system_device
            if device_index is None:
                try:
                    default_loopback = p.get_default_wasapi_loopback()
                    device_index = default_loopback['index']
                    logger.info(f"Auto-detected loopback device: {default_loopback['name']}")
                except Exception:
                    logger.warning("No default WASAPI loopback device found")
                    return

            if device_index is None:
                logger.warning("No WASAPI loopback device found")
                return

            dev_info = p.get_device_info_by_index(device_index)
            device_rate = int(dev_info['defaultSampleRate'])
            device_channels = dev_info['maxInputChannels']

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=device_channels,
                rate=device_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=pyaudio_chunk,
            )

            while self.is_recording:
                data = stream.read(pyaudio_chunk, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)

                if device_channels > 1:
                    audio = audio.reshape(-1, device_channels).mean(axis=1)

                if device_rate != self.sample_rate:
                    from scipy.signal import resample
                    target_len = int(len(audio) * self.sample_rate / device_rate)
                    audio = resample(audio, target_len).astype(np.float32)

                self._system_frames.append(audio.copy())

            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"System audio recording error: {e}")
        finally:
            p.terminate()

    def stop_recording(self):
        """Stop recording and return audio data as (mic_audio, system_audio)."""
        if not self.is_recording:
            return None, None

        self.is_recording = False

        if self._mic_thread:
            self._mic_thread.join(timeout=2.0)
        if self._system_thread:
            self._system_thread.join(timeout=2.0)

        mic_audio = np.concatenate(self._mic_frames) if self._mic_frames else np.array([], dtype=np.float32)
        system_audio = np.concatenate(self._system_frames) if self._system_frames else np.array([], dtype=np.float32)

        logger.info(f"Recording stopped. Duration: {self.get_duration():.1f}s")
        return mic_audio, system_audio

    def get_duration(self):
        """Get current recording duration in seconds."""
        if not hasattr(self, '_start_time') or self._start_time == 0:
            return 0.0
        return time.time() - self._start_time
