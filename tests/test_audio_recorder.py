def test_audio_recorder_init():
    """AudioRecorder initializes with correct defaults."""
    from src.core.audio_recorder import AudioRecorder
    recorder = AudioRecorder()
    assert recorder.sample_rate == 16000
    assert recorder.is_recording is False


def test_list_mic_devices():
    """list_mic_devices should return a list."""
    from src.core.audio_recorder import AudioRecorder
    recorder = AudioRecorder()
    devices = recorder.list_mic_devices()
    assert isinstance(devices, list)
