import wave

import pytest

from wavewright.audio_files import ffmpeg_decode_to_raw_float32


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00\x00\x40\x00\xc0")
    return path


def test_decode_maps_first_audio_stream_to_float32_stdout(audio_path):
    command = ffmpeg_decode_to_raw_float32(audio_path)

    assert command[0] == "ffmpeg"
    assert "-nostdin" in command
    assert command[command.index("-i") + 1] == str(audio_path)
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-f") + 1] == "f32le"
    assert command[command.index("-acodec") + 1] == "pcm_f32le"
    assert command[-1] == "pipe:1"
