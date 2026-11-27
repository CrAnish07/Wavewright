import wave

import pytest

from wavewright.metadata import ffprobe_gen_json_metadata


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00\x00\x40\x00\xc0")
    return path


def test_probe_requests_stream_metadata_as_json(audio_path):
    command = ffprobe_gen_json_metadata(audio_path)

    assert command[0] == "ffprobe"
    assert "-show_streams" in command
    assert command[command.index("-of") + 1] == "json"
    fields = command[command.index("-show_entries") + 1]
    assert "sample_rate" in fields
    assert "channels" in fields
    assert command[-1] == str(audio_path)
