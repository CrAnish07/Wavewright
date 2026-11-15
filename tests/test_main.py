import sys
import wave

import pytest

import main


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00\x00\x40\x00\xc0")
    return path


def test_probe_requests_first_audio_stream_as_json(audio_path):
    command = main.ffprobe_gen_json_metadata(audio_path)

    assert command[0] == "ffprobe"
    assert command[command.index("-select_streams") + 1] == "a:0"
    assert command[command.index("-of") + 1] == "json"
    fields = command[command.index("-show_entries") + 1]
    assert "sample_rate" in fields
    assert "channels" in fields
    assert command[-1] == str(audio_path)


def test_decode_maps_first_audio_stream_to_float32_stdout(audio_path):
    command = main.ffmpeg_decode_to_raw_float32(audio_path)

    assert command[0] == "ffmpeg"
    assert "-nostdin" in command
    assert command[command.index("-i") + 1] == str(audio_path)
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-f") + 1] == "f32le"
    assert command[command.index("-acodec") + 1] == "pcm_f32le"
    assert command[-1] == "pipe:1"


def test_main_creates_requested_output_directory(audio_path, tmp_path, monkeypatch):
    output = tmp_path / "reports" / "run"
    monkeypatch.setattr(sys, "argv", ["main.py", "--input", str(audio_path), "--output", str(output)])

    main.main()

    assert output.is_dir()


def test_main_requires_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as error:
        main.main()

    assert error.value.code == 2
    assert "following arguments are required: --input" in capsys.readouterr().err
