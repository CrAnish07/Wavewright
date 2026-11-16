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


def test_check_duplicate_waves_finds_identical_files(tmp_path):
    from main import check_duplicate_waves

    content = b"same audio content"

    file1 = tmp_path / "audio1.wav"
    file2 = tmp_path / "audio2.wav"

    file1.write_bytes(content)
    file2.write_bytes(content)

    result = check_duplicate_waves(tmp_path)

    assert len(result) == 1
    assert list(result.values()) == [[file1, file2]]


def test_check_duplicate_waves_ignores_unique_files(tmp_path):
    from main import check_duplicate_waves

    (tmp_path / "audio1.wav").write_bytes(b"audio one")
    (tmp_path / "audio2.wav").write_bytes(b"audio two")

    result = check_duplicate_waves(tmp_path)

    assert result == {}


def test_check_duplicate_waves_searches_nested_directories(tmp_path):
    from main import check_duplicate_waves

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    file1 = tmp_path / "audio1.wav"
    file2 = nested_dir / "audio2.mp3"

    file1.write_bytes(b"identical audio")
    file2.write_bytes(b"identical audio")

    result = check_duplicate_waves(tmp_path)

    assert len(result) == 1
    assert list(result.values()) == [[file1, file2]]


def test_check_duplicate_waves_ignores_unsupported_extensions(tmp_path):
    from main import check_duplicate_waves

    (tmp_path / "file1.txt").write_bytes(b"same content")
    (tmp_path / "file2.txt").write_bytes(b"same content")

    result = check_duplicate_waves(tmp_path)

    assert result == {}


def test_check_duplicate_waves_rejects_invalid_directory(tmp_path):
    import pytest
    from main import check_duplicate_waves

    invalid_path = tmp_path / "does_not_exist"

    with pytest.raises(NotADirectoryError):
        check_duplicate_waves(invalid_path)
