import pytest

from wavewright.duplicates import check_duplicate_waves


def test_check_duplicate_waves_finds_identical_files(tmp_path):
    content = b"same audio content"
    file1 = tmp_path / "audio1.wav"
    file2 = tmp_path / "audio2.wav"
    file1.write_bytes(content)
    file2.write_bytes(content)

    result = check_duplicate_waves(tmp_path)

    assert len(result) == 1
    assert list(result.values()) == [[file1, file2]]


def test_check_duplicate_waves_ignores_unique_files(tmp_path):
    (tmp_path / "audio1.wav").write_bytes(b"audio one")
    (tmp_path / "audio2.wav").write_bytes(b"audio two")

    assert check_duplicate_waves(tmp_path) == {}


def test_check_duplicate_waves_searches_nested_directories(tmp_path):
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
    (tmp_path / "file1.txt").write_bytes(b"same content")
    (tmp_path / "file2.txt").write_bytes(b"same content")

    assert check_duplicate_waves(tmp_path) == {}


def test_check_duplicate_waves_rejects_invalid_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        check_duplicate_waves(tmp_path / "does_not_exist")
