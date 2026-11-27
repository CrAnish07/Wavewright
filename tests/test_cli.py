import sys

import pytest

from wavewright.cli import main


def test_main_creates_requested_output_directory(tmp_path, monkeypatch):
    input_path = tmp_path / "audio"
    input_path.write_bytes(b"placeholder")
    output = tmp_path / "reports" / "run"
    monkeypatch.setattr(sys, "argv", ["wavewright", "--input", str(input_path), "--output", str(output)])

    main()

    assert output.is_dir()


def test_main_requires_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wavewright"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "following arguments are required: --input" in capsys.readouterr().err
