import json
import subprocess
from array import array
from types import SimpleNamespace

from wavewright import duplicates, fingerprints


def test_fallback_uses_full_duration_separate_from_capped_decode(tmp_path, monkeypatch):
    path = tmp_path / "long.wav"
    samples = array("f", [0.25] * 2048).tobytes()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffmpeg":
            return SimpleNamespace(returncode=0, stdout=samples, stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "audio", "duration": "240.0"}]}),
            stderr="",
        )

    monkeypatch.setattr(fingerprints.subprocess, "run", fake_run)

    duration, fingerprint_type, _, _, waveform = fingerprints._read_spectral_fingerprint(path)

    assert duration == 240.0
    assert fingerprint_type == "decoded"
    assert len(waveform) == 2048
    assert "120" in calls[0]
    assert calls[1][0] == "ffprobe"


def test_unknown_full_duration_does_not_exclude_matching_fallback_files(tmp_path, monkeypatch):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first audio bytes")
    second.write_bytes(b"second audio bytes")

    def no_chromaprint(_path):
        raise RuntimeError("no Chromaprint fingerprint")

    def same_decoded_fingerprint(_path):
        return 0.0, "decoded", [1.0] * 40, array("f", [0.25] * 512), array("f", [0.25] * 2048)

    monkeypatch.setattr(duplicates, "_read_chromaprint", no_chromaprint)
    monkeypatch.setattr(duplicates, "_read_spectral_fingerprint", same_decoded_fingerprint)

    result = duplicates.check_duplicate_waves(tmp_path)

    assert list(result.values()) == [[first, second]]


def test_unavailable_ffprobe_duration_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="ffprobe failed"),
    )

    assert fingerprints._measure_full_duration(tmp_path / "missing.wav") == 0.0
