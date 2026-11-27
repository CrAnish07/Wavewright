"""Audio file discovery and decode validation."""

import subprocess
from datetime import datetime
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".aac", ".aiff", ".alac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
}

def last_updated_file_from_duplicate(path_list: list[str | Path]) -> Path | None:
    """Convert strings to Path objects and ensure they exist and are files"""
    files = [Path(f) for f in path_list if Path(f).is_file()]

    if not files:
        print("No valid files found in the list.")
        return None

    """ Find the file with the highest st_mtime (last modified timestamp) """
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    """ Convert the timestamp to a readable date/time """
    mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)

    """ .resolve() gets the full absolute path """
    full_path = latest_file.resolve()

    print(f"Latest file: {full_path}")
    print(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return full_path


def ffmpeg_decode_to_raw_float32(path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]


def is_decodable_audio(path: Path) -> bool:
    """Return whether FFmpeg can decode the first audio stream without errors."""
    try:
        result = subprocess.run(
            ffmpeg_decode_to_raw_float32(path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is required to validate audio files.") from None

    return result.returncode == 0
