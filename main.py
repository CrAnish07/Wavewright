import argparse
import hashlib
import subprocess
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".aac", ".aiff", ".alac", ".flac", ".m4a", ".mp3", ".ogg",
    ".opus", ".wav", ".wma",
}


def check_duplicate_waves(directory_path: str | Path) -> dict[str, list[Path]]:
    """Return groups of audio files with identical contents, keyed by SHA-256."""
    directory = Path(directory_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files_by_hash: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        digest = hashlib.sha256()
        with path.open("rb") as audio_file:
            for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                digest.update(chunk)
        files_by_hash.setdefault(digest.hexdigest(), []).append(path)

    return {
        digest: paths
        for digest, paths in files_by_hash.items()
        if len(paths) > 1
    }

# Command 1: ffprobe (metadata)
def ffprobe_gen_json_metadata(path: Path) -> list[str]:
    return [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,channel_layout,duration:format=duration",
        "-of", "json", str(path),
    ]


# Command 2: ffmpeg (decode to raw float32)
def ffmpeg_decode_to_raw_float32(path: Path) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dataset_report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True) 

    if input_path.is_dir():
        duplicates = check_duplicate_waves(input_path)
        if duplicates:
            print("Duplicate audio files found:")
            for digest, paths in duplicates.items():
                print(f"SHA-256 {digest}")
                for path in paths:
                    print(f"  {path}")
        else:
            print(f"No duplicate audio files found in {input_path}")


if __name__ == "__main__":
    main()
