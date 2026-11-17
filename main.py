import argparse
import hashlib
import subprocess
import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

# Utilities
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


# Command 1: ffprobe (metadata)
def ffprobe_gen_json_metadata(path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,channel_layout,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]


# Command 2: ffmpeg (decode to raw float32)
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


# Audio extensions
AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


# Check duplicates audio waves via sha 256
def check_duplicate_waves(directory_path: str | Path) -> dict[str, list[Path]]:
    """Return groups of audio files with identical contents, keyed by SHA-256."""
    directory = Path(directory_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Find supported audio files recursively.
    audio_files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    # Nothing to hash -> nothing to return.
    if not audio_files:
        return {}

    # Derive worker count from CPU count (I/O-bound work) and cap by file count.
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    max_workers = max(1, min(max_workers, len(audio_files)))

    def hash_file(path: Path) -> tuple[str, Path]:
        """Calculate the SHA-256 hash of one audio file."""
        digest = hashlib.sha256()

        with path.open("rb") as audio_file:
            # Read in chunks to avoid loading entire audio files into memory.
            for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest(), path

    files_by_hash: dict[str, list[Path]] = {}

    # Hash multiple files concurrently using worker threads.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_hash, path in executor.map(hash_file, audio_files):
            files_by_hash.setdefault(file_hash, []).append(path)

    return {
        file_hash: paths
        for file_hash, paths in files_by_hash.items()
        if len(paths) > 1
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dataset_report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    """ Audio file set """
    audio_file_set = []

    if input_path.is_dir():
        """Returns duplicate audio waves dictionary with hash"""
        duplicates = check_duplicate_waves(input_path)
        if duplicates:
            print("Duplicate audio files found:")
            for digest, paths in duplicates.items():
                print(f"SHA-256 {digest}")
                for path in paths:
                    print(f" {path}")
        else:
            print(f"No duplicate audio files found in {input_path}")

        """ Collect ALL audio files recursively """
        all_audio_files = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]

        """ Build a set of duplicate paths to exclude """
        duplicate_paths = set()
        for paths in duplicates.values():
            """Keep the last-updated file, treat the rest as duplicates"""
            keeper = last_updated_file_from_duplicate(paths)
            for p in paths:
                if p.resolve() != keeper:
                    duplicate_paths.add(p.resolve())

        """Exclude duplicates and files that FFmpeg cannot fully decode."""
        decodable_files = []
        corrupted_paths = []
        for path in all_audio_files:
            if is_decodable_audio(path):
                decodable_files.append(path)
            else:
                corrupted_paths.append(path)

        audio_file_set = [
            p.resolve() for p in decodable_files if p.resolve() not in duplicate_paths
        ]

        print(f"\nTotal audio files found: {len(all_audio_files)}")
        print(f"Corrupted or undecodable files skipped: {len(corrupted_paths)}")
        for path in corrupted_paths:
            print(f"  {path}")
        print(f"Duplicate copies skipped: {len(duplicate_paths)}")
        print(f"Unique audio files in audio_file_set: {len(audio_file_set)}")
        for p in audio_file_set:
            print(f"  {p}")


if __name__ == "__main__":
    main()
