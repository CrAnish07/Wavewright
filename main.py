import argparse
import hashlib
import json
import subprocess
import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

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
        "-show_streams",
        "-show_format",
        "-show_entries",
        "stream=index,codec_type,codec_name,codec_long_name,profile,sample_fmt,sample_rate,channels,channel_layout,bits_per_raw_sample,bit_rate,duration:stream_disposition=default,dub,original,comment,lyrics,karaoke,forced,hearing_impaired,visual_impaired,clean_effects,attached_pic,timed_thumbnails,captions,descriptions,metadata,dependent,still_image:stream_tags:format=format_name,format_long_name,duration,size,bit_rate:format_tags",
        "-of",
        "json",
        str(path),
    ]


def probe_audio_metadata(path: Path) -> dict:
    """Read audio stream and container metadata from one decodable audio file."""
    try:
        result = subprocess.run(
            ffprobe_gen_json_metadata(path),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("FFprobe is required to collect audio metadata.") from None

    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed for {path}: {result.stderr.strip()}")

    probe_data = json.loads(result.stdout)
    streams = probe_data.get("streams", [])
    audio_format = probe_data.get("format", {})
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]

    return {
        "format": {
            "format_name": audio_format.get("format_name"),
            "format_long_name": audio_format.get("format_long_name"),
            "duration": audio_format.get("duration"),
            "size": audio_format.get("size"),
            "bit_rate": audio_format.get("bit_rate"),
            "tags": audio_format.get("tags", {}),
        },
        "audio_streams": [
            {
                "index": stream.get("index"),
                "codec_type": stream.get("codec_type"),
                "codec_name": stream.get("codec_name"),
                "codec_long_name": stream.get("codec_long_name"),
                "profile": stream.get("profile"),
                "sample_fmt": stream.get("sample_fmt"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
                "channel_layout": stream.get("channel_layout"),
                "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
                "bit_rate": stream.get("bit_rate"),
                "duration": stream.get("duration"),
                "disposition": stream.get("disposition", {}),
                "tags": stream.get("tags", {}),
            }
            for stream in audio_streams
        ],
        "audio_stream_count": len(audio_streams),
        "video_stream_count": sum(1 for stream in streams if stream.get("codec_type") == "video"),
    }


def create_audio_manifest(audio_files: list[Path], source_root: Path, output_path: Path) -> Path:
    """Probe the final audio file list and save its metadata as JSON."""
    records = []
    for audio_path in audio_files:
        stat = audio_path.stat()
        digest = hashlib.sha256()
        with audio_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)

        try:
            source_path = audio_path.relative_to(source_root).as_posix()
        except ValueError:
            source_path = str(audio_path)

        record = {
            "file_id": f"sha256:{digest.hexdigest()}",
            "source_path": source_path,
            "source_size_bytes": stat.st_size,
            "source_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
            "probe": probe_audio_metadata(audio_path),
        }
        records.append(record)

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


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

        manifest_path = create_audio_manifest(audio_file_set, Path.cwd().resolve(), output_path)
        print(f"\nAudio metadata manifest written to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
