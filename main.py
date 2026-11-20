import argparse
import hashlib
import json
import subprocess
import os
import math
from array import array

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


# Acoustic match settings. The frame comparison tolerates small encoding noise
# and alignment shifts while requiring a strong overall match.
CHROMAPRINT_SIMILARITY_THRESHOLD = 0.80
CHROMAPRINT_MAX_BIT_ERROR = 2
CHROMAPRINT_MAX_ALIGN_OFFSET = 120
CHROMAPRINT_MIN_FRAMES = 15


def chromaprint_command(path: Path) -> list[str]:
    return ["fpcalc", "-raw", "-json", str(path)]


def _read_chromaprint(path: Path) -> tuple[float, list[int]]:
    """Generate a raw Chromaprint fingerprint using the fpcalc executable."""
    try:
        result = subprocess.run(
            chromaprint_command(path), capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        raise RuntimeError("Chromaprint fpcalc is required for acoustic deduplication.") from None
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"fpcalc failed for {path}")
    try:
        data = json.loads(result.stdout)
        duration = float(data["duration"])
        fingerprint = [int(value) for value in data["fingerprint"]]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse fpcalc output for {path}: {error}") from error
    if not fingerprint:
        raise RuntimeError(f"fpcalc returned an empty fingerprint for {path}")
    return duration, fingerprint


def _read_spectral_fingerprint(path: Path) -> tuple[float, list[float]]:
    """Build a small decoded-audio fingerprint when Chromaprint has no result.

    This handles very short or synthetic material (for example pure test tones)
    that Chromaprint intentionally leaves fingerprintless.
    """
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:a:0", "-t", "120", "-ac", "1", "-ar", "11025",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is required for the short-audio fallback.") from None
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or f"FFmpeg failed for {path}")

    samples = array("f")
    samples.frombytes(result.stdout)
    if len(samples) < 2048:
        raise RuntimeError(f"Not enough decoded audio for a fallback fingerprint: {path}")
    duration = len(samples) / 11025
    # Compare temporal RMS and zero-crossing summaries. These remain stable
    # across lossless and lossy encodings for short/simple signals.
    segment_count = 20
    raw_features: list[tuple[float, float]] = []
    for index in range(segment_count):
        start = index * len(samples) // segment_count
        end = (index + 1) * len(samples) // segment_count
        segment = samples[start:end]
        if not segment:
            raw_features.append((0.0, 0.0))
            continue
        energy = sum(float(value) * float(value) for value in segment) / len(segment)
        crossings = sum(
            1 for left, right in zip(segment, segment[1:])
            if (left < 0 <= right) or (right < 0 <= left)
        )
        raw_features.append((math.sqrt(energy), crossings / max(1, len(segment) - 1)))
    rms_scale = max((rms for rms, _ in raw_features), default=0.0)
    feature = []
    for rms, crossings in raw_features:
        feature.extend((rms / rms_scale if rms_scale else 0.0, crossings))
    return duration, feature


def _chromaprint_similarity(first: list[int] | list[float], second: list[int] | list[float]) -> float:
    """Return an alignment-tolerant Chromaprint frame similarity score."""
    if len(first) == 40 and len(second) == 40:
        first_crossing_rate = sum(first[1::2]) / 20
        second_crossing_rate = sum(second[1::2]) / 20
        crossing_scale = max(first_crossing_rate, second_crossing_rate, 1e-9)
        # A normalized RMS profile alone makes different steady tones look
        # alike. Keep a tight frequency-content gate for this fallback path.
        if abs(first_crossing_rate - second_crossing_rate) / crossing_scale > 0.08:
            return 0.0
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm == 0.0 or second_norm == 0.0:
            return 1.0 if first_norm == second_norm else 0.0
        dot_product = sum(left * right for left, right in zip(first, second))
        return dot_product / (first_norm * second_norm)
    if min(len(first), len(second)) < CHROMAPRINT_MIN_FRAMES:
        return 0.0
    counts: dict[int, int] = {}
    for i, first_frame in enumerate(first):
        start = max(0, i - CHROMAPRINT_MAX_ALIGN_OFFSET)
        end = min(len(second), i + CHROMAPRINT_MAX_ALIGN_OFFSET + 1)
        for j in range(start, end):
            if (first_frame ^ second[j]).bit_count() <= CHROMAPRINT_MAX_BIT_ERROR:
                offset = i - j
                counts[offset] = counts.get(offset, 0) + 1
    return max(counts.values(), default=0) / min(len(first), len(second))


# Check duplicates through an exact hash pass followed by acoustic matching.
def check_duplicate_waves(directory_path: str | Path) -> dict[str, list[Path]]:
    """Return exact and acoustically matching audio groups."""
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

    duplicates: dict[str, list[Path]] = {
        f"sha256:{file_hash}": paths
        for file_hash, paths in files_by_hash.items()
        if len(paths) > 1
    }

    # Fingerprint one representative per SHA group, including representatives
    # from exact-duplicate groups so cross-format copies can match those groups.
    candidates = [paths[0] for paths in files_by_hash.values()]
    hash_by_representative = {
        paths[0]: file_hash for file_hash, paths in files_by_hash.items()
    }
    if not candidates:
        return duplicates

    def fingerprint_file(path: Path) -> tuple[Path, tuple[float, list[int] | list[float]] | None, str | None]:
        try:
            return path, _read_chromaprint(path), None
        except RuntimeError as error:
            try:
                return path, _read_spectral_fingerprint(path), None
            except RuntimeError as fallback_error:
                return path, None, f"{error}; fallback failed: {fallback_error}"

    with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as executor:
        fingerprint_results = list(executor.map(fingerprint_file, candidates))

    fingerprints: list[tuple[Path, float, list[int] | list[float]]] = []
    failures = []
    for path, result, error in fingerprint_results:
        if result is None:
            failures.append(f"{path}: {error}")
        else:
            duration, fingerprint = result
            fingerprints.append((path, duration, fingerprint))
    if failures:
        print("Warning: acoustic deduplication skipped for some files:")
        for failure in failures:
            print(f"  {failure}")

    # Compare candidates to group representatives; this avoids transitive
    # chaining that could combine files which do not directly match.
    group_number = 0
    remaining = list(fingerprints)
    while remaining:
        representative = remaining.pop(0)
        group = [representative[0]]
        unmatched = []
        for candidate in remaining:
            duration_ratio = min(representative[1], candidate[1]) / max(
                representative[1], candidate[1], 1.0
            )
            similarity = _chromaprint_similarity(representative[2], candidate[2])
            if len(representative[2]) == 40:
                match_threshold = 0.98
            else:
                match_threshold = CHROMAPRINT_SIMILARITY_THRESHOLD
            if duration_ratio >= 0.80 and similarity >= match_threshold:
                group.append(candidate[0])
            else:
                unmatched.append(candidate)
        remaining = unmatched
        if len(group) > 1:
            group_number += 1
            grouped_hashes = {hash_by_representative[path] for path in group}
            expanded_group = [
                path
                for file_hash in grouped_hashes
                for path in files_by_hash[file_hash]
            ]
            # Replace overlapping SHA groups with one unified duplicate group
            # so the keeper selection sees every copy exactly once.
            for file_hash in grouped_hashes:
                duplicates.pop(f"sha256:{file_hash}", None)
            duplicates[f"chromaprint:{group_number}"] = sorted(expanded_group)

    return duplicates



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
                print(f"{digest}")
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
