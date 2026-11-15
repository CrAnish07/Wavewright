#!/usr/bin/env python3
"""Inspect an audio dataset and produce per-file and dataset-level reports.

Requires ffmpeg, ffprobe, numpy, and pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
    ".wma", ".aiff", ".aif", ".caf", ".webm", ".mp4",
}


def require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing system tools: " + ", ".join(missing)
            + ". Install them with: sudo apt install ffmpeg"
        )


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP while preventing paths from escaping destination."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,channel_layout,duration:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe could not read this file")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream found")

    stream = streams[0]
    duration_value = stream.get("duration") or data.get("format", {}).get("duration")
    if duration_value is None:
        raise RuntimeError("Could not determine audio duration")

    return {
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": int(stream["channels"]) if stream.get("channels") else None,
        "channel_layout": stream.get("channel_layout", "unknown"),
        "duration_seconds": float(duration_value),
    }


def inspect_signal(path: Path, channels: int, silence_level: float) -> dict[str, float]:
    """Decode in chunks, so the whole recording is not held in memory."""
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None

    frame_bytes = channels * 4
    chunk_size = max(frame_bytes, (256 * 1024 // frame_bytes) * frame_bytes)
    carry = b""
    total_frames = 0
    silent_frames = 0
    clipped_frames = 0
    peak = 0.0
    sum_squares = 0.0
    total_samples = 0

    try:
        while True:
            block = process.stdout.read(chunk_size)
            if not block:
                break
            block = carry + block
            usable_length = len(block) - (len(block) % frame_bytes)
            carry = block[usable_length:]
            if usable_length == 0:
                continue

            samples = np.frombuffer(block[:usable_length], dtype="<f4").reshape(-1, channels)
            # A frame is considered silent only if every channel is below the threshold.
            frame_peaks = np.max(np.abs(samples), axis=1)
            total_frames += int(samples.shape[0])
            silent_frames += int(np.count_nonzero(frame_peaks < silence_level))
            clipped_frames += int(np.count_nonzero(frame_peaks >= 0.999))
            if frame_peaks.size:
                peak = max(peak, float(np.max(frame_peaks)))
            sum_squares += float(np.sum(np.square(samples, dtype=np.float64)))
            total_samples += int(samples.size)
    finally:
        process.stdout.close()
        return_code = process.wait()

    if return_code != 0 or total_frames == 0:
        raise RuntimeError("FFmpeg could not decode audio samples")

    rms = (sum_squares / total_samples) ** 0.5 if total_samples else 0.0
    return {
        "silence_ratio": silent_frames / total_frames,
        "clipped_ratio": clipped_frames / total_frames,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
    }


def load_metadata(root: Path) -> tuple[dict[str, dict[str, str]], str | None]:
    candidates = list(root.rglob("metadata.csv"))
    if not candidates:
        return {}, None

    metadata_path = candidates[0]
    rows: dict[str, dict[str, str]] = {}
    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return {}, "metadata.csv is empty or has no header"
            normalized_fields = {name.strip().lower() for name in reader.fieldnames if name}
            key_column = "filename" if "filename" in normalized_fields else (
                "path" if "path" in normalized_fields else None
            )
            if key_column is None:
                return {}, "metadata.csv needs a 'filename' or 'path' column"
            for row in reader:
                cleaned = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
                key = cleaned.get(key_column, "").replace("\\", "/")
                if key:
                    rows[key] = cleaned
    except (OSError, UnicodeError, csv.Error) as error:
        return {}, f"Could not read metadata.csv: {error}"

    return rows, None


def find_metadata_row(
    relative_path: str, metadata: dict[str, dict[str, str]]
) -> dict[str, str] | None:
    normalized = relative_path.replace("\\", "/")
    return metadata.get(normalized) or metadata.get(Path(normalized).name)


def inspect_file(
    path: Path,
    root: Path,
    metadata: dict[str, dict[str, str]],
    metadata_error: str | None,
    silence_level: float,
    silence_review_ratio: float,
    clipping_review_ratio: float,
    hashes_seen: dict[str, str],
) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    issues: list[str] = []
    result: dict[str, Any] = {
        "filename": relative_path,
        "status": "PASS",
        "issues": "",
        "file_size_bytes": path.stat().st_size,
    }

    try:
        result["sha256"] = sha256_file(path)
        previous = hashes_seen.get(result["sha256"])
        if previous:
            issues.append(f"Exact duplicate of {previous}")
        else:
            hashes_seen[result["sha256"]] = relative_path

        info = probe_audio(path)
        result.update(info)
        signal = inspect_signal(path, info["channels"], silence_level)
        result.update(signal)

        if info["duration_seconds"] <= 0:
            issues.append("Duration is zero or negative")
        if signal["silence_ratio"] >= silence_review_ratio:
            issues.append(f"High silence: {signal['silence_ratio']:.1%}")
        if signal["clipped_ratio"] >= clipping_review_ratio:
            issues.append(f"Clipping detected: {signal['clipped_ratio']:.2%} of frames")
        if previous:
            result["status"] = "FAIL"
        elif issues:
            result["status"] = "REVIEW"

        if metadata_error:
            issues.append(metadata_error)
            if result["status"] == "PASS":
                result["status"] = "REVIEW"
        elif metadata:
            row = find_metadata_row(relative_path, metadata)
            if row is None:
                issues.append("No matching row in metadata.csv")
                if result["status"] == "PASS":
                    result["status"] = "REVIEW"
            else:
                result["transcript"] = row.get("transcript", "")
                result["speaker_id"] = row.get("speaker_id", "")
                result["language"] = row.get("language", "")
                if "transcript" in row and not row["transcript"]:
                    issues.append("Transcript is empty")
                    if result["status"] == "PASS":
                        result["status"] = "REVIEW"
    except Exception as error:
        result["status"] = "FAIL"
        issues.append(f"Could not inspect file: {error}")

    result["issues"] = "; ".join(issues)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an audio dataset and generate CSV/JSON quality reports."
    )
    parser.add_argument("input", help="Directory of audio files or a .zip archive")
    parser.add_argument("-o", "--output", default="dataset_report", help="Output directory")
    parser.add_argument("--silence-level", type=float, default=0.001,
                        help="Amplitude below which a frame counts as silent (default: 0.001, about -60 dBFS)")
    parser.add_argument("--silence-review-ratio", type=float, default=0.90,
                        help="Flag files at or above this silent-frame ratio (default: 0.90)")
    parser.add_argument("--clipping-review-ratio", type=float, default=0.001,
                        help="Flag files at or above this clipped-frame ratio (default: 0.001)")
    args = parser.parse_args()

    try:
        require_tools()
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            work_root = output_path / "_extracted_dataset"
            if work_root.exists():
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True)
            safe_extract_zip(input_path, work_root)
            dataset_root = work_root
        elif input_path.is_dir():
            dataset_root = input_path
        else:
            raise ValueError("Input must be an existing directory or a .zip file")

        audio_files = sorted(
            path for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            raise ValueError("No supported audio files found in the input")

        metadata, metadata_error = load_metadata(dataset_root)
        hashes_seen: dict[str, str] = {}
        records = []
        for index, path in enumerate(audio_files, start=1):
            print(f"[{index}/{len(audio_files)}] Inspecting {path.relative_to(dataset_root)}")
            records.append(inspect_file(
                path, dataset_root, metadata, metadata_error,
                args.silence_level, args.silence_review_ratio,
                args.clipping_review_ratio, hashes_seen,
            ))

        report = pd.DataFrame(records)
        csv_path = output_path / "quality_report.csv"
        report.to_csv(csv_path, index=False)

        counts = Counter(record["status"] for record in records)
        summary = {
            "total_files": len(records),
            "pass": counts.get("PASS", 0),
            "review": counts.get("REVIEW", 0),
            "fail": counts.get("FAIL", 0),
            "sample_rate_distribution": {
                str(key): int(value)
                for key, value in report["sample_rate"].value_counts(dropna=False).items()
            } if "sample_rate" in report else {},
            "channel_distribution": {
                str(key): int(value)
                for key, value in report["channels"].value_counts(dropna=False).items()
            } if "channels" in report else {},
            "note": "PASS/REVIEW/FAIL are heuristic checks, not a guarantee that audio is suitable for every model.",
        }
        summary_path = output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nDataset inspection complete")
        print(f"  Total:  {summary['total_files']}")
        print(f"  PASS:   {summary['pass']}")
        print(f"  REVIEW: {summary['review']}")
        print(f"  FAIL:   {summary['fail']}")
        print(f"\nReports saved in: {output_path}")
        print(f"  - {csv_path.name}")
        print(f"  - {summary_path.name}")
        return 0

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
