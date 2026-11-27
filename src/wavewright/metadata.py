"""FFprobe metadata collection and manifest writing."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

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
