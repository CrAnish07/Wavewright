import argparse
import subprocess
from pathlib import Path

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
    parser.add_argument("--input", require=True)
    parser.add_argument("--output", default="dataset_report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True) 


if __name__ == "__main__":
    main()
