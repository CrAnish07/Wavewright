# Wavewright

An open-source toolkit for cleaning, reviewing, and preparing audio datasets for AI training.

Wavewright aims to help teams turn raw recordings into organized, quality-checked datasets while keeping source files and processing history traceable.

## Planned features

- Audio format and quality checks
- Optional noise reduction
- Transcription and transcript review
- Duplicate detection
- Dataset export with source metadata

Wavewright is under active development. Features and workflows may change.

## Status

Early project — implementation has not started yet.



## Requirements

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg python3-venv
```

Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Pass either a directory or a ZIP archive:

```bash
python main.py --input ./audio_dataset --output ./dataset_report
python main.py --input ./audio_dataset.zip --output ./dataset_report
```

Optional `metadata.csv` can be placed anywhere inside the dataset. It should contain a `filename` or `path` column. Optional columns include `transcript`, `speaker_id`, and `language`.

Example:

```csv
filename,transcript,speaker_id,language
clip001.wav,Hello there,speaker_1,en
clip002.wav,How are you?,speaker_2,en
```

## What it checks

- Whether FFmpeg can read and decode the file
- Duration, codec, sample rate, channels, and channel layout
- Approximate silent-frame ratio
- Approximate clipped-frame ratio
- Exact duplicates using SHA-256
- Optional metadata row and transcript presence

After duplicate copies and undecodable files are removed, the scanner writes
one record per remaining audio file, in scan order, to `<output>/manifest.json`.
Each record includes a SHA-256 file ID, source path relative to the current
working directory (`$PWD`), file size and UTC modification time, plus a `probe` object containing
container metadata, audio stream details, and audio/video stream counts. Missing
FFprobe fields are recorded as `null` (or an empty tags object).

`PASS`, `REVIEW`, and `FAIL` are simple heuristics. They do not prove that a transcript is correct or that a recording is suitable for a particular model. Keep the original dataset unchanged.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests generate a small WAV in a temporary directory and verify the CLI
arguments and FFmpeg command construction without launching FFmpeg or FFprobe.

