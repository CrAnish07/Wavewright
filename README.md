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
python checker.py ./audio_dataset --output ./dataset_report
python checker.py ./audio_dataset.zip --output ./dataset_report
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

`PASS`, `REVIEW`, and `FAIL` are simple heuristics. They do not prove that a transcript is correct or that a recording is suitable for a particular model. Keep the original dataset unchanged.

