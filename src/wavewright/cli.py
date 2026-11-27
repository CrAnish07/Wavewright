"""Command-line interface for Wavewright."""

import argparse
from pathlib import Path

from .scanner import scan_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dataset_report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    scan_dataset(input_path, output_path)
