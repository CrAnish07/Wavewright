"""Orchestrates audio scanning, duplicate filtering, and manifest output."""

from pathlib import Path

from .audio_files import AUDIO_EXTENSIONS, is_decodable_audio, last_updated_file_from_duplicate
from .duplicates import check_duplicate_waves
from .metadata import create_audio_manifest

def scan_dataset(input_path: Path, output_path: Path) -> None:
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

        """Exclude files FFmpeg cannot fully decode before choosing keepers."""
        decodable_files = []
        corrupted_paths = []
        for path in all_audio_files:
            if is_decodable_audio(path):
                decodable_files.append(path)
            else:
                corrupted_paths.append(path)

        decodable_resolved = {path.resolve() for path in decodable_files}
        """ Build a set of duplicate paths to exclude """
        duplicate_paths = set()
        for paths in duplicates.values():
            valid_paths = [p for p in paths if p.resolve() in decodable_resolved]
            """Keep the last-updated decodable file, when available."""
            keeper = last_updated_file_from_duplicate(valid_paths) if valid_paths else None
            for p in valid_paths:
                if p.resolve() != keeper:
                    duplicate_paths.add(p.resolve())

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
