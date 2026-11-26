"""Exact and acoustic duplicate detection."""

import hashlib
import os
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio_files import AUDIO_EXTENSIONS
from .fingerprints import (
    CHROMAPRINT_SIMILARITY_THRESHOLD, CHROMAPRINT_MAX_ALIGN_OFFSET,
    CHROMAPRINT_MAX_BIT_ERROR, CHROMAPRINT_MIN_FRAMES,
    _chromaprint_similarity, _decoded_feature_similarity,
    _decoded_signal_similarity, _read_chromaprint, _read_spectral_fingerprint,
)

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

    def fingerprint_file(path: Path) -> tuple[Path, tuple[float, str, list[int] | list[float], array | None, array | None] | None, str | None]:
        try:
            return path, _read_chromaprint(path), None
        except RuntimeError as error:
            try:
                return path, _read_spectral_fingerprint(path), None
            except RuntimeError as fallback_error:
                return path, None, f"{error}; fallback failed: {fallback_error}"

    with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as executor:
        fingerprint_results = list(executor.map(fingerprint_file, candidates))

    fingerprints: list[tuple[Path, float, str, list[int] | list[float], array | None, array | None]] = []
    failures = []
    for path, result, error in fingerprint_results:
        if result is None:
            failures.append(f"{path}: {error}")
        else:
            duration, fingerprint_type, fingerprint, reduced_signal, full_signal = result
            fingerprints.append((path, duration, fingerprint_type, fingerprint, reduced_signal, full_signal))
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
            first_duration, second_duration = representative[1], candidate[1]
            if first_duration <= 0 or second_duration <= 0:
                unmatched.append(candidate)
                continue
            duration_ratio = min(first_duration, second_duration) / max(
                first_duration, second_duration
            )
            if duration_ratio < 0.80:
                unmatched.append(candidate)
                continue
            if representative[2] != candidate[2]:
                unmatched.append(candidate)
                continue
            if representative[2] == "decoded":
                similarity = _decoded_feature_similarity(representative[3], candidate[3])
                verified = (
                    similarity >= 0.98
                    and representative[4] is not None
                    and candidate[4] is not None
                    and _decoded_signal_similarity(representative[4], candidate[4]) >= 0.92
                    and representative[5] is not None
                    and candidate[5] is not None
                    and _decoded_signal_similarity(representative[5], candidate[5]) >= 0.92
                )
            else:
                similarity = _chromaprint_similarity(representative[3], candidate[3])
                match_threshold = CHROMAPRINT_SIMILARITY_THRESHOLD
                verified = similarity >= match_threshold
            if verified:
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
