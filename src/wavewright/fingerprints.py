"""Chromaprint and decoded-audio fingerprint helpers."""

import json
import math
import subprocess
import sys
from array import array
from pathlib import Path

CHROMAPRINT_SIMILARITY_THRESHOLD = 0.80
CHROMAPRINT_MAX_BIT_ERROR = 2
CHROMAPRINT_MAX_ALIGN_OFFSET = 120
CHROMAPRINT_MIN_FRAMES = 15

def chromaprint_command(path: Path) -> list[str]:
    return ["fpcalc", "-raw", "-json", "-length", "0", str(path.resolve())]


def _read_chromaprint(path: Path) -> tuple[float, str, list[int], array | None, array | None]:
    """Generate a raw Chromaprint fingerprint using the fpcalc executable."""
    try:
        result = subprocess.run(
            chromaprint_command(path), capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, PermissionError):
        raise RuntimeError("Chromaprint fpcalc is required for acoustic deduplication.") from None
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"fpcalc failed for {path}")
    try:
        data = json.loads(result.stdout)
        duration = float(data["duration"])
        fingerprint = [int(value) for value in data["fingerprint"]]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse fpcalc output for {path}: {error}") from error
    if len(fingerprint) < CHROMAPRINT_MIN_FRAMES:
        raise RuntimeError(f"fpcalc returned a fingerprint that is too short for {path}")
    return duration, "chromaprint", fingerprint, None, None


def _read_spectral_fingerprint(path: Path) -> tuple[float, str, list[float], array, array]:
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
    if sys.byteorder == "big":
        samples.byteswap()
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
    # Keep a lightly downsampled waveform for a direct similarity check after
    # the coarse RMS/zero-crossing features identify a possible match.
    return duration, "decoded", feature, samples[::8], samples


def _chromaprint_similarity(first: list[int], second: list[int]) -> float:
    """Return an alignment-tolerant Chromaprint frame similarity score."""
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


def _decoded_feature_similarity(first: list[float], second: list[float]) -> float:
    """Compare the normalized coarse features used to shortlist fallback pairs."""
    first_crossing_rate = sum(first[1::2]) / 20
    second_crossing_rate = sum(second[1::2]) / 20
    crossing_scale = max(first_crossing_rate, second_crossing_rate, 1e-9)
    if abs(first_crossing_rate - second_crossing_rate) / crossing_scale > 0.08:
        return 0.0
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 1.0 if first_norm == second_norm else 0.0
    return sum(left * right for left, right in zip(first, second)) / (first_norm * second_norm)


def _decoded_signal_similarity(first: array, second: array) -> float:
    """Verify fallback matches against aligned decoded samples."""
    overlap_size = min(len(first), len(second))
    if overlap_size < 256:
        return 0.0
    max_lag = min(138, max(0, overlap_size // 5))
    best = 0.0
    lags = list(range(-max_lag, max_lag + 1, 4))
    if 0 not in lags:
        lags.append(0)
    for lag in sorted(lags):
        start_a, start_b = max(0, lag), max(0, -lag)
        size = min(len(first) - start_a, len(second) - start_b)
        if size < overlap_size * 0.8:
            continue
        dot = sum(first[start_a + i] * second[start_b + i] for i in range(size))
        norm_a = math.sqrt(sum(first[start_a + i] ** 2 for i in range(size)))
        norm_b = math.sqrt(sum(second[start_b + i] ** 2 for i in range(size)))
        if norm_a and norm_b:
            best = max(best, dot / (norm_a * norm_b))
        elif norm_a == norm_b:
            best = max(best, 1.0)
    return best
