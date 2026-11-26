"""Compatibility launcher; implementation lives in the wavewright package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wavewright.audio_files import (
    AUDIO_EXTENSIONS, ffmpeg_decode_to_raw_float32, is_decodable_audio,
    last_updated_file_from_duplicate,
)
from wavewright.cli import main
from wavewright.duplicates import check_duplicate_waves
from wavewright.fingerprints import (
    CHROMAPRINT_MAX_ALIGN_OFFSET, CHROMAPRINT_MAX_BIT_ERROR,
    CHROMAPRINT_MIN_FRAMES, CHROMAPRINT_SIMILARITY_THRESHOLD,
    _chromaprint_similarity, _decoded_feature_similarity,
    _decoded_signal_similarity, _read_chromaprint, _read_spectral_fingerprint,
    chromaprint_command,
)
from wavewright.metadata import (
    create_audio_manifest, ffprobe_gen_json_metadata, probe_audio_metadata,
)

if __name__ == "__main__":
    main()
