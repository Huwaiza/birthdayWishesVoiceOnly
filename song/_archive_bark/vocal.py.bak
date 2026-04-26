"""
Vocal generation via native Bark library (local, offline).

Uses bark.generate_audio() directly — better singing quality than
the transformers BarkModel wrapper, responds properly to [singing] tags.

Each line is generated separately for consistency and cached to disk.
"""

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

# Suppress noisy bark/transformers output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bark").setLevel(logging.ERROR)
os.environ.setdefault("SUNO_OFFLOAD_CPU", "0")   # keep on MPS/GPU
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "0")  # full quality
warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent.parent / "cache"
SAMPLE_RATE = 24000  # Bark's native sample rate

# Default speaker — warm female voice
DEFAULT_SPEAKER = "v2/en_speaker_9"

_models_loaded = False


def _ensure_models_loaded() -> None:
    """Load Bark models once into memory (cached globally by bark library)."""
    global _models_loaded
    if _models_loaded:
        return
    from bark import preload_models
    print("[vocal] Loading Bark models (first run only) …")
    preload_models(
        text_use_small=False,
        coarse_use_small=False,
        fine_use_gpu=True,
        codec_use_gpu=True,
    )
    _models_loaded = True
    print("[vocal] Bark models ready.")


def _line_cache_path(line: str, speaker: str) -> Path:
    """Cache key = MD5 of 'speaker:line' so different speakers don't collide."""
    key = hashlib.md5(f"{speaker}:{line}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}.wav"


def generate_line(
    line: str,
    speaker: str = DEFAULT_SPEAKER,
    force: bool = False,
) -> np.ndarray:
    """
    Generate audio for a single lyric line using native Bark.

    Parameters
    ----------
    line    : Lyric line already wrapped in singing prompt format
    speaker : Bark voice preset string, e.g. "v2/en_speaker_9"
    force   : If True, bypass cache and regenerate

    Returns
    -------
    np.ndarray  float32, shape (N,), at SAMPLE_RATE (24000 Hz)
    """
    cache_path = _line_cache_path(line, speaker)

    if not force and cache_path.exists():
        audio, _ = sf.read(str(cache_path), dtype="float32")
        return audio

    _ensure_models_loaded()
    from bark import generate_audio

    audio = generate_audio(
        line,
        history_prompt=speaker,
        text_temp=0.6,       # lower = more consistent melody
        waveform_temp=0.7,   # controls expressiveness
        silent=True,
    )

    audio = audio.astype(np.float32)
    # Normalise to [-1, 1] to avoid clipping when mixed
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    CACHE_DIR.mkdir(exist_ok=True)
    sf.write(str(cache_path), audio, SAMPLE_RATE)

    return audio


def generate_all_lines(
    lines: List[str],
    speaker: str = DEFAULT_SPEAKER,
    regen_index: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate audio for every line in the lyrics list.

    Parameters
    ----------
    lines        : Output of song.lyrics.build_lines()
    speaker      : Bark voice preset
    regen_index  : If set, only this line index is force-regenerated

    Returns
    -------
    List[np.ndarray]  one float32 array per line at SAMPLE_RATE
    """
    clips = []
    total = len(lines)
    for i, line in enumerate(lines):
        force = (regen_index is not None and i == regen_index)
        cached = _line_cache_path(line, speaker).exists() and not force
        label = "cached" if cached else "generating"
        print(f"[vocal] Line {i+1}/{total} ({label}): {line}")
        clips.append(generate_line(line, speaker=speaker, force=force))
    return clips
