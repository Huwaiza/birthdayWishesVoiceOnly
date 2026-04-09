"""
Vocal generation via Bark (local, offline).

Loads the Bark model once and caches it in memory.
Each line of lyrics is generated separately for consistency.
Generated WAV files are cached to disk so individual lines
can be regenerated without re-running the whole song.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

CACHE_DIR = Path(__file__).parent.parent / "cache"
SAMPLE_RATE = 24000  # Bark's native sample rate

# Default speaker — warm female voice
DEFAULT_SPEAKER = "v2/en_speaker_9"

_model_cache: dict = {}


def _get_device() -> str:
    """Return 'mps', 'cuda', or 'cpu' depending on availability."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_models(speaker: str) -> tuple:
    """Load and cache Bark processor + model. Returns (processor, model)."""
    global _model_cache
    if "processor" in _model_cache:
        return _model_cache["processor"], _model_cache["model"]

    from transformers import AutoProcessor, BarkModel
    import torch

    device = _get_device()
    print(f"[vocal] Loading Bark model on {device} …")

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained(
        "suno/bark",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = model.to(device)

    _model_cache["processor"] = processor
    _model_cache["model"] = model
    _model_cache["device"] = device

    print(f"[vocal] Bark model ready.")
    return processor, model


def _line_cache_path(line: str, speaker: str) -> Path:
    """Return the cache file path for a given line + speaker combination."""
    key = hashlib.md5(f"{speaker}:{line}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}.wav"


def generate_line(
    line: str,
    speaker: str = DEFAULT_SPEAKER,
    force: bool = False,
) -> np.ndarray:
    """
    Generate audio for a single lyric line using Bark.

    Parameters
    ----------
    line    : The lyric line, e.g. "♪ Happy birthday to you ♪"
    speaker : Bark voice preset, e.g. "v2/en_speaker_9"
    force   : If True, bypass cache and regenerate

    Returns
    -------
    np.ndarray  shape (N,), float32, sample rate = SAMPLE_RATE
    """
    cache_path = _line_cache_path(line, speaker)

    if not force and cache_path.exists():
        audio, _ = sf.read(str(cache_path), dtype="float32")
        return audio

    processor, model = _load_models(speaker)
    import torch

    device = _model_cache["device"]
    inputs = processor(line, voice_preset=speaker)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs)

    audio = output.cpu().numpy().squeeze().astype(np.float32)

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
    regen_index  : If set, only this line index is regenerated (ignores cache)

    Returns
    -------
    List[np.ndarray]  one array per line, each float32 at SAMPLE_RATE
    """
    clips = []
    total = len(lines)
    for i, line in enumerate(lines):
        force = (regen_index is not None and i == regen_index)
        status = "regenerating" if force else "generating"
        cached = _line_cache_path(line, speaker).exists() and not force
        label = "cached" if cached else status
        print(f"[vocal] Line {i+1}/{total} ({label}): {line}")
        clips.append(generate_line(line, speaker=speaker, force=force))
    return clips
