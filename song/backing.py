"""
Backing track loader and looper.

Loads an MP3 file, loops or trims it to an exact target duration,
and returns a float32 numpy array at the requested sample rate.
"""

from pathlib import Path

import numpy as np
import soundfile as sf


def load_and_loop(
    path: str,
    target_duration_s: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Load an audio file and loop/trim it to exactly target_duration_s seconds.

    Parameters
    ----------
    path              : Path to the MP3/WAV backing track
    target_duration_s : Desired output duration in seconds
    sample_rate       : Output sample rate (default 24000 to match Bark)

    Returns
    -------
    np.ndarray  shape (N,), float32, at `sample_rate`
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Backing track not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # Mix down to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    target_samples = int(target_duration_s * sample_rate)

    # Loop until long enough
    if len(audio) < target_samples:
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)

    # Trim to exact length
    audio = audio[:target_samples]

    return audio.astype(np.float32)
