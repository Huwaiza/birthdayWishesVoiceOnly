"""
Audio stitching, backing mix, and duration normalisation.
"""

from typing import List, Set

import numpy as np

SAMPLE_RATE = 24000

# Silence durations in seconds
_GAP_BETWEEN_LINES = 0.35       # between adjacent lines
_GAP_BETWEEN_SECTIONS = 1.0     # at section breaks (longer, more natural)
_INTRO_SILENCE = 4.0            # piano lead-in before first line


def _silence(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def stitch_clips(
    clips: List[np.ndarray],
    section_breaks: Set[int],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Concatenate vocal clips with natural gaps between them.

    Parameters
    ----------
    clips          : List of float32 numpy arrays (one per lyric line)
    section_breaks : Set of line indices that start a new section (longer gap before them)
    sample_rate    : Sample rate of all clips

    Returns
    -------
    np.ndarray  float32, single-channel, at sample_rate
    """
    parts = [_silence(_INTRO_SILENCE, sample_rate)]

    for i, clip in enumerate(clips):
        if i > 0:
            gap = _GAP_BETWEEN_SECTIONS if i in section_breaks else _GAP_BETWEEN_LINES
            parts.append(_silence(gap, sample_rate))
        parts.append(clip.astype(np.float32))

    return np.concatenate(parts)


def mix_backing(
    vocals: np.ndarray,
    backing: np.ndarray,
    backing_volume: float = 0.18,
) -> np.ndarray:
    """
    Mix backing track under vocals.

    Parameters
    ----------
    vocals         : float32 vocal array
    backing        : float32 backing array — must be same length as vocals
    backing_volume : Backing track gain (0.0–1.0). Default 0.18 = 18%

    Returns
    -------
    np.ndarray  float32, clipped to [-1.0, 1.0]
    """
    mixed = vocals + backing * backing_volume
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def normalise_duration(
    audio: np.ndarray,
    min_s: float = 140.0,
    max_s: float = 165.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Ensure audio falls within [min_s, max_s] seconds.

    - If too short: append silence at the end (song naturally ended early)
    - If too long: time-stretch to fit max_s using librosa

    Parameters
    ----------
    audio       : float32 numpy array
    min_s       : Minimum acceptable duration in seconds (default 140 = 2:20)
    max_s       : Maximum acceptable duration in seconds (default 165 = 2:45)
    sample_rate : Sample rate of audio

    Returns
    -------
    np.ndarray  float32
    """
    duration_s = len(audio) / sample_rate

    if duration_s < min_s:
        pad = _silence(min_s - duration_s, sample_rate)
        return np.concatenate([audio, pad])

    if duration_s > max_s:
        import librosa
        rate = duration_s / max_s  # > 1.0 means speed up
        audio = librosa.effects.time_stretch(audio, rate=rate)
        return audio[:int(max_s * sample_rate)].astype(np.float32)

    return audio
