import numpy as np
import pytest
from song.mix import stitch_clips, mix_backing, normalise_duration
from song.lyrics import SECTION_BREAKS

SAMPLE_RATE = 24000

def _make_clip(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def test_stitch_clips_concatenates():
    clips = [_make_clip(1.0), _make_clip(1.0), _make_clip(1.0)]
    result = stitch_clips(clips, SECTION_BREAKS, sample_rate=SAMPLE_RATE)
    # 3 clips × 1s + gaps > 3s
    assert len(result) > 3 * SAMPLE_RATE


def test_stitch_clips_returns_float32():
    clips = [_make_clip(0.5)]
    result = stitch_clips(clips, set(), sample_rate=SAMPLE_RATE)
    assert result.dtype == np.float32


def test_mix_backing_same_length():
    vocals = _make_clip(5.0)
    backing = _make_clip(5.0)
    result = mix_backing(vocals, backing, backing_volume=0.18)
    assert len(result) == len(vocals)


def test_mix_backing_volume_applied():
    vocals = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.5
    backing = np.ones(SAMPLE_RATE, dtype=np.float32) * 1.0
    result = mix_backing(vocals, backing, backing_volume=0.18)
    # Result should be louder than vocals alone but backing contribution small
    assert result.max() > 0.5
    assert result.max() <= 1.0  # clipped to [-1, 1]


def test_normalise_duration_short_audio_padded():
    short = _make_clip(100.0)  # 100s — under 140s minimum
    result = normalise_duration(short, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    assert len(result) >= int(140.0 * SAMPLE_RATE)


def test_normalise_duration_long_audio_trimmed():
    long = _make_clip(200.0)  # 200s — over 165s maximum
    result = normalise_duration(long, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    assert len(result) <= int(165.0 * SAMPLE_RATE) + SAMPLE_RATE  # within 1s tolerance
