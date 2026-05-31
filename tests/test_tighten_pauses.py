"""
Tests for song.vocal_isolate.tighten_pauses — the melody-safe gap shrinker.

These build synthetic WAVs (tones + silence) with pydub, so they need no
model and no ffmpeg (WAV read/write is native in pydub). They prove the two
properties that matter:

  1. Long silent gaps (the leftover instrumental space) get collapsed.
  2. Short pauses AND every bit of actual tone/melody are preserved — the
     thing that went wrong in the earlier, too-aggressive version.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pydub = pytest.importorskip("pydub")
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.silence import detect_nonsilent

from song.vocal_isolate import (
    tighten_pauses, _LONG_GAP_MS, _KEPT_GAP_MS, _SILENCE_DBFS,
)


def _tone(ms, gain_db=-3):
    # A loud tone (~-3 dBFS), well above the -55 dBFS silence floor.
    return Sine(440).to_audio_segment(duration=ms).apply_gain(gain_db)


def _write(seg, path):
    seg.export(str(path), format="wav")
    return path


def _voiced_ms(seg):
    """Total non-silent (above the silence floor) duration in ms."""
    return sum(e - s for s, e in detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=_SILENCE_DBFS))


def test_long_gap_is_collapsed(tmp_path):
    audio = _tone(800) + AudioSegment.silent(duration=3000) + _tone(800)
    src = _write(audio, tmp_path / "in.wav")
    out = tmp_path / "out.wav"

    tighten_pauses(src, out, verbose=False)
    res = AudioSegment.from_wav(str(out))

    # The 3 s gap should be gone — output meaningfully shorter than original
    # (the collapsed gap itself is asserted precisely below).
    assert len(res) < len(audio) - 1000
    # Both tones survive as two distinct phrase groups.
    groups = detect_nonsilent(res, min_silence_len=200, silence_thresh=_SILENCE_DBFS)
    assert len(groups) == 2
    # The collapsed gap is roughly one breath, not zero and not the old 3 s.
    gap_between = groups[1][0] - groups[0][1]
    assert _KEPT_GAP_MS * 0.5 < gap_between < _KEPT_GAP_MS * 2


def test_melody_is_never_clipped(tmp_path):
    audio = _tone(800) + AudioSegment.silent(duration=3000) + _tone(800)
    src = _write(audio, tmp_path / "in.wav")
    out = tmp_path / "out.wav"

    tighten_pauses(src, out, verbose=False)
    res = AudioSegment.from_wav(str(out))

    # Essentially all of the original tone (~1600 ms) must remain. Padding
    # may add a touch; the point is we never *lose* voiced audio.
    before, after = _voiced_ms(audio), _voiced_ms(res)
    assert after >= before - 50


def test_short_gap_is_preserved(tmp_path):
    # A sub-_LONG_GAP_MS pause is natural phrasing and must be left intact.
    short = _LONG_GAP_MS - 400
    audio = _tone(800) + AudioSegment.silent(duration=short) + _tone(800)
    src = _write(audio, tmp_path / "in.wav")
    out = tmp_path / "out.wav"

    tighten_pauses(src, out, verbose=False)
    res = AudioSegment.from_wav(str(out))

    # The short gap is still there (still splits into two groups at a low
    # min_silence_len) and was NOT shrunk away.
    groups = detect_nonsilent(res, min_silence_len=200, silence_thresh=_SILENCE_DBFS)
    assert len(groups) == 2
    assert groups[1][0] - groups[0][1] >= short - 100


def test_all_silence_is_copied_through(tmp_path):
    audio = AudioSegment.silent(duration=2000)
    src = _write(audio, tmp_path / "in.wav")
    out = tmp_path / "out.wav"

    result = tighten_pauses(src, out, verbose=False)
    assert Path(result).exists()
    # No crash, and we didn't emit an empty file.
    assert len(AudioSegment.from_wav(str(out))) > 0


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tighten_pauses(tmp_path / "nope.wav", tmp_path / "out.wav", verbose=False)
