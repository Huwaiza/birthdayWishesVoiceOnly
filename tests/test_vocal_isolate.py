"""Tests for song.vocal_isolate — Roformer isolation + pause tightening.

These tests do NOT require audio-separator to be installed: isolate_vocals
imports it lazily, only after input validation has run. tighten_pauses
needs pydub (a core project dependency); the tests that exercise its audio
path skip cleanly if pydub happens to be unavailable.
"""

import math
import struct
import wave

import pytest

from song.vocal_isolate import isolate_vocals, tighten_pauses


# --- isolate_vocals ---------------------------------------------------------

def test_isolate_vocals_is_callable():
    assert callable(isolate_vocals)


def test_isolate_vocals_missing_input_raises(tmp_path):
    missing = tmp_path / "nope.wav"
    out = tmp_path / "vocals.wav"
    with pytest.raises(FileNotFoundError):
        isolate_vocals(missing, out)


def test_isolate_vocals_accepts_str_paths(tmp_path):
    """Paths may be passed as plain strings — still validates cleanly."""
    with pytest.raises(FileNotFoundError):
        isolate_vocals(str(tmp_path / "nope.wav"), str(tmp_path / "out.wav"))


# --- tighten_pauses ---------------------------------------------------------

def _write_wav(path, segments, rate=16000):
    """Write a 16-bit mono WAV from (duration_s, amplitude) segments.

    amplitude 0 → silence; otherwise a 220 Hz tone at that amplitude.
    """
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for dur, amp in segments:
            for i in range(int(dur * rate)):
                v = int(amp * math.sin(2 * math.pi * 220 * i / rate)) if amp else 0
                frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))


def _wav_seconds(path):
    with wave.open(str(path)) as f:
        return f.getnframes() / float(f.getframerate())


def test_tighten_pauses_is_callable():
    assert callable(tighten_pauses)


def test_tighten_pauses_missing_input_raises(tmp_path):
    """Input validation runs before pydub is imported — testable always."""
    with pytest.raises(FileNotFoundError):
        tighten_pauses(tmp_path / "nope.wav", tmp_path / "out.wav")


def test_tighten_pauses_collapses_long_gap(tmp_path):
    pytest.importorskip("pydub")
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    # 0.5 s sung + 2.5 s instrumental gap + 0.5 s sung = 3.5 s total.
    _write_wav(src, [(0.5, 12000), (2.5, 0), (0.5, 12000)])
    result = tighten_pauses(src, out, verbose=False)
    assert result == out and out.exists()
    in_dur, out_dur = _wav_seconds(src), _wav_seconds(out)
    # The 2.5 s gap must be collapsed to a short breath...
    assert out_dur < in_dur - 1.0
    assert out_dur < 2.5
    # ...but the singing itself must survive.
    assert out_dur > 1.0


def test_tighten_pauses_keeps_continuous_audio_intact(tmp_path):
    """A clip with no long gaps comes back essentially unchanged."""
    pytest.importorskip("pydub")
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src, [(2.0, 12000)])
    tighten_pauses(src, out, verbose=False)
    in_dur, out_dur = _wav_seconds(src), _wav_seconds(out)
    # Only the small edge pads differ — within ~0.6 s of the original.
    assert abs(out_dur - in_dur) < 0.6
