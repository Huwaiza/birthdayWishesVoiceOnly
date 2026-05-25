"""Tests for song.vocal_isolate — the Roformer vocal-stem wrapper.

These tests do NOT require audio-separator to be installed. vocal_isolate
imports it lazily, only after the input-validation path has run — so the
error handling (missing input file) is fully testable on its own.
"""

import pytest

from song.vocal_isolate import isolate_vocals


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
