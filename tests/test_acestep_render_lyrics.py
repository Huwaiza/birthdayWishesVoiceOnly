"""render() must honor an explicit `lyrics` override and otherwise fall back
to build_lyrics(name). The heavy ACE-Step pipeline is faked (writes a stub WAV)
so these run with no model, no GPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import song.acestep_render as ar


class _FakePipe:
    device = "cpu"
    dtype = "float16"

    def __init__(self, captured):
        self._captured = captured

    def __call__(self, **kw):
        self._captured.update(kw)
        Path(kw["save_path"]).write_bytes(b"RIFFfakewav")


def test_render_uses_custom_lyrics_when_provided(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ar, "_get_pipeline", lambda precision="float16": _FakePipe(captured))

    def _boom(name):
        raise AssertionError("build_lyrics must NOT be called when lyrics is given")

    monkeypatch.setattr(ar, "build_lyrics", _boom)

    out = ar.render(
        "Lucy", duration_s=10.0, cache_dir=tmp_path, use_cache=False,
        verbose=False, lyrics="[verse]\nCustom subscriber line",
    )

    assert captured["lyrics"] == "[verse]\nCustom subscriber line"
    assert out.exists()


def test_render_falls_back_to_build_lyrics(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ar, "_get_pipeline", lambda precision="float16": _FakePipe(captured))
    monkeypatch.setattr(ar, "build_lyrics", lambda name: "[verse]\nBUILT FOR " + name)

    ar.render("Lucy", duration_s=10.0, cache_dir=tmp_path, use_cache=False, verbose=False)

    assert captured["lyrics"] == "[verse]\nBUILT FOR Lucy"
