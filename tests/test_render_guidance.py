"""render()'s lyric-guidance and seed-offset plumbing: params must reach the
ACE-Step pipeline, the cache key must stay byte-compatible for old renders
(guidance off), and change when guidance or the seed offset changes."""

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


def _render(tmp_path, monkeypatch, **kw):
    captured = {}
    monkeypatch.setattr(ar, "_get_pipeline",
                        lambda precision="float16": _FakePipe(captured))
    out = ar.render("Lucy", duration_s=10.0, cache_dir=tmp_path,
                    use_cache=False, verbose=False, **kw)
    return out, captured


def test_guidance_params_reach_the_pipeline(tmp_path, monkeypatch):
    _, captured = _render(tmp_path, monkeypatch,
                          guidance_scale_text=5.0, guidance_scale_lyric=1.5)
    assert captured["guidance_scale_text"] == 5.0
    assert captured["guidance_scale_lyric"] == 1.5


def test_guidance_off_passes_zero_and_keeps_old_cache_key(tmp_path, monkeypatch):
    # Explicit 0.0/0.0 must produce the exact same cache file name as not
    # passing the params at all — the 301 pre-existing cached WAVs depend
    # on this.
    out_default, captured = _render(tmp_path, monkeypatch)
    assert captured["guidance_scale_text"] == 0.0
    assert captured["guidance_scale_lyric"] == 0.0
    out_explicit, _ = _render(tmp_path, monkeypatch,
                              guidance_scale_text=0.0, guidance_scale_lyric=0.0)
    assert out_default.name == out_explicit.name


def test_guidance_on_changes_cache_key(tmp_path, monkeypatch):
    out_off, _ = _render(tmp_path, monkeypatch)
    out_on, _ = _render(tmp_path, monkeypatch,
                        guidance_scale_text=5.0, guidance_scale_lyric=1.5)
    assert out_off.name != out_on.name


def test_seed_offset_changes_seed_and_cache_key(tmp_path, monkeypatch):
    out0, cap0 = _render(tmp_path, monkeypatch, seed_offset=0)
    out1, cap1 = _render(tmp_path, monkeypatch, seed_offset=1)
    assert int(cap1["manual_seeds"]) == int(cap0["manual_seeds"]) + 1
    assert out0.name != out1.name
