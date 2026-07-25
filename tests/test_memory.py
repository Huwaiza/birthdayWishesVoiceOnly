"""Tests for song.memory — the shared memory-release helpers, and the model
singleton teardown they drive.

These must never raise: they run in a `finally` after a completed render, so a
failure here would turn a finished song into a failed job. Torch may also be
absent entirely (CI), which every helper has to tolerate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import song.acestep_render as ar
import song.vocal_isolate as vi
from song import memory


def test_free_render_memory_never_raises():
    memory.free_render_memory()


def test_rss_gb_is_a_plausible_number_or_none():
    gb = memory.rss_gb()
    assert gb is None or 0.0 < gb < 1024.0


def test_format_rss_is_log_safe():
    """Returns either a 'key=1.23GB' fragment or an empty string, so callers
    can interpolate it unconditionally."""
    out = memory.format_rss("before")
    assert out == "" or (out.startswith("before=") and out.endswith("GB"))


def test_free_all_models_drops_both_singletons(monkeypatch):
    """The whole point of the soft reload: the caching allocator can only hand
    memory back once nothing references the models, so both singletons must
    actually be cleared."""
    monkeypatch.setattr(ar, "_PIPELINE", object())
    monkeypatch.setattr(vi, "_SEPARATOR", object())

    memory.free_all_models()

    assert ar._PIPELINE is None
    assert vi._SEPARATOR is None


def test_free_all_models_is_idempotent(monkeypatch):
    monkeypatch.setattr(ar, "_PIPELINE", None)
    monkeypatch.setattr(vi, "_SEPARATOR", None)
    memory.free_all_models()
    memory.free_all_models()
    assert ar._PIPELINE is None and vi._SEPARATOR is None


def test_unload_pipeline_clears_the_cached_pipeline(monkeypatch):
    monkeypatch.setattr(ar, "_PIPELINE", object())
    ar.unload_pipeline()
    assert ar._PIPELINE is None


def test_unload_separator_clears_the_cached_separator(monkeypatch):
    monkeypatch.setattr(vi, "_SEPARATOR", object())
    vi.unload_separator()
    assert vi._SEPARATOR is None
