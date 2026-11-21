"""
Tests for song.pipeline.generate_vocal_song — the shared render → isolate →
tighten → QC → seed-retry orchestration. Every heavy stage is mocked at its
home module, exactly like the service tests do.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import song.acestep_render as ar
import song.vocal_isolate as vi
import song.qc as qc_mod
from song.pipeline import RenderAborted, generate_vocal_song
from song.qc import QCReport


def _report(passed, score=1.0, attempt=0):
    return QCReport(passed=passed, reasons=[] if passed else ["nope"],
                    sung_seconds=100.0, phrase_count=10, asr_used=False,
                    attempt=attempt, score=score)


@pytest.fixture
def stages(tmp_path, monkeypatch):
    """Mock render/isolate/tighten; record calls and the seed offsets used."""
    calls = {"order": [], "seed_offsets": [], "render_kwargs": []}

    def fake_render(name, **kw):
        calls["order"].append("render")
        calls["seed_offsets"].append(kw.get("seed_offset", 0))
        calls["render_kwargs"].append(kw)
        p = tmp_path / f"{name}_off{kw.get('seed_offset', 0)}.wav"
        p.write_bytes(b"wav")
        return p

    def fake_isolate(inp, out, verbose=True):
        calls["order"].append("isolate")
        Path(out).write_bytes(b"vocals")
        return Path(out)

    def fake_tighten(inp, out, verbose=True):
        calls["order"].append("tighten")
        Path(out).write_bytes(b"tight")
        return Path(out)

    monkeypatch.setattr(ar, "render", fake_render)
    monkeypatch.setattr(vi, "isolate_vocals", fake_isolate)
    monkeypatch.setattr(vi, "tighten_pauses", fake_tighten)
    return calls


def test_qc_pass_first_try_renders_once(stages, monkeypatch):
    monkeypatch.setattr(qc_mod, "run_qc", lambda *a, **kw: _report(True))
    tight, report = generate_vocal_song("Lucy", verbose=False)
    assert report.passed
    assert stages["seed_offsets"] == [0]
    assert stages["order"] == ["render", "isolate", "tighten"]
    assert tight.name.endswith("__vocal_tight.wav")


def test_qc_fail_then_pass_retries_with_next_seed(stages, monkeypatch):
    reports = iter([_report(False, score=1.0, attempt=0),
                    _report(True, attempt=1)])
    monkeypatch.setattr(qc_mod, "run_qc", lambda *a, **kw: next(reports))
    tight, report = generate_vocal_song("Lucy", verbose=False)
    assert report.passed and report.attempt == 1
    assert stages["seed_offsets"] == [0, 1]
    assert "off1" in tight.name


def test_all_attempts_fail_returns_best_attempt(stages, monkeypatch):
    reports = iter([_report(False, score=2.0, attempt=0),
                    _report(False, score=1.0, attempt=1)])
    monkeypatch.setattr(qc_mod, "run_qc", lambda *a, **kw: next(reports))
    tight, report = generate_vocal_song("Lucy", verbose=False)
    assert not report.passed
    assert report.attempt == 0            # attempt 0 scored higher
    assert "off0" in tight.name
    assert stages["seed_offsets"] == [0, 1]


def test_qc_disabled_is_single_shot(stages, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("QC must not run when disabled")
    monkeypatch.setattr(qc_mod, "run_qc", boom)
    tight, report = generate_vocal_song("Lucy", verbose=False, qc_enabled=False)
    assert report is None
    assert stages["seed_offsets"] == [0]


def test_guidance_and_lyrics_are_passed_through(stages, monkeypatch):
    captured = {}

    def capture_qc(tight, **kw):
        captured.update(kw)
        return _report(True)

    monkeypatch.setattr(qc_mod, "run_qc", capture_qc)
    generate_vocal_song("Lucy", verbose=False,
                        guidance_scale_text=5.0, guidance_scale_lyric=1.5,
                        lyrics="[verse]\ncustom")
    kw = stages["render_kwargs"][0]
    assert kw["guidance_scale_text"] == 5.0
    assert kw["guidance_scale_lyric"] == 1.5
    assert kw["lyrics"] == "[verse]\ncustom"
    assert captured["custom_lyrics"] is True   # custom lyrics relax QC


def test_abort_after_render_skips_the_remaining_stages(stages, monkeypatch):
    """A job the watchdog already force-failed must stop at the next stage
    boundary. Before this, the worker went on to isolate, QC and even
    re-render a song nobody was waiting for, while real jobs sat queued."""
    def boom(*a, **kw):
        raise AssertionError("QC must not run for an abandoned job")
    monkeypatch.setattr(qc_mod, "run_qc", boom)

    def abort_once_rendered():
        return "render" in stages["order"]

    with pytest.raises(RenderAborted):
        generate_vocal_song("Lucy", verbose=False,
                            should_abort=abort_once_rendered)
    assert stages["order"] == ["render"]


def test_abort_prevents_the_qc_retry_render(stages, monkeypatch):
    """The QC retry is a second full render, the single most expensive thing
    an abandoned job could still do."""
    state = {"abandoned": False}

    def failing_qc(*a, **kw):
        state["abandoned"] = True          # watchdog fires while QC runs
        return _report(False)

    monkeypatch.setattr(qc_mod, "run_qc", failing_qc)
    with pytest.raises(RenderAborted):
        generate_vocal_song("Lucy", verbose=False,
                            should_abort=lambda: state["abandoned"])
    assert stages["seed_offsets"] == [0]   # no second render


def test_no_abort_hook_keeps_the_existing_behaviour(stages, monkeypatch):
    monkeypatch.setattr(qc_mod, "run_qc", lambda *a, **kw: _report(True))
    tight, report = generate_vocal_song("Lucy", verbose=False,
                                        should_abort=lambda: False)
    assert report.passed
    assert stages["order"] == ["render", "isolate", "tighten"]
