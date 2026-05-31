"""
Tests for the FastAPI vocal-only service (service/app.py).

The heavy bits — the ACE-Step model load and the actual render — are mocked
out, so these tests run in milliseconds and need no GPU, no model weights and
no ffmpeg. We verify the *service behaviour*: queueing, the background worker,
job dedupe, caching, error handling and the endpoints. The real render
pipeline order is checked separately in test_generate_pipeline_order().
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient


def _poll(client, job_id, want=("done", "error"), timeout=5.0):
    """Poll GET /jobs/{id} until it reaches a terminal status (or times out)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {want} within {timeout}s")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """A fresh service module with the model + render mocked, and output
    pointed at a temp dir. Yields (TestClient, module)."""
    from service import app as module

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "out")
    # Never load the real 4 GB model in tests — just mark it ready.
    monkeypatch.setattr(module, "_warm_model", lambda: module._MODEL_READY.set())

    # Fast fake render: write a tiny stand-in MP3 where the real one would go.
    def fake_generate(job):
        p = module._mp3_path_for(job.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ID3\x04\x00fake-mp3-bytes")
        return p

    monkeypatch.setattr(module, "_generate", fake_generate)

    with module._JOBS_LOCK:
        module._JOBS.clear()

    with TestClient(module.app) as client:
        yield client, module


# --- helpers ----------------------------------------------------------------

def test_safe_name_and_job_id_are_deterministic():
    from service import app as module
    assert module._safe_name("  John Doe ") == "john_doe"
    a = module._job_id("Mujtaba", None, 150.0)
    b = module._job_id("mujtaba", None, 150.0)          # case/space-insensitive
    assert a == b
    assert a != module._job_id("Mujtaba", 1, 150.0)     # voice changes the id
    assert a != module._job_id("Mujtaba", None, 120.0)  # duration changes the id


def test_mp3_path_is_username_based(tmp_path, monkeypatch):
    from service import app as module
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    assert module._mp3_path_for("Sara Khan") == tmp_path / "happy_birthday_sara_khan.mp3"


# --- endpoints --------------------------------------------------------------

def test_health(svc):
    client, _ = svc
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert isinstance(body["queued_or_running"], int)


def test_submit_then_render_completes(svc):
    client, module = svc
    r = client.post("/jobs", json={"name": "Mujtaba"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("queued", "done")
    job_id = body["job_id"]

    done = _poll(client, job_id)
    assert done["status"] == "done"
    assert done["mp3_path"] is not None
    mp3 = Path(done["mp3_path"])
    assert mp3.exists() and mp3.name == "happy_birthday_mujtaba.mp3"


def test_submit_is_idempotent(svc):
    client, _ = svc
    a = client.post("/jobs", json={"name": "Talha"}).json()
    b = client.post("/jobs", json={"name": "talha"}).json()  # same after normalise
    assert a["job_id"] == b["job_id"]


def test_missing_name_is_400(svc):
    client, _ = svc
    r = client.post("/jobs", json={"name": "   "})
    assert r.status_code == 400


def test_cached_mp3_returns_done_without_queueing(svc):
    client, module = svc
    # Pre-create the output MP3 → submit should report done immediately.
    mp3 = module._mp3_path_for("Yasir")
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"already-here")
    body = client.post("/jobs", json={"name": "Yasir"}).json()
    assert body["status"] == "done"
    assert body["mp3_path"] == str(mp3)


def test_status_unknown_job_is_404(svc):
    client, _ = svc
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_render_error_is_reported(svc):
    client, module = svc

    def boom(job):
        raise RuntimeError("render exploded")

    # Replace the fake with a failing one (worker reads the global at call time).
    module._generate = boom
    job_id = client.post("/jobs", json={"name": "Boom"}).json()["job_id"]
    done = _poll(client, job_id)
    assert done["status"] == "error"
    assert "render exploded" in done["error"]


def test_audio_endpoint(svc):
    client, module = svc
    # Not-done job → 409.
    with module._JOBS_LOCK:
        module._JOBS["pending"] = module.Job(
            job_id="pending", name="Pending", voice=None, duration=150.0,
            status="running",
        )
    assert client.get("/jobs/pending/audio").status_code == 409

    # Completed job → 200 with the bytes.
    job_id = client.post("/jobs", json={"name": "Audio"}).json()["job_id"]
    _poll(client, job_id)
    r = client.get(f"/jobs/{job_id}/audio")
    assert r.status_code == 200
    assert r.content == b"ID3\x04\x00fake-mp3-bytes"

    # Unknown job → 404.
    assert client.get("/jobs/nope/audio").status_code == 404


# --- the real orchestration (song functions mocked, not the model) ----------

def test_generate_pipeline_order(tmp_path, monkeypatch):
    """_generate must call render → isolate_vocals → tighten_pauses → encode,
    in that order, and write a username-based MP3."""
    import song.acestep_render as ar
    import song.vocal_isolate as vi
    from service import app as module

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "out")
    calls = []

    def fake_render(name, voice_index=None, duration_s=150.0, use_cache=True, verbose=True):
        calls.append("render")
        p = tmp_path / "render.wav"
        p.write_bytes(b"wav")
        return p

    def fake_isolate(inp, out, verbose=True):
        calls.append("isolate")
        Path(out).write_bytes(b"vocals")
        return Path(out)

    def fake_tighten(inp, out, verbose=True):
        calls.append("tighten")
        Path(out).write_bytes(b"tight")
        return Path(out)

    monkeypatch.setattr(ar, "render", fake_render)
    monkeypatch.setattr(vi, "isolate_vocals", fake_isolate)
    monkeypatch.setattr(vi, "tighten_pauses", fake_tighten)

    # Fake pydub so no ffmpeg is needed: from_wav(...).export(mp3) writes a file.
    class _FakeSeg:
        @staticmethod
        def from_wav(path):
            return _FakeSeg()

        def export(self, out, **kw):
            calls.append("encode")
            Path(out).write_bytes(b"mp3")

    import pydub
    monkeypatch.setattr(pydub, "AudioSegment", _FakeSeg)

    job = module.Job(job_id="x", name="Adam", voice=None, duration=150.0)
    mp3 = module._generate(job)

    assert calls == ["render", "isolate", "tighten", "encode"]
    assert mp3 == module._mp3_path_for("Adam")
    assert mp3.exists()
