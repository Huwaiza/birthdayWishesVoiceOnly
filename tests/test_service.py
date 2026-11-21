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
        p = module._mp3_path_for(job.name, job.slug)
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


def test_watchdog_force_fails_a_render_past_the_hard_limit(svc, monkeypatch):
    """A render that has been 'running' longer than RENDER_HARD_LIMIT_S is
    force-failed so the caller gets a clean error instead of polling forever."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_HARD_LIMIT_S", 100.0)

    job = module.Job(job_id="wedged1", name="Wedge", voice=None, duration=1.0,
                     status="running")
    job.render_started_at = time.time() - 250.0  # started well past the limit
    with module._JOBS_LOCK:
        module._JOBS[job.job_id] = job

    failed = module._watchdog_sweep(time.time())

    assert "wedged1" in failed
    assert module._JOBS["wedged1"].status == "error"
    assert "hard limit" in (module._JOBS["wedged1"].error or "")


def test_watchdog_leaves_a_fresh_render_running(svc, monkeypatch):
    """A render still within the limit must NOT be touched by the watchdog."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_HARD_LIMIT_S", 100.0)

    job = module.Job(job_id="fresh1", name="Fresh", voice=None, duration=1.0,
                     status="running")
    job.render_started_at = time.time() - 5.0  # well within the limit
    with module._JOBS_LOCK:
        module._JOBS[job.job_id] = job

    failed = module._watchdog_sweep(time.time())

    assert "fresh1" not in failed
    assert module._JOBS["fresh1"].status == "running"


def test_free_render_memory_is_safe_to_call(svc):
    """The between-jobs memory release must never raise (torch may be absent)."""
    client, module = svc
    module._free_render_memory()  # should be a no-op-safe call


def test_infer_step_60_keeps_legacy_job_ids(svc):
    """The step count joins the job id only away from 60, so every id minted
    before infer_step was tunable still resolves — otherwise switching to this
    build would orphan the existing render cache."""
    client, module = svc
    legacy = module._job_id("Lucy", None, 150.0)
    assert module._job_id("Lucy", None, 150.0, infer_step=60) == legacy


def test_infer_step_changes_the_job_id(svc):
    """Different step counts produce different audio, so they must not dedupe
    onto one another."""
    client, module = svc
    base = module._job_id("Lucy", None, 150.0, infer_step=60)
    assert module._job_id("Lucy", None, 150.0, infer_step=30) != base
    assert (module._job_id("Lucy", None, 150.0, infer_step=30)
            != module._job_id("Lucy", None, 150.0, infer_step=20))


def test_submit_records_infer_step_on_the_job(svc):
    """A per-request infer_step must survive onto the queued Job."""
    client, module = svc
    job_id = client.post("/jobs",
                         json={"name": "Stepped", "infer_step": 24}).json()["job_id"]
    with module._JOBS_LOCK:
        assert module._JOBS[job_id].infer_step == 24


def test_submit_defaults_infer_step_to_the_configured_default(svc):
    """The Django caller posts only {"name": ...}; it must inherit
    DEFAULT_INFER_STEP (i.e. RENDER_INFER_STEP) rather than a hardcoded 60."""
    client, module = svc
    job_id = client.post("/jobs", json={"name": "Defaulted"}).json()["job_id"]
    with module._JOBS_LOCK:
        assert module._JOBS[job_id].infer_step == module.DEFAULT_INFER_STEP


def test_generate_forwards_infer_step_to_the_pipeline(tmp_path, monkeypatch):
    """The step count has to reach generate_vocal_song. Plumbing that stops at
    the Job object would leave every render at 60 while the API claimed
    otherwise — silent, and invisible except as an unchanged fan."""
    from service import app as module
    import song.pipeline as pipeline_mod

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "out")
    captured = {}

    def fake_generate_vocal_song(name, **kw):
        captured.update(kw)
        p = tmp_path / "tight.wav"
        p.write_bytes(b"wav")
        return p, None

    monkeypatch.setattr(pipeline_mod, "generate_vocal_song",
                        fake_generate_vocal_song)

    class _FakeSeg:
        @staticmethod
        def from_wav(path):
            return _FakeSeg()

        def export(self, out, **kw):
            Path(out).write_bytes(b"mp3")

    import pydub
    monkeypatch.setattr(pydub, "AudioSegment", _FakeSeg)

    job = module.Job(job_id="x", name="Adam", voice=None, duration=150.0,
                     infer_step=24)
    module._generate(job)

    assert captured["infer_step"] == 24


def test_soft_reload_due_at_the_threshold_when_idle(svc, monkeypatch):
    """The guard that actually applies to a bare `uvicorn`: after N jobs, with
    nothing queued, drop the models in-process."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_SOFT_RELOAD_AFTER_JOBS", 10)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 10)
    while not module._WORK_QUEUE.empty():
        module._WORK_QUEUE.get_nowait()
    assert module._soft_reload_due() is True


def test_soft_reload_recurs_every_n_jobs(svc, monkeypatch):
    """Unlike the one-shot hard recycle, the soft reload has to fire again on
    every multiple — the process keeps running and keeps accumulating."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_SOFT_RELOAD_AFTER_JOBS", 10)
    while not module._WORK_QUEUE.empty():
        module._WORK_QUEUE.get_nowait()
    for completed, expected in [(10, True), (11, False), (20, True), (25, False)]:
        monkeypatch.setattr(module, "_JOBS_COMPLETED", completed)
        assert module._soft_reload_due() is expected, f"at {completed} jobs"


def test_soft_reload_not_due_before_any_job(svc, monkeypatch):
    """0 % N == 0 — guard against reloading models that were never used."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_SOFT_RELOAD_AFTER_JOBS", 10)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 0)
    assert module._soft_reload_due() is False


def test_soft_reload_not_due_while_work_is_pending(svc, monkeypatch):
    """Never stall a queued render behind a model reload."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_SOFT_RELOAD_AFTER_JOBS", 10)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 10)
    module._WORK_QUEUE.put("pending-job")
    try:
        assert module._soft_reload_due() is False
    finally:
        module._WORK_QUEUE.get_nowait()


def test_soft_reload_can_be_disabled(svc, monkeypatch):
    client, module = svc
    monkeypatch.setattr(module, "RENDER_SOFT_RELOAD_AFTER_JOBS", 0)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 9999)
    assert module._soft_reload_due() is False


def test_soft_reload_drops_the_model_singletons(svc, monkeypatch):
    """_soft_reload() must clear the singletons, not just call empty_cache() —
    that is the difference between reclaiming the ~7 GB and not."""
    client, module = svc
    import song.acestep_render as ar
    import song.vocal_isolate as vi
    monkeypatch.setattr(ar, "_PIPELINE", object())
    monkeypatch.setattr(vi, "_SEPARATOR", object())

    module._soft_reload()

    assert ar._PIPELINE is None
    assert vi._SEPARATOR is None


def test_restart_disabled_by_default(svc, monkeypatch):
    """With RENDER_RESTART_AFTER_JOBS=0 the service never recycles itself, no
    matter how many jobs it has done — a bare uvicorn must not quit unexpectedly."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_RESTART_AFTER_JOBS", 0)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 9999)
    assert module._restart_due() is False


def test_restart_due_when_threshold_reached_and_idle(svc, monkeypatch):
    client, module = svc
    monkeypatch.setattr(module, "RENDER_RESTART_AFTER_JOBS", 10)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 10)
    # Drain the queue so the recycle is safe (won't drop a pending render).
    while not module._WORK_QUEUE.empty():
        module._WORK_QUEUE.get_nowait()
    assert module._restart_due() is True


def test_restart_not_due_while_a_job_is_queued(svc, monkeypatch):
    """Even past the threshold, don't recycle while work is pending."""
    client, module = svc
    monkeypatch.setattr(module, "RENDER_RESTART_AFTER_JOBS", 10)
    monkeypatch.setattr(module, "_JOBS_COMPLETED", 50)
    module._WORK_QUEUE.put("pending-job")
    try:
        assert module._restart_due() is False
    finally:
        module._WORK_QUEUE.get_nowait()  # leave the queue clean for other tests


def test_completed_counter_increments_after_a_job(svc):
    client, module = svc
    before = module._JOBS_COMPLETED
    job_id = client.post("/jobs", json={"name": "Counted"}).json()["job_id"]
    _poll(client, job_id)
    # The counter bumps in the worker's finally, just after status flips to done.
    deadline = time.time() + 2.0
    while module._JOBS_COMPLETED <= before and time.time() < deadline:
        time.sleep(0.02)
    assert module._JOBS_COMPLETED > before


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
    """_generate must call render → isolate_vocals → tighten_pauses → qc →
    encode, in that order, and write a username-based MP3."""
    import song.acestep_render as ar
    import song.vocal_isolate as vi
    import song.qc as qc_mod
    from service import app as module

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "out")
    calls = []

    def fake_render(name, **kw):
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

    def fake_qc(tight, **kw):
        calls.append("qc")
        return qc_mod.QCReport(passed=True, reasons=[], sung_seconds=100.0,
                               phrase_count=10, asr_used=False)

    monkeypatch.setattr(ar, "render", fake_render)
    monkeypatch.setattr(vi, "isolate_vocals", fake_isolate)
    monkeypatch.setattr(vi, "tighten_pauses", fake_tighten)
    monkeypatch.setattr(qc_mod, "run_qc", fake_qc)

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

    assert calls == ["render", "isolate", "tighten", "qc", "encode"]
    assert mp3 == module._mp3_path_for("Adam")
    assert mp3.exists()
    assert job.qc_passed is True
    assert job.qc_info["sung_seconds"] == 100.0


def test_job_id_changes_with_lyrics_and_slug():
    from service import app as module
    plain = module._job_id("Lucy", None, 150.0)
    custom = module._job_id("Lucy", None, 150.0, lyrics="[verse]\nx", slug="lucy_300")
    assert plain != custom
    # Deterministic for the same custom inputs.
    assert custom == module._job_id("Lucy", None, 150.0, lyrics="[verse]\nx", slug="lucy_300")


def test_job_id_stable_at_defaults_but_changes_with_guidance_and_qc():
    """Ids minted before the guidance/qc params existed must stay reachable:
    default values add nothing to the key. Non-default values fork the id,
    which is what makes A/B submissions distinct jobs."""
    from service import app as module
    old_style = module._job_id("Lucy", None, 150.0)
    defaults = module._job_id("Lucy", None, 150.0,
                              guidance_scale_text=0.0, guidance_scale_lyric=0.0,
                              qc=True)
    assert old_style == defaults
    guided = module._job_id("Lucy", None, 150.0,
                            guidance_scale_text=5.0, guidance_scale_lyric=1.5)
    assert guided != old_style
    no_qc = module._job_id("Lucy", None, 150.0, qc=False)
    assert no_qc != old_style and no_qc != guided


def test_submit_carries_guidance_and_qc_to_the_job(svc):
    client, module = svc
    body = client.post("/jobs", json={
        "name": "Guided", "slug": "guided_on",
        "guidance_scale_text": 5.0, "guidance_scale_lyric": 1.5,
        "qc": True, "qc_max_retries": 2,
    }).json()
    with module._JOBS_LOCK:
        job = module._JOBS[body["job_id"]]
    assert job.guidance_scale_text == 5.0
    assert job.guidance_scale_lyric == 1.5
    assert job.qc is True and job.qc_max_retries == 2
    # The payload exposes the QC fields (null until a real render fills them).
    assert "qc_passed" in body and "qc" in body


def test_mp3_path_uses_slug_when_given(tmp_path, monkeypatch):
    from service import app as module
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    assert module._mp3_path_for("Lucy", "lucy_300_subscribers") == tmp_path / "lucy_300_subscribers.mp3"
    # No slug → unchanged username-based path.
    assert module._mp3_path_for("Lucy") == tmp_path / "happy_birthday_lucy.mp3"


def test_custom_lyrics_job_renders_to_slug_file(svc):
    client, module = svc
    r = client.post("/jobs", json={
        "name": "Lucy",
        "slug": "lucy_300_subscribers",
        "lyrics": "[verse]\nHappy three hundred subscribers",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    done = _poll(client, job_id)
    assert done["status"] == "done"
    assert Path(done["mp3_path"]).name == "lucy_300_subscribers.mp3"


def test_generate_abandons_a_job_once_the_watchdog_fails_it(tmp_path, monkeypatch):
    """_generate must hand the pipeline an abort hook tied to the job status:
    False while the job runs, True once the watchdog has force-failed it.
    Without it the worker finished (and QC-retried) dead jobs for an hour
    while the queue behind it starved."""
    from service import app as module
    import song.pipeline as pipeline_mod

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "out")
    job = module.Job(job_id="x", name="Adam", voice=None, duration=150.0)
    job.status = "running"
    seen = {}

    def fake_generate_vocal_song(name, **kw):
        should_abort = kw["should_abort"]
        seen["while_running"] = should_abort()
        job.status = "error"                    # what the watchdog sweep does
        seen["after_watchdog"] = should_abort()
        raise pipeline_mod.RenderAborted("abandoned")

    monkeypatch.setattr(pipeline_mod, "generate_vocal_song",
                        fake_generate_vocal_song)

    with pytest.raises(pipeline_mod.RenderAborted):
        module._generate(job)
    assert seen == {"while_running": False, "after_watchdog": True}


def test_worker_moves_on_after_abandoning_a_force_failed_job(svc, monkeypatch):
    """End to end through the worker: the watchdog fails a slow render, the
    pipeline aborts at the next stage, the job keeps the watchdog's error,
    and the next queued job is rendered."""
    client, module = svc
    import song.pipeline as pipeline_mod
    real_generate = module._generate

    def slow_then_abandoned(job):
        if job.name != "Slow":
            return real_generate(job)
        with module._JOBS_LOCK:                 # the watchdog's sweep
            job.status = "error"
            job.error = "render exceeded hard limit (1500s)"
        raise pipeline_mod.RenderAborted("'Slow' abandoned before vocal isolation")

    monkeypatch.setattr(module, "_generate", slow_then_abandoned)
    slow = client.post("/jobs", json={"name": "Slow"}).json()["job_id"]
    nxt = client.post("/jobs", json={"name": "Next"}).json()["job_id"]

    assert _poll(client, nxt)["status"] == "done"
    failed = client.get(f"/jobs/{slow}").json()
    assert failed["status"] == "error"
    assert "hard limit" in failed["error"]      # not overwritten by the abort
