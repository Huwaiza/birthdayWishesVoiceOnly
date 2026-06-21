"""
Vocal-only birthday-song service.

A thin FastAPI wrapper around the existing render pipeline so another
project (e.g. a Django + Celery app) can request songs *asynchronously*
over HTTP. None of the song code is modified — this service only *calls*
the existing functions in the same order the `--acapella` CLI does:

    render() → isolate_vocals() → tighten_pauses() → encode MP3

Design constraints this service is built around
-----------------------------------------------
* The ACE-Step model is heavy (~4 GB) and is a per-process singleton, so
  we run a SINGLE uvicorn worker and load the model once at startup.
* A render takes minutes, and the model can't safely run two at once, so
  jobs go through an in-process queue and a single background worker —
  they execute strictly one at a time.
* Because jobs are long, the API is submit-then-poll (never a blocking
  10-minute request): POST /jobs returns immediately, GET /jobs/{id}
  reports progress. This maps cleanly onto Celery's retry/countdown.
* Output MP3s are written to a shared folder; the status response hands
  back the absolute path so the caller (on the same machine) reads the
  file directly — no download step needed.

Run it (from the repo root, in the ML venv that has ACE-Step):

    uvicorn service.app:app --host 127.0.0.1 --port 8000 --workers 1

`--workers 1` is important: one process == one model load == renders
serialized through this service's own queue.
"""

from __future__ import annotations

import gc
import hashlib
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the repo root importable so `song` resolves no matter where uvicorn
# is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Where finished MP3s land. Override with BIRTHDAY_OUTPUT_DIR. This is the
# "shared folder" the Django side reads from.
OUTPUT_DIR = Path(os.environ.get("BIRTHDAY_OUTPUT_DIR", _REPO_ROOT / "output"))

# Render defaults — match the CLI so service output == `--acapella` output.
DEFAULT_DURATION_S = 150.0

# --- Render watchdog --------------------------------------------------------
# A single worker renders jobs one at a time, and a render is a heavy in-process
# model call with no built-in timeout. If one hangs (model deadlock, resource
# exhaustion) the worker is wedged forever and every queued job behind it
# starves — the caller just polls "running" until it times out. The watchdog
# bounds that: any job rendering longer than RENDER_HARD_LIMIT_S is force-failed
# so the caller gets a clean error and can retry / self-heal. Normal renders are
# ~8-9 min; the 25-min default leaves generous headroom for a slow (not hung)
# render to still finish on its own.
RENDER_HARD_LIMIT_S = float(os.environ.get("RENDER_HARD_LIMIT_S", "1500"))
_WATCHDOG_INTERVAL_S = float(os.environ.get("RENDER_WATCHDOG_INTERVAL_S", "10"))
# The wedged worker THREAD can't be killed in-process. Set this (only when the
# service runs under a process supervisor that restarts it) to also exit on a
# hang, so a fresh process reloads the model and drains the backed-up queue.
_WATCHDOG_HARD_RESTART = os.environ.get("RENDER_WATCHDOG_HARD_RESTART", "0") == "1"

# --- Proactive recycle ------------------------------------------------------
# Memory pressure accumulates across renders (MPS/RAM the allocator never hands
# back), which is what eventually swaps a render into a multi-hour/hung one.
# _free_render_memory() after every job is the primary mitigation. As a belt-
# and-suspenders guarantee of a clean slate, the service can also recycle itself
# after this many completed jobs: it exits cleanly once the queue is idle so a
# supervisor (run_service.sh) respawns a fresh process. 0 = disabled (so a bare
# `uvicorn` never quits unexpectedly); the supervisor wrapper sets it.
RENDER_RESTART_AFTER_JOBS = int(os.environ.get("RENDER_RESTART_AFTER_JOBS", "0"))
# Grace period before a recycle exit, so a client polling for the last job's
# "done" sees it before the process goes down. Must exceed the client poll
# interval (~15s).
_RESTART_GRACE_S = float(os.environ.get("RENDER_RESTART_GRACE_S", "20"))


# --- Job bookkeeping --------------------------------------------------------

@dataclass
class Job:
    job_id: str
    name: str
    voice: Optional[int]
    duration: float
    lyrics: Optional[str] = None
    slug: Optional[str] = None
    status: str = "queued"          # queued → running → done | error
    mp3_path: Optional[str] = None
    error: Optional[str] = None
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: Optional[str] = None
    render_started_at: Optional[float] = None   # monotonic-ish wall clock; set when status→running


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_WORK_QUEUE: "queue.Queue[str]" = queue.Queue()

# Count of jobs this process has finished (done or error), for the recycle gate.
_JOBS_COMPLETED = 0

# Set once the ACE-Step model has finished loading.
_MODEL_READY = threading.Event()

# Guards against starting the warmer/worker more than once. Production runs a
# single lifespan so it never matters; under the test client, which enters
# lifespan once per TestClient context, it keeps us to exactly one worker.
_BG_STARTED = False


def _safe_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def _job_id(name: str, voice: Optional[int], duration: float,
            lyrics: Optional[str] = None, slug: Optional[str] = None) -> str:
    """Deterministic id from the render parameters, so re-submitting the
    same request dedupes onto the same job instead of rendering twice."""
    lyrics_key = hashlib.md5(lyrics.encode("utf-8")).hexdigest() if lyrics else ""
    key = f"{_safe_name(name)}|{voice}|{duration:.1f}|{slug or ''}|{lyrics_key}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _mp3_path_for(name: str, slug: Optional[str] = None) -> Path:
    """Predictable output path the video step picks up. With `slug`, write
    output/<slug>.mp3 (custom occasion) so it never collides with the bare
    name's happy_birthday_<name>.mp3 (e.g. a real "Lucy" birthday)."""
    stem = _safe_name(slug) if slug else f"happy_birthday_{_safe_name(name)}"
    return OUTPUT_DIR / f"{stem}.mp3"


# --- The actual pipeline (calls the untouched song functions) ---------------

def _generate(job: Job) -> Path:
    """render → isolate → tighten → mp3, exactly like generate_birthday.py
    --acapella. Reuses the on-disk caches keyed by render parameters."""
    from pydub import AudioSegment
    from song.acestep_render import render
    from song.vocal_isolate import isolate_vocals, tighten_pauses

    wav = render(
        name=job.name,
        voice_index=job.voice,
        duration_s=job.duration,
        use_cache=True,
        verbose=False,
        lyrics=job.lyrics,
    )

    vocals = wav.with_name(wav.stem + "__vocals.wav")
    if not vocals.exists():
        isolate_vocals(wav, vocals, verbose=False)

    tight = wav.with_name(wav.stem + "__vocal_tight.wav")
    if not tight.exists():
        tighten_pauses(vocals, tight, verbose=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mp3 = _mp3_path_for(job.name, job.slug)
    AudioSegment.from_wav(str(tight)).export(str(mp3), format="mp3", bitrate="192k")
    return mp3


# --- Background threads -----------------------------------------------------

def _warm_model() -> None:
    """Load the ACE-Step pipeline once, up front, so the first real job
    isn't penalised by the multi-GB load."""
    try:
        from song.acestep_render import _get_pipeline
        _get_pipeline()
    except Exception as e:  # noqa: BLE001 — warming is best-effort
        print(f"[service] model warm-up failed (will retry on first job): {e}")
    finally:
        _MODEL_READY.set()


def _free_render_memory() -> None:
    """Release per-render memory back to the OS after every job. Without this
    the MPS/CUDA caching allocator holds onto each render's working set, so
    pressure only climbs across a session until a render swaps and hangs — the
    exact failure documented in song/acestep_render.py. The model singletons are
    NOT touched (we only drop intermediates + the per-call separator)."""
    gc.collect()
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort; torch may be absent in tests
        pass


def _restart_due() -> bool:
    """True when the process has finished enough jobs to warrant a clean restart
    AND nothing is waiting — so recycling won't drop a queued render. No-op
    unless RENDER_RESTART_AFTER_JOBS > 0 (set only by the supervisor wrapper)."""
    if RENDER_RESTART_AFTER_JOBS <= 0:
        return False
    with _JOBS_LOCK:
        completed = _JOBS_COMPLETED
    return completed >= RENDER_RESTART_AFTER_JOBS and _WORK_QUEUE.empty()


def _worker() -> None:
    """Single consumer — pulls one job at a time and runs it to completion."""
    global _JOBS_COMPLETED
    while True:
        job_id = _WORK_QUEUE.get()
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            _WORK_QUEUE.task_done()
            continue
        try:
            with _JOBS_LOCK:
                job.status = "running"
                job.render_started_at = time.time()
            print(f"[service] rendering job {job_id} ({job.name})")
            t0 = time.time()
            mp3 = _generate(job)
            with _JOBS_LOCK:
                # The watchdog may have force-failed this job for running too
                # long; don't resurrect it to "done" (the caller has moved on).
                # The MP3 is still written to disk, so a later run picks it up.
                if job.status == "running":
                    job.status = "done"
                    job.mp3_path = str(mp3)
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
            print(f"[service] job {job_id} done in {time.time() - t0:.0f}s → {mp3}")
        except Exception as e:  # noqa: BLE001 — report failure, keep serving
            with _JOBS_LOCK:
                if job.status == "running":
                    job.status = "error"
                    job.error = str(e)
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
            print(f"[service] job {job_id} FAILED: {e}")
        finally:
            _WORK_QUEUE.task_done()
            _free_render_memory()
            with _JOBS_LOCK:
                _JOBS_COMPLETED += 1

        # Recycle to a fresh process once we've done enough jobs and the queue
        # has drained — keeps long sessions from accumulating into a hung render.
        if _restart_due():
            print(f"[service] recycling after {_JOBS_COMPLETED} jobs to reset "
                  f"memory; supervisor will respawn a clean process.")
            time.sleep(_RESTART_GRACE_S)   # let a poller fetch the last 'done'
            if _WORK_QUEUE.empty():
                os._exit(0)


def _watchdog_sweep(now: float) -> list[str]:
    """Force-fail any job that has been rendering longer than
    RENDER_HARD_LIMIT_S. Returns the ids it failed. Kept free of sleeping and
    threads so the policy is unit-testable in isolation."""
    wedged: list[str] = []
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if (job.status == "running" and job.render_started_at is not None
                    and now - job.render_started_at > RENDER_HARD_LIMIT_S):
                job.status = "error"
                job.error = (
                    f"render exceeded hard limit ({RENDER_HARD_LIMIT_S:.0f}s); "
                    "worker wedged — caller should retry; service may need restart"
                )
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                wedged.append(job.job_id)
    return wedged


def _watchdog() -> None:
    """Periodically sweep for renders that have hung past the hard limit."""
    while True:
        time.sleep(_WATCHDOG_INTERVAL_S)
        for job_id in _watchdog_sweep(time.time()):
            print(f"[service] WATCHDOG force-failed job {job_id} "
                  f"(> {RENDER_HARD_LIMIT_S:.0f}s).")
            if _WATCHDOG_HARD_RESTART:
                print("[service] WATCHDOG hard-restart: exiting so a supervisor "
                      "respawns a clean worker and drains the queue.")
                os._exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BG_STARTED
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _BG_STARTED:
        threading.Thread(target=_warm_model, name="model-warmer", daemon=True).start()
        threading.Thread(target=_worker, name="render-worker", daemon=True).start()
        threading.Thread(target=_watchdog, name="render-watchdog", daemon=True).start()
        _BG_STARTED = True
    yield


app = FastAPI(title="Birthday vocal-only service", lifespan=lifespan)


# --- API --------------------------------------------------------------------

class JobRequest(BaseModel):
    name: str
    voice: Optional[int] = None
    duration: float = DEFAULT_DURATION_S
    lyrics: Optional[str] = None
    slug: Optional[str] = None


def _status_payload(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "mp3_path": job.mp3_path,
        "error": job.error,
        "submitted_at": job.submitted_at,
        "finished_at": job.finished_at,
    }


@app.get("/health")
def health() -> dict:
    with _JOBS_LOCK:
        active = sum(1 for j in _JOBS.values() if j.status in ("queued", "running"))
    return {
        "status": "ok",
        "model_ready": _MODEL_READY.is_set(),
        "queued_or_running": active,
    }


@app.post("/jobs")
def submit(req: JobRequest) -> dict:
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    job_id = _job_id(req.name, req.voice, req.duration, req.lyrics, req.slug)

    with _JOBS_LOCK:
        existing = _JOBS.get(job_id)
        # Already known and still relevant → just return its state (dedupe).
        if existing and existing.status in ("queued", "running", "done"):
            return _status_payload(existing)

        # Completed in a previous run (process restarted) but the MP3 is
        # still on disk → report done without re-rendering.
        mp3 = _mp3_path_for(req.name, req.slug)
        if mp3.exists():
            job = Job(job_id=job_id, name=req.name, voice=req.voice,
                      duration=req.duration, lyrics=req.lyrics, slug=req.slug,
                      status="done", mp3_path=str(mp3),
                      finished_at=datetime.now().isoformat(timespec="seconds"))
            _JOBS[job_id] = job
            return _status_payload(job)

        # Fresh (or a previously-errored) job → (re)queue it.
        job = Job(job_id=job_id, name=req.name, voice=req.voice,
                  duration=req.duration, lyrics=req.lyrics, slug=req.slug,
                  status="queued")
        _JOBS[job_id] = job

    _WORK_QUEUE.put(job_id)
    return _status_payload(job)


@app.get("/jobs/{job_id}")
def status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return _status_payload(job)


@app.get("/jobs/{job_id}/audio")
def audio(job_id: str):
    """Optional convenience: stream the MP3 over HTTP. Not needed in the
    shared-folder setup (read mp3_path directly), but handy for testing or
    if the caller is ever moved off-box."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if job.status != "done" or not job.mp3_path:
        raise HTTPException(status_code=409, detail=f"job not done (status={job.status})")
    return FileResponse(job.mp3_path, media_type="audio/mpeg",
                        filename=Path(job.mp3_path).name)
