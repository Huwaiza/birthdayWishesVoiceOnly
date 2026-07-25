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

# --- Soft reload ------------------------------------------------------------
# The hard recycle above only helps when something respawns the process, i.e.
# when the service is started via run_service.sh. Started as a bare `uvicorn`
# (no supervisor) it is necessarily disabled, which left the accumulation
# problem completely unguarded in exactly the setup most people run.
#
# The soft reload is the supervisor-free equivalent: after this many completed
# jobs, once the queue is idle, drop every model singleton and release their
# memory in-process. That is what actually returns the ~7 GB to the OS — the
# caching allocator can only hand back blocks nothing still references — so the
# next render starts from a clean, unfragmented slate. Cost is one model reload
# (~1-2 min) per N songs; at ~20 songs/day that is a couple of minutes a day.
# Set to 0 to disable.
RENDER_SOFT_RELOAD_AFTER_JOBS = int(
    os.environ.get("RENDER_SOFT_RELOAD_AFTER_JOBS", "10")
)


# --- Job bookkeeping --------------------------------------------------------

@dataclass
class Job:
    job_id: str
    name: str
    voice: Optional[int]
    duration: float
    lyrics: Optional[str] = None
    slug: Optional[str] = None
    # ACE-Step double-condition (lyric-adherence) guidance. 0.0/0.0 = off =
    # identical to pre-existing behaviour; ACE-Step activates it only when
    # both are > 1.0 (suggested starting point: text=5.0, lyric=1.5).
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    # Post-render QC gate (sung-time + transcript checks) with seed-retry.
    qc: bool = True
    qc_max_retries: int = 1
    status: str = "queued"          # queued → running → done | error
    mp3_path: Optional[str] = None
    error: Optional[str] = None
    qc_passed: Optional[bool] = None    # None = QC off / not run (e.g. cached MP3)
    qc_info: Optional[dict] = None
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
            lyrics: Optional[str] = None, slug: Optional[str] = None,
            guidance_scale_text: float = 0.0, guidance_scale_lyric: float = 0.0,
            qc: bool = True) -> str:
    """Deterministic id from the render parameters, so re-submitting the
    same request dedupes onto the same job instead of rendering twice.
    The newer parameters join the key only at non-default values, so every
    id minted before they existed stays reachable."""
    lyrics_key = hashlib.md5(lyrics.encode("utf-8")).hexdigest() if lyrics else ""
    key = f"{_safe_name(name)}|{voice}|{duration:.1f}|{slug or ''}|{lyrics_key}"
    if guidance_scale_text or guidance_scale_lyric:
        key += f"|gst{guidance_scale_text:.2f}|gsl{guidance_scale_lyric:.2f}"
    if not qc:
        key += "|noqc"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _mp3_path_for(name: str, slug: Optional[str] = None) -> Path:
    """Predictable output path the video step picks up. With `slug`, write
    output/<slug>.mp3 (custom occasion) so it never collides with the bare
    name's happy_birthday_<name>.mp3 (e.g. a real "Lucy" birthday)."""
    stem = _safe_name(slug) if slug else f"happy_birthday_{_safe_name(name)}"
    return OUTPUT_DIR / f"{stem}.mp3"


# --- The actual pipeline (calls the untouched song functions) ---------------

def _generate(job: Job) -> Path:
    """render → isolate → tighten → QC/retry → mp3, via the shared
    song.pipeline used by the CLI. Reuses the on-disk caches keyed by
    render parameters."""
    from pydub import AudioSegment
    from song.pipeline import generate_vocal_song

    tight, report = generate_vocal_song(
        job.name,
        voice_index=job.voice,
        duration_s=job.duration,
        guidance_scale_text=job.guidance_scale_text,
        guidance_scale_lyric=job.guidance_scale_lyric,
        use_cache=True,
        verbose=False,
        lyrics=job.lyrics,
        qc_enabled=job.qc,
        qc_max_retries=job.qc_max_retries,
    )
    if report is not None:
        with _JOBS_LOCK:
            job.qc_passed = report.passed
            job.qc_info = report.to_dict()

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
    NOT touched (only per-render intermediates).

    The implementation lives in song.memory so the batch runner releases memory
    identically; this wrapper stays for the service's own call sites and tests.
    """
    from song.memory import free_render_memory
    free_render_memory()


def _soft_reload_due() -> bool:
    """True when enough jobs have completed to warrant dropping the models
    in-process AND nothing is waiting, so the reload can't stall a queued
    render. No-op when RENDER_SOFT_RELOAD_AFTER_JOBS is 0."""
    if RENDER_SOFT_RELOAD_AFTER_JOBS <= 0:
        return False
    with _JOBS_LOCK:
        completed = _JOBS_COMPLETED
    return (completed > 0
            and completed % RENDER_SOFT_RELOAD_AFTER_JOBS == 0
            and _WORK_QUEUE.empty())


def _soft_reload() -> None:
    """Drop every model singleton and reclaim their memory. The next job pays
    the reload cost; see RENDER_SOFT_RELOAD_AFTER_JOBS for the rationale."""
    from song.memory import format_rss, free_all_models
    before = format_rss("before")
    free_all_models()
    after = format_rss("after")
    print(f"[service] soft reload after {_JOBS_COMPLETED} jobs — models dropped, "
          f"memory released ({before} {after}). Next render reloads them.")


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
            from song.memory import format_rss
            print(f"[service] job {job_id} done in {time.time() - t0:.0f}s → {mp3} "
                  f"{format_rss()}")
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
        # Only ever enabled under a supervisor that respawns us (run_service.sh).
        if _restart_due():
            print(f"[service] recycling after {_JOBS_COMPLETED} jobs to reset "
                  f"memory; supervisor will respawn a clean process.")
            time.sleep(_RESTART_GRACE_S)   # let a poller fetch the last 'done'
            if _WORK_QUEUE.empty():
                os._exit(0)

        # No supervisor (or not due yet) → get the same clean slate in-process
        # by dropping the models. This is the guard that actually applies to a
        # bare `uvicorn`, where the hard recycle above is necessarily disabled.
        if _soft_reload_due():
            _soft_reload()


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


def _log_memory_guards() -> None:
    """Print which memory guards are actually live.

    Worth the four lines: the hard recycle and the hang restart are opt-in via
    env vars that only run_service.sh sets, so a bare `uvicorn` silently ran
    with both disabled and no way to tell from the logs.
    """
    hard = (f"recycle every {RENDER_RESTART_AFTER_JOBS} jobs"
            if RENDER_RESTART_AFTER_JOBS > 0 else "OFF (no supervisor)")
    soft = (f"soft reload every {RENDER_SOFT_RELOAD_AFTER_JOBS} jobs"
            if RENDER_SOFT_RELOAD_AFTER_JOBS > 0 else "soft reload: OFF")
    print(f"[service] memory guards — per-job release: ON; {soft}; "
          f"process {hard}; hang restart: "
          f"{'ON' if _WATCHDOG_HARD_RESTART else 'OFF'}")
    if RENDER_RESTART_AFTER_JOBS <= 0 and RENDER_SOFT_RELOAD_AFTER_JOBS <= 0:
        print("[service] WARNING: no periodic reset is active — memory pressure "
              "will accumulate across renders. Run ./run_service.sh, or set "
              "RENDER_SOFT_RELOAD_AFTER_JOBS.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BG_STARTED
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _BG_STARTED:
        _log_memory_guards()
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
    # Lyric-adherence (double-condition) guidance. Defaults keep the exact
    # pre-existing render behaviour; set text=5.0, lyric=1.5 to enable.
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    # QC gate + seed retry. qc=false reproduces the old fire-and-hope path.
    qc: bool = True
    qc_max_retries: int = 1


def _status_payload(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "mp3_path": job.mp3_path,
        "error": job.error,
        "submitted_at": job.submitted_at,
        "finished_at": job.finished_at,
        "qc_passed": job.qc_passed,
        "qc": job.qc_info,
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

    job_id = _job_id(req.name, req.voice, req.duration, req.lyrics, req.slug,
                     req.guidance_scale_text, req.guidance_scale_lyric, req.qc)

    with _JOBS_LOCK:
        existing = _JOBS.get(job_id)
        # Already known and still relevant → just return its state (dedupe).
        if existing and existing.status in ("queued", "running", "done"):
            return _status_payload(existing)

        # Completed in a previous run (process restarted) but the MP3 is
        # still on disk → report done without re-rendering. NB: the MP3 path
        # depends only on name/slug, so A/B arms of the same name MUST use
        # distinct slugs or the second arm short-circuits here.
        mp3 = _mp3_path_for(req.name, req.slug)
        if mp3.exists():
            job = Job(job_id=job_id, name=req.name, voice=req.voice,
                      duration=req.duration, lyrics=req.lyrics, slug=req.slug,
                      guidance_scale_text=req.guidance_scale_text,
                      guidance_scale_lyric=req.guidance_scale_lyric,
                      qc=req.qc, qc_max_retries=req.qc_max_retries,
                      status="done", mp3_path=str(mp3),
                      finished_at=datetime.now().isoformat(timespec="seconds"))
            _JOBS[job_id] = job
            return _status_payload(job)

        # Fresh (or a previously-errored) job → (re)queue it.
        job = Job(job_id=job_id, name=req.name, voice=req.voice,
                  duration=req.duration, lyrics=req.lyrics, slug=req.slug,
                  guidance_scale_text=req.guidance_scale_text,
                  guidance_scale_lyric=req.guidance_scale_lyric,
                  qc=req.qc, qc_max_retries=req.qc_max_retries,
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
