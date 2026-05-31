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


# --- Job bookkeeping --------------------------------------------------------

@dataclass
class Job:
    job_id: str
    name: str
    voice: Optional[int]
    duration: float
    status: str = "queued"          # queued → running → done | error
    mp3_path: Optional[str] = None
    error: Optional[str] = None
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: Optional[str] = None


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_WORK_QUEUE: "queue.Queue[str]" = queue.Queue()

# Set once the ACE-Step model has finished loading.
_MODEL_READY = threading.Event()

# Guards against starting the warmer/worker more than once. Production runs a
# single lifespan so it never matters; under the test client, which enters
# lifespan once per TestClient context, it keeps us to exactly one worker.
_BG_STARTED = False


def _safe_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def _job_id(name: str, voice: Optional[int], duration: float) -> str:
    """Deterministic id from the render parameters, so re-submitting the
    same request dedupes onto the same job instead of rendering twice."""
    key = f"{_safe_name(name)}|{voice}|{duration:.1f}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _mp3_path_for(name: str) -> Path:
    """Predictable, username-based output path — what the video step picks
    up. Note: voice/duration are NOT in the filename, so changing those for
    the same name overwrites the previous MP3 (fine for one-song-per-user)."""
    return OUTPUT_DIR / f"happy_birthday_{_safe_name(name)}.mp3"


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
    )

    vocals = wav.with_name(wav.stem + "__vocals.wav")
    if not vocals.exists():
        isolate_vocals(wav, vocals, verbose=False)

    tight = wav.with_name(wav.stem + "__vocal_tight.wav")
    if not tight.exists():
        tighten_pauses(vocals, tight, verbose=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mp3 = _mp3_path_for(job.name)
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


def _worker() -> None:
    """Single consumer — pulls one job at a time and runs it to completion."""
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
            print(f"[service] rendering job {job_id} ({job.name})")
            t0 = time.time()
            mp3 = _generate(job)
            with _JOBS_LOCK:
                job.status = "done"
                job.mp3_path = str(mp3)
                job.finished_at = datetime.now().isoformat(timespec="seconds")
            print(f"[service] job {job_id} done in {time.time() - t0:.0f}s → {mp3}")
        except Exception as e:  # noqa: BLE001 — report failure, keep serving
            with _JOBS_LOCK:
                job.status = "error"
                job.error = str(e)
                job.finished_at = datetime.now().isoformat(timespec="seconds")
            print(f"[service] job {job_id} FAILED: {e}")
        finally:
            _WORK_QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BG_STARTED
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _BG_STARTED:
        threading.Thread(target=_warm_model, name="model-warmer", daemon=True).start()
        threading.Thread(target=_worker, name="render-worker", daemon=True).start()
        _BG_STARTED = True
    yield


app = FastAPI(title="Birthday vocal-only service", lifespan=lifespan)


# --- API --------------------------------------------------------------------

class JobRequest(BaseModel):
    name: str
    voice: Optional[int] = None
    duration: float = DEFAULT_DURATION_S


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

    job_id = _job_id(req.name, req.voice, req.duration)

    with _JOBS_LOCK:
        existing = _JOBS.get(job_id)
        # Already known and still relevant → just return its state (dedupe).
        if existing and existing.status in ("queued", "running", "done"):
            return _status_payload(existing)

        # Completed in a previous run (process restarted) but the MP3 is
        # still on disk → report done without re-rendering.
        mp3 = _mp3_path_for(req.name)
        if mp3.exists():
            job = Job(job_id=job_id, name=req.name, voice=req.voice,
                      duration=req.duration, status="done", mp3_path=str(mp3),
                      finished_at=datetime.now().isoformat(timespec="seconds"))
            _JOBS[job_id] = job
            return _status_payload(job)

        # Fresh (or a previously-errored) job → (re)queue it.
        job = Job(job_id=job_id, name=req.name, voice=req.voice,
                  duration=req.duration, status="queued")
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
