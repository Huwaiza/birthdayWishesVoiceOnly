#!/usr/bin/env bash
#
# Supervisor for the vocal-only service.
#
# Why this exists: a render is a heavy in-process model call, and memory
# pressure (MPS/RAM the allocator never hands back) accumulates across renders
# until one swaps to disk and turns into a multi-hour / hung render — which
# wedges the single worker for everything behind it. Two guards handle that, and
# both need a process that comes BACK after it exits — that's this loop:
#
#   * RENDER_RESTART_AFTER_JOBS — proactively recycle after N renders, while the
#     queue is idle, so each batch starts on a clean process (model reloads
#     once, ~1-2 min).
#   * RENDER_WATCHDOG_HARD_RESTART — reactively exit if a render hangs past the
#     hard limit, so a wedge can't strand the queue.
#
# Run THIS instead of bare uvicorn:  ./run_service.sh
# Stop for good with Ctrl-C (the loop exits on SIGINT).
set -u
cd "$(dirname "$0")"

# Recycle after this many renders (each name = 1 render). 0 disables. The Django
# batch is 10 names/run, so 10 lines each batch up with a fresh process.
export RENDER_RESTART_AFTER_JOBS="${RENDER_RESTART_AFTER_JOBS:-10}"
# Also bail out if a single render hangs past the watchdog's hard limit.
export RENDER_WATCHDOG_HARD_RESTART="${RENDER_WATCHDOG_HARD_RESTART:-1}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
PYBIN="${PYBIN:-venv-diffsinger/bin/uvicorn}"

trap 'echo "[run_service] stopping."; exit 0' INT TERM

while true; do
  echo "[run_service] starting uvicorn on ${HOST}:${PORT} " \
       "(recycle after ${RENDER_RESTART_AFTER_JOBS} jobs, hard-restart-on-hang=${RENDER_WATCHDOG_HARD_RESTART})…"
  "$PYBIN" service.app:app --host "$HOST" --port "$PORT" --workers 1
  code=$?
  echo "[run_service] service exited (${code}); respawning in 2s…"
  sleep 2
done
