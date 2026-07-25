"""
Shared memory-release helpers for the render pipeline.

Why this module exists
----------------------
A render is a heavy in-process model call, and on Apple Silicon there is no
separate VRAM pool — MPS allocates out of the same unified memory the OS and
every other app share. Torch's caching allocator keeps freed blocks *reserved*
rather than handing them back, so across a long session of back-to-back renders
the process's footprint only climbs. Past a certain point macOS starts swapping,
a render that took ~8 minutes crawls for hours, and the machine surfaces the
"force quit applications to free memory" dialog.

`service/app.py` already had a private copy of this release routine; it now
lives here so the service, the batch runner, and anything added later all
release memory the same way instead of drifting apart.

Nothing here touches the model singletons — see `free_all_models()` for that.
"""

from __future__ import annotations

import gc
from typing import Optional


def free_render_memory() -> None:
    """Release a finished render's working set back to the OS.

    Drops Python-side intermediates, then asks torch's caching allocator to
    return its reserved-but-unused blocks. Best-effort by design: torch may be
    absent (unit tests) or the backend unavailable, and neither is worth
    failing a completed render over.

    The model singletons are deliberately NOT touched — only per-render
    intermediates. Use `free_all_models()` when you want a full reset.
    """
    gc.collect()
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort; torch may be absent
        pass


def free_all_models() -> None:
    """Drop every cached model singleton, then release their memory.

    This is the in-process equivalent of restarting the service: the next
    render pays the model-load cost again (~1-2 min) but starts from a clean,
    unfragmented allocator instead of inheriting hours of accumulated pressure.

    Used by the periodic soft reload so a bare `uvicorn` (no supervisor to
    respawn it) can still get a clean slate without the process exiting.
    """
    # Imported here, not at module scope, so this module stays importable in
    # environments where the heavy render deps aren't installed.
    try:
        from . import acestep_render

        acestep_render.unload_pipeline()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    try:
        from . import vocal_isolate

        vocal_isolate.unload_separator()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    free_render_memory()


def rss_gb() -> Optional[float]:
    """Current process resident set size in GB, or None if psutil is absent.

    Purely diagnostic: the whole point of the fixes around this module is that
    the footprint stops climbing across renders, and that claim is only worth
    anything if it is actually measured. Callers log this per render.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:  # noqa: BLE001 — diagnostics must never break a render
        return None


def format_rss(prefix: str = "rss") -> str:
    """`'rss=7.42GB'`, or an empty string when psutil isn't installed, so
    callers can drop this straight into a log line without a conditional."""
    gb = rss_gb()
    return f"{prefix}={gb:.2f}GB" if gb is not None else ""
