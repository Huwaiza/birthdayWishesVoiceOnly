"""
Shared vocals-only pipeline: render → isolate → tighten → QC → retry.

One function all three entry points (generate_birthday.py, batch_render.py,
service/app.py) call, so the QC gate and its seed-retry policy live in
exactly one place.

Retry semantics: a QC failure means ACE-Step drew a bad seed (dropped or
substituted lyric lines), so the retry re-renders with seed_offset=attempt —
still fully deterministic per (name, voice, attempt), and cached separately
because the seed participates in the render cache key. If every attempt
fails QC, the best-scoring attempt is returned (with its failing report)
rather than raising: a slightly-flawed song beats no song, and the caller
can see qc.passed == False and decide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .qc import QCReport


def generate_vocal_song(
    name: str,
    *,
    voice_index: Optional[int] = None,
    duration_s: float = 150.0,
    infer_step: int = 60,
    guidance_scale: float = 15.0,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    use_cache: bool = True,
    verbose: bool = True,
    precision: str = "float16",
    lyrics: Optional[str] = None,
    qc_enabled: bool = True,
    qc_max_retries: int = 1,
) -> Tuple[Path, Optional[QCReport]]:
    """
    Produce the final tightened vocals-only WAV for one song.

    Returns
    -------
    (tight_wav, report)
        report is None when qc_enabled is False. When QC ran, report.passed
        says whether the returned WAV cleared the gate (False = best of the
        failed attempts).
    """
    # Module-attribute imports (not `from x import y`) so tests can
    # monkeypatch the underlying functions on their home modules.
    from song import acestep_render as ar
    from song import vocal_isolate as vi

    attempts = (max(0, qc_max_retries) + 1) if qc_enabled else 1
    best: Optional[Tuple[Path, QCReport]] = None

    for attempt in range(attempts):
        wav = ar.render(
            name=name,
            voice_index=voice_index,
            duration_s=duration_s,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            use_cache=use_cache,
            verbose=verbose,
            precision=precision,
            lyrics=lyrics,
            seed_offset=attempt,
        )

        vocals = wav.with_name(wav.stem + "__vocals.wav")
        if not (use_cache and vocals.exists()):
            vi.isolate_vocals(wav, vocals, verbose=verbose)
        elif verbose:
            print(f"[pipeline] cached vocal stem: {vocals.name}")

        tight = wav.with_name(wav.stem + "__vocal_tight.wav")
        if not (use_cache and tight.exists()):
            vi.tighten_pauses(vocals, tight, verbose=verbose)
        elif verbose:
            print(f"[pipeline] cached tightened vocal: {tight.name}")

        if not qc_enabled:
            return tight, None

        from song import qc as qc_mod
        report = qc_mod.run_qc(
            tight,
            name=name,
            duration_s=duration_s,
            custom_lyrics=lyrics is not None,
            use_cache=use_cache,
            verbose=verbose,
            attempt=attempt,
        )
        if report.passed:
            return tight, report

        if best is None or report.score > best[1].score:
            best = (tight, report)
        if attempt < attempts - 1 and verbose:
            print(f"[pipeline] QC failed (attempt {attempt + 1}/{attempts}) — "
                  f"re-rendering with seed offset {attempt + 1}")

    if verbose:
        print(f"[pipeline] all {attempts} attempt(s) failed QC — "
              f"keeping best attempt {best[1].attempt}")
    return best
