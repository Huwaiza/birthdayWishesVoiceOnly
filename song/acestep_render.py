"""
ACE-Step renderer for the birthday song generator (v7).

One call → one ~2:30 birthday song WAV. ACE-Step generates vocals AND
backing in a single pass conditioned on a natural-language prompt and
[verse]/[bridge]/[chorus]/[outro]-tagged lyrics.

Voice rotation is via prompt variation + seed (see voice_profiles.py).
Caching is keyed on (name, voice_index, lyrics_hash, prompt_hash) so a
prompt or lyric tweak invalidates the cache for everyone.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .lyrics import build_lyrics
from .voice_profiles import VOICE_PROFILES, VoiceProfile, pick_voice, seed_for

# Singleton — the ACE-Step pipeline is loaded once per Python process and
# reused. It auto-detects MPS on Apple Silicon.
_PIPELINE = None


def _get_pipeline(precision: str = "float16"):
    """Lazy-load the ACE-Step pipeline. First call downloads ~4GB of weights
    from Hugging Face into ~/.cache/ace-step/checkpoints — only happens once.

    precision : "float16" (default) or "float32".
        ACE-Step normally forces float32 on Apple Silicon (MPS). That makes
        the 3.5B model ~14 GB in RAM — on a Mac with <=18 GB it overflows
        memory and macOS swaps to disk, turning a few-minute render into a
        multi-hour one. We override to float16 via the ACE_PIPELINE_DTYPE
        env var that ACE-Step honours: the model drops to ~7 GB and fits in
        RAM with no swapping. float16 is the standard precision for diffusion
        inference, so the quality difference versus float32 is negligible.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # Must be set BEFORE the pipeline is constructed — ACE-Step reads this
    # env var inside its __init__. setdefault means an externally-set value
    # (e.g. `export ACE_PIPELINE_DTYPE=float32` in your shell) still wins.
    os.environ.setdefault("ACE_PIPELINE_DTYPE", precision)

    # Defer the heavy import until first call so unit tests / CLI --help
    # don't pay the 5–10s torch import cost.
    from acestep.pipeline_ace_step import ACEStepPipeline

    _PIPELINE = ACEStepPipeline(
        checkpoint_dir=None,           # auto = ~/.cache/ace-step/checkpoints
        dtype="bfloat16",              # see ACE_PIPELINE_DTYPE override above
        torch_compile=False,           # MPS doesn't benefit from compile
        cpu_offload=False,             # offload trades speed for RAM — slower
        overlapped_decode=False,
    )
    try:
        print(f"[acestep_render] pipeline ready — device={_PIPELINE.device} "
              f"dtype={_PIPELINE.dtype}")
    except Exception:
        pass
    return _PIPELINE


def _hash(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:12]


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time, e.g. 4483.0 -> '1h 14m 43s', 92 -> '1m 32s'."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def render(
    name: str,
    voice_index: Optional[int] = None,
    duration_s: float = 150.0,
    output_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    infer_step: int = 60,
    guidance_scale: float = 15.0,
    use_cache: bool = True,
    verbose: bool = True,
    precision: str = "float16",
    lyrics: Optional[str] = None,
) -> Path:
    """
    Render one birthday song end-to-end.

    Parameters
    ----------
    name : str
        Recipient's name. Injected into [verse]/[chorus]/[outro] lyrics.
    lyrics : str or None
        Explicit ACE-Step tagged lyrics. None → build_lyrics(name) (birthday).
    voice_index : int or None
        Pin a specific voice profile (0..N-1). None → hash(name)-based rotation.
    duration_s : float
        Target song length, seconds. ACE-Step honours this fairly precisely.
    output_path : Path or None
        Where to write the WAV. If None, writes to cache_dir/<key>.wav.
    cache_dir : Path or None
        Cache root. Default: <project_root>/cache/acestep/.
    infer_step : int
        Diffusion steps. 60 is the documented sweet-spot.
    guidance_scale : float
        Classifier-free guidance. 15 is the documented default.
    use_cache : bool
        If True, skip render when a matching cached WAV already exists.
    verbose : bool
        Print progress lines.
    precision : str
        "float16" (default) or "float32". float16 halves the model's RAM
        footprint (~7 GB vs ~14 GB) and prevents disk swapping on Macs with
        <=18 GB. See _get_pipeline() for the full rationale.

    Returns
    -------
    Path to the rendered WAV.
    """
    profile = pick_voice(name, override_index=voice_index)
    seed = seed_for(profile, name)
    lyrics = lyrics if lyrics is not None else build_lyrics(name)

    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "cache" / "acestep"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _hash(
        name.strip().lower(),
        profile.name,
        str(seed),
        profile.prompt,
        lyrics,
        f"{duration_s:.1f}",
        f"{infer_step}",
        f"{guidance_scale:.2f}",
    )
    cached_wav = cache_dir / f"{name.lower().replace(' ', '_')}__{profile.name}__{cache_key}.wav"

    target_wav = Path(output_path) if output_path else cached_wav

    if use_cache and cached_wav.exists():
        if verbose:
            print(f"[acestep_render] cache hit: {cached_wav.name}")
        if target_wav != cached_wav:
            import shutil
            shutil.copy2(cached_wav, target_wav)
        return target_wav

    if verbose:
        print(f"[acestep_render] rendering '{name}' → voice={profile.name} seed={seed}")
        print(f"[acestep_render] duration={duration_s:.0f}s steps={infer_step} cfg={guidance_scale}")

    pipeline = _get_pipeline(precision=precision)

    started_at = datetime.now()
    t0 = time.time()
    if verbose:
        print(f"[acestep_render] render started  {started_at:%Y-%m-%d %H:%M:%S}")
    pipeline(
        format="wav",
        audio_duration=float(duration_s),
        prompt=profile.prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=str(seed),
        save_path=str(cached_wav),
        batch_size=1,
    )
    dt = time.time() - t0
    finished_at = datetime.now()

    if not cached_wav.exists():
        raise RuntimeError(f"ACE-Step did not produce {cached_wav}")

    if verbose:
        size_mb = cached_wav.stat().st_size / (1024 * 1024)
        print(f"[acestep_render] render finished {finished_at:%Y-%m-%d %H:%M:%S}")
        print(f"[acestep_render] render took {format_duration(dt)} "
              f"({dt:.0f}s) — {size_mb:.1f} MB")

    if target_wav != cached_wav:
        import shutil
        shutil.copy2(cached_wav, target_wav)

    return target_wav


def list_voices():
    """Return the list of voice profiles for CLI display."""
    return list(VOICE_PROFILES)
