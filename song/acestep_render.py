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
from pathlib import Path
from typing import Optional

from .lyrics import build_lyrics
from .voice_profiles import VOICE_PROFILES, VoiceProfile, pick_voice, seed_for

# Singleton — ACE-Step takes ~30s to warm up, ~3GB RAM. We load once per
# Python process. The pipeline auto-detects MPS on Apple Silicon.
_PIPELINE = None


def _get_pipeline():
    """Lazy-load the ACE-Step pipeline. First call downloads ~4GB of weights
    from Hugging Face into ~/.cache/ace-step/checkpoints — only happens once."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # Defer the heavy import until first call so unit tests / CLI --help
    # don't pay the 5–10s torch import cost.
    from acestep.pipeline_ace_step import ACEStepPipeline

    _PIPELINE = ACEStepPipeline(
        checkpoint_dir=None,           # auto = ~/.cache/ace-step/checkpoints
        dtype="bfloat16",              # auto-falls back to float32 on MPS
        torch_compile=False,           # MPS doesn't benefit from compile
        cpu_offload=False,
        overlapped_decode=False,
    )
    return _PIPELINE


def _hash(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:12]


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
) -> Path:
    """
    Render one birthday song end-to-end.

    Parameters
    ----------
    name : str
        Recipient's name. Injected into [verse]/[chorus]/[outro] lyrics.
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

    Returns
    -------
    Path to the rendered WAV.
    """
    profile = pick_voice(name, override_index=voice_index)
    seed = seed_for(profile, name)
    lyrics = build_lyrics(name)

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

    pipeline = _get_pipeline()

    t0 = time.time()
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

    if not cached_wav.exists():
        raise RuntimeError(f"ACE-Step did not produce {cached_wav}")

    if verbose:
        size_mb = cached_wav.stat().st_size / (1024 * 1024)
        print(f"[acestep_render] done in {dt:.1f}s — {size_mb:.1f} MB")

    if target_wav != cached_wav:
        import shutil
        shutil.copy2(cached_wav, target_wav)

    return target_wav


def list_voices():
    """Return the list of voice profiles for CLI display."""
    return list(VOICE_PROFILES)
