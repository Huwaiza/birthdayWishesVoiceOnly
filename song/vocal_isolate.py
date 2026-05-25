"""
Vocal isolation for the birthday song generator (v7).

ACE-Step always renders a full mix — vocals AND backing instruments — in
one pass. To get a true instrument-free (a cappella) result we take that
finished mix and run it through a vocal-separation model, keeping only the
vocal stem.

This is a *guaranteed* instrument-free result: unlike prompting ACE-Step
for "a cappella" (which only nudges a generative model), the separator
physically pulls the recorded audio apart into stems and we discard
everything that isn't the voice.

Why Roformer, not Demucs
------------------------
We use a **Roformer** model (via the `audio-separator` package) rather than
Demucs. Roformer is the current state of the art for vocal separation: it
leaves essentially no instrument bleed — no faint background pads, and no
leaking of an instrumental intro/outro into the vocal stem (the two things
Demucs's default model is weakest at).

We deliberately do a **single pass**. A second "cleanup" pass would scrub
a touch more residue but also starts adding a faint, watery artefact to
the voice itself — not worth it when the whole point is a pristine vocal.

`audio-separator` is imported lazily inside isolate_vocals(), so this
module imports fine even when the package isn't installed — the dependency
is only needed at render time, and the error message points at the fix:

    pip install "audio-separator[cpu]"
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

# BS-Roformer — a state-of-the-art vocal-separation model (SDR ~12.98).
# Its weights (~200 MB) download once on first use into the cache dir below.
# To try a different model, run `audio-separator --list_models` and swap
# this string (the WAV cache for --acapella is keyed separately, so a model
# change just means the next render re-isolates).
_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Persist downloaded model weights here so they survive reboots — the
# audio-separator default (/tmp) does not, which would re-download ~200 MB
# every restart.
_MODEL_CACHE_DIR = Path.home() / ".cache" / "audio-separator-models"


def isolate_vocals(
    input_wav: Path,
    output_wav: Path,
    verbose: bool = True,
) -> Path:
    """
    Separate and keep only the vocal stem of a full-mix WAV.

    Uses a Roformer separation model (state of the art for vocal
    isolation) in a single pass — instruments are removed cleanly while
    the vocal itself is left untouched.

    Parameters
    ----------
    input_wav : Path
        The full-mix WAV (vocals + instruments) produced by ACE-Step.
    output_wav : Path
        Where to write the isolated-vocal WAV.
    verbose : bool
        Print progress lines.

    Returns
    -------
    Path to output_wav.

    Raises
    ------
    FileNotFoundError
        If input_wav does not exist.
    RuntimeError
        If audio-separator is not installed or fails to produce a vocal stem.
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"input WAV not found: {input_wav}")

    # Lazy import: keeps this module importable without audio-separator
    # installed (the unit tests rely on that), and defers a heavy import.
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise RuntimeError(
            "audio-separator is not installed — it powers --acapella.\n"
            "    pip install \"audio-separator[cpu]\""
        ) from e

    # The separator writes <input>_(Vocals)_<model>.wav and the matching
    # _(Instrumental)_ file into output_dir; we keep only the vocal one.
    work_dir = output_wav.parent / "_separator_tmp"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[vocal_isolate] separating vocals from {input_wav.name} "
              f"(Roformer) — first run downloads the model (~200 MB)...")
    t0 = time.time()

    separator = Separator(
        output_dir=str(work_dir),
        output_format="WAV",
        model_file_dir=str(_MODEL_CACHE_DIR),
        log_level=logging.INFO if verbose else logging.WARNING,
    )
    try:
        separator.load_model(model_filename=_ROFORMER_MODEL)
        outputs = separator.separate(str(input_wav))
    except Exception as e:  # noqa: BLE001 — surface any separator failure cleanly
        raise RuntimeError(
            f"vocal separation failed: {e}\n"
            f"If the model name is unrecognised, run `audio-separator "
            f"--list_models` and update _ROFORMER_MODEL in song/vocal_isolate.py."
        ) from e

    # Identify the vocal stem. The separator returns the output file names;
    # the vocal one is simply the file that is NOT the instrumental.
    vocal_file = _pick_vocal_stem(outputs, work_dir)
    if vocal_file is None:
        raise RuntimeError(
            f"separation ran but no vocal stem was found in {work_dir}"
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(vocal_file), str(output_wav))
    shutil.rmtree(work_dir, ignore_errors=True)

    dt = time.time() - t0
    if verbose:
        size_mb = output_wav.stat().st_size / (1024 * 1024)
        print(f"[vocal_isolate] done in {dt:.0f}s — {output_wav.name} "
              f"({size_mb:.1f} MB, instruments removed)")
    return output_wav


def _pick_vocal_stem(outputs, work_dir: Path):
    """Return the vocal-stem WAV from a separator run, or None.

    `outputs` is whatever Separator.separate() returned (file names or
    paths); we also fall back to scanning work_dir directly. The vocal
    stem is the produced WAV that is not the instrumental one.
    """
    candidates = []
    for entry in (outputs or []):
        p = Path(entry)
        if not p.is_absolute():
            p = work_dir / p.name
        candidates.append(p)
    if not candidates:
        candidates = sorted(work_dir.glob("*.wav"))

    for p in candidates:
        name = p.name.lower()
        if not p.exists():
            continue
        if "instrumental" in name or "no_vocal" in name or "(no vocals)" in name:
            continue
        if "vocal" in name:
            return p
    # Last resort: any produced WAV that isn't the instrumental.
    for p in candidates:
        if p.exists() and "instrumental" not in p.name.lower():
            return p
    return None
