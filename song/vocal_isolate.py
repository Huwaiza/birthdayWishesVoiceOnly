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
# this string (then re-run with --no-cache so the new model re-isolates —
# the isolated-vocal WAV is otherwise cached by filename).
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
            "audio-separator is not installed — it powers the default "
            "vocals-only mode.\n"
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


# --- Pause tightening -------------------------------------------------------
#
# ACE-Step composes a song *arrangement*, so even an a cappella-prompted
# render leaves some instrumental space — a short intro, fills between
# sections, an outro. Once the backing is separated out, those stretches
# become long silences. tighten_pauses() collapses them so the vocal-only
# track is continuous and production-ready.

# A silent stretch longer than this (ms) is treated as a leftover
# instrumental gap and collapsed; anything shorter is natural phrasing
# and is kept exactly as the singer performed it.
_LONG_GAP_MS = 700
# Collapsed gaps become this much silence — one natural breath.
_KEPT_GAP_MS = 350
# Leading / trailing silence is trimmed down to this small, even pad.
_EDGE_PAD_MS = 150
# A little audio kept on each side of a phrase so consonants at the
# start/end of a line aren't clipped off.
_PHRASE_PAD_MS = 60


def _silence(ms, like):
    """A silent AudioSegment matching another segment's audio format."""
    from pydub import AudioSegment
    seg = AudioSegment.silent(duration=ms, frame_rate=like.frame_rate)
    return seg.set_channels(like.channels).set_sample_width(like.sample_width)


def tighten_pauses(
    input_wav: Path,
    output_wav: Path,
    verbose: bool = True,
) -> Path:
    """
    Collapse the long silent gaps in an isolated-vocal WAV.

    Detects silences longer than ~0.7 s — the ghosts of the instrumental
    intro / fills / outro — and shrinks each to a single natural breath
    (~0.35 s), then trims the silent head and tail. Short pauses are left
    untouched, so the result still breathes like a real performance
    instead of sounding rushed.

    Parameters
    ----------
    input_wav : Path
        Isolated-vocal WAV (the output of isolate_vocals()).
    output_wav : Path
        Where to write the tightened WAV.
    verbose : bool
        Print progress lines.

    Returns
    -------
    Path to output_wav.

    Raises
    ------
    FileNotFoundError
        If input_wav does not exist.
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"input WAV not found: {input_wav}")

    # pydub is a hard project dependency, but imported lazily so this
    # module stays importable on its own (the unit tests rely on that).
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    t0 = time.time()
    audio = AudioSegment.from_wav(str(input_wav))
    total_ms = len(audio)

    # Threshold relative to the track's own loudness, so it adapts to
    # whatever absolute level the separator produced.
    silence_thresh = audio.dBFS - 16

    # Only silences >= _LONG_GAP_MS split the audio; shorter pauses stay
    # inside a non-silent run and are therefore preserved untouched.
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=_LONG_GAP_MS,
        silence_thresh=silence_thresh,
        seek_step=10,
    )

    if not nonsilent:
        # No singing detected (or the file is unexpectedly quiet) — copy
        # it through untouched rather than emit an empty file.
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_wav), str(output_wav))
        if verbose:
            print(f"[tighten_pauses] no audio detected — copied "
                  f"{input_wav.name} unchanged")
        return output_wav

    kept_gap = _silence(_KEPT_GAP_MS, audio)
    out = AudioSegment.empty()
    for i, (start, end) in enumerate(nonsilent):
        # Pad each phrase slightly so onsets/tails aren't clipped.
        s = max(0, start - _PHRASE_PAD_MS)
        e = min(total_ms, end + _PHRASE_PAD_MS)
        if i > 0:
            out += kept_gap
        out += audio[s:e]

    edge = _silence(_EDGE_PAD_MS, audio)
    out = edge + out + edge

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(output_wav), format="wav")

    dt = time.time() - t0
    if verbose:
        trimmed = (total_ms - len(out)) / 1000.0
        print(f"[tighten_pauses] {len(nonsilent)} phrase group(s), "
              f"{total_ms / 1000:.1f}s → {len(out) / 1000:.1f}s "
              f"(removed {trimmed:.1f}s of gaps) in {dt:.0f}s")
    return output_wav
