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

The Separator is a process-wide singleton
-----------------------------------------
Loading a model is expensive (weights off disk, tensors allocated) and this
process renders song after song, so the separator is built ONCE and reused —
the same treatment ACE-Step and Whisper already get. Rebuilding it per render
re-allocated the model ~20x a day inside a long-lived process, and torch's
caching allocator does not hand that memory back cleanly, so the churn showed
up as steadily climbing memory pressure over a session.

Reuse is safe by the library's own design: `Separator.separate()` calls
`clear_gpu_cache()` and `clear_file_specific_paths()` after every file, so no
per-file state survives into the next call. The one constraint is that
`output_dir` is baked into the model instance at `load_model()` time and cannot
change per call — hence the fixed work directory below, out of which each
finished stem is moved to wherever the caller asked for it.
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

# Fixed scratch directory the separator writes its stems into. It has to be
# fixed (not derived from each output path) because `output_dir` is captured
# when the model loads and can't be varied per call on a reused instance — see
# the module docstring. Stems are moved out of here immediately, so it only
# ever holds one render's pair of files.
_WORK_DIR = Path(__file__).resolve().parent.parent / "cache" / "_separator_tmp"

# Process-wide Separator singleton, loaded on first use and reused for every
# subsequent render. None = not built yet.
_SEPARATOR = None


def _get_separator(verbose: bool = True):
    """Build (once) and return the Roformer separator.

    Raises RuntimeError with an actionable message when audio-separator isn't
    installed — the import is deliberately lazy so this module stays importable
    without it (the unit tests rely on that).
    """
    global _SEPARATOR
    if _SEPARATOR is not None:
        return _SEPARATOR

    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise RuntimeError(
            "audio-separator is not installed — it powers --acapella.\n"
            "    pip install \"audio-separator[cpu]\""
        ) from e

    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[vocal_isolate] loading Roformer separator (once per process) — "
              f"first run downloads the model (~200 MB)...")
    t0 = time.time()
    separator = Separator(
        output_dir=str(_WORK_DIR),
        output_format="WAV",
        model_file_dir=str(_MODEL_CACHE_DIR),
        log_level=logging.INFO if verbose else logging.WARNING,
    )
    try:
        separator.load_model(model_filename=_ROFORMER_MODEL)
    except Exception as e:  # noqa: BLE001 — surface any load failure cleanly
        raise RuntimeError(
            f"loading the vocal separation model failed: {e}\n"
            f"If the model name is unrecognised, run `audio-separator "
            f"--list_models` and update _ROFORMER_MODEL in song/vocal_isolate.py."
        ) from e

    _SEPARATOR = separator
    if verbose:
        print(f"[vocal_isolate] separator ready in {time.time() - t0:.0f}s")
    return _SEPARATOR


def unload_separator() -> None:
    """Drop the cached separator so its memory can be reclaimed.

    Called by song.memory.free_all_models() during a periodic soft reload. The
    next isolate_vocals() rebuilds it. Safe to call when nothing is loaded.
    """
    global _SEPARATOR
    _SEPARATOR = None


def _clear_work_dir() -> None:
    """Empty the scratch directory without removing it — the loaded model holds
    the path, so the directory itself must survive between renders. Clearing is
    what stops a previous render's stems from being mistaken for this one's."""
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    for leftover in _WORK_DIR.iterdir():
        try:
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.unlink()
        except OSError:
            pass


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

    # Built once per process and reused; see the module docstring for why.
    separator = _get_separator(verbose=verbose)

    # The separator writes <input>_(Vocals)_<model>.wav and the matching
    # _(Instrumental)_ file into its fixed work dir; we keep only the vocal one.
    # Clear first so a previous render's stems can't be picked up as this one's.
    _clear_work_dir()

    if verbose:
        print(f"[vocal_isolate] separating vocals from {input_wav.name} (Roformer)...")
    t0 = time.time()

    try:
        outputs = separator.separate(str(input_wav))
    except Exception as e:  # noqa: BLE001 — surface any separator failure cleanly
        raise RuntimeError(f"vocal separation failed: {e}") from e

    # Identify the vocal stem. The separator returns the output file names;
    # the vocal one is simply the file that is NOT the instrumental.
    vocal_file = _pick_vocal_stem(outputs, _WORK_DIR)
    if vocal_file is None:
        raise RuntimeError(
            f"separation ran but no vocal stem was found in {_WORK_DIR}"
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(vocal_file), str(output_wav))
    # Drop the instrumental stem now rather than leaving a full-length WAV on
    # disk until the next render needs the directory.
    _clear_work_dir()

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
# Even with an a cappella result, ACE-Step composed a song *arrangement*, so
# the isolated vocal still has the ghosts of the instrumental intro, fills,
# and outro — long stretches that are now pure silence. tighten_pauses()
# shrinks ONLY those long silent gaps so the song doesn't drag, while leaving
# every sung note exactly as performed.
#
# Melody safety is the whole point here. The detector is deliberately timid:
#
#   * Threshold is _SILENCE_DBFS, an ABSOLUTE floor far below any sung note.
#     A clean Roformer stem's silence sits near digital zero (~-90 dBFS),
#     while even the softest sung tail stays well above -50 dBFS — so notes
#     are never mistaken for silence and cut. (My earlier attempt used a
#     level *relative* to the track's loudness, which clipped soft singing.)
#   * Only gaps >= _LONG_GAP_MS (a full second) are touched. Shorter pauses —
#     breaths, phrase ends, the space between words — are left untouched.
#   * Each kept phrase is padded by _PHRASE_PAD_MS so note onsets and decay
#     tails survive the cut, then given a tiny _FADE_MS fade purely to avoid
#     a click at the splice (not enough to be audible as a fade).

# A silent stretch at least this long (ms) is a leftover instrumental gap and
# gets collapsed; anything shorter is natural phrasing and is kept verbatim.
_LONG_GAP_MS = 1000
# Collapsed gaps become this much silence — one natural breath.
_KEPT_GAP_MS = 450
# Leading / trailing silence is trimmed down to this small, even pad.
_EDGE_PAD_MS = 250
# Audio kept on each side of a phrase so note onsets/tails aren't clipped.
_PHRASE_PAD_MS = 120
# Tiny fade on each phrase edge to declick the splice (inaudible as a fade).
_FADE_MS = 8
# Absolute silence floor (dBFS). Well below the softest sung note, well above
# a clean vocal stem's near-digital-silence noise floor.
_SILENCE_DBFS = -55.0


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
    Collapse only the long silent gaps in an isolated-vocal WAV.

    Detects silences at least ~1 s long — the ghosts of the instrumental
    intro / fills / outro — and shrinks each to a single natural breath
    (~0.45 s), then trims the silent head and tail. Short pauses and every
    sung note are left untouched, so the melody is preserved exactly and the
    song simply stops dragging.

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

    # pydub is a hard project dependency, but imported lazily so this module
    # stays importable on its own (the unit tests rely on that).
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    t0 = time.time()
    audio = AudioSegment.from_wav(str(input_wav))
    total_ms = len(audio)

    # Only silences >= _LONG_GAP_MS split the audio; shorter pauses stay
    # inside a non-silent run and are therefore preserved untouched.
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=_LONG_GAP_MS,
        silence_thresh=_SILENCE_DBFS,
        seek_step=5,
    )

    if not nonsilent:
        # No singing detected (or the file is unexpectedly quiet) — copy it
        # through untouched rather than emit an empty file.
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_wav), str(output_wav))
        if verbose:
            print(f"[tighten_pauses] no audio detected — copied "
                  f"{input_wav.name} unchanged")
        return output_wav

    kept_gap = _silence(_KEPT_GAP_MS, audio)
    out = AudioSegment.empty()
    for i, (start, end) in enumerate(nonsilent):
        # Pad each phrase so onsets/decay tails survive; fade only to declick.
        s = max(0, start - _PHRASE_PAD_MS)
        e = min(total_ms, end + _PHRASE_PAD_MS)
        chunk = audio[s:e].fade_in(_FADE_MS).fade_out(_FADE_MS)
        if i > 0:
            out += kept_gap
        out += chunk

    edge = _silence(_EDGE_PAD_MS, audio)
    out = edge + out + edge

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(output_wav), format="wav")

    dt = time.time() - t0
    if verbose:
        trimmed = (total_ms - len(out)) / 1000.0
        print(f"[tighten_pauses] {len(nonsilent)} phrase group(s), "
              f"{total_ms / 1000:.1f}s → {len(out) / 1000:.1f}s "
              f"(removed {trimmed:.1f}s of dead air) in {dt:.0f}s")
    return output_wav
