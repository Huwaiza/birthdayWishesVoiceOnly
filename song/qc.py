"""
Post-render quality control for the birthday song generator (v7).

ACE-Step's lyric-following is probabilistic and seed-dependent: on a bad
seed it drops or substitutes lines — usually the personalised "dear {name}"
ones — or spends half the render on instrumentals (which the vocal isolation
then turns into dead air). Nothing in render()/isolate/tighten can see that,
so without a check those songs ship silently.

This module measures the finished (tightened) vocal and decides pass/fail:

  1. Sung time — total voiced seconds via the same absolute -55 dBFS floor
     used by tighten_pauses(). Nearly free (pydub, no model). A 150 s render
     of the birthday template carries ~100-130 s of singing; well under that
     means whole sections went missing.
  2. Transcript checks (birthday template only) — Whisper (CPU, ~1 min)
     transcribes the stem, then we count "happy birthday" occurrences, look
     for one marker word per non-repeated section, and fuzzy-match the name
     (ASR mangles rare names — Jubayel comes back "Baile" — so this is a
     lenient similarity match, not an exact one).

The expensive measurements (sung time, transcript) are cached in a JSON
sidecar next to the tightened WAV, so re-running QC on a cached render is
instant. Thresholds are evaluated fresh each call, so tuning a constant
below doesn't force re-transcription.

Whisper is optional: if transformers/librosa are missing the transcript
checks are skipped (reported in the QCReport) and only sung time gates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

# Reuse the isolation module's silence floor so "sung" here means exactly
# what tighten_pauses() kept.
from .vocal_isolate import _SILENCE_DBFS

# --- Thresholds (evaluated fresh on every call; safe to tune) ---------------

# Minimum voiced seconds at the default 150 s duration; scaled linearly for
# other durations. Healthy renders measure ~90-130 s; the failure cases sit
# near 55 s.
MIN_SUNG_S = 80.0
# The template sings "happy birthday" 10+ times; ASR reliably catches most.
MIN_HB_COUNT = 5
# ...and an ABSURD count means Whisper repetition-looped, which it does on
# degenerate chant-like/mushy renders (the "eating words" failure: Shahrukh
# measured 102 in 94 s of singing — physically impossible). Healthy renders
# measure 13-26 against the template's 16, so 40 is comfortably above noise.
MAX_HB_COUNT = 40
# Lenient fuzzy floor for the name (see name_match_score). Catches "the name
# was never sung", tolerates ASR misspellings.
MIN_NAME_SCORE = 0.5
# How many of the three non-repeated sections must be detected.
MIN_SECTIONS = 2

# Gaps >= this (ms) separate phrases when measuring sung time. Shorter than
# tighten_pauses' 1 s so we measure singing, not phrase groups.
_PHRASE_GAP_MS = 300

# Whisper checkpoint. base is the sweet spot: tiny mishears singing badly,
# small triples the CPU time for little gain on this fixed lyric sheet.
ASR_MODEL = "openai/whisper-base"

# One marker hit from any tuple = that section was sung. Words chosen from
# lines unique to each section (the repeated "happy birthday to you" verses
# can't be distinguished from each other, so they're covered by MIN_HB_COUNT).
_SECTION_MARKERS = {
    "verse2": ("dream",),                                     # "may your dreams come true"
    "bridge": ("laughter", "happiness", "glad", "in our lives"),
    "chorus": ("hooray", "special day", "wishes", "sing for you"),
}

# Module-level ASR singleton: None = not loaded yet, False = unavailable.
_ASR = None


@dataclass
class QCReport:
    passed: bool
    reasons: List[str]                 # empty when passed
    sung_seconds: float
    phrase_count: int
    asr_used: bool
    hb_count: Optional[int] = None     # None when transcript checks skipped
    name_score: Optional[float] = None
    sections_found: Optional[List[str]] = None
    transcript: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    attempt: int = 0
    # Composite used to pick the least-bad attempt when every retry fails.
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        bits = [f"sung={self.sung_seconds:.0f}s", f"phrases={self.phrase_count}"]
        if self.hb_count is not None:
            bits.append(f"happy-birthday x{self.hb_count}")
        if self.name_score is not None:
            bits.append(f"name~{self.name_score:.2f}")
        if self.sections_found is not None:
            bits.append(f"sections={','.join(self.sections_found) or 'none'}")
        if not self.asr_used:
            bits.append("asr=off")
        line = f"{verdict}  ({'; '.join(bits)})"
        if self.reasons:
            line += "  reasons: " + "; ".join(self.reasons)
        return line


# --- Measurements ------------------------------------------------------------

def measure_sung_seconds(wav_path: Path) -> Tuple[float, int]:
    """Total voiced seconds and phrase count of a vocal WAV, using the same
    absolute silence floor as tighten_pauses()."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_wav(str(wav_path))
    spans = detect_nonsilent(
        audio,
        min_silence_len=_PHRASE_GAP_MS,
        silence_thresh=_SILENCE_DBFS,
        seek_step=5,
    )
    sung_s = sum(e - s for s, e in spans) / 1000.0
    return sung_s, len(spans)


def _get_asr():
    """Lazy-load the Whisper pipeline once per process (CPU). Returns None if
    the optional deps aren't installed — QC then degrades to sung-time only."""
    global _ASR
    if _ASR is False:
        return None
    if _ASR is None:
        try:
            from transformers import pipeline as hf_pipeline
            _ASR = hf_pipeline(
                "automatic-speech-recognition", model=ASR_MODEL, device=-1
            )
        except Exception as e:  # noqa: BLE001 — ASR is best-effort
            print(f"[qc] ASR unavailable ({e.__class__.__name__}: {e}) — "
                  f"transcript checks disabled; sung-time check still applies")
            _ASR = False
            return None
    return _ASR


def transcribe(wav_path: Path) -> Optional[str]:
    """Whisper transcript of a vocal WAV, or None when ASR is unavailable."""
    asr = _get_asr()
    if asr is None:
        return None
    try:
        import librosa
        y, _sr = librosa.load(str(wav_path), sr=16000, mono=True)
    except Exception as e:  # noqa: BLE001 — treat a load failure like no ASR
        print(f"[qc] could not load audio for ASR ({e}) — skipping transcript")
        return None
    out = asr(
        {"raw": y, "sampling_rate": 16000},
        chunk_length_s=30,               # required for clips > 30 s
        return_timestamps=True,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    return out["text"].strip()


# --- Transcript evaluation ----------------------------------------------------

def _norm_words(text: str) -> List[str]:
    cleaned = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text.lower())
    return cleaned.split()


def count_happy_birthday(transcript: str) -> int:
    return " ".join(_norm_words(transcript)).count("happy birthday")


def name_match_score(transcript: str, name: str) -> float:
    """Best fuzzy similarity (0..1) between the name and any transcript
    word n-gram. Lenient by design: ASR misspells sung names, so we only
    want to detect 'the name was never sung at all'."""
    words = _norm_words(transcript)
    target = "".join(_norm_words(name))
    if not target or not words:
        return 0.0
    n = max(1, len(_norm_words(name)))
    best = 0.0
    for size in range(max(1, n - 1), n + 2):
        for i in range(len(words) - size + 1):
            cand = "".join(words[i:i + size])
            best = max(best, SequenceMatcher(None, target, cand).ratio())
    return best


def sections_present(transcript: str) -> List[str]:
    joined = " ".join(_norm_words(transcript))
    return [sec for sec, markers in _SECTION_MARKERS.items()
            if any(m in joined for m in markers)]


def evaluate(
    *,
    name: str,
    duration_s: float,
    sung_seconds: float,
    phrase_count: int,
    transcript: Optional[str],
    custom_lyrics: bool = False,
    attempt: int = 0,
) -> QCReport:
    """Apply the pass/fail policy to measurements. Pure — no I/O, no model —
    so thresholds can be re-evaluated against cached measurements."""
    reasons: List[str] = []
    notes: List[str] = []

    min_sung = MIN_SUNG_S * (duration_s / 150.0)
    if sung_seconds < min_sung:
        reasons.append(f"only {sung_seconds:.0f}s sung (need >= {min_sung:.0f}s "
                       f"for a {duration_s:.0f}s render) — sections likely dropped")
    score = min(sung_seconds / max(min_sung, 1.0), 1.5)

    hb_count = name_score = sections = None
    asr_used = transcript is not None
    if custom_lyrics:
        notes.append("custom lyrics — transcript checks skipped (sung-time only)")
    elif transcript is None:
        notes.append("ASR unavailable — transcript checks skipped (sung-time only)")
    else:
        hb_count = count_happy_birthday(transcript)
        if hb_count < MIN_HB_COUNT:
            reasons.append(f"'happy birthday' heard only {hb_count}x "
                           f"(need >= {MIN_HB_COUNT})")
        elif hb_count > MAX_HB_COUNT:
            reasons.append(
                f"'happy birthday' heard {hb_count}x (> {MAX_HB_COUNT}) — "
                f"ASR repetition loop; render is likely garbled/chant-like")
        name_score = name_match_score(transcript, name)
        if name_score < MIN_NAME_SCORE:
            reasons.append(f"name '{name}' not heard "
                           f"(best fuzzy match {name_score:.2f} < {MIN_NAME_SCORE})")
        sections = sections_present(transcript)
        if len(sections) < MIN_SECTIONS:
            reasons.append(f"only {len(sections)}/3 distinct sections heard "
                           f"({', '.join(sections) or 'none'}; need >= {MIN_SECTIONS})")
        score += min(hb_count / MIN_HB_COUNT, 1.0) + name_score + len(sections) / 3.0

    passed = not reasons
    if passed:
        score += 10.0
    return QCReport(
        passed=passed, reasons=reasons,
        sung_seconds=sung_seconds, phrase_count=phrase_count,
        asr_used=asr_used, hb_count=hb_count, name_score=name_score,
        sections_found=sections, transcript=transcript,
        notes=notes, attempt=attempt, score=score,
    )


# --- Entry point ---------------------------------------------------------------

def _sidecar_path(tight_wav: Path) -> Path:
    return tight_wav.with_name(tight_wav.stem + "__qc.json")


def run_qc(
    tight_wav: Path,
    *,
    name: str,
    duration_s: float = 150.0,
    custom_lyrics: bool = False,
    use_cache: bool = True,
    verbose: bool = True,
    attempt: int = 0,
) -> QCReport:
    """Measure (or load cached measurements for) a tightened vocal WAV and
    evaluate the QC policy against it."""
    tight_wav = Path(tight_wav)
    sidecar = _sidecar_path(tight_wav)

    measured = None
    if use_cache and sidecar.exists():
        try:
            measured = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt sidecar just re-measures
            measured = None

    if measured is None:
        t0 = time.time()
        sung_s, phrases = measure_sung_seconds(tight_wav)
        transcript = None if custom_lyrics else transcribe(tight_wav)
        measured = {
            "sung_seconds": sung_s,
            "phrase_count": phrases,
            "transcript": transcript,
            "asr_model": ASR_MODEL if transcript is not None else None,
        }
        sidecar.write_text(json.dumps(measured, indent=2), encoding="utf-8")
        if verbose:
            print(f"[qc] measured {tight_wav.name} in {time.time() - t0:.0f}s")
    elif verbose:
        print(f"[qc] cached measurements: {sidecar.name}")

    report = evaluate(
        name=name,
        duration_s=duration_s,
        sung_seconds=measured["sung_seconds"],
        phrase_count=measured["phrase_count"],
        transcript=measured.get("transcript"),
        custom_lyrics=custom_lyrics,
        attempt=attempt,
    )
    if verbose:
        print(f"[qc] {report.summary()}")
    return report
