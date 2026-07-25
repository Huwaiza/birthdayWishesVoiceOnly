"""
Tests for song.qc — the post-render QC gate.

The evaluation policy is pure (no model, no I/O), so most tests feed it
fabricated measurements, including transcripts modelled on real Whisper
output from renders that shipped with dropped lyrics. Sung-time measurement
uses synthetic pydub audio (native WAV, no ffmpeg). ASR itself is never
loaded here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pydub = pytest.importorskip("pydub")
from pydub import AudioSegment
from pydub.generators import Sine

from song import qc


# A healthy render's transcript: every section, the name, plenty of HBs.
GOOD_TRANSCRIPT = (
    "Happy birthday to you. Happy birthday to you. Happy birthday dear Abriella. "
    "Happy birthday to you. Happy birthday to you. Happy birthday dear Abriella. "
    "May your dreams come true. May your day be filled with joy. "
    "Laughter, love and happiness. We're so glad you're in our lives. "
    "Happy birthday Abriella. Hip hip hooray it's your special day. "
    "We sing for you Abriella in every way. May all your wishes come true today. "
    "Happy birthday, happy birthday to you. Happy birthday dear Abriella."
)

# Modelled on the real Whisper output of a bad render: bridge partially
# there, "dreams" verse missing, outro personalisation replaced by "to you".
BAD_TRANSCRIPT = (
    "Happy birthday to you. Happy birthday to you. Happy birthday to you. "
    "Happy birthday to you. Happy birthday to you. Hip hip hooray. "
    "It's your special day. Happy birthday to you. Happy birthday to you."
)


# --- transcript primitives ----------------------------------------------------

def test_count_happy_birthday():
    assert qc.count_happy_birthday(GOOD_TRANSCRIPT) >= 10
    assert qc.count_happy_birthday("hello world") == 0
    # Punctuation/case must not break the count.
    assert qc.count_happy_birthday("HAPPY BIRTHDAY! happy, birthday?") == 2


def test_name_match_tolerates_asr_misspelling():
    # Whisper heard "Arbriela" / "Gabriella" for Abriella — must still match.
    assert qc.name_match_score("happy birthday dear Arbriela", "Abriella") >= 0.5
    assert qc.name_match_score("happy birthday to your Gabriella", "Abriella") >= 0.5


def test_name_match_detects_absent_name():
    assert qc.name_match_score(BAD_TRANSCRIPT, "Xavier") < 0.5


def test_name_match_multiword_name():
    s = qc.name_match_score("we sing for you Anup Singh in every way", "Anup Singh")
    assert s >= 0.9


def test_sections_present():
    assert set(qc.sections_present(GOOD_TRANSCRIPT)) == {"verse2", "bridge", "chorus"}
    assert qc.sections_present(BAD_TRANSCRIPT) == ["chorus"]


# --- evaluation policy ----------------------------------------------------------

def test_evaluate_passes_a_healthy_render():
    r = qc.evaluate(name="Abriella", duration_s=150.0, sung_seconds=110.0,
                    phrase_count=20, transcript=GOOD_TRANSCRIPT)
    assert r.passed and r.reasons == []
    assert r.asr_used and r.hb_count >= 10 and r.name_score >= 0.5


def test_evaluate_fails_short_sung_time():
    r = qc.evaluate(name="Abriella", duration_s=150.0, sung_seconds=55.0,
                    phrase_count=21, transcript=GOOD_TRANSCRIPT)
    assert not r.passed
    assert any("sung" in reason for reason in r.reasons)


def test_evaluate_fails_asr_repetition_loop():
    # Modelled on the real Shahrukh render: Whisper repetition-looped and
    # produced 102 "happy birthday"s in 94 s of singing — physically
    # impossible, and a reliable marker of garbled chant-like audio.
    looped = "happy birthday to you, " * 90 + GOOD_TRANSCRIPT
    r = qc.evaluate(name="Shahrukh", duration_s=150.0, sung_seconds=94.0,
                    phrase_count=18, transcript=looped)
    assert not r.passed
    assert any("repetition loop" in reason for reason in r.reasons)


def test_evaluate_fails_dropped_sections_and_name():
    r = qc.evaluate(name="Xavier", duration_s=150.0, sung_seconds=110.0,
                    phrase_count=10, transcript=BAD_TRANSCRIPT)
    assert not r.passed
    joined = " ".join(r.reasons)
    assert "name" in joined and "sections" in joined


def test_evaluate_min_sung_scales_with_duration():
    # 55 s of singing fails a 150 s render but passes a 90 s one.
    long = qc.evaluate(name="A", duration_s=150.0, sung_seconds=55.0,
                       phrase_count=9, transcript=None)
    short = qc.evaluate(name="A", duration_s=90.0, sung_seconds=55.0,
                        phrase_count=9, transcript=None)
    assert not long.passed and short.passed


def test_evaluate_without_asr_uses_sung_time_only():
    r = qc.evaluate(name="Abriella", duration_s=150.0, sung_seconds=110.0,
                    phrase_count=20, transcript=None)
    assert r.passed and not r.asr_used
    assert r.hb_count is None and r.name_score is None
    assert any("ASR unavailable" in n for n in r.notes)


def test_evaluate_custom_lyrics_skips_transcript_checks():
    r = qc.evaluate(name="Lucy", duration_s=150.0, sung_seconds=110.0,
                    phrase_count=20, transcript=None, custom_lyrics=True)
    assert r.passed
    assert any("custom lyrics" in n for n in r.notes)


def test_passing_report_outscores_failing_one():
    good = qc.evaluate(name="Abriella", duration_s=150.0, sung_seconds=110.0,
                       phrase_count=20, transcript=GOOD_TRANSCRIPT)
    bad = qc.evaluate(name="Abriella", duration_s=150.0, sung_seconds=55.0,
                      phrase_count=21, transcript=BAD_TRANSCRIPT)
    assert good.score > bad.score


# --- measurement ---------------------------------------------------------------

def test_measure_sung_seconds(tmp_path):
    tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-3)
    audio = tone + AudioSegment.silent(duration=800) + tone
    wav = tmp_path / "in.wav"
    audio.export(str(wav), format="wav")

    sung_s, phrases = qc.measure_sung_seconds(wav)
    assert phrases == 2
    assert 1.8 <= sung_s <= 2.3


# --- run_qc sidecar caching -----------------------------------------------------

def test_run_qc_caches_measurements(tmp_path, monkeypatch):
    wav = tmp_path / "x__vocal_tight.wav"
    wav.write_bytes(b"not really a wav")  # never decoded thanks to the mocks

    monkeypatch.setattr(qc, "measure_sung_seconds", lambda p: (110.0, 20))
    monkeypatch.setattr(qc, "transcribe", lambda p: GOOD_TRANSCRIPT)
    first = qc.run_qc(wav, name="Abriella", duration_s=150.0, verbose=False)
    assert first.passed
    assert qc._sidecar_path(wav).exists()

    # Second call must come from the sidecar — measuring again is an error.
    def boom(p):
        raise AssertionError("measurement should have been cached")
    monkeypatch.setattr(qc, "measure_sung_seconds", boom)
    monkeypatch.setattr(qc, "transcribe", boom)
    second = qc.run_qc(wav, name="Abriella", duration_s=150.0, verbose=False)
    assert second.passed
    assert second.sung_seconds == first.sung_seconds

    # use_cache=False re-measures.
    monkeypatch.setattr(qc, "measure_sung_seconds", lambda p: (55.0, 21))
    monkeypatch.setattr(qc, "transcribe", lambda p: BAD_TRANSCRIPT)
    fresh = qc.run_qc(wav, name="Abriella", duration_s=150.0,
                      use_cache=False, verbose=False)
    assert not fresh.passed
