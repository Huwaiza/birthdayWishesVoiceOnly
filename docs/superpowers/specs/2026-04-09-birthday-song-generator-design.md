# Birthday Song Generator — Design Spec
**Date:** 2026-04-09  
**Status:** Approved

---

## Problem

The existing `generate_birthday.py` (v5) calls the YuE HuggingFace Space API remotely. This has three problems:
1. Depends on an external service (queue times of 2–10 min, service downtime)
2. Cannot guarantee output duration
3. Requires internet connection

Previous approaches (Bark, Deepsinger, edge-tts + Whisper pitch-matching, UberDuck API) either failed on pitch consistency, had no duration control, or were external APIs.

---

## Goal

A **fully local, offline** Python script that generates a personalised ~2.5-minute birthday song with:
- A dynamic name (e.g. Huwaiza, Armeen, Usama) sung naturally in the lyrics
- Soft instrumental backing (piano)
- No external API calls
- Output suitable for YouTube (2–3 min, royalty-free backing)

---

## Architecture

```
[1] Generate vocals        [2] Mix backing track      [3] Export
Bark (local, MPS/CPU)  →  pydub overlay (piano loop) →  MP3 output
```

### Stage 1 — Bark Vocal Generation
- Model: `suno/bark` (Hugging Face, ~5 GB one-time download, cached)
- Apple Silicon MPS acceleration auto-detected; falls back to CPU
- Default speaker preset: `v2/en_speaker_9` (warm female voice)
- Lyrics split into individual short lines, each wrapped in `♪` markers
- Each line generated separately → more consistent pitch and tempo
- Lines concatenated with natural breathing gaps (0.2–0.4s silence between lines, 0.8–1.2s between sections)

### Stage 2 — Backing Track Mix
- 3 bundled royalty-free piano loops in `assets/backing/` (~3 min each)
- Sourced from pixabay.com (royalty-free, YouTube-safe)
- Loop trimmed/extended to match exact vocal length
- Mixed at 18% volume under vocals using `pydub`

### Stage 3 — Export
- Final mix exported as MP3
- Output filename: `happy_birthday_{name}.mp3`

---

## Song Structure (~2.5 min)

```
[intro]     Piano lead-in (4s)
[verse 1]   Happy birthday to you
            Happy birthday to you
            Happy birthday dear {name}
            Happy birthday to you

[verse 2]   Happy birthday to you
            Happy birthday to you
            Happy birthday dear {name}
            May your dreams come true

[bridge]    May your day be filled with joy
            Laughter, love, and happiness
            We're so glad you're in our lives
            Happy birthday, {name}

[verse 3]   Happy birthday to you
            Happy birthday to you
            Happy birthday dear {name}
            Happy birthday to you

[chorus]    Hip hip hooray, it's your special day
            We sing for you, {name}, in every way
            May all your wishes all come true
            Happy birthday, happy birthday to you

[outro]     Happy birthday dear {name}
            Happy birthday dear {name}  (softer, fading)
```

Each line is a separate Bark inference call. Sections separated by longer pauses for natural feel.

---

## Duration Control

- After stitching all clips, total duration is measured
- Target: **2:20–2:45**
- If under 2:00 → extend inter-section pauses
- If over 3:00 → lightly time-stretch clips using `librosa` (no pitch distortion)
- `--regen-line N` flag regenerates a single line without re-running the whole song (cached per-line WAV files in `cache/`)

---

## Backing Track Styles

| `--style` flag | Description |
|---|---|
| `simple` (default) | Single piano melody, minimal, clean |
| `warm` | Gentle acoustic piano, slow tempo, emotional |
| `upbeat` | Bright piano + light percussion, celebratory |

Backing tracks bundled in `assets/backing/simple.mp3`, `assets/backing/warm.mp3`, `assets/backing/upbeat.mp3`.

---

## CLI Interface

```bash
# Basic usage
python generate_birthday.py --name "Huwaiza"

# With style
python generate_birthday.py --name "Armeen" --style upbeat

# Custom speaker voice
python generate_birthday.py --name "Usama" --style warm --speaker v2/en_speaker_5

# Regenerate just one line (0-indexed)
python generate_birthday.py --name "Huwaiza" --regen-line 3

# List available speaker presets
python generate_birthday.py --list-speakers
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Bark model loading |
| `bark` (`suno-bark`) | Vocal generation |
| `pydub` | Audio stitching, mixing, export |
| `librosa` | Time-stretching for duration control |
| `soundfile` | Audio I/O |
| `numpy` | Signal processing |
| `torch` | Bark backend (MPS on Apple Silicon) |

---

## File Layout

```
birthdayVoiceOnly/
├── generate_birthday.py       # main script (replaces current v5)
├── assets/
│   └── backing/
│       ├── simple.mp3
│       ├── warm.mp3
│       └── upbeat.mp3
├── cache/                     # per-line WAV cache (gitignored)
├── requirements.txt           # updated
└── docs/superpowers/specs/
    └── 2026-04-09-birthday-song-generator-design.md
```

---

## Out of Scope (for now)

- RVC voice conversion post-processing (Option C — revisit if Bark quality is insufficient)
- Video generation / YouTube upload automation
- Multiple language support
