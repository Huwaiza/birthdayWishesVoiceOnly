# Birthday Song Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current HuggingFace API-dependent `generate_birthday.py` with a fully local, offline Bark-based script that generates a personalised ~2.5-minute birthday song with soft piano backing.

**Architecture:** Bark (local, MPS-accelerated on Apple Silicon) generates vocals line-by-line with `♪` markers for singing style; lines are stitched with natural gaps using `pydub`; a bundled royalty-free piano loop is mixed at 18% volume underneath; duration is normalised to 2:20–2:45 via `librosa` time-stretching if needed.

**Tech Stack:** Python 3.10+, `suno-bark`, `transformers`, `torch` (MPS), `pydub`, `librosa`, `soundfile`, `numpy`, `ffmpeg` (system)

---

## File Structure

```
birthdayVoiceOnly/
├── generate_birthday.py          # REPLACE — new entry point + CLI
├── song/
│   ├── __init__.py               # CREATE — empty
│   ├── lyrics.py                 # CREATE — lyrics template + line builder
│   ├── vocal.py                  # CREATE — Bark model loading + per-line generation + cache
│   ├── mix.py                    # CREATE — stitch lines, mix backing, duration control
│   └── backing.py                # CREATE — backing track loader + looper
├── assets/
│   └── backing/
│       ├── simple.mp3            # DOWNLOAD — royalty-free piano (pixabay)
│       ├── warm.mp3              # DOWNLOAD — royalty-free piano (pixabay)
│       └── upbeat.mp3            # DOWNLOAD — royalty-free piano (pixabay)
├── cache/                        # CREATE — per-line WAV cache (gitignored)
├── tests/
│   ├── test_lyrics.py            # CREATE
│   ├── test_mix.py               # CREATE
│   └── test_backing.py           # CREATE
├── requirements.txt              # MODIFY
└── .gitignore                    # MODIFY — add cache/
```

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `song/__init__.py`
- Create: `cache/.gitkeep`
- Create: `assets/backing/` (directory)

- [ ] **Step 1: Update requirements.txt**

Replace the entire file contents with:

```
suno-bark>=0.0.1
transformers>=4.38.0
torch>=2.2.0
pydub>=0.25.1
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
scipy>=1.11.0
```

- [ ] **Step 2: Create song package**

```bash
mkdir -p song
touch song/__init__.py
```

- [ ] **Step 3: Create cache and assets directories**

```bash
mkdir -p cache assets/backing
touch cache/.gitkeep
```

- [ ] **Step 4: Update .gitignore**

If `.gitignore` does not exist, create it. Add these lines:

```
cache/
*.wav
__pycache__/
venv/
*.pyc
```

- [ ] **Step 5: Install dependencies**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without error. `torch` will be large (~700 MB). Bark model itself downloads on first use.

- [ ] **Step 6: Verify torch MPS availability**

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected output on Apple Silicon: `True`

- [ ] **Step 7: Commit**

```bash
git init  # only if not already a git repo
git add requirements.txt .gitignore song/__init__.py cache/.gitkeep
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Download Backing Tracks

**Files:**
- Create: `assets/backing/simple.mp3`
- Create: `assets/backing/warm.mp3`
- Create: `assets/backing/upbeat.mp3`

- [ ] **Step 1: Download royalty-free backing tracks from Pixabay**

Run each command and verify the file exists and is > 100 KB:

```bash
# Simple — gentle solo piano
curl -L "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0c6ff1bab.mp3" \
     -o assets/backing/simple.mp3

# Warm — soft emotional piano
curl -L "https://cdn.pixabay.com/download/audio/2021/11/25/audio_5b3e636228.mp3" \
     -o assets/backing/warm.mp3

# Upbeat — bright celebratory piano
curl -L "https://cdn.pixabay.com/download/audio/2022/03/15/audio_1a609c5b9e.mp3" \
     -o assets/backing/upbeat.mp3
```

> **Note:** If any URL is broken (Pixabay changes CDN paths), search pixabay.com for "birthday piano" or "happy piano" under the free license, download manually, and save to the paths above. The files must be MP3, ~2–4 minutes long, instrumental only.

- [ ] **Step 2: Verify files**

```bash
ls -lh assets/backing/
```

Expected: three `.mp3` files, each between 1 MB and 10 MB.

- [ ] **Step 3: Commit**

```bash
git add assets/backing/
git commit -m "feat: add royalty-free backing tracks (simple, warm, upbeat)"
```

---

## Task 3: Lyrics Module

**Files:**
- Create: `song/lyrics.py`
- Create: `tests/test_lyrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_lyrics.py`:

```python
import pytest
from song.lyrics import build_lines

def test_name_injected_in_lines():
    lines = build_lines("Huwaiza")
    full = " ".join(lines)
    assert "Huwaiza" in full

def test_returns_list_of_strings():
    lines = build_lines("Sara")
    assert isinstance(lines, list)
    assert all(isinstance(l, str) for l in lines)

def test_each_line_has_music_markers():
    lines = build_lines("Adam")
    for line in lines:
        assert line.startswith("♪"), f"Missing leading ♪ in: {line}"
        assert line.endswith("♪"), f"Missing trailing ♪ in: {line}"

def test_minimum_line_count():
    lines = build_lines("Tom")
    assert len(lines) >= 18  # full song structure

def test_section_separators_not_in_lines():
    lines = build_lines("Layla")
    for line in lines:
        assert not line.startswith("["), f"Section header leaked into lines: {line}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_lyrics.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `song.lyrics` does not exist yet.

- [ ] **Step 3: Implement lyrics.py**

Create `song/lyrics.py`:

```python
"""
Lyrics builder for the birthday song generator.

Produces a list of singable lines, each wrapped in ♪ markers,
with the recipient's name injected at the correct positions.
"""

from typing import List

# Full song structure — each string is one Bark inference line.
# {name} is replaced with the actual name at runtime.
_SONG_TEMPLATE = [
    # --- verse 1 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday to you ♪",
    # --- verse 2 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ May your dreams come true ♪",
    # --- bridge ---
    "♪ May your day be filled with joy ♪",
    "♪ Laughter, love, and happiness ♪",
    "♪ We're so glad you're in our lives ♪",
    "♪ Happy birthday {name} ♪",
    # --- verse 3 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday to you ♪",
    # --- chorus ---
    "♪ Hip hip hooray, it's your special day ♪",
    "♪ We sing for you {name}, in every way ♪",
    "♪ May all your wishes all come true ♪",
    "♪ Happy birthday, happy birthday to you ♪",
    # --- outro ---
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday dear {name} ♪",
]

# Which line indices start a new section (used for longer pauses during mixing)
SECTION_BREAKS = {0, 4, 8, 12, 16, 20}


def build_lines(name: str) -> List[str]:
    """
    Return the full list of singable lines with {name} replaced.

    Parameters
    ----------
    name : str
        The recipient's name, e.g. "Huwaiza"

    Returns
    -------
    List[str]
        One string per Bark inference call, each wrapped in ♪ markers.
    """
    return [line.format(name=name) for line in _SONG_TEMPLATE]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_lyrics.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add song/lyrics.py tests/test_lyrics.py
git commit -m "feat: add lyrics module with full song structure and name injection"
```

---

## Task 4: Vocal Generation Module

**Files:**
- Create: `song/vocal.py`

> No unit tests for this module — Bark requires a GPU/CPU and model download; integration is verified manually in Task 7.

- [ ] **Step 1: Create song/vocal.py**

```python
"""
Vocal generation via Bark (local, offline).

Loads the Bark model once and caches it in memory.
Each line of lyrics is generated separately for consistency.
Generated WAV files are cached to disk so individual lines
can be regenerated without re-running the whole song.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

CACHE_DIR = Path(__file__).parent.parent / "cache"
SAMPLE_RATE = 24000  # Bark's native sample rate

# Default speaker — warm female voice
DEFAULT_SPEAKER = "v2/en_speaker_9"

_model_cache: dict = {}


def _get_device() -> str:
    """Return 'mps', 'cuda', or 'cpu' depending on availability."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_models(speaker: str) -> tuple:
    """Load and cache Bark processor + model. Returns (processor, model)."""
    global _model_cache
    if "processor" in _model_cache:
        return _model_cache["processor"], _model_cache["model"]

    from transformers import AutoProcessor, BarkModel
    import torch

    device = _get_device()
    print(f"[vocal] Loading Bark model on {device} …")

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained(
        "suno/bark",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = model.to(device)
    model.enable_cpu_offload()  # reduces VRAM/MPS memory pressure

    _model_cache["processor"] = processor
    _model_cache["model"] = model
    _model_cache["device"] = device

    print(f"[vocal] Bark model ready.")
    return processor, model


def _line_cache_path(line: str, speaker: str) -> Path:
    """Return the cache file path for a given line + speaker combination."""
    key = hashlib.md5(f"{speaker}:{line}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}.wav"


def generate_line(
    line: str,
    speaker: str = DEFAULT_SPEAKER,
    force: bool = False,
) -> np.ndarray:
    """
    Generate audio for a single lyric line using Bark.

    Parameters
    ----------
    line    : The lyric line, e.g. "♪ Happy birthday to you ♪"
    speaker : Bark voice preset, e.g. "v2/en_speaker_9"
    force   : If True, bypass cache and regenerate

    Returns
    -------
    np.ndarray  shape (N,), float32, sample rate = SAMPLE_RATE
    """
    cache_path = _line_cache_path(line, speaker)

    if not force and cache_path.exists():
        audio, _ = sf.read(str(cache_path), dtype="float32")
        return audio

    processor, model = _load_models(speaker)
    import torch

    device = _model_cache["device"]
    inputs = processor(line, voice_preset=speaker)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs)

    audio = output.cpu().numpy().squeeze().astype(np.float32)

    CACHE_DIR.mkdir(exist_ok=True)
    sf.write(str(cache_path), audio, SAMPLE_RATE)

    return audio


def generate_all_lines(
    lines: List[str],
    speaker: str = DEFAULT_SPEAKER,
    regen_index: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate audio for every line in the lyrics list.

    Parameters
    ----------
    lines        : Output of song.lyrics.build_lines()
    speaker      : Bark voice preset
    regen_index  : If set, only this line index is regenerated (ignores cache)

    Returns
    -------
    List[np.ndarray]  one array per line, each float32 at SAMPLE_RATE
    """
    clips = []
    total = len(lines)
    for i, line in enumerate(lines):
        force = (regen_index is not None and i == regen_index)
        status = "regenerating" if force else "generating"
        cached = _line_cache_path(line, speaker).exists() and not force
        label = "cached" if cached else status
        print(f"[vocal] Line {i+1}/{total} ({label}): {line}")
        clips.append(generate_line(line, speaker=speaker, force=force))
    return clips
```

- [ ] **Step 2: Commit**

```bash
git add song/vocal.py
git commit -m "feat: add vocal generation module with Bark + per-line disk cache"
```

---

## Task 5: Backing Track Module

**Files:**
- Create: `song/backing.py`
- Create: `tests/test_backing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backing.py`:

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from song.backing import load_and_loop


def test_returns_numpy_array():
    # Use a tiny synthetic audio array instead of a real file
    fake_audio = np.zeros(24000, dtype=np.float32)  # 1s at 24kHz
    with patch("song.backing.sf.read", return_value=(fake_audio, 24000)):
        with patch("song.backing.Path.exists", return_value=True):
            result = load_and_loop("fake.mp3", target_duration_s=3.0, sample_rate=24000)
    assert isinstance(result, np.ndarray)


def test_output_length_matches_target():
    fake_audio = np.zeros(24000, dtype=np.float32)  # 1s
    with patch("song.backing.sf.read", return_value=(fake_audio, 24000)):
        with patch("song.backing.Path.exists", return_value=True):
            result = load_and_loop("fake.mp3", target_duration_s=3.0, sample_rate=24000)
    expected_samples = int(3.0 * 24000)
    assert abs(len(result) - expected_samples) <= 100  # within ~4ms


def test_missing_file_raises():
    with patch("song.backing.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_and_loop("nonexistent.mp3", target_duration_s=10.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_backing.py -v
```

Expected: `ModuleNotFoundError` — `song.backing` does not exist yet.

- [ ] **Step 3: Implement backing.py**

Create `song/backing.py`:

```python
"""
Backing track loader and looper.

Loads an MP3 file, loops or trims it to an exact target duration,
and returns a float32 numpy array at the requested sample rate.
"""

from pathlib import Path

import numpy as np
import soundfile as sf


def load_and_loop(
    path: str,
    target_duration_s: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Load an audio file and loop/trim it to exactly target_duration_s seconds.

    Parameters
    ----------
    path              : Path to the MP3/WAV backing track
    target_duration_s : Desired output duration in seconds
    sample_rate       : Output sample rate (default 24000 to match Bark)

    Returns
    -------
    np.ndarray  shape (N,), float32, at `sample_rate`
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Backing track not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # Mix down to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    target_samples = int(target_duration_s * sample_rate)

    # Loop until long enough
    if len(audio) < target_samples:
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)

    # Trim to exact length
    audio = audio[:target_samples]

    return audio.astype(np.float32)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_backing.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add song/backing.py tests/test_backing.py
git commit -m "feat: add backing track loader with loop/trim to target duration"
```

---

## Task 6: Mix Module

**Files:**
- Create: `song/mix.py`
- Create: `tests/test_mix.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mix.py`:

```python
import numpy as np
import pytest
from song.mix import stitch_clips, mix_backing, normalise_duration
from song.lyrics import SECTION_BREAKS

SAMPLE_RATE = 24000

def _make_clip(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def test_stitch_clips_concatenates():
    clips = [_make_clip(1.0), _make_clip(1.0), _make_clip(1.0)]
    result = stitch_clips(clips, SECTION_BREAKS, sample_rate=SAMPLE_RATE)
    # 3 clips × 1s + gaps > 3s
    assert len(result) > 3 * SAMPLE_RATE


def test_stitch_clips_returns_float32():
    clips = [_make_clip(0.5)]
    result = stitch_clips(clips, set(), sample_rate=SAMPLE_RATE)
    assert result.dtype == np.float32


def test_mix_backing_same_length():
    vocals = _make_clip(5.0)
    backing = _make_clip(5.0)
    result = mix_backing(vocals, backing, backing_volume=0.18)
    assert len(result) == len(vocals)


def test_mix_backing_volume_applied():
    vocals = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.5
    backing = np.ones(SAMPLE_RATE, dtype=np.float32) * 1.0
    result = mix_backing(vocals, backing, backing_volume=0.18)
    # Result should be louder than vocals alone but backing contribution small
    assert result.max() > 0.5
    assert result.max() <= 1.0  # clipped to [-1, 1]


def test_normalise_duration_short_audio_padded():
    short = _make_clip(100.0)  # 100s — under 140s minimum
    result = normalise_duration(short, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    assert len(result) >= int(140.0 * SAMPLE_RATE)


def test_normalise_duration_long_audio_trimmed():
    long = _make_clip(200.0)  # 200s — over 165s maximum
    result = normalise_duration(long, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    assert len(result) <= int(165.0 * SAMPLE_RATE) + SAMPLE_RATE  # within 1s tolerance
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_mix.py -v
```

Expected: `ModuleNotFoundError` — `song.mix` does not exist yet.

- [ ] **Step 3: Implement mix.py**

Create `song/mix.py`:

```python
"""
Audio stitching, backing mix, and duration normalisation.
"""

from typing import List, Set

import numpy as np

SAMPLE_RATE = 24000

# Silence durations in seconds
_GAP_BETWEEN_LINES = 0.35       # between adjacent lines
_GAP_BETWEEN_SECTIONS = 1.0     # at section breaks (longer, more natural)
_INTRO_SILENCE = 4.0            # piano lead-in before first line


def _silence(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def stitch_clips(
    clips: List[np.ndarray],
    section_breaks: Set[int],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Concatenate vocal clips with natural gaps between them.

    Parameters
    ----------
    clips          : List of float32 numpy arrays (one per lyric line)
    section_breaks : Set of line indices that start a new section (longer gap before them)
    sample_rate    : Sample rate of all clips

    Returns
    -------
    np.ndarray  float32, single-channel, at sample_rate
    """
    parts = [_silence(_INTRO_SILENCE, sample_rate)]

    for i, clip in enumerate(clips):
        if i > 0:
            gap = _GAP_BETWEEN_SECTIONS if i in section_breaks else _GAP_BETWEEN_LINES
            parts.append(_silence(gap, sample_rate))
        parts.append(clip.astype(np.float32))

    return np.concatenate(parts)


def mix_backing(
    vocals: np.ndarray,
    backing: np.ndarray,
    backing_volume: float = 0.18,
) -> np.ndarray:
    """
    Mix backing track under vocals.

    Parameters
    ----------
    vocals         : float32 vocal array
    backing        : float32 backing array — must be same length as vocals
    backing_volume : Backing track gain (0.0–1.0). Default 0.18 = 18%

    Returns
    -------
    np.ndarray  float32, clipped to [-1.0, 1.0]
    """
    mixed = vocals + backing * backing_volume
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def normalise_duration(
    audio: np.ndarray,
    min_s: float = 140.0,
    max_s: float = 165.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Ensure audio falls within [min_s, max_s] seconds.

    - If too short: append silence at the end (song naturally ended early)
    - If too long: time-stretch to fit max_s using librosa

    Parameters
    ----------
    audio       : float32 numpy array
    min_s       : Minimum acceptable duration in seconds (default 140 = 2:20)
    max_s       : Maximum acceptable duration in seconds (default 165 = 2:45)
    sample_rate : Sample rate of audio

    Returns
    -------
    np.ndarray  float32
    """
    duration_s = len(audio) / sample_rate

    if duration_s < min_s:
        pad = _silence(min_s - duration_s, sample_rate)
        return np.concatenate([audio, pad])

    if duration_s > max_s:
        import librosa
        rate = duration_s / max_s  # > 1.0 means speed up
        audio = librosa.effects.time_stretch(audio, rate=rate)
        return audio[:int(max_s * sample_rate)].astype(np.float32)

    return audio
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_mix.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add song/mix.py tests/test_mix.py
git commit -m "feat: add mix module — stitch, backing overlay, duration normalisation"
```

---

## Task 7: Main Entry Point

**Files:**
- Replace: `generate_birthday.py`

- [ ] **Step 1: Replace generate_birthday.py**

```python
#!/usr/bin/env python3
"""
Birthday Song Generator  v6  —  Local Bark + Piano Backing
===========================================================
Generates a personalised ~2.5-minute birthday song entirely offline.
No external API. No GPU required (runs on Apple Silicon MPS or CPU).

One-time model download: ~5 GB (cached by HuggingFace hub).

Usage
-----
  python generate_birthday.py --name "Huwaiza"
  python generate_birthday.py --name "Armeen"   --style upbeat
  python generate_birthday.py --name "Usama"    --style warm
  python generate_birthday.py --name "Huwaiza"  --speaker v2/en_speaker_5
  python generate_birthday.py --name "Sara"     --regen-line 2
  python generate_birthday.py --list-speakers
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

SCRIPT_DIR   = Path(__file__).parent.resolve()
ASSETS_DIR   = SCRIPT_DIR / "assets" / "backing"
SAMPLE_RATE  = 24000  # Bark native rate

STYLES = {
    "simple":  ASSETS_DIR / "simple.mp3",
    "warm":    ASSETS_DIR / "warm.mp3",
    "upbeat":  ASSETS_DIR / "upbeat.mp3",
}

SPEAKERS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
    "v2/en_speaker_9",  # default — warm female
]


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(mp3_path), format="mp3", bitrate="192k")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a personalised birthday song locally using Bark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        EXAMPLES
          python generate_birthday.py --name "Huwaiza"
          python generate_birthday.py --name "Armeen"  --style upbeat
          python generate_birthday.py --name "Usama"   --style warm
          python generate_birthday.py --name "Sara"    --speaker v2/en_speaker_5
          python generate_birthday.py --name "Huwaiza" --regen-line 2
          python generate_birthday.py --list-speakers

        NOTES
          • First run downloads the Bark model (~5 GB). Subsequent runs use cache.
          • Each lyric line is cached in cache/ — use --regen-line N to redo one line.
          • Target duration: 2:20–2:45 (suitable for YouTube monetisation).
          • Apple Silicon MPS is used automatically when available.
        """),
    )
    ap.add_argument("--name",          default=None,  help="Name to sing  (required)")
    ap.add_argument("--style",         default="simple", choices=list(STYLES.keys()),
                    help="Backing track style: simple | warm | upbeat  (default: simple)")
    ap.add_argument("--speaker",       default="v2/en_speaker_9",
                    help="Bark speaker preset  (default: v2/en_speaker_9)")
    ap.add_argument("--output",        default=None,
                    help="Output MP3 filename  (default: happy_birthday_<name>.mp3)")
    ap.add_argument("--regen-line",    type=int, default=None, dest="regen_line",
                    metavar="N", help="Regenerate only line N (0-indexed), ignoring cache")
    ap.add_argument("--list-speakers", action="store_true", dest="list_speakers",
                    help="Print available speaker presets and exit")
    args = ap.parse_args()

    if args.list_speakers:
        print("\nAvailable speaker presets:")
        for s in SPEAKERS:
            marker = "  ← default" if s == "v2/en_speaker_9" else ""
            print(f"  {s}{marker}")
        print()
        return

    if not args.name:
        ap.error("--name is required")

    output_mp3 = Path(args.output or f"happy_birthday_{args.name.lower().replace(' ', '_')}.mp3")
    if not output_mp3.is_absolute():
        output_mp3 = SCRIPT_DIR / output_mp3

    backing_path = STYLES[args.style]
    if not backing_path.exists():
        sys.exit(
            f"[ERROR] Backing track not found: {backing_path}\n"
            f"  Run Task 2 of the implementation plan to download backing tracks."
        )

    print("\n" + "=" * 60)
    print("  Birthday Song Generator  v6  (local Bark)")
    print(f"  Name    : {args.name}")
    print(f"  Style   : {args.style}")
    print(f"  Speaker : {args.speaker}")
    print(f"  Output  : {output_mp3}")
    print("=" * 60 + "\n")

    # --- 1. Build lyrics ---
    from song.lyrics import build_lines, SECTION_BREAKS
    lines = build_lines(args.name)
    print(f"[main] {len(lines)} lyric lines built.\n")

    # --- 2. Generate vocals ---
    from song.vocal import generate_all_lines
    clips = generate_all_lines(lines, speaker=args.speaker, regen_index=args.regen_line)

    # --- 3. Stitch vocals ---
    from song.mix import stitch_clips, mix_backing, normalise_duration
    vocals = stitch_clips(clips, SECTION_BREAKS, sample_rate=SAMPLE_RATE)
    duration_s = len(vocals) / SAMPLE_RATE
    print(f"\n[main] Vocals stitched: {duration_s:.1f}s")

    # --- 4. Load and loop backing track ---
    from song.backing import load_and_loop
    backing = load_and_loop(str(backing_path), target_duration_s=duration_s, sample_rate=SAMPLE_RATE)

    # --- 5. Mix ---
    mixed = mix_backing(vocals, backing, backing_volume=0.18)

    # --- 6. Normalise duration to 2:20–2:45 ---
    mixed = normalise_duration(mixed, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    final_duration = len(mixed) / SAMPLE_RATE
    print(f"[main] Final duration: {final_duration:.1f}s ({final_duration/60:.1f} min)")

    # --- 7. Export WAV then convert to MP3 ---
    tmp_wav = output_mp3.with_suffix(".wav")
    sf.write(str(tmp_wav), mixed, SAMPLE_RATE)
    _wav_to_mp3(tmp_wav, output_mp3)
    tmp_wav.unlink()

    size_mb = output_mp3.stat().st_size / (1024 * 1024)
    print(f"\n✓  Done!  {size_mb:.1f} MB  →  {output_mp3}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test with a short name**

```bash
source venv/bin/activate
python generate_birthday.py --name "Test"
```

Expected:
- First run: downloads Bark model (~5 GB, takes several minutes)
- Generates 22 lyric lines (progress printed per line)
- Outputs `happy_birthday_test.mp3` in the project directory
- Prints final duration between 2:20 and 2:45

- [ ] **Step 3: Verify output with a real name**

```bash
python generate_birthday.py --name "Huwaiza" --style warm
```

Expected: `happy_birthday_huwaiza.mp3` generated. Second run is faster (lines cached).

- [ ] **Step 4: Test regen-line flag**

```bash
python generate_birthday.py --name "Huwaiza" --regen-line 2
```

Expected: only line 2 (`♪ Happy birthday dear Huwaiza ♪`) is regenerated; all others use cache.

- [ ] **Step 5: Commit**

```bash
git add generate_birthday.py
git commit -m "feat: add main entry point — local Bark song generation v6"
```

---

## Task 8: Run Full Test Suite

**Files:** None new — verification only.

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass. Output should show:
```
tests/test_lyrics.py::test_name_injected_in_lines PASSED
tests/test_lyrics.py::test_returns_list_of_strings PASSED
tests/test_lyrics.py::test_each_line_has_music_markers PASSED
tests/test_lyrics.py::test_minimum_line_count PASSED
tests/test_lyrics.py::test_section_separators_not_in_lines PASSED
tests/test_backing.py::test_returns_numpy_array PASSED
tests/test_backing.py::test_output_length_matches_target PASSED
tests/test_backing.py::test_missing_file_raises PASSED
tests/test_mix.py::test_stitch_clips_concatenates PASSED
tests/test_mix.py::test_stitch_clips_returns_float32 PASSED
tests/test_mix.py::test_mix_backing_same_length PASSED
tests/test_mix.py::test_mix_backing_volume_applied PASSED
tests/test_mix.py::test_normalise_duration_short_audio_padded PASSED
tests/test_mix.py::test_normalise_duration_long_audio_trimmed PASSED
```

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: all tests passing — birthday song generator v6 complete"
```

---

## Self-Review Notes

- **Spec coverage:** All sections covered — architecture (Tasks 1,4,5,6,7), lyrics structure (Task 3), duration control (Task 6), backing styles (Tasks 2,5,7), CLI flags (Task 7), file layout (Task 1)
- **No placeholders:** All code blocks are complete and runnable
- **Type consistency:** `SECTION_BREAKS` exported from `song/lyrics.py` and imported in `generate_birthday.py` and `tests/test_mix.py` consistently; `SAMPLE_RATE=24000` defined in `song/vocal.py` and `song/mix.py` independently (avoids circular import)
- **One gap addressed:** Task 2 backing track URLs may break (Pixabay CDN); note added with manual fallback instructions
