# Birthday Song Generator (v7)

Generates a personalised, production-quality 2:30 birthday song from a name,
locally on an Apple Silicon Mac. Free for unlimited renders — no API, no
per-song cost.

Powered by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step), an
open-source music foundation model (Apache 2.0).

---

## Quick start

After the one-time install (see below), this is your daily workflow:

```bash
source venv-diffsinger/bin/activate

# One song
python generate_birthday.py --name "Huwaiza"

# A batch
python batch_render.py --names names.txt --out output/
```

A 2:30 song renders in ~1–3 minutes on M-series silicon (depends on `--steps`).

---

## How it works

ACE-Step is a diffusion model for music — same family as Stable Diffusion,
but trained on songs instead of images. You give it a text prompt
(describing the vibe and singer) and a lyrics blob with section tags. It
returns a complete WAV with vocals **and** backing track in one pass.

The pipeline:

```
name "Huwaiza"
    │
    ▼
song/lyrics.py            → Builds tagged lyrics:
                              [verse] Happy birthday to you ...
                              [bridge] May your day be filled with joy ...
                              [chorus] Hip hip hooray it's your special day ...
                              [outro] Happy birthday dear Huwaiza ...
    │
    ▼
song/voice_profiles.py    → MD5(name) % 4 picks one of 4 female-vocal
                            prompts ("warm-alto", "bright-soprano",
                            "intimate-mezzo", "folk-clear") + a pinned seed.
                            Same name → same voice forever (deterministic).
    │
    ▼
song/acestep_render.py    → Calls ACE-Step with prompt + lyrics + seed,
                            caches the WAV by hash(everything).
    │
    ▼
generate_birthday.py      → Encodes the WAV to a 192 kbps MP3.
```

That's the whole thing. No backing-track loops, no separate vocal/mix
stages, no MIDI authoring — ACE-Step composes melody, sings it, and mixes
the band on every render.

---

## Voice rotation

Four pinned voice profiles in `song/voice_profiles.py`:

| Index | Name              | Vibe                                       |
|-------|-------------------|--------------------------------------------|
| 0     | `warm-alto`       | Soulful mid-range, expressive vibrato       |
| 1     | `bright-soprano`  | Clear high notes, youthful pop              |
| 2     | `intimate-mezzo`  | Smooth and breathy, indie pop, close-mic'd  |
| 3     | `folk-clear`      | Natural folk-pop, sincere phrasing          |

By default, each name gets a deterministic voice via `MD5(name) % 4`. Pin
a specific voice with `--voice 0..3` if you want to override:

```bash
python generate_birthday.py --name "Sara" --voice 2
python generate_birthday.py --list-voices   # see full prompt for each
```

To audition all four for the same name:

```bash
for v in 0 1 2 3; do
  python generate_birthday.py --name "Huwaiza" --voice $v \
    --output "huwaiza_voice_${v}.mp3"
done
```

---

## Caching

Rendered WAVs are cached in `cache/acestep/`. The cache key is a hash of
`(name, voice profile, prompt, lyrics, duration, steps, guidance)`. So
calling the same name + voice twice is instant; tweaking the lyrics or a
voice prompt invalidates everyone correctly. Force a re-render with
`--no-cache`.

---

## CLI reference

`generate_birthday.py` — single name:

| Flag             | Default                      | Notes                                  |
|------------------|------------------------------|----------------------------------------|
| `--name`         | required                     | The recipient's name                   |
| `--voice N`      | `MD5(name) % 4`              | Pin a specific voice profile           |
| `--duration`     | `150` (seconds, = 2:30)      | Song length                            |
| `--steps`        | `60`                         | Diffusion steps. Lower = faster, lower quality |
| `--guidance`     | `15.0`                       | Classifier-free guidance scale         |
| `--output`       | `happy_birthday_<name>.mp3`  | Output MP3 path                        |
| `--no-cache`     | off                          | Skip the cache, always re-render       |
| `--list-voices`  | —                            | Print profiles and exit                |

`batch_render.py` — multi-name:

| Flag             | Default       | Notes                                       |
|------------------|---------------|---------------------------------------------|
| `--names`        | required      | Path to a text file (one name per line)     |
| `--out`          | required      | Output directory for MP3s                   |
| `--duration`     | `150`         | Per-song length                             |
| `--steps`        | `60`          | Diffusion steps                             |
| `--guidance`     | `15.0`        | CFG scale                                   |
| `--retries`      | `1`           | Per-name retry count on failure             |
| `--voice N`      | rotate        | Pin all renders to one voice                |
| `--no-cache`     | off           | Always re-render                            |
| `--start N`      | `0`           | Skip the first N names (resume)             |
| `--limit N`      | unlimited     | Stop after N names                          |

---

## Project layout

```
birthdayVoiceOnly/
├── generate_birthday.py    Single-name CLI
├── batch_render.py         Multi-name runner (resume, retry, validate)
├── names.txt               Sample names list
├── PLAN.md                 Architecture + 2-week plan + kill criteria
├── ANALYSIS.md             v6 (Bark) post-mortem — kept for context
├── README.md               This file
├── requirements.txt        Project deps (ACE-Step itself installs from git)
│
├── song/                   Python package
│   ├── lyrics.py             Tagged lyrics builder
│   ├── voice_profiles.py     4 female voice prompts + deterministic rotation
│   └── acestep_render.py     ACE-Step pipeline wrapper + cache
│
├── external/
│   └── ACE-Step/           Cloned by scripts/install_acestep.sh
│
├── scripts/
│   ├── install_acestep.sh    One-shot installer
│   └── smoke_test.py         30-s render to verify the install
│
├── cache/acestep/          WAV cache, keyed by inputs hash
├── tests/                  pytest unit tests (lyrics, voice_profiles)
└── venv-diffsinger/        Python 3.10 venv (despite the legacy name)
```

The model weights themselves (~4 GB) live outside the project, in
`~/.cache/ace-step/checkpoints/`. They download once on the first render.

---

## Install (one-time)

If you're starting on a fresh Mac:

```bash
# 1. System deps
brew install python@3.10 ffmpeg rust

# 2. Project venv
cd "/path/to/birthdayVoiceOnly"
/opt/homebrew/bin/python3.10 -m venv venv-diffsinger
source venv-diffsinger/bin/activate
pip install --upgrade pip wheel setuptools

# 3. PyTorch (with MPS for Apple Silicon)
pip install torch torchvision torchaudio torchcodec

# 4. ACE-Step + project deps
bash scripts/install_acestep.sh

# 5. Smoke test (downloads ~4 GB of weights on first run)
python scripts/smoke_test.py
```

If the smoke test produces a playable WAV at `scripts/smoke_test_out.wav`,
you're good. Run `python generate_birthday.py --name "YourName"` for your
first real song.

---

## Tuning

- **Faster but rougher renders**: `--steps 30 --guidance 12`
- **Slower but cleaner renders**: `--steps 90 --guidance 18`
- **Different vocal style**: edit `song/voice_profiles.py` — each profile is
  just a natural-language prompt + a base seed. Re-render after editing
  invalidates the cache automatically.
- **Different lyrics**: edit `_SONG_TEMPLATE` in `song/lyrics.py`. Keep the
  `[verse]` / `[bridge]` / `[chorus]` / `[outro]` tags — they're how
  ACE-Step understands song structure.

---

## Tests

```bash
source venv-diffsinger/bin/activate
pytest -v
```

Tests cover lyrics building and voice rotation. The ACE-Step render path
itself isn't unit-tested (it's a 4 GB neural network) — `scripts/smoke_test.py`
serves as the integration test.

---

## Credits

- Music synthesis: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) by ACE Studio & StepFun AI (Apache 2.0)
- Audio I/O: [pydub](https://github.com/jiaaro/pydub), [soundfile](https://github.com/bastibe/python-soundfile)
- Apple Silicon acceleration: [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
