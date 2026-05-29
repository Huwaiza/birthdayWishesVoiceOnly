# Birthday Song Generator (v7)

Generates a personalised, production-quality birthday song from a name,
locally on an Apple Silicon Mac. Songs are **vocals-only** (a cappella) by
default; pass `--with-music` for the full backing band. Free for unlimited
renders — no API, no per-song cost.

Powered by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step), an
open-source music foundation model (Apache 2.0).

---

## Quick start

After the one-time install (see below), this is your daily workflow:

```bash
source venv-diffsinger/bin/activate

# One song — vocals only (a cappella), the default
python generate_birthday.py --name "Huwaiza"

# One song — with the full backing band
python generate_birthday.py --name "Huwaiza" --with-music

# A batch (vocals only)
python batch_render.py --names names.txt --out output/
```

Once the model is loaded, the diffusion render takes roughly 5–10 minutes
on M-series silicon (depends on `--steps` and the chip). The default
vocals-only mode then spends another ~2–5 minutes isolating the voice and
tightening pauses. If a render is far slower than that, see **Performance
& memory** below — it's almost always a RAM-swapping problem with a
one-flag fix.

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
song/vocal_isolate.py     → DEFAULT (vocals-only): a Roformer model
                            strips the backing, then tighten_pauses()
                            collapses the leftover instrumental gaps.
                            Skipped when you pass --with-music.
    │
    ▼
generate_birthday.py      → Encodes the final WAV to a 192 kbps MP3.
```

ACE-Step composes the melody, sings it, and mixes a band on every render
— no backing-track loops, no MIDI authoring. For the default vocals-only
output, that band is then separated back out and the song tightened into
a clean a cappella track.

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

## Vocals only vs. with music

**Vocals only (a cappella) is the default.** Run the CLI with no mode
flag and you get a clean, instrument-free vocal:

```bash
python generate_birthday.py --name "Huwaiza"
python batch_render.py --names names.txt --out output/
```

Pass `--with-music` to instead get the full song — the voice plus a warm
piano-ballad backing band:

```bash
python generate_birthday.py --name "Huwaiza" --with-music
python batch_render.py --names names.txt --out output/ --with-music
```

### How vocals-only works

ACE-Step is a *song* model — it always wants to compose an arrangement.
So the vocals-only pipeline takes three steps:

1. **Render** with an a cappella-oriented prompt — the model is asked for
   continuous solo singing with no instrumental intro, breaks, or outro.
   That already minimises the instrumental space.
2. **Isolate** with a **Roformer** model (via
   [audio-separator](https://pypi.org/project/audio-separator/)) — the
   current state of the art for vocal separation. It physically pulls the
   recording into stems and keeps only the voice, so the result is
   instrument-free for real. One pass only: a second cleanup pass would
   scrub marginally more residue but add a watery artefact to the voice.
3. **Tighten** with `tighten_pauses()` — even an a cappella-prompted
   render leaves a little instrumental space, which becomes silence once
   the backing is gone. This pass collapses any gap longer than ~0.7 s to
   a single natural breath (~0.35 s) and trims the silent head/tail.
   Short, natural pauses between phrases are kept, so it still breathes
   like a real performance.

The result is a continuous, production-ready vocal with no awkward gaps
where a backing track used to be.

### Notes

- Vocals-only adds roughly 2–5 minutes per song on top of the render —
  isolation runs on CPU.
- It needs `audio-separator`: `pip install "audio-separator[cpu]"`. A
  fresh install via `scripts/install_acestep.sh` already includes it
  (it's in `requirements.txt`). The Roformer model (~200 MB) downloads
  once, on first use, into `~/.cache/audio-separator-models/`.
- The default vocals-only song is naturally short (~1–1.5 min). The
  `--with-music` song aims for ~2:30 — both adjustable via `--duration`.
- `--with-music` output gets a `_with_music` suffix
  (`happy_birthday_<name>_with_music.mp3`) so it never overwrites the
  vocals-only file.
- Every intermediate WAV (full render, isolated stem, tightened vocal) is
  cached, so re-running is instant. Force a redo with `--no-cache`.

---

## Caching

Rendered WAVs are cached in `cache/acestep/`. The cache key is a hash of
`(name, voice profile, mode, prompt, lyrics, duration, steps, guidance)`.
So
calling the same name + voice twice is instant; tweaking the lyrics or a
voice prompt invalidates everyone correctly. Force a re-render with
`--no-cache`.

---

## Performance & memory

ACE-Step's model is 3.5B parameters. On Apple Silicon (MPS) the library
defaults to `float32`, which makes the model **~14 GB in RAM**. On a Mac
with 16–18 GB that overflows physical memory, so macOS swaps to disk —
and a render that should take minutes takes **hours** (you'll see the
diffusion bar crawling at 60+ s/iteration).

The fix: render in **`float16`**, which halves the model to **~7 GB** and
keeps it entirely in RAM. This is now the **default**. `float16` is the
standard precision for diffusion inference, so audio quality is
effectively identical to `float32` — you just stop swapping. Under the
hood this sets the `ACE_PIPELINE_DTYPE=float16` env var that ACE-Step
honours.

```bash
# float16 is the default — nothing to do
python generate_birthday.py --name "Huwaiza"

# force full float32 (only worth it on a Mac with 32 GB+)
python generate_birthday.py --name "Huwaiza" --precision float32
```

Other tips for high throughput:

- **Use `batch_render.py`, not a loop of `generate_birthday.py`.** The
  ~1–2 min model load happens *once* per process; a batch amortises it
  across every name. Running the single-name CLI 30 times reloads the
  model 30 times — that alone wastes ~30–60 minutes.
- **Quit other memory-hungry apps** (browsers especially) before a big
  batch. Every GB of free RAM is one less GB that might swap.
- **Run big batches overnight.** Budget ~7–15 min per vocals-only song
  (render + isolation + tightening), so 20–30 songs is a 3–7 hour
  unattended run — start it, collect the MP3s in the morning.
- The first render of any session also downloads ~4 GB of weights (once,
  ever) and loads them — budget a few extra minutes for run #1.

---

## CLI reference

`generate_birthday.py` — single name:

| Flag             | Default                      | Notes                                  |
|------------------|------------------------------|----------------------------------------|
| `--name`         | required                     | The recipient's name                   |
| `--voice N`      | `MD5(name) % 4`              | Pin a specific voice profile           |
| `--duration`     | `100` vocals / `150` music   | Song length in seconds                 |
| `--steps`        | `60`                         | Diffusion steps. Lower = faster, lower quality |
| `--guidance`     | `15.0`                       | Classifier-free guidance scale         |
| `--precision`    | `float16`                    | `float16` (~7 GB RAM) or `float32` (~14 GB) |
| `--with-music`   | off                          | Add the backing band. Default (off) = vocals only |
| `--output`       | `happy_birthday_<name>.mp3`  | Output MP3 path                        |
| `--no-cache`     | off                          | Skip the cache, always re-render       |
| `--list-voices`  | —                            | Print profiles and exit                |

`batch_render.py` — multi-name:

| Flag             | Default       | Notes                                       |
|------------------|---------------|---------------------------------------------|
| `--names`        | required      | Path to a text file (one name per line)     |
| `--out`          | required      | Output directory for MP3s                   |
| `--duration`     | `100`/`150`   | Per-song length (vocals-only / with music)  |
| `--steps`        | `60`          | Diffusion steps                             |
| `--guidance`     | `15.0`        | CFG scale                                   |
| `--precision`    | `float16`     | `float16` (~7 GB RAM) or `float32` (~14 GB)  |
| `--with-music`   | off           | Add the backing band. Default (off) = vocals only |
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
│   ├── acestep_render.py     ACE-Step pipeline wrapper + cache
│   └── vocal_isolate.py      Roformer isolation + pause tightening
│
├── external/
│   └── ACE-Step/           Cloned by scripts/install_acestep.sh
│
├── scripts/
│   ├── install_acestep.sh    One-shot installer
│   └── smoke_test.py         30-s render to verify the install
│
├── cache/acestep/          WAV cache, keyed by inputs hash
├── tests/                  pytest unit tests (lyrics, voices, isolation)
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

- **Render speed, no quality cost**: keep the default `--precision float16`
  (see **Performance & memory**). This is the single biggest speedup on a
  Mac with 16–18 GB RAM.
- **Faster but rougher renders**: `--steps 30 --guidance 12`
- **Slower but cleaner renders**: `--steps 90 --guidance 18`
- **Different vocal style**: edit `song/voice_profiles.py` — each profile
  is a vocalist descriptor + a base seed, expanded into two prompts (a
  cappella and full-band). Re-render after editing invalidates the cache.
- **Pause tightening**: the gap thresholds live at the top of
  `song/vocal_isolate.py` (`_LONG_GAP_MS`, `_KEPT_GAP_MS`, `_EDGE_PAD_MS`)
  — raise `_KEPT_GAP_MS` for more breathing room, lower it for a tighter
  read.
- **Different lyrics**: edit `_SONG_TEMPLATE` in `song/lyrics.py`. Keep the
  `[verse]` / `[bridge]` / `[chorus]` / `[outro]` tags — they're how
  ACE-Step understands song structure.

---

## Tests

```bash
source venv-diffsinger/bin/activate
pytest -v
```

Tests cover lyrics building, voice rotation, and the vocal-isolation /
pause-tightening helpers. The ACE-Step render path itself isn't
unit-tested (it's a 4 GB neural network) and neither is Roformer
separation — `scripts/smoke_test.py` serves as the integration test.

---

## Credits

- Music synthesis: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) by ACE Studio & StepFun AI (Apache 2.0)
- Vocal isolation: [audio-separator](https://pypi.org/project/audio-separator/) with a BS-Roformer model (default vocals-only mode)
- Audio I/O: [pydub](https://github.com/jiaaro/pydub), [soundfile](https://github.com/bastibe/python-soundfile)
- Apple Silicon acceleration: [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
