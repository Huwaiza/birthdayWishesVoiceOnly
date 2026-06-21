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

# One song (vocals only — the default)
python generate_birthday.py --name "Huwaiza"

# A batch
python batch_render.py --names names.txt --out output/

# Keep the backing band instead
python generate_birthday.py --name "Huwaiza" --with-music
```

Once the model is loaded, a 2:30 song renders in roughly 5–10 minutes on
M-series silicon (depends on `--steps` and the chip). If yours is far
slower than that, see **Performance & memory** below — it's almost always
a RAM-swapping problem with a one-flag fix.

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
song/vocal_isolate.py     → DEFAULT (vocals only): a Roformer model strips
                            every instrument, then tighten_pauses() collapses
                            the long silent gaps the backing left behind.
                            (Skipped with --with-music.)
    │
    ▼
generate_birthday.py      → Encodes the WAV to a 192 kbps MP3.
```

ACE-Step composes the melody, sings it, and mixes the band in one pass; by
default we then lift just the vocal back out and tidy the gaps, so the
finished song is a clean a cappella. Add `--with-music` to keep the band.

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

## A cappella (vocals only) — the default

By default the finished song is **vocals only**. ACE-Step still renders a
full mix (voice + backing band), then two post-steps run automatically:

```bash
python generate_birthday.py --name "Huwaiza"               # vocals only (default)
python batch_render.py --names names.txt --out output/     # vocals only (default)

python generate_birthday.py --name "Huwaiza" --with-music  # keep the band
```

**1. Isolate** — the full mix goes through a **Roformer** vocal-separation
model (via [audio-separator](https://pypi.org/project/audio-separator/)),
keeping only the vocal stem. Roformer is the current state of the art: it
physically pulls the recording apart into stems, so the result is
instrument-free for real — far cleaner than prompting ACE-Step for "a
cappella" (which a generative model may not honour, and which actually
*degrades* the separation by starving it of a real mix to work on). One
pass is deliberate; a second would add a faint watery artefact to the voice.

**2. Tighten pauses** — isolating the vocal leaves long silences where the
instrumental intro/fills/outro used to be. `tighten_pauses()` collapses only
those long gaps (≥1 s) down to a natural breath, leaving every sung note and
short pause untouched, so the song stops dragging without ever clipping the
melody.

Trade-offs:

- Adds roughly 2–5 minutes per song — separation runs on CPU.
- Needs `audio-separator`: `pip install "audio-separator[cpu]"`. A fresh
  install via `scripts/install_acestep.sh` already includes it (it's in
  `requirements.txt`). The Roformer model (~200 MB) downloads once, on
  first use, into `~/.cache/audio-separator-models/`.
- The full-song version (`--with-music`) goes to
  `happy_birthday_<name>_with_music.mp3`, so the two never overwrite.

The isolated and tightened WAVs are cached alongside the full render, so
re-running the same name is instant. Force a re-render with `--no-cache`.

---

## Caching

Rendered WAVs are cached in `cache/acestep/`. The cache key is a hash of
`(name, voice profile, prompt, lyrics, duration, steps, guidance)`. So
calling the same name + voice twice is instant; tweaking the lyrics or a
voice prompt invalidates everyone correctly. Force a re-render with
`--no-cache`.

---

## Running as a service (HTTP)

For on-demand use from another local process, run the bundled HTTP service.
It keeps the model loaded once, renders one song at a time, and writes the
finished MP3 to a shared folder another process can read directly.

```bash
# Install the extra deps (into the same venv that has ACE-Step), once:
pip install -r service/requirements.txt

# Start it (one worker = model loaded once; renders are serialized):
uvicorn service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Because a render takes minutes, the API is **submit-then-poll** — you never
hold a long request open:

| Method & path           | Purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `GET  /health`          | `{status, model_ready, queued_or_running}`                         |
| `POST /jobs`            | Submit `{name, voice?, duration?, lyrics?, slug?}` → `{job_id, status, mp3_path?}` |
| `GET  /jobs/{job_id}`   | Poll → `{status, mp3_path, error, ...}`                            |
| `GET  /jobs/{job_id}/audio` | Optional: stream the MP3 over HTTP (not needed locally)        |

`status` moves `queued → running → done` (or `error`). When `done`,
`mp3_path` is the absolute path of the finished file.

**Custom (non-birthday) songs.** `lyrics` and `slug` are optional. Pass
`lyrics` (a full ACE-Step `[verse]/[bridge]/[chorus]/[outro]` blob) to render
something other than the birthday template — when omitted, the service builds
the birthday lyrics from `name` exactly as before. Pass `slug` to write the
output as `<slug>.mp3` instead of `happy_birthday_<name>.mp3`, so a custom song
never collides with a real person's birthday of the same name. These power the
`generate_custom_video` campaign command in the `birthdaygen` repo.

**Where files land.** Finished MP3s are written to `output/` (override with
the `BIRTHDAY_OUTPUT_DIR` env var) as `happy_birthday_<name>.mp3` — or
`<slug>.mp3` when a `slug` is given. The folder is created automatically.
Intermediate WAVs reuse the normal `cache/acestep/` cache, so re-requesting the
same parameters returns instantly.

**Idempotent & crash-safe.** Submitting the same
`(name, voice, duration, lyrics, slug)` twice returns the same job; if the MP3
already exists (even after a restart), the service reports `done` without
re-rendering.

```bash
# Submit a song, then poll until it's done:
curl -s -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"name": "Mujtaba"}'
# → {"job_id":"ab12cd34ef56","status":"queued","mp3_path":null,...}

curl -s localhost:8000/jobs/ab12cd34ef56
# → {"status":"done","mp3_path":".../output/happy_birthday_mujtaba.mp3",...}
```

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
- **Run big batches overnight.** At ~5–10 min/song, 20–30 songs is a
  3–5 hour unattended run — start it, collect the MP3s in the morning.
- The first render of any session also downloads ~4 GB of weights (once,
  ever) and loads them — budget a few extra minutes for run #1.

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
| `--precision`    | `float16`                    | `float16` (~7 GB RAM) or `float32` (~14 GB) |
| `--with-music`   | off                          | Keep the backing band (default is vocals only) |
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
| `--precision`    | `float16`     | `float16` (~7 GB RAM) or `float32` (~14 GB)  |
| `--with-music`   | off           | Keep the backing band (default is vocals only) |
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
│   └── vocal_isolate.py      Roformer vocal-stem isolation + pause tightening
│
├── service/                HTTP service (see "Running as a service")
│   ├── app.py                FastAPI app — submit/poll jobs, warm model
│   └── requirements.txt      fastapi + uvicorn (on top of the main deps)
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

- **Render speed, no quality cost**: keep the default `--precision float16`
  (see **Performance & memory**). This is the single biggest speedup on a
  Mac with 16–18 GB RAM.
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

Tests cover lyrics building, voice rotation, pause tightening
(`test_tighten_pauses.py`, synthetic audio — no model), and the HTTP service
(`test_service.py`, with the model + render mocked). The ACE-Step render path
itself isn't unit-tested (it's a 4 GB neural network) — `scripts/smoke_test.py`
serves as the integration test.

---

## Credits

- Music synthesis: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) by ACE Studio & StepFun AI (Apache 2.0)
- Vocal isolation: [audio-separator](https://pypi.org/project/audio-separator/) with a BS-Roformer model (default vocals-only mode)
- Audio I/O: [pydub](https://github.com/jiaaro/pydub), [soundfile](https://github.com/bastibe/python-soundfile)
- Apple Silicon acceleration: [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
