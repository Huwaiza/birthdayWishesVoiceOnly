# Render optimisation — findings and handoff

**Branch:** `huwaiza/optimization`
**Last updated:** 2026-07-26
**Status:** ACE-Step 1.5 spike **completed and rejected on quality**. All 1.5
artifacts deleted. Memory/compute fixes from the earlier round are **kept and
committed**.

---

## TL;DR for whoever picks this up

1. **The v1 pipeline is untouched and is what runs in production.** Nothing in
   `song/`, `service/`, or `run_service.sh` references ACE-Step 1.5.
2. **ACE-Step 1.5 turbo was 4.4x faster (8 min → 1.8 min) but the vocals were
   rejected** — audible backing music and a "suppressed voice" artifact after
   separation. Three prompt/separation strategies were tried; all failed the
   listening test.
3. **The unresolved problem is vocal isolation, not generation speed.** Every
   approach that strips backing after the fact damages the voice.
4. The one approach **not yet tried** is native stem extraction
   (`acestep-v15-base`, `task_type="extract"`). Commands to resume are at the
   bottom.

---

## Part 1 — What was kept (committed, in production path)

Commits `f85323e` and `bcb8f91` on `huwaiza/optimization`. These change memory
behaviour only — **audio output is byte-identical to before**, because
`infer_step` still defaults to 60, the value that was previously hardcoded.

| Change | File | Effect |
|---|---|---|
| Shared memory-release helpers | `song/memory.py` (new) | One implementation of `free_render_memory()` / `free_all_models()` used by both the service and the batch runner |
| Roformer separator singleton | `song/vocal_isolate.py` | Was rebuilt on **every** render (~20x/day in a long-lived process). Now loaded once. Measured: 2nd separation 3.6s vs 8.1s |
| Periodic soft reload | `service/app.py` | `RENDER_SOFT_RELOAD_AFTER_JOBS` (default 10) drops model singletons in-process and reclaims memory. Works under bare `uvicorn`, unlike the pre-existing hard recycle |
| Startup guard banner | `service/app.py` | Prints which memory guards are active — the hard recycle and hang restart are env-gated and were silently OFF under bare `uvicorn` |
| `infer_step` exposed | `service/app.py` | `RENDER_INFER_STEP` env var + per-request field. Was hardcoded at 60 with no way to tune it. Job IDs at 60 are unchanged, so the existing render cache stays valid |
| Batch-mode memory hygiene | `batch_render.py` | `--reload-every` (default 10), `--mem-log` |

**Honest assessment of these:** they fix real memory accumulation, but they did
**not** solve the user's actual complaint (heat, fan, slowness). That complaint
is dominated by per-render compute, which these do not reduce.

Still-unmeasured lever in the v1 path: **the QC retry rate**. `qc_max_retries`
defaults to 1, so a QC failure means a *second full render* — an up-to-2x cost
that nobody has quantified. `bench_render.py` (kept) sweeps `infer_step` and
scores results with the production QC gate.

---

## Part 2 — The compute analysis (still valid for v1)

Read from `external/ACE-Step/acestep/pipeline_ace_step.py` (~L1200-1330):

Per diffusion step the transformer runs 1 forward pass outside the guidance
interval, 2 inside it (cond + uncond), or 3 with double-condition guidance.
With `guidance_interval=0.5` and `infer_step=60`:

```
30 steps inside  x 2 = 60
30 steps outside x 1 = 30
                total = 90 transformer forward passes per render
```

Enabling lyric guidance (`guidance_scale_text/lyric > 1.0`) makes it 120 passes
(**+33%**). It is off by default and the Django caller does not set it — keep it
that way unless it earns its cost.

**Not a saving:** the 150s duration is fully used. The Django side renders a
long-form video (uses all 150s) and a Short (truncated to 59s) from the *same*
audio render. Shortening the song is a product change, not free.

---

## Part 3 — The ACE-Step 1.5 spike (rejected)

### Measured results — the speed was real

Same voice prompt, lyrics, name (Ayesha), 150s, seed as production.

| Stage | v1 (3.5B, PyTorch MPS) | 1.5 turbo (2B, MLX) |
|---|---|---|
| Model load | 277s (under memory pressure) | 95-100s |
| **Generation** | ~8 min (reported) | **106s** (3 runs: 107/105/106) |
| Roformer isolate + tighten | ~131s | ~131s (unchanged) |
| **End-to-end** | ~10 min | **~4 min** |
| Peak RSS | ~7 GB | 3.0-6.4 GB |

Caveat: the v1 ~8 min is the *reported* production figure, not a controlled
measurement taken the same day. Generation-vs-generation ≈ 4.4x; end-to-end
≈ 2.5x, because isolation was unchanged and became the new bottleneck.

### Why it was rejected

**The vocals were not acceptable.** After isolation the voice sounded
suppressed/damaged, and backing music was still audible. v1 quality was judged
clearly better. Three strategies were tried:

| Strategy | Result |
|---|---|
| Full mix → Roformer isolation (production architecture) | Rejected — voice suppressed |
| A-cappella prompt ("no instruments, no music, no backing track") | Rejected — music still present |
| Sparse-backing prompt ("single quiet piano, no drums/bass/strings") | Rejected — still suppressed |

Measured evidence that negative prompting is ignored, via near-silent frame
ratio (a cappella has real gaps between phrases; a full mix fills them):

| Output | Near-silent frames |
|---|---|
| Full mix | 6.1% |
| A-cappella prompt | **5.2%** — no better than the full mix |
| Sparse prompt (mix) | 8.0% |
| Roformer-isolated | 16.2% |

**Conclusion: prompting a diffusion model to omit instruments does not work.**

### Key insight for whoever resumes

**Production never generated vocals-only.** The voice profile prompt explicitly
asks for `warm piano ballad, light strings`, and Roformer is what strips it.
Any "just tell the model not to make music" approach has now been tested and
failed. The real problem is that **post-hoc separation is lossy** — it removes
backing that is spectrally entangled with the voice and takes vocal with it.

### The untried approach

ACE-Step 1.5 has native track support:

```python
TRACK_CHOICES = ["vocals", "backing_vocals", "drums", "bass", "guitar", ...]
task_type    = "extract" | "lego"
```

This is model-native, trained jointly with the generator — it should not produce
the separation artifact. **But `acestep-v15-turbo` does not support it.** From
the 1.5 README capability matrix:

| DiT Model | Steps | Extract/Lego | Quality |
|---|---|---|---|
| `acestep-v15-turbo` | 8 | ❌ | Very High |
| `acestep-v15-sft` | 50 | ❌ | High |
| `acestep-v15-base` | 50 | ✅ | Medium |

So native extraction requires `acestep-v15-base`: 4.79 GB, 50 steps (~6x slower
than turbo), and only *Medium* generation quality. The suggested experiment is
**generate with turbo, extract with base** — keeping the good generator and
borrowing base only for its extraction capability. This was downloaded but
**never run**; the spike was stopped first.

### Operational lessons

- **The OOM kills were a disk problem.** Two runs died with `Killed: 9` /
  exit 137 during model load. macOS swap lives on the boot volume, which was at
  8.5 GB free — the kernel could not page out, so it killed the process. With
  78 GB free the same command loaded first try at only 3 GB peak RSS. **Keep
  well clear of a full disk before any model work.**
- **ACE-Step 1.5 overrides download exclusions.** Its loader calls
  `ensure_main_model()`, which re-fetches the whole unified repo for any
  component it thinks is missing — it pulled the 3.76 GB LM we had excluded.
- **`ACESTEP_CHECKPOINTS_DIR` must be set**, or 1.5 resolves its checkpoint dir
  from the cwd and writes weights **inside the git repo**.
- The machine reports **13.3 GB unified memory** available to the GPU, not the
  full 18 GB. That is the real ceiling; `acestep-v15-xl-turbo` (19.95 GB) is not
  viable.

### What was deleted

~17 GB reclaimed: `~/.cache/ace-step-1.5/`, `venv-acestep15/`,
`external/ACE-Step-1.5/`, `output_v15*/`. Production (`~/.cache/ace-step/`,
`external/ACE-Step`, `venv-diffsinger`, `output/`) untouched and verified.

`bench_acestep15.py` is **kept** as the reproduction script — it holds the
working handler-init sequence, the three prompt variants, and the
offload/quantization fallbacks, so resuming does not mean rediscovering the API.

---

## Part 4 — How to resume (manual commands)

### Step 1 — recreate the environment

ACE-Step 1.5 requires Python >=3.11,<3.13, so it **cannot** share
`venv-diffsinger` (3.10). Check disk first — keep 20 GB+ free or you will hit
the OOM-on-load described above.

```bash
cd /Users/huwaizatahir/personal/work/birthdayVoiceOnly
df -h /                                    # need 20GB+ free

git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git external/ACE-Step-1.5
/opt/homebrew/bin/python3.12 -m venv venv-acestep15
venv-acestep15/bin/pip install -e external/ACE-Step-1.5
```

### Step 2 — download the models

```bash
export ACESTEP_CHECKPOINTS_DIR=/Users/huwaizatahir/.cache/ace-step-1.5/checkpoints

# Core (turbo DiT + VAE + Qwen3 embedding + 1.7B LM) — ~10 GB
venv-acestep15/bin/hf download ACE-Step/Ace-Step1.5 --local-dir "$ACESTEP_CHECKPOINTS_DIR"

# Small LM (1.37 GB) — lighter on 18 GB RAM than the bundled 1.7B
venv-acestep15/bin/hf download ACE-Step/acestep-5Hz-lm-0.6B \
  --local-dir "$ACESTEP_CHECKPOINTS_DIR/acestep-5Hz-lm-0.6B"

# Base model (4.79 GB) — ONLY this one supports extract/lego
venv-acestep15/bin/hf download ACE-Step/acestep-v15-base \
  --local-dir "$ACESTEP_CHECKPOINTS_DIR/acestep-v15-base"
```

Run these **sequentially in one terminal** — the `export` does not carry to a
second terminal, and parallel downloads compete for disk and bandwidth.

### Step 3 — reproduce the baseline (should give ~106s generation)

```bash
ACESTEP_CHECKPOINTS_DIR=/Users/huwaizatahir/.cache/ace-step-1.5/checkpoints \
  venv-acestep15/bin/python bench_acestep15.py --name Ayesha --steps 8
```

Flags available: `--acapella`, `--sparse` (both tested and rejected),
`--offload`, `--offload-dit`, `--quantization int8_weight_only` (memory
fallbacks), `--duration`, `--seed`, `--save-dir`.

### Step 4 — the actual next experiment (native extraction)

`bench_acestep15.py` does **not** implement this yet. It needs a second handler
on `acestep-v15-base` with `task_type="extract"`, `extract_track="vocals"`, and
`src_audio` pointing at the turbo output. See `external/ACE-Step-1.5/cli.py`
(~L1318-1420) for the initialisation sequence and `acestep/inference.py`
(`GenerationParams`) for the parameter names.

---

## Recommendation

**Do not resume the 1.5 migration on speed grounds alone.** The speed was proven
(4.4x on generation) and it still lost on quality, which was the binding
requirement. Resume only if native `extract` is worth testing — that is the one
remaining idea that attacks the actual defect rather than working around it.

If the goal is purely "less heat, less time, same quality", the cheaper bets on
the **existing v1 pipeline** are, in order:

1. **Measure the QC retry rate.** An up-to-2x cost nobody has quantified.
2. **Sweep `infer_step`** with `bench_render.py` — now tunable via
   `RENDER_INFER_STEP` without touching source.
3. **Faster separation.** Isolation costs ~131s/song and is pure overhead. A
   Demucs MLX port claims 34x realtime. This does not fix the suppression
   artifact, but it is the largest remaining time saving that carries no
   quality risk.
4. **Keep the disk clear.** Swap lives on it; a full disk is what turned a
   working render into a `Killed: 9`.

And the honest structural point: 20 songs/day of diffusion inference is a server
workload running on an 18 GB laptop that is also a daily driver. A rented GPU
would do the same batch in a fraction of the time, and the heat problem stops
being a software problem.
