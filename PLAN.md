# Birthday Song Generator — v7 Rebuild Plan
**Date:** 2026-04-25
**Status:** Implementation in progress (post-pivot)
**Supersedes:** docs/superpowers/specs/2026-04-09-birthday-song-generator-design.md (v6 / Bark)

---

## Why this rebuild

v6 used Bark with `[singing]` tags. Bark is a speech model, not a singing model — the tag is unreliable, causing pitch drift, no vibrato, robotic timbre. See `ANALYSIS.md` for the full diagnosis.

## Pivot from earlier draft of this plan

An earlier version of this document specified a **DiffSinger + RVC** pipeline: render vocals from a MIDI of "Happy Birthday" through a singing-synthesis voicebank, then convert to one of N voice models for rotation. That plan died on Day 1 due to install fragility on Apple Silicon (DiffSinger PyTorch deps, OpenUtau voicebank format issues, Python 3.10 vs 3.11+ incompatibilities).

**Post-pivot decision (2026-04-25):** drop the requirement for the *literal traditional Happy Birthday melody* in exchange for a working production pipeline. We use **ACE-Step 1.5** (Apache 2.0, music foundation model, released 2026-04-02) which generates vocal + backing in one pass from a text prompt and tagged lyrics. The lyrics still preserve the song's *content* ("Happy birthday to you / Happy birthday dear {name}…") but the melody is composed by the model rather than transplanted from the public-domain HB tune.

This is a real product trade-off — captured in writing so the next reviewer (us, in two weeks) doesn't have to rediscover it.

## Requirements (post-pivot)

Locked with user on 2026-04-25:

1. **Production music quality** — no compromise.
2. **2 weeks dev budget.**
3. **Free for YouTube-volume daily uploads** (~50-100 songs/day, zero per-song cost).
4. ~~Traditional Happy Birthday melody must be recognisable in verse sections.~~ **DROPPED** — replaced by "song clearly says 'Happy birthday' and the recipient's name; melody is AI-composed."
5. **Voice rotation** across 3-4 female voice profiles.
6. **Apple Silicon Mac** (M-series) as target hardware.

## Architecture (v7 — ACE-Step)

```
name
 │
 ▼
[1] song/lyrics.py           build_lyrics(name) → tagged blob with
 │                            [verse] [bridge] [chorus] [outro]
 ▼
[2] song/voice_profiles.py   pick_voice(name) → VoiceProfile
 │                            (4 distinct female prompts + base seeds;
 │                             rotation via hash(name) % 4)
 ▼
[3] song/acestep_render.py   render(name, voice_idx) → WAV
 │                            ACE-Step generates vocals + backing in
 │                            one diffusion pass, conditioned on prompt
 │                            + lyrics. Caches by (name, voice, prompt,
 │                            lyrics, duration, steps) hash.
 ▼
[4] generate_birthday.py     WAV → 192 kbps MP3 via pydub
```

That's the whole pipeline. No separate vocal/backing/mix stages — ACE-Step does it all in one shot.

### Why this stack

| Stage | Tool | Why |
|-------|------|-----|
| Vocal + backing synthesis | **ACE-Step 1.5** | Music foundation model, Apache 2.0, runs on MPS. Free, local, unlimited. Produces a complete song from a prompt + tagged lyrics. |
| Voice rotation | **Prompt variation + seed** | ACE-Step's "voice" is steered by natural-language descriptors. Four pinned prompts ("warm alto", "bright soprano", etc.) rotate by hash(name). |
| Encoding | **pydub** | Stable, well-known WAV → MP3 conversion. Unchanged from v6. |

### Why NOT DiffSinger (per the earlier draft)

DiffSinger gives MIDI-driven control over the exact melody — which is what requirement #4 originally demanded. But on Apple Silicon, getting a production-grade English female DiffSinger voicebank rendering in 2026 is days-to-weeks of yak-shaving (OpenUtau dotnet GUI, vocoder downloads, voicebank format compatibility, Python dep hell). We tried for a day, got nowhere productive, and the user accepted the trade.

### Why NOT pure DiffSinger with 3-4 voicebanks

Production-grade English DiffSinger voicebanks are a narrow field. Relying on 3-4 of them is fragile. ACE-Step ships with one model file — voice variety comes from prompt-engineering against that single weight set.

### Why NOT keep RVC for voice conversion

RVC was paired with DiffSinger to expand the voice pool. With ACE-Step we don't need it — prompt variations get us the rotation, and adding RVC post-processing would risk artefacts on top of an already-good vocal.

---

## 2-week timeline (revised)

### Week 1 — Working pipeline + first song

**Phase 1 (Day 1, complete-ish): ACE-Step install**

- ✅ Python 3.10 venv (`venv-diffsinger`) on Apple Silicon
- ✅ PyTorch with MPS verified
- ⏳ `pip install -e external/ACE-Step` via `scripts/install_acestep.sh`
- ⏳ `scripts/smoke_test.py` renders a 30-s sample WAV

**Phase 2 (Day 1-2): Pipeline scaffolding**

- ✅ `song/lyrics.py` — tagged lyrics builder
- ✅ `song/voice_profiles.py` — 4 voice profiles + deterministic rotation
- ✅ `song/acestep_render.py` — pipeline wrapper with cache
- ✅ `generate_birthday.py` — single-name CLI
- ✅ `batch_render.py` — multi-name runner with retry/cache

**Phase 3 (Day 2-3): First end-to-end render**

- Run `python generate_birthday.py --name "Huwaiza"` — get a real MP3
- A/B four voices, pick favourites
- Tune duration, infer_step, guidance_scale by ear

**Phase 4 (Day 4-5): Lyrics + name pronunciation tuning**

- Test with hard names: Xiomara, Anastasia, Jo, Luc, Mohammed, Gwyneth
- Iterate on lyrics phrasing if pronunciation is off
- Confirm `{name}` slot fits without forcing the model into weird timing

**Phase 5 (Day 6-7): Voice profile tuning**

- Render the same name through all 4 profiles, listen blind, score them
- Adjust prompts until the four are clearly distinct from each other AND each consistently good across names

### Week 2 — Production polish

**Phase 6 (Days 8-9): Quality pass**

- Loudness normalise (LUFS -14 for YouTube monetisation)
- Spot-check 10 random batch outputs end-to-end
- Tune `infer_step` for quality vs. render time
- A/B against a commercial birthday song

**Phase 7 (Days 10-12): Stress test + batch infrastructure**

- 50 names overnight via `batch_render.py`
- Failure mode analysis: any silent renders? wrong-pronunciation names? clipping?
- Resume support, output validation

**Phase 8 (Day 13-14): Slack buffer**

Always ship with slack. Things will go wrong.

---

## Directory layout (current state)

```
birthdayVoiceOnly/
├── generate_birthday.py            # ACE-Step single-name CLI (REWRITTEN)
├── batch_render.py                 # NEW — batch runner
├── requirements.txt                # SLIMMED to project-only deps
├── PLAN.md                         # this file
├── ANALYSIS.md                     # v6 post-mortem
│
├── song/                           # Python package
│   ├── __init__.py
│   ├── lyrics.py                   # REWRITTEN for ACE-Step tags
│   ├── voice_profiles.py           # NEW
│   ├── acestep_render.py           # NEW — main pipeline wrapper
│   ├── backing.py                  # OBSOLETE (kept until tests are migrated)
│   ├── vocal.py                    # OBSOLETE (kept until tests are migrated)
│   ├── mix.py                      # OBSOLETE (kept until tests are migrated)
│   └── _archive_bark/              # backup of v6 vocal.py + reqs
│
├── external/
│   └── ACE-Step/                   # populated by install_acestep.sh
│
├── scripts/
│   ├── install_acestep.sh          # one-shot installer
│   └── smoke_test.py               # 30s sample render to confirm install
│
├── docs/
│   └── DIFFSINGER_SETUP.md         # historical — kept for reference
│
├── cache/
│   └── acestep/                    # WAV cache, key = hash(name+voice+prompt+lyrics)
│
└── tests/                          # to be updated (see Phase 2)
```

---

## Out of scope (for v7)

- True traditional Happy Birthday melody (would require DiffSinger or hand-authored MIDI + a working singing synth on Mac).
- YouTube upload automation.
- Video generation (visuals, name cards).
- Multiple language support.
- Web UI.
- RVC post-processing (no longer needed; voice rotation is in-prompt).

## Kill criteria

Reasons to abandon ACE-Step and fall back to **Option C** (accept Bark quality, ship as-is) from ANALYSIS.md:

- ACE-Step won't install in venv-diffsinger after one Phase 1 retry.
- Smoke test produces silence or unintelligible output on Mac MPS.
- Per-song render time is > 10 minutes on this hardware (kills YouTube-volume requirement).
- Prompt-driven voice rotation can't yield 3 audibly distinct female voices.

If any of these hit, stop and reassess.
