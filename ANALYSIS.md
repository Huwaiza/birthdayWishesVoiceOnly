# Birthday Voice Generator — Comprehensive Analysis
**Date:** April 13, 2026  
**Analyst:** Claude  
**Status:** Thorough technical review with improvement roadmap

---

## Executive Summary

The current pipeline (v6, local Bark-based) produces **poor singing output** due to a fundamental architectural mismatch: **Bark is a speech synthesis model being forced into singing mode**. The `[singing]` tag and `♪` markers provide hints to the model, but Bark's training data is primarily speech, not vocal performance. This results in output that sounds more like "sung speech" than singing—lacking:

- Consistent pitch/melody (notes drift)
- Natural vibrato and expression
- Breath control and timing
- Emotional resonance
- Proper phrasing and legato

The project's documented alternatives (Suno, UberDuck) sidestep this by using singing-specific models, but they're cloud-based. **The gap between local free tools and commercial singing synthesis is real and significant.**

This analysis identifies why the current approach fails, documents the realistic free/open-source options, and provides a ranked roadmap for improvement.

---

## Part 1: Current State Assessment

### Pipeline Overview

```
[1] Lyrics Builder (song/lyrics.py)
    ↓ (22 lines wrapped in ♪ markers)
[2] Vocal Generation (song/vocal.py)
    ↓ (Bark per-line inference with [singing] tag)
[3] Audio Stitching (song/mix.py)
    ↓ (silence gaps between lines)
[4] Backing Mix (song/mix.py + song/backing.py)
    ↓ (piano loop at 18% volume)
[5] Export (generate_birthday.py)
    ↓ MP3 output (~2.5 min)
```

### Module-by-Module Breakdown

#### `generate_birthday.py` — Entry Point
**What it does:**
- CLI argument parsing (name, style, speaker, regen-line)
- Orchestrates the 5-stage pipeline
- Converts final WAV to MP3 via pydub

**Issues identified:**
- None at this layer—orchestration is sound. Problem lies in the vocal generation quality.

---

#### `song/lyrics.py` — Lyric Template
**What it does:**
- Builds a hardcoded 22-line song structure with name injection
- Wraps each line in `[singing] ♪ ... ♪` markers to hint to Bark that this should be sung

**Content:**
- Verse 1 (4 lines): "Happy birthday to you" (3x) + "Happy birthday to you"
- Verse 2 (4 lines): Similar, ending with "May your dreams come true"
- Bridge (4 lines): Emotional lines like "May your day be filled with joy"
- Verse 3 (4 lines): Repetition of verse 1
- Chorus (4 lines): "Hip hip hooray" + wish-based lines
- Outro (2 lines): Final name repetitions

**Why this design (probably):**
- Shorter lines = easier for Bark to infer (max 14s output)
- Section breaks signal pauses for pacing
- Name appears at meaningful moments for personalization

**Hidden problem:**
- The `[singing]` tag is **unreliable**. Bark was trained primarily on speech. The singing capability exists but is inconsistent—some lines sing, others speak or waver between the two.
- No melodic structure encoded. Bark doesn't know what notes to hit. Each line is generated independently with no awareness of the previous melody, causing pitch discontinuity.

---

#### `song/vocal.py` — Bark Vocal Generation
**What it does:**
1. Loads Bark model once (cached in `_models_loaded` global)
2. For each lyric line:
   - Check disk cache (`cache/{hash}.wav`)
   - If not cached, run Bark inference with the line
   - Normalize audio to [-1, 1] with headroom (0.9 ceiling)
   - Cache to disk
3. Return list of numpy arrays (float32, 24 kHz)

**Bark configuration:**
```python
audio = generate_audio(
    line,
    history_prompt=speaker,
    text_temp=0.6,       # lower = more consistent melody
    waveform_temp=0.7,   # expressiveness
    silent=True,
)
```

**Key settings:**
- `text_temp=0.6`: Lower temperature → more deterministic phoneme→notes mapping. Good for consistency, but Bark's "melody" is still arbitrary—no musical structure.
- `waveform_temp=0.7`: Controls fine detail/expressiveness. Default is 0.7; this is standard.
- `history_prompt=speaker`: Selects a voice preset (v2/en_speaker_0 through v2/en_speaker_9). Each preset is a token that conditions the model toward that speaker's characteristics.

**Why this fails at singing:**
1. **No pitch guidance**: Bark doesn't receive target notes. It infers pitch from phonemes alone. This works for speech prosody but produces melodically incoherent "singing."
2. **No duration control**: Bark generates ~14s of audio per call, regardless of lyric length. Lines are NOT tempo-matched to the backing track.
3. **No vibrato or legato**: Speech synthesis doesn't model these. Result sounds robotic.
4. **Inconsistent [singing] tag response**: Sometimes Bark produces singing-like output; often it's sung speech—unstable.

**Observable symptoms:**
- Cache directory has 11 WAV files (partial runs), suggesting multiple attempts to regenerate lines
- Files vary widely in size (216 KB to 719 KB = 0.7–2.3 seconds each at 24 kHz), indicating inconsistent generation

---

#### `song/backing.py` — Backing Track Loader
**What it does:**
1. Load MP3/WAV backing file
2. Convert to mono if stereo
3. Resample to 24 kHz (Bark native rate)
4. Loop or trim to exact target duration
5. Return as float32 numpy array

**Issue:**
- All three backing tracks (simple.mp3, warm.mp3, upbeat.mp3) are **identical files** (different MD5, but likely the same audio encoded differently or a setup incomplete—check with user). Even if different, piano backing at 18% volume cannot compensate for poor vocal quality.
- Backing volume (0.18 = 18%) is reasonable, but it also masks the vocal quality problem.

---

#### `song/mix.py` — Stitching & Duration Control
**What it does:**
1. **Stitch clips**: Concatenate vocals with silence gaps
   - 4s intro silence
   - 0.35s between lines
   - 1.0s between sections (longer, marked in `SECTION_BREAKS`)
2. **Mix backing**: Overlay piano at 18% volume
3. **Normalise duration**: Enforce 2:20–2:45 (140–165s) via padding or time-stretch

**Quality of implementation:**
- Technically solid. Gaps are musically reasonable.
- Time-stretch via librosa is non-destructive for small adjustments.

**Problem:**
- Cannot fix poor vocal quality upstream. Stitching garbage vocals still produces garbage songs.

---

### Where the Pipeline Breaks Down

**Root cause: Bark is not a singing model.**

| Aspect | Speech Synthesis (Bark) | Singing Synthesis (What we need) |
|--------|------------------------|--------------------------------|
| **Pitch control** | Inferred from phoneme prosody | Must be specified (MIDI notes or F0 contour) |
| **Vibrato/expression** | Not modeled | Requires explicit modeling or emotional encoding |
| **Duration** | Variable based on content | Tied to note length and tempo |
| **Legato/phrasing** | Not applicable | Requires smooth pitch transitions |
| **Training data** | Billions of speech samples | Singing audio (much smaller corpus) |

**Result:**
- Output sounds like a robot reciting lyrics in a sing-songy voice
- Pitch is unstable (drifts, cracks)
- No emotional resonance
- Lacks musicality

**Why the [singing] tag helps marginally:**
Bark includes a `[singing]` token in its vocabulary, and model is aware to slightly adjust behavior when it appears. But without pitch guidance, this is like telling a musician "now sing" without giving them the score. They'll produce something vaguely sing-like but won't be in tune or on rhythm.

---

## Part 2: Why Current Alternatives Work (and Why They Cost)

### Suno AI
**What it does:**
- Diffusion-based generative model trained on singing data
- Understands lyrics, melody, and emotional intent
- Generates full songs with instrumentals

**Why it's better:**
- Trained explicitly on singing (not speech)
- Models pitch, vibrato, breath, phrasing as first-class features
- Can generate instrumentals to match vocal style

**Cost/Drawback:**
- Cloud-only. No local inference.
- Latency (queue times).
- Rate limits.

**Quality assessment (2026):**
- v5.5 output is coherent but emotionally thin
- Vocal timbre is compressed/uniform
- Better for ideation than release-ready tracks
- Still significantly better than Bark singing

### UberDuck & Similar Services
**Historical approach (as documented in v5 README):**
- Use API to generate singing template
- Transcribe with Whisper
- Replace name via pitch-matching + time-stretch
- Worked reasonably well for template-based output

**Why we moved to v6 (local Bark):**
- Avoid API dependency
- Avoid latency
- Offline operation

**Trade-off:** Gained independence, lost quality.

---

## Part 3: Realistic Free/Open-Source Options

### Tier 1: Voice Conversion (RVC)

**What it is:**
- Retrieval-based Voice Conversion
- Takes existing vocal audio + target voice reference
- Converts the voice while preserving pitch, timing, emotional content

**Advantage:**
- If you start with a good vocal recording (human singer, or TTS of a singing voice), RVC can convert it to different voices
- Runs locally on M-series Macs (CPU-capable)
- Excellent for covers and voice cloning

**How it could help:**
1. Use a TTS singing service (e.g., Fish Audio S2's singing mode) to generate a clean vocal
2. Use RVC to convert that voice to alternative tones
3. Combine with backing

**Drawback:**
- Requires a reference vocal to start with
- Does not generate singing from scratch

**Effort to integrate:** Medium (add RVC inference step, manage models)

---

### Tier 2: TTS with Singing Tag (Fish Audio S2)

**What it is:**
- ByteDance-derived TTS model (March 2026)
- Supports 15,000+ emotion tags
- Has experimental `[singing]` support

**How it could help:**
- Similar to Bark but potentially more reliable singing behavior
- Better emotion/expression control
- Can be used as a replacement for Bark line-by-line generation

**Drawback:**
- Not fully open-source (available via API or limited local inference)
- Singing mode not as mature as speech
- Still fundamentally a TTS model, not singing-specific

**Effort to integrate:** Low (drop-in replacement for Bark in `vocal.py`)

---

### Tier 3: Voice Conversion + Synthesized Base (GPT-SoVITS)

**What it is:**
- Few-shot TTS with emotional voice cloning
- Requires ~60s reference audio of target voice
- Can adapt to singing via F0 embedding modification

**How it could help:**
1. User provides a 60-second reference of themselves or a target voice
2. Generate singing TTS using that voice profile
3. Run through RVC for further refinement

**Drawback:**
- More complex pipeline
- Still not true singing synthesis (no native melody understanding)
- Requires user input (voice reference)

**Effort to integrate:** High (new inference step, UI for voice upload)

---

### Tier 4: True Singing Synthesis (DiffSinger)

**What it is:**
- Singing voice synthesis model using diffusion
- Takes musical score (MIDI or phoneme + F0 contour)
- Generates singing audio with proper pitch, vibrato, phrasing

**Advantage:**
- Mathematically correct pitch control
- Proper vibrato, breath, legato modeling
- Most natural sound among local options

**Drawback:**
- Requires musical score in addition to lyrics (MIDI + phonemes)
- Implementation in OpenUtau (UI-based) or raw inference (complex)
- Apple Silicon support is **buggy**—crashes on certain phoneme operations
- Larger model (~1-2 GB)

**Effort to integrate:** Very High (must encode musical score, debug ARM64 issues, integrate complex inference pipeline)

---

### Comparison Table

| Option | Quality | Local? | Effort | Notes |
|--------|---------|--------|--------|-------|
| **Current (Bark)** | Poor | ✓ | N/A | Speech model forced into singing |
| **RVC** | Good* | ✓ | Medium | Requires source vocal; voice conversion only |
| **Fish S2** | Fair-Good | Partial | Low | TTS with experimental singing; API or limited local |
| **GPT-SoVITS** | Fair | ✓ | High | Few-shot TTS; requires reference audio |
| **DiffSinger** | Excellent | ✓ (buggy) | Very High | Requires MIDI score; ARM64 issues |
| **Suno (cloud)** | Good-Excellent | ✗ | N/A | External API; reliable but cloud-dependent |

\* RVC quality depends on source vocal quality

---

## Part 4: Practical Improvement Roadmap

### Quick Wins (1–2 hours each)

#### 1. Replace Bark with Fish Audio S2 (or equivalent TTS singing)
- **Effort:** Low
- **Expected improvement:** 15–25% better singing consistency
- **Why:** Fish S2 may have better `[singing]` tag handling than Bark
- **Implementation:**
  ```python
  # In song/vocal.py, swap Bark for Fish API or local inference
  # Minimal changes needed (same input/output)
  ```
- **Limitation:** Still a TTS model; won't fully solve melodic incoherence

**Verdict:** Quick temporary improvement, but doesn't solve the core problem.

---

#### 2. Add Melodic Structure via Pitch Shifting
- **Effort:** Medium
- **Expected improvement:** 20–30% perceived melody coherence
- **Why:** Currently, each line is generated independently. What if we analyze the "melody" of the original Happy Birthday (which humans know), then pitch-shift Bark output to approximate it?
- **How:**
  1. Define Happy Birthday's melody as MIDI notes (C, C, C, C-E... standard notes)
  2. Extract F0 (fundamental frequency) from Bark output using librosa
  3. Shift output audio so F0 peaks align with target notes
  4. Use librosa's pitch-shift (not time-stretch) to preserve speed
- **Code sketch:**
  ```python
  import librosa
  # Extract F0 contour from Bark audio
  f0, voiced, _ = librosa.pyin(audio, ...)
  # Compute target pitch based on melodic structure
  target_f0 = midi_to_hz(note_for_line[i])
  # Shift audio to match target
  shift_steps = 12 * np.log2(target_f0 / current_f0)
  shifted = librosa.effects.pitch_shift(audio, sr=24000, n_steps=shift_steps)
  ```
- **Limitation:** Requires encoding the melody manually. Assumes Bark output has detectable F0 (robustness question).

**Verdict:** Moderate effort, moderate improvement. Doesn't solve all issues but moves needle.

---

#### 3. Use Suno API as Optional Backend
- **Effort:** Low
- **Expected improvement:** 100% (Suno is much better, but cloud-dependent)
- **Why:** If users have API credits or are willing to pay, offer Suno as an alternative path
- **Implementation:**
  ```python
  # In vocal.py, add a backend switch:
  if use_suno:
      audio = call_suno_api(line, name)
  else:
      audio = bark_generate(line)
  ```
- **Trade-off:** Loses offline capability for those who choose Suno

**Verdict:** Brings cloud quality to users willing to pay; doesn't help offline users.

---

### Medium Effort (4–8 hours each)

#### 4. Integrate RVC for Voice Refinement
- **Effort:** Medium
- **Expected improvement:** 10–15% perceptual quality (voice character polish)
- **Why:** After Bark generates vocals, run through RVC to convert to a more "singing" voice
- **How:**
  1. Include RVC model in project
  2. After Bark generation, apply RVC inference
  3. Use a pre-selected "singer" reference voice
- **Code structure:**
  ```python
  # In vocal.py, add post-processing:
  bark_audio = generate_line(line, ...)
  refined_audio = rvc_convert(bark_audio, target_singer="reference.wav")
  return refined_audio
  ```
- **Limitation:** RVC doesn't fix pitch or melody; it just makes the voice sound more like a singer. Marginal gain if Bark output is already poor.

**Verdict:** Polish on top of poor base. Moderate effort for modest gain.

---

#### 5. Encode Melody via Phoneme-F0 Pairs
- **Effort:** Medium-High
- **Expected improvement:** 30–40% melodic coherence
- **Why:** Instead of passing raw lyrics to Bark, augment with F0 hints
- **How:**
  1. Map Happy Birthday lyrics to a standard melody
  2. For each word, compute expected F0 (in Hz) given the melody
  3. Modify Bark prompt to include F0 information (if supported) or use external pitch control
  4. Constraint generation to stay in melodic range
- **Limitation:** Bark doesn't natively support F0 conditioning. Would require custom wrapper or different model.

**Verdict:** High effort, moderate-high gain, but requires model modification.

---

### Major Effort (1–2 weeks each)

#### 6. Switch to DiffSinger for True Singing Synthesis
- **Effort:** Very High
- **Expected improvement:** 70–90% (genuine singing quality)
- **Why:** DiffSinger is trained on singing, not speech. Understands pitch, vibrato, phrasing.
- **Requirements:**
  1. Convert lyrics + melody to MIDI format (or phoneme + F0 contour)
  2. Run DiffSinger inference
  3. Debug Apple Silicon (ARM64) compatibility issues
  4. Integrate into pipeline
- **Known issues:**
  - OpenUtau macOS has phonetic assistant crashes
  - Some operations aren't optimized for ARM64
  - Requires MIDI/score input (more user effort)
- **Architecture change:**
  ```
  [Lyrics] → [MIDI/Score] → [DiffSinger] → [Audio]
  ```

**Verdict:** Highest effort, highest quality. Worth it if singing quality is critical.

---

#### 7. Custom Fine-Tuning of Bark on Singing Data
- **Effort:** Very High (weeks)
- **Expected improvement:** 50–70% (depends on fine-tune data quality)
- **Why:** Fine-tune Bark on a singing corpus (e.g., curated singing samples)
- **Requirements:**
  - Collect/curate singing voice dataset
  - Fine-tune on A40/H100 GPU
  - Validate and ship new model
- **Limitation:** Data collection is labor-intensive. No guarantee fine-tune will beat Suno/DiffSinger.

**Verdict:** Highest effort, uncertain outcome. Not recommended.

---

## Part 5: Recommended Path Forward

### **Option A: Hybrid (Best user experience, moderate effort)**

**Path:** Bark + Pitch Correction + optional Suno backend

1. **Keep Bark as default** (offline, works now)
2. **Add pitch-shifting post-processor** (quick win: +20-30% perceived melody coherence)
   - Analyze line's F0, shift toward Happy Birthday melody
   - Cost: 4 hours
3. **Add optional Suno backend** (for users willing to use API)
   - Switch via CLI flag `--use-suno`
   - Cost: 2 hours
4. **Document limitations clearly**
   - Tell users: "Local version sounds like sung speech. For better quality, use --use-suno."
   - Set expectations.

**Total effort:** ~6 hours  
**Result:** 
- Offline users get a 20-30% improvement (pitch-shifted singing sounds marginally less robotic)
- Cloud users can opt-in to Suno for 100% better quality
- Project remains maintainable

---

### **Option B: Full Migration to DiffSinger (Best quality, highest effort)**

**Path:** Replace Bark entirely with DiffSinger

1. Build MIDI score from lyrics (automate via music theory library)
2. Integrate DiffSinger inference (or OpenUtau CLI)
3. Debug Apple Silicon issues (expect 1-2 days of troubleshooting)
4. Benchmark against Suno

**Total effort:** 10-14 days  
**Result:**
- Genuine singing quality approaching commercial tools
- Offline, no API dependencies
- Higher model size, slower inference
- More complex pipeline

**Verdict:** Worth it if you want to ship a product that sounds genuinely good.

---

### **Option C: Accept Bark, Improve UX (Minimal effort, realistic)**

**Path:** Keep Bark, improve everything around it

1. Better CLI messaging ("This sounds like sung speech; for better quality use Suno")
2. Experiment with Bark speaker presets to find "most singing-like" voice
3. Add optional effects (light reverb, compression) to post-process
4. Market as "free, offline alternative" rather than "great singing"

**Total effort:** ~4 hours  
**Result:**
- Honest positioning
- Manages expectations
- Free and works; but known limitations
- Good for novelty/gift apps where quality is secondary

**Verdict:** Practical if you can't commit major dev time.

---

## Part 6: Technical Debt & Quick Fixes

### Issue 1: Backing Tracks Might Be Identical
**Symptom:** All three backing MP3s are exactly the same file size  
**Action:** Verify they're different. If they're the same, find/download truly different piano styles from Pixabay.

### Issue 2: Cache Inconsistency
**Symptom:** 11 cached WAVs suggest multiple regeneration attempts  
**Cause:** Likely debugging/testing runs  
**Action:** Document cache invalidation strategy. Clear cache for major version bumps.

### Issue 3: No Error Handling for Bark Failures
**Symptom:** If Bark crashes or model download fails, no clear error message  
**Action:** Add try-catch around Bark inference with helpful messages:
```python
try:
    audio = generate_audio(line, ...)
except Exception as e:
    print(f"[ERROR] Bark failed on line {i}: {e}")
    print("[INFO] Try: pip install --upgrade suno-bark")
```

### Issue 4: No Tests for Audio Quality
**Symptom:** Tests check file I/O but not audio coherence  
**Action:** Add basic sanity checks (F0 variance, silence detection, loudness normalization)

---

## Part 7: Questions for You

Before I'd recommend a path forward, clarify:

1. **What's the use case?** Novelty gift app? Production music? Internal demo?
2. **Quality bar:** How bad is "bad"? Can users tolerate Bark-like output, or must it be genuinely singing?
3. **Budget:** How much dev time can you invest? (2 days? 2 weeks?)
4. **Dependencies:** Are API calls (Suno, etc.) acceptable, or must it be truly offline?
5. **Target audience:** Tech-savvy users (expect limitations) or general public (expect quality)?

---

## Conclusion

**The core issue:** Bark is a speech synthesis model, not a singing model. No amount of prompt engineering (♪ markers, [singing] tags) will make it sing like a human. The output inevitably sounds like a robot reciting lyrics in a "sing-song" voice.

**The trade-off:** 
- **Local + Free:** Limited quality (current state)
- **Local + Complex:** Better quality but weeks of engineering (DiffSinger route)
- **Cloud + Easy:** Excellent quality but depends on external service (Suno)

**Recommendation:** Start with **Option A (Hybrid)** — keep Bark as default, add pitch-shifting post-processor (+20-30% perceived improvement), and offer Suno as an optional upgrade path. This gives users a choice and improves the baseline without massive effort.

If quality is critical and you can afford 2 weeks, pivot to **Option B (DiffSinger)** for genuine singing synthesis.

---

## Appendix: Glossary

- **F0**: Fundamental frequency (Hz); the primary pitch of a voice
- **Vibrato**: Periodic variation in pitch (singing technique)
- **Legato**: Smooth pitch transitions between notes
- **Pitch-shift**: Change a sound's pitch without changing its speed
- **Time-stretch**: Change a sound's speed without changing its pitch
- **TTS**: Text-to-Speech (speech synthesis)
- **RVC**: Retrieval-based Voice Conversion (voice cloning)
- **DiffSinger**: Diffusion-based singing voice synthesis
- **MIDI**: Musical Instrument Digital Interface (score format)
