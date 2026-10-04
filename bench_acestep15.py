#!/usr/bin/env python3
"""
Time ACE-Step 1.5 (2B turbo, MLX) on the real birthday workload.

Runs under venv-acestep15 (Python 3.12) — NOT venv-diffsinger. The two stacks
share no packages and cannot share an interpreter: ACE-Step 1.5 requires
Python >=3.11,<3.13 while the production env is 3.10.

The point is a fair comparison, so this uses the *production* prompt and lyrics
(copied in below rather than imported, because song/ lives in the other venv)
and the same 150s duration. What it reports:

  - model load time      (one-off per process; production amortises it)
  - generation time      (the number that decides fan noise and throughput)
  - peak RSS             (decides whether 18 GB of RAM copes)

It writes a real audio file. **Listen to it.** Timing parity is meaningless if
the singing is worse, and no automated score substitutes for hearing whether it
says "happy birthday <name>" convincingly.

Usage:
    ACESTEP_CHECKPOINTS_DIR=~/.cache/ace-step-1.5/checkpoints \
      venv-acestep15/bin/python bench_acestep15.py --name Ayesha --steps 8
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Production voice profile "bright-soprano" (song/voice_profiles.py).
# NOTE the instrument terms: production deliberately generates a full mix and
# then strips the backing with Roformer (song/vocal_isolate.py). The model was
# never asked for vocals only.
PROMPT = (
    "female vocalist, bright soprano voice, clear high notes, youthful, "
    "crisp articulation, cheerful pop delivery, happy birthday song, "
    "warm piano ballad, light strings, studio recording, clean mix, "
    "90 bpm, major key, joyful, professional production, loudness normalised"
)

# A-cappella variant: same voice, instruments removed from the prompt and
# actively negated. If the model honours this, the whole Roformer isolation
# stage can be dropped — it costs minutes per song and is only there to undo
# backing the model was asked to produce.
# Sparse-backing variant. Separation damage scales with how much the backing
# overlaps the voice spectrally — dense pads and strings sit right on top of it,
# a single quiet piano does not. Same voice as production, minimal accompaniment,
# so Roformer has less to remove and takes less of the vocal with it.
PROMPT_SPARSE = (
    "solo female vocalist, bright soprano voice, clear high notes, youthful, "
    "crisp articulation, cheerful delivery, happy birthday song, "
    "very sparse minimal accompaniment, single quiet soft piano only, "
    "no drums, no bass, no strings, no synth, no pads, "
    "vocal forward in the mix, dry close-mic studio vocal, "
    "90 bpm, major key, joyful"
)

PROMPT_ACAPELLA = (
    "solo female vocalist, a cappella, unaccompanied singing, voice only, "
    "no instruments, no music, no backing track, no piano, no drums, "
    "bright soprano voice, clear high notes, youthful, crisp articulation, "
    "cheerful delivery, happy birthday song, dry studio vocal recording, "
    "close mic, clean isolated vocal, 90 bpm, major key, joyful"
)

LYRICS_TEMPLATE = """[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
Happy birthday to you

[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
May your dreams come true

[bridge]
May your day be filled with joy
Laughter, love, and happiness
We're so glad you're in our lives
Happy birthday {name}

[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
Happy birthday to you

[chorus]
Hip hip hooray it's your special day
We sing for you {name} in every way
May all your wishes come true today
Happy birthday, happy birthday to you
"""


class PeakRSS:
    """Sample RSS on a thread; the high-water mark is what decides swapping."""

    def __init__(self, interval: float = 1.0):
        self.interval, self.peak = interval, 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        import psutil
        p = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, p.memory_info().rss / 1024**3)
            except Exception:  # noqa: BLE001
                return
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ayesha")
    ap.add_argument("--steps", type=int, default=8,
                    help="Diffusion steps. Turbo is distilled for ~8; the old "
                         "v1 pipeline used 60.")
    ap.add_argument("--duration", type=float, default=150.0,
                    help="Seconds, matching the production default")
    ap.add_argument("--save-dir", default="output_v15")
    ap.add_argument("--seed", type=int, default=20260202,
                    help="Production base_seed for the bright-soprano profile")
    # Memory-pressure fallbacks. The first attempts on an 18 GB Mac were
    # SIGKILLed (exit 137) during model load, so these exist to trade speed for
    # a smaller peak footprint when the default config won't fit.
    ap.add_argument("--offload", action="store_true",
                    help="Offload non-active components to CPU RAM. Slower, "
                         "but lowers peak unified-memory usage.")
    ap.add_argument("--offload-dit", action="store_true",
                    help="Also offload the DiT itself — last resort, slowest.")
    ap.add_argument("--quantization", default=None,
                    choices=["int8_weight_only", "fp8_weight_only", "w8a8_dynamic"],
                    help="Quantize DiT weights to shrink the resident model.")
    ap.add_argument("--acapella", action="store_true",
                    help="Ask the model for unaccompanied vocals instead of a "
                         "full mix. If it obeys, the Roformer isolation stage "
                         "becomes unnecessary.")
    ap.add_argument("--sparse", action="store_true",
                    help="Ask for minimal accompaniment so the later separation "
                         "step has less to strip and damages the vocal less.")
    ap.add_argument("--tag", default="", help="Suffix for the output filename")
    args = ap.parse_args()

    ck = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if not ck:
        print("ERROR: set ACESTEP_CHECKPOINTS_DIR to the 1.5 checkpoints dir.")
        return 1
    ck = os.path.expanduser(ck)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ACE-Step 1.5 (2B turbo, MLX) — birthday workload")
    print("=" * 70)
    print(f"  checkpoints : {ck}")
    print(f"  name        : {args.name}")
    print(f"  steps       : {args.steps}   (v1 production used 60)")
    print(f"  duration    : {args.duration:.0f}s")
    print("=" * 70, flush=True)

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    project_root = str(Path(__file__).resolve().parent / "external" / "ACE-Step-1.5")

    with PeakRSS() as rss:
        t0 = time.time()
        # use_mlx_dit=True is the default and the whole point — it runs the
        # diffusion transformer through MLX instead of PyTorch MPS.
        dit_handler = AceStepHandler()
        dit_handler.initialize_service(
            project_root=project_root,
            config_path="acestep-v15-turbo",
            device="mps",
            use_mlx_dit=True,
            offload_to_cpu=args.offload,
            offload_dit_to_cpu=args.offload_dit,
            quantization=args.quantization,
        )
        llm_handler = LLMHandler()
        llm_handler.initialize(
            checkpoint_dir=ck,
            lm_model_path="acestep-5Hz-lm-0.6B",
            backend="mlx",
            device="mps",
        )
        t_load = time.time() - t0
        print(f"\n  model load: {t_load:.0f}s ({t_load/60:.1f} min)", flush=True)

        params = GenerationParams(
            caption=(PROMPT_ACAPELLA if args.acapella
                     else PROMPT_SPARSE if args.sparse
                     else PROMPT),
            lyrics=LYRICS_TEMPLATE.format(name=args.name),
            duration=args.duration,
            inference_steps=args.steps,
            seed=args.seed,
            vocal_language="en",
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False,
                                  seeds=[args.seed], audio_format="mp3")

        t1 = time.time()
        result = generate_music(dit_handler, llm_handler, params, config,
                                save_dir=args.save_dir)
        t_gen = time.time() - t1

    print("\n" + "=" * 70)
    print(f"  model load     : {t_load:6.0f}s  ({t_load/60:.1f} min)")
    print(f"  GENERATION     : {t_gen:6.0f}s  ({t_gen/60:.1f} min)   <-- compare to v1")
    print(f"  peak RSS       : {rss.peak:6.2f} GB")
    print(f"  output         : {getattr(result, 'audio_paths', args.save_dir)}")
    print("=" * 70)
    print("  Now LISTEN to the output. Speed without quality parity is not a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
