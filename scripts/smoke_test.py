#!/usr/bin/env python3
"""
ACE-Step smoke test.

Renders a 30-second sample to confirm:
  1. ACE-Step is importable.
  2. Weights download from Hugging Face on first run (~4 GB).
  3. MPS is detected on Apple Silicon and the pipeline produces a WAV.

Run from project root:
    source venv-diffsinger/bin/activate
    python scripts/smoke_test.py

First run takes 5-15 minutes (weight download + warm-up + render).
Subsequent runs take ~30-90 seconds.
"""

import os
import sys
import time
from pathlib import Path

# Run the model in float16 (~7 GB) instead of ACE-Step's default float32
# (~14 GB) on Apple Silicon. float32 overflows RAM on Macs with <=18 GB and
# forces disk swapping — the cause of multi-hour renders. Must be set before
# the acestep import below.
os.environ.setdefault("ACE_PIPELINE_DTYPE", "float16")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "scripts" / "smoke_test_out.wav"


def main():
    print("=" * 60)
    print("  ACE-Step smoke test")
    print("=" * 60)

    print("\n[1/4] Checking torch + MPS...")
    import torch
    print(f"      torch       = {torch.__version__}")
    print(f"      mps avail.  = {torch.backends.mps.is_available()}")
    print(f"      mps built   = {torch.backends.mps.is_built()}")
    if not torch.backends.mps.is_available():
        print("      WARNING: MPS not available — render will run on CPU (very slow)")

    print("\n[2/4] Importing ACE-Step pipeline...")
    t0 = time.time()
    from acestep.pipeline_ace_step import ACEStepPipeline
    print(f"      imported in {time.time()-t0:.1f}s")

    print("\n[3/4] Loading model (downloads ~4 GB on first run)...")
    t0 = time.time()
    pipeline = ACEStepPipeline(
        checkpoint_dir=None,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
        overlapped_decode=False,
    )
    print(f"      pipeline constructed in {time.time()-t0:.1f}s "
          f"(weights load lazily on first call)")
    print(f"      device={pipeline.device}  dtype={pipeline.dtype}")
    if str(pipeline.dtype) != "torch.float16":
        print("      WARNING: expected float16 — float32 will be slow on <=18 GB Macs")

    print("\n[4/4] Rendering 30s sample...")
    print("      First call also lazy-loads weights — this can take several minutes.")
    t0 = time.time()
    pipeline(
        format="wav",
        audio_duration=30.0,
        prompt=("female vocalist, warm alto voice, soulful, "
                "happy birthday song, warm piano ballad, 90 bpm, major key, joyful"),
        lyrics=("[verse]\n"
                "Happy birthday to you\n"
                "Happy birthday to you\n"
                "Happy birthday dear friend\n"
                "Happy birthday to you\n"),
        infer_step=30,                # short test → fewer steps
        guidance_scale=15.0,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds="42",
        save_path=str(OUTPUT),
        batch_size=1,
    )
    dt = time.time() - t0

    if not OUTPUT.exists():
        print(f"\n✗ FAIL: pipeline returned but {OUTPUT} not written")
        sys.exit(1)

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n✓  Render succeeded in {dt:.0f}s ({dt/60:.1f} min)")
    print(f"   Output: {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"\nListen to it:  open {OUTPUT}")


if __name__ == "__main__":
    main()
