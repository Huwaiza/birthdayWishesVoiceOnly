#!/usr/bin/env python3
"""
batch_render.py — render birthday songs for a list of names.

Reads a names file (one name per line, blank lines and #-comments ignored),
renders each via ACE-Step, encodes to 192 kbps MP3, and writes to an output
directory. Songs are vocals-only (a cappella) by DEFAULT; pass --with-music
for the full backing band. Re-runs are nearly free thanks to the WAV cache.

Usage
-----
  python batch_render.py --names names.txt --out output/
  python batch_render.py --names names.txt --out output/ --with-music
  python batch_render.py --names names.txt --out output/ --retries 2

The model loads once and is reused across the whole batch — overhead is
amortised over N names.
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List


def _read_names(names_file: Path) -> List[str]:
    if not names_file.exists():
        sys.exit(f"[ERROR] names file not found: {names_file}")
    out = []
    for raw in names_file.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    if not out:
        sys.exit(f"[ERROR] no names in {names_file}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-render birthday songs")
    ap.add_argument("--names", required=True, type=Path,
                    help="Path to names.txt (one name per line)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for MP3s")
    ap.add_argument("--duration", type=float, default=150.0,
                    help="Target duration per song in seconds (default 150)")
    ap.add_argument("--steps", type=int, default=60,
                    help="ACE-Step inference steps (default 60)")
    ap.add_argument("--guidance", type=float, default=15.0,
                    help="Classifier-free guidance (default 15)")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16",
                    help="Model precision (default float16: ~7 GB RAM, fast). "
                         "float32 doubles RAM use; on Macs with <=18 GB that "
                         "causes disk swapping and very slow renders.")
    ap.add_argument("--with-music", action="store_true", dest="with_music",
                    help="Keep the backing band. The DEFAULT is vocals only: "
                         "strip every instrument with a Roformer model after "
                         "each render, then tighten the long silent gaps "
                         "(vocals-only needs `pip install \"audio-separator[cpu]\"`)")
    ap.add_argument("--retries", type=int, default=1,
                    help="Per-name retry count on failure (default 1)")
    ap.add_argument("--voice", type=int, default=None,
                    help="Pin all renders to a specific voice index (default: rotate)")
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Force re-render every name")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first N names (resume support)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N names")
    args = ap.parse_args()

    names = _read_names(args.names)
    if args.start:
        names = names[args.start:]
    if args.limit:
        names = names[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)

    batch_started = datetime.now()
    print("\n" + "=" * 60)
    print("  Batch Birthday Song Render")
    print(f"  Names    : {len(names)} (file: {args.names})")
    print(f"  Output   : {args.out}")
    print(f"  Duration : {args.duration:.0f}s   Steps: {args.steps}   CFG: {args.guidance}")
    print(f"  Precision: {args.precision}")
    print(f"  Mode     : {'full song — vocals + backing' if args.with_music else 'a cappella — vocals only (default)'}")
    print(f"  Voice    : {'index ' + str(args.voice) + ' (pinned)' if args.voice is not None else 'auto-rotated'}")
    print(f"  Started  : {batch_started:%Y-%m-%d %H:%M:%S}")
    print("=" * 60 + "\n")

    # Heavy imports deferred so --help is snappy.
    from song.acestep_render import render, format_duration
    from song.voice_profiles import pick_voice
    from pydub import AudioSegment
    if not args.with_music:
        from song.vocal_isolate import isolate_vocals, tighten_pauses

    t_batch = time.time()
    succeeded, failed, cached = [], [], []

    for i, name in enumerate(names, start=1):
        safe = name.lower().replace(" ", "_").replace("/", "_")
        mp3_suffix = "_with_music" if args.with_music else ""
        mp3_path = args.out / f"happy_birthday_{safe}{mp3_suffix}.mp3"
        chosen = pick_voice(name, override_index=args.voice)

        print(f"\n[{i}/{len(names)}] {name}  →  voice={chosen.name}  →  {mp3_path.name}")
        if mp3_path.exists() and not args.no_cache:
            print(f"  ⤷ MP3 already exists, skipping")
            cached.append(name)
            continue

        last_err = None
        for attempt in range(1, args.retries + 2):  # retries=1 → 2 total attempts
            try:
                t0 = time.time()
                wav_path = render(
                    name=name,
                    voice_index=args.voice,
                    duration_s=args.duration,
                    infer_step=args.steps,
                    guidance_scale=args.guidance,
                    use_cache=not args.no_cache,
                    verbose=False,
                    precision=args.precision,
                )
                # Validate the WAV is non-empty
                if wav_path.stat().st_size < 50_000:
                    raise RuntimeError(f"WAV looks too small: {wav_path.stat().st_size} bytes")

                if not args.with_music:
                    vocals_wav = wav_path.with_name(wav_path.stem + "__vocals.wav")
                    if not (vocals_wav.exists() and not args.no_cache):
                        isolate_vocals(wav_path, vocals_wav, verbose=False)
                    tight_wav = wav_path.with_name(wav_path.stem + "__vocal_tight.wav")
                    if not (tight_wav.exists() and not args.no_cache):
                        tighten_pauses(vocals_wav, tight_wav, verbose=False)
                    wav_path = tight_wav

                AudioSegment.from_wav(str(wav_path)).export(
                    str(mp3_path), format="mp3", bitrate="192k"
                )
                dt = time.time() - t0
                print(f"  ✓ rendered in {format_duration(dt)} ({dt:.0f}s)  "
                      f"[finished {datetime.now():%H:%M:%S}]")
                succeeded.append(name)
                break
            except Exception as e:  # noqa: BLE001 — top-level batch loop
                last_err = e
                print(f"  ✗ attempt {attempt} failed: {e}")
                if attempt > args.retries:
                    print(traceback.format_exc())
        else:
            failed.append((name, str(last_err)))

    total = time.time() - t_batch
    batch_finished = datetime.now()
    print("\n" + "=" * 60)
    print(f"  Batch complete")
    print(f"  Started  : {batch_started:%Y-%m-%d %H:%M:%S}")
    print(f"  Finished : {batch_finished:%Y-%m-%d %H:%M:%S}")
    print(f"  Total    : {format_duration(total)}  ({total:.0f}s)")
    print(f"  Rendered : {len(succeeded)}")
    print(f"  Cached   : {len(cached)}")
    print(f"  Failed   : {len(failed)}")
    if failed:
        print("\n  Failures:")
        for n, err in failed:
            print(f"    - {n}: {err}")
    print("=" * 60 + "\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
