#!/usr/bin/env python3
"""
Birthday Song Generator  v7  —  ACE-Step (local, free)
======================================================
One name in → one ~2:30 personalised birthday song MP3 out.
ACE-Step generates vocals + backing in a single pass on Apple Silicon
(MPS auto-detected). 100% local, no API calls, free for unlimited renders.

First run downloads ACE-Step weights (~4 GB) into ~/.cache/ace-step/checkpoints.
Subsequent runs are cached.

Usage
-----
  python generate_birthday.py --name "Huwaiza"
  python generate_birthday.py --name "Sara" --voice 2
  python generate_birthday.py --name "Adam" --duration 165
  python generate_birthday.py --list-voices
"""

import argparse
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(mp3_path), format="mp3", bitrate="192k")


def main() -> None:
    script_t0 = time.time()
    started_at = datetime.now()
    ap = argparse.ArgumentParser(
        description="Generate a personalised birthday song locally with ACE-Step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        EXAMPLES
          python generate_birthday.py --name "Huwaiza"
          python generate_birthday.py --name "Sara" --voice 2
          python generate_birthday.py --name "Adam" --duration 165
          python generate_birthday.py --list-voices

        NOTES
          • First run downloads the ACE-Step model (~4 GB) into ~/.cache/ace-step.
          • Voices rotate deterministically by hash(name) across 4 profiles.
          •   --voice N pins a specific profile (0..3).
          • Renders are cached — same (name, voice, lyrics) → instant return.
          • Apple Silicon MPS is used automatically when available.
        """),
    )
    ap.add_argument("--name", default=None,
                    help="Name to sing (required unless --list-voices)")
    ap.add_argument("--voice", type=int, default=None, metavar="N",
                    help="Pin voice profile 0..3 (default: hash(name) rotation)")
    ap.add_argument("--duration", type=float, default=150.0,
                    help="Target duration in seconds (default 150 = 2:30)")
    ap.add_argument("--output", default=None,
                    help="Output MP3 path (default: happy_birthday_<name>.mp3)")
    ap.add_argument("--steps", type=int, default=60,
                    help="ACE-Step inference steps (default 60; lower = faster, lower quality)")
    ap.add_argument("--guidance", type=float, default=15.0,
                    help="Classifier-free guidance scale (default 15)")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16",
                    help="Model precision (default float16: ~7 GB RAM, fast). "
                         "float32 doubles RAM use to ~14 GB; on Macs with "
                         "<=18 GB that causes disk swapping and very slow renders.")
    ap.add_argument("--acapella", action="store_true",
                    help="Vocals only: render the full song, then strip every "
                         "instrument with a Roformer model "
                         "(needs `pip install \"audio-separator[cpu]\"`)")
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Force re-render even if cached")
    ap.add_argument("--list-voices", action="store_true", dest="list_voices",
                    help="List voice profiles and exit")
    args = ap.parse_args()

    # Defer heavy imports until after arg parsing so --list-voices and --help
    # don't pay the torch import tax.
    from song.voice_profiles import VOICE_PROFILES, pick_voice

    if args.list_voices:
        print("\nAvailable voice profiles:")
        for i, v in enumerate(VOICE_PROFILES):
            print(f"  [{i}] {v.name}")
            for line in textwrap.wrap(v.prompt, width=72,
                                      initial_indent="      ",
                                      subsequent_indent="      "):
                print(line)
            print()
        return

    if not args.name:
        ap.error("--name is required (or use --list-voices)")

    safe = args.name.lower().replace(" ", "_")
    suffix = "_acapella" if args.acapella else ""
    output_mp3 = Path(args.output or f"happy_birthday_{safe}{suffix}.mp3")
    if not output_mp3.is_absolute():
        output_mp3 = SCRIPT_DIR / output_mp3

    chosen = pick_voice(args.name, override_index=args.voice)

    print("\n" + "=" * 60)
    print("  Birthday Song Generator  v7  (ACE-Step, local)")
    print(f"  Name     : {args.name}")
    print(f"  Voice    : {chosen.name}"
          + (f"  (pinned)" if args.voice is not None else "  (auto-rotated)"))
    print(f"  Duration : {args.duration:.0f}s")
    print(f"  Precision: {args.precision}")
    print(f"  Mode     : {'a cappella — vocals only' if args.acapella else 'full song — vocals + backing'}")
    print(f"  Started  : {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Output   : {output_mp3}")
    print("=" * 60 + "\n")

    from song.acestep_render import render, format_duration
    wav_path = render(
        name=args.name,
        voice_index=args.voice,
        duration_s=args.duration,
        infer_step=args.steps,
        guidance_scale=args.guidance,
        use_cache=not args.no_cache,
        verbose=True,
        precision=args.precision,
    )

    print(f"\n[main] WAV ready: {wav_path}")

    if args.acapella:
        from song.vocal_isolate import isolate_vocals
        vocals_wav = wav_path.with_name(wav_path.stem + "__vocals.wav")
        if vocals_wav.exists() and not args.no_cache:
            print(f"[main] cached vocal stem: {vocals_wav.name}")
        else:
            print(f"[main] Isolating vocals — removing all instruments...")
            isolate_vocals(wav_path, vocals_wav, verbose=True)
        wav_path = vocals_wav

    print(f"[main] Encoding 192 kbps MP3 → {output_mp3}")
    _wav_to_mp3(wav_path, output_mp3)

    size_mb = output_mp3.stat().st_size / (1024 * 1024)
    finished_at = datetime.now()
    elapsed = time.time() - script_t0
    print("\n" + "=" * 60)
    print(f"  ✓  Done!  {size_mb:.1f} MB  →  {output_mp3}")
    print(f"  Started  : {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Finished : {finished_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Total    : {format_duration(elapsed)}  ({elapsed:.0f}s)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
