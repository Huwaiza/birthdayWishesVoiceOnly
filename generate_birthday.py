#!/usr/bin/env python3
"""
Birthday Song Generator  v7  —  ACE-Step (local, free)
======================================================
One name in → one personalised birthday song MP3 out. By DEFAULT the song
is vocals only (a cappella): ACE-Step renders the full mix, a Roformer model
strips every instrument, and the long silent gaps are tightened. Pass
--with-music to keep the backing band. ACE-Step runs on Apple Silicon (MPS
auto-detected). 100% local, no API calls, free for unlimited renders.

First run downloads ACE-Step weights (~4 GB) into ~/.cache/ace-step/checkpoints.
Subsequent runs are cached.

Usage
-----
  python generate_birthday.py --name "Huwaiza"
  python generate_birthday.py --name "Sara" --voice 2
  python generate_birthday.py --name "Adam" --with-music
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
          python generate_birthday.py --name "Adam" --with-music
          python generate_birthday.py --list-voices

        NOTES
          • DEFAULT is vocals only (a cappella); pass --with-music for the band.
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
    ap.add_argument("--guidance-text", type=float, default=0.0, dest="guidance_text",
                    help="ACE-Step double-condition guidance, text scale "
                         "(default 0 = off). Enable BOTH this and "
                         "--guidance-lyric > 1 for stronger lyric adherence; "
                         "upstream suggests 5.0 / 1.5. Adds ~33%% diffusion time.")
    ap.add_argument("--guidance-lyric", type=float, default=0.0, dest="guidance_lyric",
                    help="ACE-Step double-condition guidance, lyric scale "
                         "(default 0 = off). See --guidance-text.")
    ap.add_argument("--no-qc", action="store_true", dest="no_qc",
                    help="Skip the post-render QC gate (sung-time + Whisper "
                         "transcript checks) and its seed-retry. QC only runs "
                         "in vocals-only mode.")
    ap.add_argument("--qc-retries", type=int, default=1, dest="qc_retries",
                    help="Max re-renders (with a new seed) when QC fails "
                         "(default 1)")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16",
                    help="Model precision (default float16: ~7 GB RAM, fast). "
                         "float32 doubles RAM use to ~14 GB; on Macs with "
                         "<=18 GB that causes disk swapping and very slow renders.")
    ap.add_argument("--with-music", action="store_true", dest="with_music",
                    help="Keep the backing band. The DEFAULT is vocals only: "
                         "render the full song, strip every instrument with a "
                         "Roformer model, then tighten the long silent gaps "
                         "(vocals-only needs `pip install \"audio-separator[cpu]\"`)")
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
    suffix = "_with_music" if args.with_music else ""
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
    print(f"  Mode     : {'full song — vocals + backing' if args.with_music else 'a cappella — vocals only (default)'}")
    print(f"  Started  : {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Output   : {output_mp3}")
    print("=" * 60 + "\n")

    from song.acestep_render import format_duration
    qc_report = None
    if args.with_music:
        from song.acestep_render import render
        wav_path = render(
            name=args.name,
            voice_index=args.voice,
            duration_s=args.duration,
            infer_step=args.steps,
            guidance_scale=args.guidance,
            guidance_scale_text=args.guidance_text,
            guidance_scale_lyric=args.guidance_lyric,
            use_cache=not args.no_cache,
            verbose=True,
            precision=args.precision,
        )
        print(f"\n[main] WAV ready: {wav_path}")
    else:
        from song.pipeline import generate_vocal_song
        wav_path, qc_report = generate_vocal_song(
            args.name,
            voice_index=args.voice,
            duration_s=args.duration,
            infer_step=args.steps,
            guidance_scale=args.guidance,
            guidance_scale_text=args.guidance_text,
            guidance_scale_lyric=args.guidance_lyric,
            use_cache=not args.no_cache,
            verbose=True,
            precision=args.precision,
            qc_enabled=not args.no_qc,
            qc_max_retries=args.qc_retries,
        )
        print(f"\n[main] tightened vocal ready: {wav_path}")

    print(f"[main] Encoding 192 kbps MP3 → {output_mp3}")
    _wav_to_mp3(wav_path, output_mp3)

    size_mb = output_mp3.stat().st_size / (1024 * 1024)
    finished_at = datetime.now()
    elapsed = time.time() - script_t0
    print("\n" + "=" * 60)
    print(f"  ✓  Done!  {size_mb:.1f} MB  →  {output_mp3}")
    if qc_report is not None:
        print(f"  QC       : {qc_report.summary()}")
    print(f"  Started  : {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Finished : {finished_at:%Y-%m-%d %H:%M:%S}")
    print(f"  Total    : {format_duration(elapsed)}  ({elapsed:.0f}s)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
