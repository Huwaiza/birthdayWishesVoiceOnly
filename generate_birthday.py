#!/usr/bin/env python3
"""
Birthday Song Generator  v6  —  Local Bark + Piano Backing
===========================================================
Generates a personalised ~2.5-minute birthday song entirely offline.
No external API. No GPU required (runs on Apple Silicon MPS or CPU).

One-time model download: ~5 GB (cached by HuggingFace hub).

Usage
-----
  python generate_birthday.py --name "Huwaiza"
  python generate_birthday.py --name "Armeen"   --style upbeat
  python generate_birthday.py --name "Usama"    --style warm
  python generate_birthday.py --name "Huwaiza"  --speaker v2/en_speaker_5
  python generate_birthday.py --name "Sara"     --regen-line 2
  python generate_birthday.py --list-speakers
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

SCRIPT_DIR   = Path(__file__).parent.resolve()
ASSETS_DIR   = SCRIPT_DIR / "assets" / "backing"
SAMPLE_RATE  = 24000  # Bark native rate

STYLES = {
    "simple":  ASSETS_DIR / "simple.mp3",
    "warm":    ASSETS_DIR / "warm.mp3",
    "upbeat":  ASSETS_DIR / "upbeat.mp3",
}

SPEAKERS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
    "v2/en_speaker_9",  # default — warm female
]


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(mp3_path), format="mp3", bitrate="192k")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a personalised birthday song locally using Bark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        EXAMPLES
          python generate_birthday.py --name "Huwaiza"
          python generate_birthday.py --name "Armeen"  --style upbeat
          python generate_birthday.py --name "Usama"   --style warm
          python generate_birthday.py --name "Sara"    --speaker v2/en_speaker_5
          python generate_birthday.py --name "Huwaiza" --regen-line 2
          python generate_birthday.py --list-speakers

        NOTES
          • First run downloads the Bark model (~5 GB). Subsequent runs use cache.
          • Each lyric line is cached in cache/ — use --regen-line N to redo one line.
          • Target duration: 2:20–2:45 (suitable for YouTube monetisation).
          • Apple Silicon MPS is used automatically when available.
        """),
    )
    ap.add_argument("--name",          default=None,  help="Name to sing  (required)")
    ap.add_argument("--style",         default="simple", choices=list(STYLES.keys()),
                    help="Backing track style: simple | warm | upbeat  (default: simple)")
    ap.add_argument("--speaker",       default="v2/en_speaker_9",
                    help="Bark speaker preset  (default: v2/en_speaker_9)")
    ap.add_argument("--output",        default=None,
                    help="Output MP3 filename  (default: happy_birthday_<name>.mp3)")
    ap.add_argument("--regen-line",    type=int, default=None, dest="regen_line",
                    metavar="N", help="Regenerate only line N (0-indexed), ignoring cache")
    ap.add_argument("--list-speakers", action="store_true", dest="list_speakers",
                    help="Print available speaker presets and exit")
    args = ap.parse_args()

    if args.list_speakers:
        print("\nAvailable speaker presets:")
        for s in SPEAKERS:
            marker = "  ← default" if s == "v2/en_speaker_9" else ""
            print(f"  {s}{marker}")
        print()
        return

    if not args.name:
        ap.error("--name is required")

    output_mp3 = Path(args.output or f"happy_birthday_{args.name.lower().replace(' ', '_')}.mp3")
    if not output_mp3.is_absolute():
        output_mp3 = SCRIPT_DIR / output_mp3

    backing_path = STYLES[args.style]
    if not backing_path.exists():
        sys.exit(
            f"[ERROR] Backing track not found: {backing_path}\n"
            f"  Run Task 2 of the implementation plan to download backing tracks."
        )

    print("\n" + "=" * 60)
    print("  Birthday Song Generator  v6  (local Bark)")
    print(f"  Name    : {args.name}")
    print(f"  Style   : {args.style}")
    print(f"  Speaker : {args.speaker}")
    print(f"  Output  : {output_mp3}")
    print("=" * 60 + "\n")

    # --- 1. Build lyrics ---
    from song.lyrics import build_lines, SECTION_BREAKS
    lines = build_lines(args.name)
    print(f"[main] {len(lines)} lyric lines built.\n")

    # --- 2. Generate vocals ---
    from song.vocal import generate_all_lines
    clips = generate_all_lines(lines, speaker=args.speaker, regen_index=args.regen_line)

    # --- 3. Stitch vocals ---
    from song.mix import stitch_clips, mix_backing, normalise_duration
    vocals = stitch_clips(clips, SECTION_BREAKS, sample_rate=SAMPLE_RATE)
    duration_s = len(vocals) / SAMPLE_RATE
    print(f"\n[main] Vocals stitched: {duration_s:.1f}s")

    # --- 4. Load and loop backing track ---
    from song.backing import load_and_loop
    backing = load_and_loop(str(backing_path), target_duration_s=duration_s, sample_rate=SAMPLE_RATE)

    # --- 5. Mix ---
    mixed = mix_backing(vocals, backing, backing_volume=0.18)

    # --- 6. Normalise duration to 2:20–2:45 ---
    mixed = normalise_duration(mixed, min_s=140.0, max_s=165.0, sample_rate=SAMPLE_RATE)
    final_duration = len(mixed) / SAMPLE_RATE
    print(f"[main] Final duration: {final_duration:.1f}s ({final_duration/60:.1f} min)")

    # --- 7. Export WAV then convert to MP3 ---
    tmp_wav = output_mp3.with_suffix(".wav")
    sf.write(str(tmp_wav), mixed, SAMPLE_RATE)
    _wav_to_mp3(tmp_wav, output_mp3)
    tmp_wav.unlink()

    size_mb = output_mp3.stat().st_size / (1024 * 1024)
    print(f"\n✓  Done!  {size_mb:.1f} MB  →  {output_mp3}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
