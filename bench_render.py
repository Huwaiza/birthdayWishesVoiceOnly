#!/usr/bin/env python3
"""
Measure the compute/quality trade-off across render settings.

Why this exists
---------------
The pipeline's cost is dominated by one number — `infer_step` — and nobody had
measured what lowering it actually costs in quality. "Fewer steps is probably
fine" and "60 steps is probably needed" are both guesses, and picking between
them by ear across a 2.5-minute song is unreliable. This sweeps the settings,
times each render, and scores the result with the same QC gate production uses,
so the trade-off is a table instead of an opinion.

What it measures
----------------
- **wall time per render** — the thing that heats the laptop. Compute scales
  ~linearly with infer_step, so this is the lever with the most headroom.
- **peak RSS** — sampled on a background thread, to confirm a cheaper config
  isn't just trading time for memory.
- **QC score + pass** — the quality proxy. `song.qc` transcribes the vocal and
  scores sung-seconds, "happy birthday" count, fuzzy name match and section
  coverage. It is not a substitute for listening, but it reliably catches the
  failure mode that matters: a cheaper config that stops singing the words.

Method notes
------------
Each (config, name) pair is exactly ONE render: retries are disabled so a
config can't hide a bad first attempt behind a second roll, and the cache is
bypassed so every number is a real render. Seeds are fixed per name, so across
configs you are comparing the same song rendered different ways, not different
songs.

Usage
-----
    # The main question: how few steps can we get away with?
    python bench_render.py --names Ayesha,Bilal --steps 60,45,30,20

    # Does a shorter song buy proportional time?
    python bench_render.py --names Ayesha --steps 60 --durations 150,120,90

    python bench_render.py --names Ayesha --steps 60,30 --json bench.json

Budget the time before starting: a 60-step/150s render is ~8 min, and the run
is (configs x names) renders. The 4-step x 2-name sweep above is ~40-50 min.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass
class Result:
    """One render at one config."""
    name: str
    infer_step: int
    duration_s: float
    seconds: float
    peak_rss_gb: Optional[float]
    qc_passed: Optional[bool]
    qc_score: Optional[float]
    qc_summary: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class _RSSSampler:
    """Sample process RSS on a background thread while a render runs.

    Peak matters more than the value at any instant: the render allocates and
    frees repeatedly, and it is the high-water mark that decides whether the
    machine starts swapping.
    """

    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            return self          # psutil absent → peak stays None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, proc.memory_info().rss / (1024 ** 3))
            except Exception:  # noqa: BLE001 — sampling must never kill a bench
                return
            self._stop.wait(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return False

    @property
    def peak_gb(self) -> Optional[float]:
        return self._peak or None


def _run_one(name: str, infer_step: int, duration_s: float,
             verbose: bool) -> Result:
    """Render one song at one config and score it. Never raises — a config that
    blows up is a data point, not a reason to lose the rest of the sweep."""
    from song.memory import free_render_memory
    from song.pipeline import generate_vocal_song

    print(f"\n  ▶ {name} @ {infer_step} steps / {duration_s:.0f}s ...", flush=True)
    t0 = time.time()
    try:
        with _RSSSampler() as sampler:
            _tight, report = generate_vocal_song(
                name,
                duration_s=duration_s,
                infer_step=infer_step,
                use_cache=False,      # every number must be a real render
                verbose=verbose,
                qc_enabled=True,
                qc_max_retries=0,     # exactly one render per data point
            )
        dt = time.time() - t0
        res = Result(
            name=name, infer_step=infer_step, duration_s=duration_s,
            seconds=dt, peak_rss_gb=sampler.peak_gb,
            qc_passed=(report.passed if report else None),
            qc_score=(report.score if report else None),
            qc_summary=(str(report) if report else ""),
        )
        flag = "✓" if res.qc_passed else "✗"
        print(f"    {flag} {dt / 60:.1f} min  score={res.qc_score:.2f}"
              f"  peak={res.peak_rss_gb or float('nan'):.1f}GB", flush=True)
    except Exception as e:  # noqa: BLE001 — record and carry on
        dt = time.time() - t0
        res = Result(name=name, infer_step=infer_step, duration_s=duration_s,
                     seconds=dt, peak_rss_gb=None, qc_passed=None,
                     qc_score=None, error=str(e))
        print(f"    ! failed after {dt:.0f}s: {e}", flush=True)

    free_render_memory()
    return res


def _summarise(results: List[Result], songs_per_day: int) -> str:
    """Group by config and render the comparison table, cheapest config last so
    the eye lands on the trade-off."""
    configs = {}
    for r in results:
        configs.setdefault((r.infer_step, r.duration_s), []).append(r)

    # Baseline = the most expensive config actually run, so speedups are honest
    # even when the sweep doesn't include the production default.
    baseline_key = max(configs, key=lambda k: (k[0], k[1]))
    base_times = [r.seconds for r in configs[baseline_key] if r.ok]
    base_mean = statistics.mean(base_times) if base_times else None

    lines = [
        "",
        "=" * 78,
        "  RESULTS",
        "=" * 78,
        f"  {'steps':>5} {'dur':>6} {'n':>3} {'mean min':>9} {'vs base':>8} "
        f"{'QC pass':>8} {'score':>6} {'peak GB':>8}",
        "  " + "-" * 74,
    ]
    for key in sorted(configs, key=lambda k: (-k[0], -k[1])):
        rs = [r for r in configs[key] if r.ok]
        if not rs:
            lines.append(f"  {key[0]:>5} {key[1]:>6.0f}  — all renders failed")
            continue
        mean_s = statistics.mean(r.seconds for r in rs)
        passes = sum(1 for r in rs if r.qc_passed)
        scores = [r.qc_score for r in rs if r.qc_score is not None]
        peaks = [r.peak_rss_gb for r in rs if r.peak_rss_gb]
        speed = f"{base_mean / mean_s:.2f}x" if base_mean else "—"
        lines.append(
            f"  {key[0]:>5} {key[1]:>6.0f} {len(rs):>3} {mean_s / 60:>9.1f} "
            f"{speed:>8} {f'{passes}/{len(rs)}':>8} "
            f"{statistics.mean(scores) if scores else 0:>6.2f} "
            f"{statistics.mean(peaks) if peaks else 0:>8.1f}"
        )

    lines += ["", "  Daily GPU load at %d songs/day:" % songs_per_day]
    for key in sorted(configs, key=lambda k: (-k[0], -k[1])):
        rs = [r for r in configs[key] if r.ok]
        if not rs:
            continue
        hours = statistics.mean(r.seconds for r in rs) * songs_per_day / 3600
        lines.append(f"    {key[0]:>3} steps / {key[1]:>4.0f}s → {hours:5.1f} h "
                     f"of sustained load")

    lines += [
        "",
        "  Read this as: pick the cheapest row whose QC pass rate and score",
        "  match the baseline, then confirm by listening before adopting it.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep render settings and measure time vs quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="Ayesha,Bilal",
                    help="Comma-separated names to render (default: Ayesha,Bilal). "
                         "More names = less noise, proportionally more time.")
    ap.add_argument("--steps", default="60,45,30,20",
                    help="Comma-separated infer_step values to compare "
                         "(default: 60,45,30,20). 60 is the production default.")
    ap.add_argument("--durations", default="150",
                    help="Comma-separated song lengths in seconds (default: 150)")
    ap.add_argument("--songs-per-day", type=int, default=20, dest="songs_per_day",
                    help="Used only to project daily sustained load (default: 20)")
    ap.add_argument("--json", dest="json_path",
                    help="Also write the raw per-render results here")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress the per-render pipeline chatter")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    durations = [float(d) for d in args.durations.split(",") if d.strip()]

    total = len(names) * len(steps) * len(durations)
    print("=" * 78)
    print("  RENDER BENCHMARK")
    print("=" * 78)
    print(f"  names     : {', '.join(names)}")
    print(f"  steps     : {steps}")
    print(f"  durations : {durations}")
    print(f"  renders   : {total}  (cache bypassed, 1 render each, no retries)")
    print(f"\n  At roughly 8 min for a 60-step/150s render, budget "
          f"~{total * 8} min worst case.")
    print("=" * 78)

    results: List[Result] = []
    t_start = time.time()
    try:
        for duration_s in durations:
            for infer_step in steps:
                for name in names:
                    results.append(
                        _run_one(name, infer_step, duration_s, not args.quiet))
    except KeyboardInterrupt:
        print("\n  Interrupted — summarising what completed so far.")

    if not results:
        print("  No renders completed.")
        return 1

    print(_summarise(results, args.songs_per_day))
    print(f"  Total bench time: {(time.time() - t_start) / 60:.0f} min")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([asdict(r) for r in results], indent=2))
        print(f"  Raw results → {args.json_path}")

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n  {len(failed)} render(s) failed:")
        for r in failed:
            print(f"    {r.name} @ {r.infer_step} steps: {r.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
