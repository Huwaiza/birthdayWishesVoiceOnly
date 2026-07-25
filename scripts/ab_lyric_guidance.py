#!/usr/bin/env python3
"""
A/B the lyric-adherence guidance through the running HTTP service.

For each name, submits two jobs against a running service instance:

  arm A (baseline): guidance off  → output/ab_<name>_guidance_off.mp3
  arm B (guided):   text=5.0, lyric=1.5 → output/ab_<name>_guidance_on.mp3

Each arm gets its own slug because the service's output path (and its
"MP3 already on disk → done" idempotency) is keyed by name/slug — without
distinct slugs the second arm would just return the first arm's file.

Jobs are submitted up front (the service queues and renders them one at a
time), then polled until both arms of every name are terminal. Finish by
listening to the pairs side by side and comparing the qc summaries printed
at the end.

Usage (service must be running, see README "Running as a service"):

    python scripts/ab_lyric_guidance.py Huwaiza Sara
    python scripts/ab_lyric_guidance.py --host http://127.0.0.1:8000 \
        --text 5.0 --lyric 1.5 Huwaiza

Stdlib only — no requests/jq needed.
"""

import argparse
import json
import sys
import time
import urllib.request


def _post(host, payload):
    req = urllib.request.Request(
        f"{host}/jobs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _get(host, job_id):
    with urllib.request.urlopen(f"{host}/jobs/{job_id}") as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description="A/B lyric guidance via the service")
    ap.add_argument("names", nargs="+", help="Names to render in both arms")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--text", type=float, default=5.0,
                    help="guidance_scale_text for the ON arm (default 5.0)")
    ap.add_argument("--lyric", type=float, default=1.5,
                    help="guidance_scale_lyric for the ON arm (default 1.5)")
    ap.add_argument("--poll", type=float, default=15.0,
                    help="Seconds between polls (default 15)")
    args = ap.parse_args()

    jobs = {}  # (name, arm) -> job_id
    for name in args.names:
        safe = name.strip().lower().replace(" ", "_")
        arms = {
            "off": {"name": name, "slug": f"ab_{safe}_guidance_off"},
            "on": {"name": name, "slug": f"ab_{safe}_guidance_on",
                   "guidance_scale_text": args.text,
                   "guidance_scale_lyric": args.lyric},
        }
        for arm, payload in arms.items():
            body = _post(args.host, payload)
            jobs[(name, arm)] = body["job_id"]
            print(f"submitted {name} [{arm:3s}] → job {body['job_id']} "
                  f"({body['status']})")

    print(f"\npolling every {args.poll:.0f}s — renders run one at a time, "
          f"expect ~5-10 min per job...")
    pending = dict(jobs)
    results = {}
    while pending:
        time.sleep(args.poll)
        for key, job_id in list(pending.items()):
            body = _get(args.host, job_id)
            if body["status"] in ("done", "error"):
                results[key] = body
                del pending[key]
                qc = body.get("qc") or {}
                verdict = ("QC " + ("PASS" if body.get("qc_passed")
                                    else "FAIL" if body.get("qc_passed") is False
                                    else "n/a"))
                print(f"  {key[0]} [{key[1]:3s}] {body['status']} — {verdict} "
                      f"— {body.get('mp3_path') or body.get('error')}"
                      + (f" (sung {qc.get('sung_seconds', 0):.0f}s, "
                         f"name~{qc.get('name_score')})" if qc else ""))

    print("\n=== A/B pairs — listen side by side ===")
    failed = False
    for name in args.names:
        print(f"\n{name}:")
        for arm in ("off", "on"):
            body = results[(name, arm)]
            failed |= body["status"] == "error"
            print(f"  [{arm:3s}] {body.get('mp3_path') or body.get('error')}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
