#!/usr/bin/env bash
# One-shot cleanup of v6 (Bark) leftovers.
# Safe to run multiple times — every command is idempotent.
#
# Usage:
#   bash scripts/cleanup_bark_leftovers.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "==> Cleaning $PROJECT_ROOT"

# --- Bark-only Python modules (replaced by song/acestep_render.py) ---
rm -fv song/vocal.py song/backing.py song/mix.py

# --- Bark backing tracks (ACE-Step generates its own backing) ---
rm -rfv assets/backing
rmdir assets 2>/dev/null && echo "removed empty assets/" || true

# --- Old per-line Bark cache (NOT cache/acestep/) ---
find cache/ -maxdepth 1 -type f \( -name '*.wav' -o -name '*.npy' \) -print -delete

# --- Stray root-level MP3s from old runs ---
rm -fv b2a4a162-2e45-444c-a23d-b713dc00dd4d.mp3
rm -fv happy_birthday_huwaiza.mp3

# --- Empty leftover dirs from the abandoned DiffSinger plan ---
rmdir models 2>/dev/null && echo "removed empty models/" || true
rmdir midi 2>/dev/null && echo "removed empty midi/" || true

# --- macOS cruft ---
find . -name '.DS_Store' -print -delete 2>/dev/null || true

# --- Stale design docs (superseded by PLAN.md) ---
rm -rfv docs/superpowers
rmdir docs 2>/dev/null && echo "removed empty docs/" || true

# --- Test backups (current tests pass; git holds the history) ---
rm -rfv tests/_archive_bark

# --- The original Python 3.14 Bark venv (broken, replaced by venv-diffsinger) ---
# Comment out if you want to keep it:
rm -rfv venv

echo
echo "✓ Cleanup done."
echo "Remaining: see 'ls -la' in $PROJECT_ROOT"
