#!/usr/bin/env bash
# Install ACE-Step into the active venv on Apple Silicon.
# Usage:
#   source venv-diffsinger/bin/activate     # or whatever venv you're using
#   bash scripts/install_acestep.sh
#
# This script is idempotent — re-running it is safe.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
ACE_DIR="$EXTERNAL_DIR/ACE-Step"

echo "==> Project root: $PROJECT_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: no virtualenv active. Activate one first, e.g.:"
    echo "    source $PROJECT_ROOT/venv-diffsinger/bin/activate"
    exit 1
fi
echo "==> Using venv: $VIRTUAL_ENV"
python --version

mkdir -p "$EXTERNAL_DIR"
if [ ! -d "$ACE_DIR/.git" ]; then
    echo "==> Cloning ACE-Step..."
    git clone --depth 1 https://github.com/ace-step/ACE-Step.git "$ACE_DIR"
else
    echo "==> ACE-Step already cloned at $ACE_DIR — pulling latest"
    git -C "$ACE_DIR" pull --ff-only || echo "    (skipped — local changes or detached HEAD)"
fi

echo "==> Installing ACE-Step (editable)..."
pip install --upgrade pip wheel setuptools
pip install -e "$ACE_DIR"

echo "==> Installing project deps..."
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "==> Verifying ACE-Step import..."
python -c "from acestep.pipeline_ace_step import ACEStepPipeline; print('OK: ACEStepPipeline imports')"

echo
echo "✓  ACE-Step installed."
echo "Next: python scripts/smoke_test.py"
