#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-3.12}"
VENV="$ROOT/.venv-rl"

cd "$ROOT"

git submodule update --init --recursive \
  vendor/unitree_rl_mjlab

# unitree_rl_mjlab 1425b15 pins mujoco-warp==3.5.0 in setup.py.
# MJLab 1.2 still uses wp.context, which requires Warp 1.12. Keep this venv
# exclusively for policy training; the live Newton simulator has its own
# current stack in .venv-sim (see setup-sim-stack.sh).
uv venv --python "$PYTHON" "$VENV"

uv pip install --python "$VENV/bin/python" \
  "warp-lang==1.12.0" \
  "mujoco==3.5.0" \
  "mujoco-warp==3.5.0.2" \
  "mjlab==1.2.0" \
  "scipy" \
  "usd-core" \
  "pycollada"

uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/vendor/unitree_rl_mjlab"

echo
echo "RL stack installed in $VENV"
echo "Verify with: $ROOT/scripts/verify-rl-stack.sh"
