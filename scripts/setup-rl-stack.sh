#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-3.12}"
VENV="$ROOT/.venv-rl"

cd "$ROOT"

git submodule update --init --recursive \
  vendor/unitree_rl_mjlab \
  vendor/newton

# unitree_rl_mjlab 1425b15 pins mujoco-warp==3.5.0 in setup.py.
# mjlab 1.2.0 itself supports mujoco-warp>=3.5.0, and Newton v1.0.0 uses
# mujoco==3.5.0 + mujoco-warp==3.5.0.2.  Install the compatible common
# runtime explicitly, then install Unitree editable with --no-deps so its
# unnecessarily strict 3.5.0 pin does not downgrade MuJoCo Warp.
uv venv --python "$PYTHON" "$VENV"

uv pip install --python "$VENV/bin/python" \
  "warp-lang==1.12.0" \
  "mujoco==3.5.0" \
  "mujoco-warp==3.5.0.2" \
  "mjlab==1.2.0" \
  "scipy" \
  "usd-core" \
  "pycollada"

uv pip install --python "$VENV/bin/python" -e "$ROOT/vendor/newton[sim]"
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/vendor/unitree_rl_mjlab"

echo
echo "RL stack installed in $VENV"
echo "Verify with: $ROOT/scripts/verify-rl-stack.sh"
