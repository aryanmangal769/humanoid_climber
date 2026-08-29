#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vendor/mujoco_playground"

uv venv --python 3.12 .venv
uv --no-config sync --extra cuda
"$ROOT/vendor/mujoco_playground/.venv/bin/python" -c \
  "from mujoco_playground._src.mjx_env import ensure_menagerie_exists; ensure_menagerie_exists()"
echo "MuJoCo Playground is ready. Next: $ROOT/scripts/verify-g1.sh"
