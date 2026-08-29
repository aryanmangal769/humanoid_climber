#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_ENVS="${NUM_ENVS:-4096}"
TASK="${TASK:-Unitree-G1-Flat}"
LOGGER="${LOGGER:-tensorboard}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT/vendor/unitree_rl_mjlab"
exec "$ROOT/.venv-rl/bin/python" scripts/train.py "$TASK" \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.logger="$LOGGER" \
  "$@"
