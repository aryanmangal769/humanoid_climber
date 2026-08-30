#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

if [[ ! -x .venv-sim/bin/python ]]; then
  echo "Missing .venv-sim; run scripts/setup-sim-stack.sh first." >&2
  exit 1
fi

if [[ ! -f maps/everest_local_terrain.json || ! -f maps/everest_macro_terrain.json ]]; then
  .venv-sim/bin/python maps/build_unity_terrain.py
fi

exec .venv-sim/bin/python -m simulation.unity_bridge "$@"
