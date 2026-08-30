#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# WSL exposes the NVIDIA driver libraries here. Warp can otherwise initialize
# without seeing CUDA and silently build the Newton snow patch on CPU.
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/everest-warp-cache}"
exec "$ROOT/.venv-rl/bin/python" -m dashboard.server "$@"
