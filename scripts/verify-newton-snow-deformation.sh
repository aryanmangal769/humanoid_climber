#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/everest-matplotlib}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/everest-warp-cache}"
cd "$ROOT"
exec "$ROOT/.venv-sim/bin/python" -m scripts.verify_newton_snow_deformation
