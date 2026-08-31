#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# WSL's CUDA driver shim is not always in the dynamic loader's default search
# path even though nvidia-smi and /dev/dxg are available. Warp otherwise falls
# back to CPU with CUDA error 100. Match the proven main-backend environment.
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/everest-demo-warp-cache}"

for argument in "$@"; do
  if [[ "$argument" == "--disable-newton" ]]; then
    echo "The autonomous Everest demo requires live Newton snow; --disable-newton is only for the normal MuJoCo-only track." >&2
    exit 2
  fi
done

exec scripts/start-simulation-backend.sh --demo autonomous-showcase "$@"
