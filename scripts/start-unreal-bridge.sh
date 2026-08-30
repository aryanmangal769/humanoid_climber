#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv-sim/bin/python ]]; then
  echo "Missing .venv-sim; run scripts/setup-sim-stack.sh first." >&2
  exit 1
fi
# WSL ships the real host NVIDIA driver shim here. Ubuntu may also have a
# distro libcuda.so installed; Warp dlopen()s libcuda.so directly and can pick
# that stale library first unless the WSL driver path leads LD_LIBRARY_PATH.
if [[ -f /usr/lib/wsl/lib/libcuda.so ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="/usr/lib/wsl/lib:$PATH"
fi
export PYTHONPATH="$ROOT:$ROOT/vendor/newton${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-sim/bin/python studio/unreal/bridge.py "$@"
