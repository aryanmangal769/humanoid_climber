#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /usr/lib/wsl/lib/libcuda.so ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="/usr/lib/wsl/lib:$PATH"
fi
export PYTHONPATH="$ROOT:$ROOT/vendor/newton${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/7] Checking RL/Newton environment"
if [[ ! -x .venv-sim/bin/python ]]; then
  ./scripts/setup-sim-stack.sh
fi

echo "[2/7] Verifying Python dependencies"
.venv-sim/bin/python - <<'PY'
mods = ("numpy", "PIL", "mujoco", "warp", "websockets", "trimesh")
missing = []
for name in mods:
    try:
        __import__(name)
    except Exception as exc:
        missing.append((name, repr(exc)))
if missing:
    raise SystemExit("Missing runtime dependencies: " + ", ".join(f"{n}: {e}" for n, e in missing))
print("Python runtime dependencies OK")
PY

echo "[3/7] Regenerating true-scale Everest terrain"
.venv-sim/bin/python studio/unreal/tools/build_true_scale_terrain.py
.venv-sim/bin/python studio/unreal/tools/export_everest_heightmap.py

echo "[4/7] Preparing Unitree G1 render assets"
.venv-sim/bin/python studio/unreal/tools/prepare_g1_assets.py

echo "[5/7] Verifying RTX / Newton MPM"
.venv-sim/bin/python - <<'PY'
import warp as wp
wp.init()
devices = [str(d) for d in wp.get_devices()]
if not any(d.startswith("cuda") for d in devices):
    raise SystemExit(f"Warp cannot see CUDA. Devices: {devices}")
print("Warp CUDA devices:", devices)
PY
PROBE="$(mktemp)"
trap 'rm -f "$PROBE"' EXIT
./scripts/start-unreal-bridge.sh --probe >"$PROBE"
.venv-sim/bin/python - "$PROBE" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
start = text.find("{")
if start < 0:
    raise SystemExit("Bridge probe produced no JSON")
data = json.loads(text[start:])
mpm = data["snow"]["mpm"]
assert data["engine"] == "newton+mujoco", data["engine"]
assert mpm["active"], mpm
assert mpm["cuda"], mpm
print(f"Bridge OK: {mpm['solver']} on {mpm['device']} ({mpm['particle_count']} particles)")
PY

echo "[6/7] Mirroring Unreal project onto the Windows filesystem"
WINDOWS_PROJECT="$(./scripts/sync-unreal-windows.sh || true)"
if [[ -n "$WINDOWS_PROJECT" ]]; then
  echo "Windows project: $WINDOWS_PROJECT"
fi

echo "[7/7] Discovering Unreal Editor"
if EDITOR="$(./scripts/find-unreal-editor.sh 2>/dev/null)"; then
  echo "Unreal Editor: $EDITOR"
  echo "Everest Unreal setup is complete. Run: ./scripts/start-unreal.sh"
else
  echo "Unreal Editor is the only missing component."
  echo "Install Unreal Engine 5.6 from Epic Games Launcher, preferably on D:, then run ./scripts/start-unreal.sh."
  exit 2
fi
