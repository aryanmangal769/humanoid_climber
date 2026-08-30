#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/studio/omniverse/apps/everest.studio.kit"
WIN_ISAAC="/mnt/c/Users/auverus/venvs/everest-isaacsim/Scripts/isaacsim.exe"
WIN_ROOT="/mnt/c/Users/auverus/everest-studio-runtime"

if [[ -x "$WIN_ISAAC" ]] && command -v wslpath >/dev/null 2>&1; then
  mkdir -p \
    "$WIN_ROOT/studio/omniverse/apps" \
    "$WIN_ROOT/studio/omniverse/exts" \
    "$WIN_ROOT/studio/omniverse/assets" \
    "$WIN_ROOT/maps" \
    "$WIN_ROOT/vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1"

  rsync -a --delete "$ROOT/studio/omniverse/apps/" "$WIN_ROOT/studio/omniverse/apps/"
  rsync -a --delete "$ROOT/studio/omniverse/exts/" "$WIN_ROOT/studio/omniverse/exts/"
  cp "$ROOT/maps/everest_terrain.json" "$WIN_ROOT/maps/everest_terrain.json"
  cp "$ROOT/studio/omniverse/assets/everest_terrain.usda" "$WIN_ROOT/studio/omniverse/assets/everest_terrain.usda"
  rsync -a --delete \
    "$ROOT/vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/" \
    "$WIN_ROOT/vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/"

  EXTS="$(wslpath -w "$WIN_ROOT/studio/omniverse/exts")"
  export OMNI_KIT_ACCEPT_EULA=YES
  export WSLENV="${WSLENV:+$WSLENV:}OMNI_KIT_ACCEPT_EULA"
  exec "$WIN_ISAAC" isaacsim.exp.full \
    --ext-folder "$EXTS" \
    --enable everest.studio \
    --enable isaacsim.physics.newton \
    --enable isaacsim.physics.newton.ui \
    --/exts/isaacsim.physics.newton/auto_switch_on_startup=true \
    --/exts/isaacsim.physics.newton/capture_graph_physics_step=true \
    "$@"
fi

if [[ -x "$ROOT/.venv-isaacsim/bin/isaacsim" ]]; then
  ISAAC="$ROOT/.venv-isaacsim/bin/isaacsim"
  ISAAC_PIP=1
elif [[ -n "${ISAAC_SIM_PATH:-}" && -x "$ISAAC_SIM_PATH/isaac-sim.sh" ]]; then
  ISAAC="$ISAAC_SIM_PATH/isaac-sim.sh"
elif [[ -x "$HOME/isaacsim/isaac-sim.sh" ]]; then
  ISAAC="$HOME/isaacsim/isaac-sim.sh"
else
  ISAAC="$(find "$HOME" -maxdepth 5 -type f -name isaac-sim.sh -perm -u+x 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${ISAAC:-}" ]]; then
  echo "Isaac Sim runtime not found." >&2
  echo "Install Isaac Sim 6.x or set ISAAC_SIM_PATH, then rerun." >&2
  exit 2
fi

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
if [[ "${ISAAC_PIP:-0}" == "1" ]]; then
  export OMNI_KIT_ACCEPT_EULA=YES
  exec "$ISAAC" "$APP" "$@"
fi
exec "$ISAAC" --experience "$APP" "$@"
