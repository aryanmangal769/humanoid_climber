#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINUX_PROJECT="$ROOT/studio/unreal/EverestSim/EverestSim.uproject"

if ! EDITOR="$($ROOT/scripts/find-unreal-editor.sh 2>/dev/null)"; then
  echo "UnrealEditor not found. Run ./scripts/setup-unreal.sh after installing Unreal Engine 5.6." >&2
  echo "Linux source project: $LINUX_PROJECT" >&2
  exit 2
fi

PROJECT="$LINUX_PROJECT"
if [[ "$EDITOR" == *.exe ]]; then
  MIRROR="$($ROOT/scripts/sync-unreal-windows.sh)"
  PROJECT="$MIRROR/EverestSim.uproject"
fi

"$ROOT/scripts/start-unreal-bridge.sh" >"${TMPDIR:-/tmp}/everest-unreal-bridge.log" 2>&1 &
BRIDGE_PID=$!
trap 'kill "$BRIDGE_PID" 2>/dev/null || true' EXIT INT TERM

# Give the bridge a bounded chance to finish Newton/Warp warmup before the editor connects.
for _ in $(seq 1 120); do
  if grep -q "Everest Unreal bridge ws://" "${TMPDIR:-/tmp}/everest-unreal-bridge.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    cat "${TMPDIR:-/tmp}/everest-unreal-bridge.log" >&2
    exit 1
  fi
  sleep 0.25
done

if [[ "$EDITOR" == *.exe ]]; then
  PROJECT_ARG="$(wslpath -w "$PROJECT")"
else
  PROJECT_ARG="$PROJECT"
fi

exec "$EDITOR" "$PROJECT_ARG" -log
