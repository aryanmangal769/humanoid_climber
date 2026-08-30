#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITY_VERSION="${UNITY_VERSION:-2022.3.0f1}"
UNITY_EXE="${UNITY_EXE:-/mnt/c/Program Files/Unity/Hub/Editor/$UNITY_VERSION/Editor/Unity.exe}"
WINDOWS_USER="${EVEREST_WINDOWS_USER:-auverus}"
WEB_PROJECT="${EVEREST_UNITY_WEB_PROJECT:-/mnt/c/Users/$WINDOWS_USER/Documents/EverestUnityWeb}"

if [[ ! -x "$UNITY_EXE" ]]; then
  echo "Unity editor not found: $UNITY_EXE" >&2
  exit 1
fi

EVEREST_UNITY_WINDOWS_DIR="$WEB_PROJECT" "$ROOT/scripts/sync-unity-windows.sh" >/dev/null
WIN_PROJECT="$(wslpath -w "$WEB_PROJECT")"
WIN_LOG="$WIN_PROJECT\\webgl-build.log"

"$UNITY_EXE" \
  -batchmode \
  -quit \
  -projectPath "$WIN_PROJECT" \
  -executeMethod EverestSim.Editor.EverestWebBuild.BuildWebGL \
  -logFile "$WIN_LOG"

OUTPUT="$WEB_PROJECT/Builds/WebGL"
if [[ ! -f "$OUTPUT/index.html" ]]; then
  echo "Unity reported success but $OUTPUT/index.html is missing" >&2
  exit 1
fi
printf '%s\n' "$OUTPUT"
