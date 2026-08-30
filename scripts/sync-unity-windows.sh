#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/studio/unity/EverestSim"

if [[ ! -d /mnt/c/Users ]]; then
  echo "Windows filesystem is not mounted; open $SOURCE directly in Unity." >&2
  exit 0
fi

WINDOWS_USER="${EVEREST_WINDOWS_USER:-$USER}"
if [[ ! -d "/mnt/c/Users/$WINDOWS_USER" ]]; then
  WINDOWS_USER="$(find /mnt/c/Users -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | grep -Ev '^(All Users|Default|Default User|Public)$' | head -1 || true)"
fi
if [[ -z "$WINDOWS_USER" || ! -d "/mnt/c/Users/$WINDOWS_USER" ]]; then
  echo "Could not determine the Windows user profile." >&2
  exit 1
fi

DEST="${EVEREST_UNITY_WINDOWS_DIR:-/mnt/c/Users/$WINDOWS_USER/Documents/EverestUnity}"
mkdir -p "$DEST"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude Library \
    --exclude Temp \
    --exclude Obj \
    --exclude Logs \
    --exclude UserSettings \
    --exclude Build \
    --exclude Builds \
    --exclude webgl-build.log \
    "$SOURCE/" "$DEST/"
else
  rm -rf "$DEST/Assets" "$DEST/Packages" "$DEST/ProjectSettings"
  cp -a "$SOURCE/Assets" "$SOURCE/Packages" "$SOURCE/ProjectSettings" "$DEST/"
fi

printf '%s\n' "$DEST"
