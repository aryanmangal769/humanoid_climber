#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/studio/unreal/EverestSim"

if [[ ! -d /mnt/c/Users ]]; then
  echo "Windows filesystem is not mounted; no Windows Unreal mirror is needed." >&2
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

DEST="${EVEREST_UNREAL_WINDOWS_DIR:-/mnt/c/Users/$WINDOWS_USER/Documents/EverestSim}"
mkdir -p "$DEST"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude Binaries \
    --exclude Intermediate \
    --exclude Saved \
    --exclude DerivedDataCache \
    "$SOURCE/" "$DEST/"
else
  rm -rf "$DEST/Config" "$DEST/Content" "$DEST/Source" "$DEST/SourceData"
  cp -a "$SOURCE/Config" "$SOURCE/Content" "$SOURCE/Source" "$SOURCE/SourceData" "$DEST/"
  cp -a "$SOURCE/EverestSim.uproject" "$DEST/"
fi

printf '%s\n' "$DEST"
