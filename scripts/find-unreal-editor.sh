#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${UNREAL_EDITOR:-}" && -e "$UNREAL_EDITOR" ]]; then
  printf '%s\n' "$UNREAL_EDITOR"
  exit 0
fi

for candidate in \
  "$(command -v UnrealEditor 2>/dev/null || true)" \
  /opt/UnrealEngine/Engine/Binaries/Linux/UnrealEditor \
  /opt/UnrealEngine-5.6/Engine/Binaries/Linux/UnrealEditor; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done

for drive in c d e f; do
  for version in 5.6 5.7 5.5 5.4; do
    candidate="/mnt/$drive/Program Files/Epic Games/UE_${version}/Engine/Binaries/Win64/UnrealEditor.exe"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      exit 0
    fi
  done
done

for drive in c d e f; do
  for root in UnrealEngine UE_5.6 UE5 Unreal; do
    candidate="/mnt/$drive/$root/Engine/Binaries/Win64/UnrealEditor.exe"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      exit 0
    fi
  done
done

exit 1
