#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINDOWS_USER="${EVEREST_WINDOWS_USER:-$USER}"
PROFILE="/mnt/c/Users/$WINDOWS_USER"
if [[ ! -d "$PROFILE" ]]; then
  echo "Windows profile not found: $PROFILE" >&2
  exit 1
fi

TEMP_LINUX="$PROFILE/AppData/Local/Temp/everest-unreal-prereqs.ps1"
TEMP_WIN="C:\\Users\\$WINDOWS_USER\\AppData\\Local\\Temp\\everest-unreal-prereqs.ps1"
cp "$ROOT/scripts/setup-unreal-windows.ps1" "$TEMP_LINUX"
rm -f "$PROFILE/AppData/Local/Temp/everest-unreal-prereqs.done" "$PROFILE/AppData/Local/Temp/everest-unreal-prereqs.log"

# This intentionally requests elevation. Windows may display a UAC consent dialog.
"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -Command \
  "Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"$TEMP_WIN\" -OpenEpicLauncher'"

echo "Elevated Unreal prerequisite installer launched."
echo "Log: $PROFILE/AppData/Local/Temp/everest-unreal-prereqs.log"
