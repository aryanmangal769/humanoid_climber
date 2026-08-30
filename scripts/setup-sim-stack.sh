#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-3.12}"
VENV="$ROOT/.venv-sim"

cd "$ROOT"

if [[ ! -f "$ROOT/vendor/newton/pyproject.toml" ]]; then
  echo "Newton submodule is missing; run: git submodule update --init vendor/newton" >&2
  exit 1
fi

uv venv --clear --python "$PYTHON" "$VENV"
uv pip install --python "$VENV/bin/python" \
  -e "$ROOT/vendor/newton[sim]" \
  "numpy" \
  "onnx" \
  "pillow" \
  "scipy" \
  "trimesh" \
  "websockets"

echo
echo "Everest simulation stack installed in $VENV"
echo "Verify with: $ROOT/scripts/verify-newton-mujoco.sh"
