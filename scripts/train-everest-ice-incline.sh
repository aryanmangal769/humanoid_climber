#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${EVEREST_ICE_INCLINE_CHECKPOINT:-/home/auverus/git/humanoid_climber_safety_ckpts/ckpt/exported/ice_incline.onnx}"
exec .venv-rl/bin/python -m training.everest_ppo \
  --checkpoint "$CHECKPOINT" \
  "$@"
