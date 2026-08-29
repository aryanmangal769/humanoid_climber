#!/bin/sh
set -eu

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
RUN_NAME="${RUN_NAME:-slope-0.2-friction-0.1}"
HF_OUTPUT_REPO="${HF_OUTPUT_REPO:-Aryanmangal1234/humanoid-climber-policy}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"

apt-get update
apt-get install -y --no-install-recommends git libegl1 libgl1
rm -rf /var/lib/apt/lists/*

rm -rf /job
cp -R /workspace /job
cd /job

uv sync
mkdir -p "$LOG_ROOT/g1_ice/pretrained"
cp ckpt/g1_velocity_model_final.pt "$LOG_ROOT/g1_ice/pretrained/model_0.pt"

UV_PROJECT_ENVIRONMENT=.venv uv run hum-climber-train \
  HumClimber-Velocity-Ice-Unitree-G1 \
  --log-root "$LOG_ROOT" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.resume True \
  --agent.load-run pretrained \
  --agent.load-checkpoint model_0.pt \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger tensorboard \
  --agent.upload-model False

uvx --from huggingface_hub hf upload \
  "$HF_OUTPUT_REPO" \
  "$LOG_ROOT/g1_ice" \
  --include "**/*.pt" "**/*.yaml" \
  --commit-message "Upload G1 ice training checkpoints"
