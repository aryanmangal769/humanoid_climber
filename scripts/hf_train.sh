#!/bin/sh
set -eu

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
RUN_NAME="${RUN_NAME:-slope-0.2-friction-0.1}"
TASK_ID="${TASK_ID:-HumClimber-Velocity-Ice-Unitree-G1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_ice}"
HF_OUTPUT_REPO="${HF_OUTPUT_REPO:-Aryanmangal1234/humanoid-climber-policy}"
HF_OUTPUT_PATH="${HF_OUTPUT_PATH:-training/$EXPERIMENT_NAME/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"
UPLOAD_INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}"
RESUME_FROM_BASE="${RESUME_FROM_BASE:-true}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-ckpt/g1_velocity_model_final.pt}"
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

apt-get update
apt-get install -y --no-install-recommends git libegl1 libgl1
rm -rf /var/lib/apt/lists/*

cd "$PROJECT_DIR"

uv sync
if [ "$RESUME_FROM_BASE" = "true" ]; then
  mkdir -p "$LOG_ROOT/$EXPERIMENT_NAME/pretrained"
  cp "$PRETRAINED_CHECKPOINT" \
    "$LOG_ROOT/$EXPERIMENT_NAME/pretrained/model_0.pt"
fi

upload_checkpoints() {
  uvx --from huggingface_hub hf upload \
    "$HF_OUTPUT_REPO" \
    "$LOG_ROOT/$EXPERIMENT_NAME" \
    "$HF_OUTPUT_PATH" \
    --include "**/*.pt" "**/*.yaml" \
    --commit-message "Update private training checkpoints" >/dev/null
}

periodic_upload() {
  while kill -0 "$1" 2>/dev/null; do
    sleep "$UPLOAD_INTERVAL_SECONDS"
    upload_checkpoints || true
  done
}

run_training() {
  if [ "$RESUME_FROM_BASE" = "true" ]; then
    UV_PROJECT_ENVIRONMENT=.venv uv run hum-climber-train \
      "$TASK_ID" \
      --log-root "$LOG_ROOT" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.resume True \
      --agent.load-run pretrained \
      --agent.load-checkpoint model_0.pt \
      --agent.max-iterations "$MAX_ITERATIONS" \
      --agent.run-name "$RUN_NAME" \
      --agent.logger tensorboard \
      --agent.upload-model False
  else
    UV_PROJECT_ENVIRONMENT=.venv uv run hum-climber-train \
      "$TASK_ID" \
      --log-root "$LOG_ROOT" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.max-iterations "$MAX_ITERATIONS" \
      --agent.run-name "$RUN_NAME" \
      --agent.logger tensorboard \
      --agent.upload-model False
  fi
}

run_training &
train_pid="$!"
periodic_upload "$train_pid" &
uploader_pid="$!"

handle_termination() {
  trap - INT TERM
  kill "$train_pid" "$uploader_pid" 2>/dev/null || true
  wait "$train_pid" 2>/dev/null || true
  wait "$uploader_pid" 2>/dev/null || true
  upload_checkpoints || true
  exit 143
}
trap handle_termination INT TERM

set +e
wait "$train_pid"
train_status="$?"
set -e
trap - INT TERM
kill "$uploader_pid" 2>/dev/null || true
wait "$uploader_pid" 2>/dev/null || true
upload_checkpoints

exit "$train_status"
