#!/bin/sh
set -eu

export TASK_ID="HumClimber-Velocity-Flat-Wind-Unitree-G1"
export EXPERIMENT_NAME="g1_flat_wind"
export RUN_NAME="flat-wind-base-finetune"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
export HF_OUTPUT_REPO="${HF_OUTPUT_REPO:-Aryanmangal1234/humanoid-climber-policy}"
export HF_OUTPUT_PATH="${HF_OUTPUT_PATH:-training/g1_flat_wind/base-finetune}"
export UPLOAD_INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec sh "$(dirname "$0")/hf_train.sh"
