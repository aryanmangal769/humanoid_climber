#!/bin/sh
set -eu

export TASK_ID="HumClimber-Tracking-Recovery-Unitree-G1"
export EXPERIMENT_NAME="g1_recovery"
export RUN_NAME="supine-native-tracking"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
export RESUME_FROM_BASE="false"
export HUM_CLIMBER_RECOVERY_MOTION="${HUM_CLIMBER_RECOVERY_MOTION:-private_assets/recovery/g1_humanup_getup_50hz.npz}"
export HF_OUTPUT_REPO="${HF_OUTPUT_REPO:-Aryanmangal1234/humanoid-climber-policy}"
export HF_OUTPUT_PATH="${HF_OUTPUT_PATH:-training/g1_recovery/supine-native-tracking}"
export UPLOAD_INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec sh "$(dirname "$0")/hf_train.sh"