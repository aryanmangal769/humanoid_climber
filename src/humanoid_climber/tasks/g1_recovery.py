"""Native MjLab Unitree G1 supine recovery tracking task."""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1 import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_tracking_ppo_runner_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "HumClimber-Tracking-Recovery-Unitree-G1"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOTION_FILE = (
  _PROJECT_ROOT / "private_assets" / "recovery" / "g1_humanup_getup_50hz.npz"
)


def unitree_g1_recovery_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build a 29-action G1 task that tracks a supine-to-standing reference."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  # Supine recovery creates more simultaneous ground and self contacts than the
  # stock tracking motion. Keep enough capacity to avoid dropping contacts.
  cfg.sim.nconmax = 70
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = os.environ.get(
    "HUM_CLIMBER_RECOVERY_MOTION", str(DEFAULT_MOTION_FILE)
  )

  # Match the complete 8-second reference and avoid unrelated random pushes while
  # learning the basic recovery behavior. Startup pose/velocity randomization in the
  # tracking command still supplies robust reference-state initialization in training.
  cfg.episode_length_s = 8.0 if not play else int(1e9)
  cfg.events.pop("push_robot", None)

  return cfg


def unitree_g1_recovery_ppo_runner_cfg():
  """Return the dedicated PPO configuration for recovery tracking."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_recovery"
  cfg.save_interval = 250
  cfg.max_iterations = 20_000
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_recovery_env_cfg(),
  play_env_cfg=unitree_g1_recovery_env_cfg(play=True),
  rl_cfg=unitree_g1_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
