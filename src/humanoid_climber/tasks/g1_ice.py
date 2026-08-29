"""Low-friction Unitree G1 velocity task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

TASK_ID = "HumClimber-Velocity-Ice-Unitree-G1"
TRAIN_FRICTION_RANGE = (0.1, 1.0)
EVAL_FRICTION = 0.2
EVAL_FORWARD_VELOCITY = 2.0


def unitree_g1_ice_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the stock flat G1 task with lower foot friction.

  Training randomizes friction for robustness. Playback fixes friction at 0.2 so
  repeated evaluations use the same ice-like surface.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)
  friction = (EVAL_FRICTION, EVAL_FRICTION) if play else TRAIN_FRICTION_RANGE
  cfg.events["foot_friction"].params["ranges"] = friction

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (
      EVAL_FORWARD_VELOCITY,
      EVAL_FORWARD_VELOCITY,
    )
    # Viser requires a positive range for its optional joystick sliders. The
    # forward-only override below still sets both values to exactly zero.
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
    twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)
    twist_cmd.ranges.heading = None
    twist_cmd.heading_command = False
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_world_envs = 0.0
    twist_cmd.rel_forward_envs = 1.0

  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_ice_env_cfg(),
  play_env_cfg=unitree_g1_ice_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
