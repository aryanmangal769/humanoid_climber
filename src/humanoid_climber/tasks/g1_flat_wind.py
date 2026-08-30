"""Flat-ground Unitree G1 task with a physical crosswind force."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from humanoid_climber import mdp as climber_mdp
from humanoid_climber.tasks.g1_ice import EVAL_WIND_FORCE_RANGES

TASK_ID = "HumClimber-Velocity-Flat-Wind-Unitree-G1"
EVAL_FRICTION = 0.15
TRAIN_FRICTION_RANGE = (0.15, 1.0)
TRAIN_WIND_FORCE_RANGES = {
  "x": (-4.0, 4.0),
  "y": (-16.0, 16.0),
  "z": (0.0, 0.0),
}


def unitree_g1_flat_wind_env_cfg(*, play: bool = False):
  """Build the stock flat-ground G1 task with an external torso crosswind."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.events["foot_friction"].params["ranges"] = (
    (EVAL_FRICTION, EVAL_FRICTION) if play else TRAIN_FRICTION_RANGE
  )
  if not play:
    cfg.events.pop("push_robot", None)
  cfg.events["wind"] = EventTermCfg(
    func=climber_mdp.apply_wind_force,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "force_ranges": EVAL_WIND_FORCE_RANGES if play else TRAIN_WIND_FORCE_RANGES,
    },
  )
  return cfg


def unitree_g1_flat_wind_ppo_runner_cfg():
  """Create a dedicated PPO configuration for flat-wind fine-tuning."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_flat_wind"
  cfg.max_iterations = 5_000
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_flat_wind_env_cfg(),
  play_env_cfg=unitree_g1_flat_wind_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_wind_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)