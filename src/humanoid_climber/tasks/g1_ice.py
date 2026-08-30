"""Low-friction Unitree G1 velocity task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains import HfPyramidSlopedTerrainCfg, TerrainGeneratorCfg

from humanoid_climber import mdp as climber_mdp

TASK_ID = "HumClimber-Velocity-Ice-Unitree-G1"
WIND_TASK_ID = "HumClimber-Velocity-Ice-Wind-Unitree-G1"
TRAIN_FRICTION_RANGE = (0.1, 1.0)
TRAIN_SLOPE_RANGE = (0.0, 0.2)
TRAIN_TERRAIN_SIZE = (8.0, 8.0)
TRAIN_WIND_FORCE_RANGES = {
  "x": (-4.0, 4.0),
  "y": (-12.0, 12.0),
  "z": (0.0, 0.0),
}
EVAL_FRICTION = 0.1
EVAL_SLOPE_GRADIENT = 0.2
EVAL_TERRAIN_SIZE = (16.0, 16.0)
EVAL_WIND_FORCE_RANGES = {
  "x": (0.0, 0.0),
  "y": (16.0, 16.0),
  "z": (0.0, 0.0),
}


def unitree_g1_ice_env_cfg(
  *, play: bool = False, wind: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build the stock flat G1 task with lower foot friction.

  Training randomizes friction for robustness. Playback fixes friction at 0.1
  and places the robot at the bottom of an 11.3-degree slope so repeated
  evaluations use the same ice-like uphill test.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)
  friction = (EVAL_FRICTION, EVAL_FRICTION) if play else TRAIN_FRICTION_RANGE
  cfg.events["foot_friction"].params["ranges"] = friction

  if wind:
    cfg.events["wind"] = EventTermCfg(
      func=climber_mdp.apply_wind_force,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_ranges": (
          EVAL_WIND_FORCE_RANGES if play else TRAIN_WIND_FORCE_RANGES
        ),
      },
    )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"

  if play:
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      seed=42,
      size=EVAL_TERRAIN_SIZE,
      num_rows=1,
      num_cols=1,
      difficulty_range=(1.0, 1.0),
      sub_terrains={
        "ice_slope": HfPyramidSlopedTerrainCfg(
          proportion=1.0,
          slope_range=(EVAL_SLOPE_GRADIENT, EVAL_SLOPE_GRADIENT),
          platform_width=1.0,
          inverted=True,
          horizontal_scale=0.1,
        )
      },
    )
  else:
    cfg.events.pop("push_robot", None)
    cfg.scene.terrain.max_init_terrain_level = 2
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      seed=42,
      curriculum=True,
      size=TRAIN_TERRAIN_SIZE,
      border_width=2.0,
      num_rows=10,
      num_cols=1,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "ice_slope": HfPyramidSlopedTerrainCfg(
          proportion=1.0,
          slope_range=TRAIN_SLOPE_RANGE,
          platform_width=1.0,
          inverted=True,
          horizontal_scale=0.1,
        )
      },
    )
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    )

  return cfg


def unitree_g1_ice_ppo_runner_cfg():
  """Create the PPO configuration for ice-slope fine-tuning."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_ice"
  cfg.max_iterations = 5_000
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_ice_env_cfg(),
  play_env_cfg=unitree_g1_ice_env_cfg(play=True),
  rl_cfg=unitree_g1_ice_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=WIND_TASK_ID,
  env_cfg=unitree_g1_ice_env_cfg(wind=True),
  play_env_cfg=unitree_g1_ice_env_cfg(play=True, wind=True),
  rl_cfg=unitree_g1_ice_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
