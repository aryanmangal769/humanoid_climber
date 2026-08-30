from humanoid_climber.tasks.g1_ice import (
  EVAL_FRICTION,
  EVAL_SLOPE_GRADIENT,
  EVAL_TERRAIN_SIZE,
  EVAL_WIND_FORCE_RANGES,
  TRAIN_FRICTION_RANGE,
  TRAIN_SLOPE_RANGE,
  TRAIN_TERRAIN_SIZE,
  TRAIN_WIND_FORCE_RANGES,
  unitree_g1_ice_env_cfg,
  unitree_g1_ice_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.terrains import HfPyramidSlopedTerrainCfg

from humanoid_climber.tasks.g1_flat_wind import (
  EVAL_FRICTION as FLAT_WIND_EVAL_FRICTION,
  TRAIN_FRICTION_RANGE as FLAT_WIND_TRAIN_FRICTION_RANGE,
  TRAIN_WIND_FORCE_RANGES as FLAT_WIND_TRAIN_FORCE_RANGES,
  unitree_g1_flat_wind_env_cfg,
  unitree_g1_flat_wind_ppo_runner_cfg,
)
from humanoid_climber.tasks.g1_recovery import (
  DEFAULT_MOTION_FILE,
  unitree_g1_recovery_env_cfg,
  unitree_g1_recovery_ppo_runner_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def test_ice_task_keeps_stock_policy_interface() -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  ice = unitree_g1_ice_env_cfg(play=True)

  assert tuple(ice.observations["actor"].terms) == tuple(
    stock.observations["actor"].terms
  )
  assert tuple(ice.observations["critic"].terms) == tuple(
    stock.observations["critic"].terms
  )
  assert tuple(ice.actions) == tuple(stock.actions)


def test_ice_friction_ranges() -> None:
  train = unitree_g1_ice_env_cfg()
  play = unitree_g1_ice_env_cfg(play=True)

  assert train.events["foot_friction"].params["ranges"] == TRAIN_FRICTION_RANGE
  assert play.events["foot_friction"].params["ranges"] == (
    EVAL_FRICTION,
    EVAL_FRICTION,
  )


def test_ice_play_keeps_stock_randomized_commands() -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  play = unitree_g1_ice_env_cfg(play=True)

  assert play.commands["twist"] == stock.commands["twist"]


def test_ice_play_uses_fixed_uphill_slope() -> None:
  play = unitree_g1_ice_env_cfg(play=True)
  assert play.scene.terrain is not None
  assert play.scene.terrain.terrain_type == "generator"
  generator = play.scene.terrain.terrain_generator
  assert generator is not None
  assert generator.size == EVAL_TERRAIN_SIZE
  assert generator.difficulty_range == (1.0, 1.0)

  slope = generator.sub_terrains["ice_slope"]
  assert isinstance(slope, HfPyramidSlopedTerrainCfg)
  assert slope.slope_range == (EVAL_SLOPE_GRADIENT, EVAL_SLOPE_GRADIENT)
  assert slope.inverted
  assert slope.horizontal_scale == 0.1


def test_ice_training_uses_slope_curriculum() -> None:
  train = unitree_g1_ice_env_cfg()
  assert train.scene.terrain is not None
  assert train.scene.terrain.terrain_type == "generator"
  assert train.scene.terrain.max_init_terrain_level == 2

  generator = train.scene.terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum
  assert generator.size == TRAIN_TERRAIN_SIZE
  assert generator.num_rows == 10
  assert "terrain_levels" in train.curriculum
  assert "push_robot" not in train.events

  slope = generator.sub_terrains["ice_slope"]
  assert isinstance(slope, HfPyramidSlopedTerrainCfg)
  assert slope.slope_range == TRAIN_SLOPE_RANGE
  assert slope.inverted


def test_wind_task_applies_crosswind_to_torso() -> None:
  train = unitree_g1_ice_env_cfg(wind=True)
  play = unitree_g1_ice_env_cfg(play=True, wind=True)

  assert "wind" not in unitree_g1_ice_env_cfg().events
  assert train.events["wind"].mode == "reset"
  assert hasattr(train.events["wind"].func, "debug_vis")
  assert train.events["wind"].params["force_ranges"] == TRAIN_WIND_FORCE_RANGES
  assert play.events["wind"].params["force_ranges"] == EVAL_WIND_FORCE_RANGES
  assert train.events["wind"].params["asset_cfg"].body_names == ("torso_link",)


def test_wind_task_keeps_stock_policy_interface() -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  wind = unitree_g1_ice_env_cfg(play=True, wind=True)

  assert tuple(wind.observations["actor"].terms) == tuple(
    stock.observations["actor"].terms
  )
  assert tuple(wind.observations["critic"].terms) == tuple(
    stock.observations["critic"].terms
  )
  assert tuple(wind.actions) == tuple(stock.actions)


def test_flat_wind_task_keeps_stock_terrain_and_policy_interface() -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  wind = unitree_g1_flat_wind_env_cfg(play=True)

  assert wind.scene.terrain is not None
  assert stock.scene.terrain is not None
  assert wind.scene.terrain.terrain_type == stock.scene.terrain.terrain_type
  assert wind.scene.terrain.terrain_generator == stock.scene.terrain.terrain_generator
  assert wind.commands == stock.commands
  assert tuple(wind.observations["actor"].terms) == tuple(
    stock.observations["actor"].terms
  )
  assert tuple(wind.observations["critic"].terms) == tuple(
    stock.observations["critic"].terms
  )
  assert tuple(wind.actions) == tuple(stock.actions)
  assert wind.events["wind"].params["force_ranges"] == EVAL_WIND_FORCE_RANGES
  assert wind.events["foot_friction"].params["ranges"] == (
    FLAT_WIND_EVAL_FRICTION,
    FLAT_WIND_EVAL_FRICTION,
  )


def test_flat_wind_training_covers_evaluation_conditions() -> None:
  train = unitree_g1_flat_wind_env_cfg()

  assert train.events["foot_friction"].params["ranges"] == (
    FLAT_WIND_TRAIN_FRICTION_RANGE
  )
  assert "push_robot" not in train.events
  assert train.events["wind"].params["force_ranges"] == (
    FLAT_WIND_TRAIN_FORCE_RANGES
  )
  assert FLAT_WIND_TRAIN_FORCE_RANGES["y"] == (-16.0, 16.0)


def test_flat_wind_runner_has_dedicated_experiment() -> None:
  runner = unitree_g1_flat_wind_ppo_runner_cfg()

  assert runner.experiment_name == "g1_flat_wind"


def test_ice_runner_has_dedicated_experiment() -> None:
  runner = unitree_g1_ice_ppo_runner_cfg()
  assert runner.experiment_name == "g1_ice"
  assert runner.max_iterations == 5_000


def test_recovery_task_uses_native_g1_tracking_interface() -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  recovery = unitree_g1_recovery_env_cfg()
  motion = recovery.commands["motion"]

  assert isinstance(motion, MotionCommandCfg)
  assert motion.motion_file == str(DEFAULT_MOTION_FILE)
  assert recovery.episode_length_s == 8.0
  assert recovery.sim.nconmax == 70
  assert "push_robot" not in recovery.events
  assert tuple(recovery.actions) == tuple(stock.actions)


def test_recovery_runner_has_dedicated_experiment() -> None:
  runner = unitree_g1_recovery_ppo_runner_cfg()

  assert runner.experiment_name == "g1_recovery"
  assert runner.max_iterations == 20_000
  assert runner.save_interval == 250