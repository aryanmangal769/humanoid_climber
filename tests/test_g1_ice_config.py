from humanoid_climber.tasks.g1_ice import (
  EVAL_FRICTION,
  EVAL_SLOPE_GRADIENT,
  EVAL_TERRAIN_SIZE,
  TRAIN_FRICTION_RANGE,
  TRAIN_SLOPE_RANGE,
  TRAIN_TERRAIN_SIZE,
  unitree_g1_ice_env_cfg,
  unitree_g1_ice_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.terrains import HfPyramidSlopedTerrainCfg


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


def test_ice_runner_has_dedicated_experiment() -> None:
  runner = unitree_g1_ice_ppo_runner_cfg()
  assert runner.experiment_name == "g1_ice"
  assert runner.max_iterations == 5_000