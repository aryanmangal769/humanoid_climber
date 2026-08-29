from humanoid_climber.tasks.g1_ice import (
  EVAL_FRICTION,
  EVAL_SLOPE_GRADIENT,
  TRAIN_FRICTION_RANGE,
  unitree_g1_ice_env_cfg,
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
  assert generator.difficulty_range == (1.0, 1.0)

  slope = generator.sub_terrains["ice_slope"]
  assert isinstance(slope, HfPyramidSlopedTerrainCfg)
  assert slope.slope_range == (EVAL_SLOPE_GRADIENT, EVAL_SLOPE_GRADIENT)
  assert slope.inverted
  assert slope.horizontal_scale == 0.1