from humanoid_climber.tasks.g1_ice import (
  EVAL_FRICTION,
  TRAIN_FRICTION_RANGE,
  unitree_g1_ice_env_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


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