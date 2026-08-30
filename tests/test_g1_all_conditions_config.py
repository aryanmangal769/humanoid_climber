import math

import mujoco
import pytest
import torch

from mjlab.scene import Scene
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from humanoid_climber.tasks.g1_all_conditions import (
  CONTROLLED_FORWARD_SPEED,
  CONTROLLED_FRICTION,
  CONTROLLED_SEED,
  CONTROLLED_WIND_FORCE_RANGES,
  EVENT_PATCH_HALF_WIDTH_M,
  HAZARD_WIDTH_MULTIPLIER,
  ICE_PATCH_LENGTH_M,
  ICE_PATCH_HALF_WIDTH_M,
  RANDOM_ACTION_DELAY_STEPS,
  RANDOM_ARMATURE_SCALE_RANGE,
  RANDOM_DENSITY_LOG_SCALE_RANGE,
  RANDOM_ENCODER_BIAS_RANGE,
  RANDOM_EVENT_BREAK_DURATION_S,
  RANDOM_EVENT_DURATION_S,
  RANDOM_EVENT_PATCH_AHEAD_RANGE_M,
  RANDOM_ICE_FRICTION_RANGE,
  RANDOM_JOINT_DAMPING_SCALE_RANGE,
  RANDOM_JOINT_FRICTION_SCALE_RANGE,
  RANDOM_JOINT_POSITION_RANGE,
  RANDOM_JOINT_VELOCITY_RANGE,
  RANDOM_PD_GAIN_SCALE_RANGE,
  NORMAL_FRICTION_RANGE,
  PLAYBACK_JOINT_POSITION_RANGE,
  PLAYBACK_JOINT_VELOCITY_RANGE,
  PLAYBACK_ROOT_POSE_RANGES,
  PLAYBACK_ROOT_VELOCITY_RANGES,
  RANDOM_ROOT_POSE_RANGES,
  RANDOM_ROOT_VELOCITY_RANGES,
  RANDOM_ACTIVE_SLOPE_MAGNITUDE_RANGE,
  RANDOM_SLOPE_INNER_FRACTION_RANGE,
  RANDOM_SLOPE_OUTER_FRACTION_RANGE,
  RANDOM_ROCK_HEIGHT_RANGE_M,
  RANDOM_ROUGH_FORWARD_RANGE_M,
  RANDOM_ROUGH_LATERAL_RANGE_M,
  RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M,
  RANDOM_SLOPE_GRADIENT_RANGE,
  RANDOM_WIND_FORCE_RANGES,
  TREADMILL_FORWARD_SPEED,
  TREADMILL_ROCK_COUNT,
  TREADMILL_ROUGH_CLEAR_DELTA_PER_STEP_M,
  TREADMILL_ROUGH_COLS,
  TREADMILL_ROUGH_MOUND_COUNT,
  TREADMILL_ROUGH_ROWS,
  SLOPE_PATCH_LENGTH_M,
  SLOPE_PATCH_HALF_WIDTH_M,
  SLOPE_PATCH_SEGMENT_COUNT,
  TREADMILL_SIZE,
  TREADMILL_SLOPE_DELTA_PER_STEP,
  FUTURE_SNOW_EVENT_COLOR,
  HIDDEN_TRAIL_RGBA,
  MOUNTAIN_COUNT,
  MOUNTAIN_RIDGE_GEOM_NAMES,
  SNOW_SURFACE_TILE,
  WINTER_GROUND_BASE_COLOR,
  WINTER_GROUND_MATERIAL_NAME,
  WINTER_GROUND_TEXTURE_NAME,
  unitree_g1_controlled_env_cfg,
  unitree_g1_controlled_ppo_runner_cfg,
  unitree_g1_randomized_env_cfg,
  unitree_g1_randomized_ppo_runner_cfg,
)
from humanoid_climber.safety import CENTERLINE_MAX_OFFSET_M
from humanoid_climber.mdp import advance_infinite_trail, orchestrated_policy_sequence
from humanoid_climber.orchestrator import (
  HIGH_WIND_FORCE_RANGES,
  INCLINE_ANGLE_RANGE_DEG,
  INCLINE_FRICTION_RANGE,
  INCLINE_GRADIENT_RANGE,
  POLICY_ANNOUNCEMENT_DELAY_SECONDS,
  SHOWCASE_STAGES,
  STAGE_DURATION_SECONDS,
  ShowcaseClock,
)
from humanoid_climber.trail import TRAIL_HALF_WIDTH_M, TRAIL_SEGMENTS


def _assert_stock_policy_interface(cfg) -> None:
  stock = unitree_g1_flat_env_cfg(play=True)
  assert tuple(cfg.observations["actor"].terms) == tuple(
    stock.observations["actor"].terms
  )
  assert tuple(cfg.observations["critic"].terms) == tuple(
    stock.observations["critic"].terms
  )
  assert tuple(cfg.actions) == tuple(stock.actions)


def test_infinite_trail_skips_nonfinite_robot_position() -> None:
  recycler = object.__new__(advance_infinite_trail)
  recycler._robot = type(
    "Robot",
    (),
    {"data": type("Data", (), {"root_link_pos_w": torch.tensor([[float("nan"), 0.0, 0.0]])})()},
  )()
  recycler._env = type(
    "Env",
    (),
    {"scene": type("Scene", (), {"env_origins": torch.zeros((1, 3))})()},
  )()
  assert recycler._unpitched_local_x(0) is None


def test_controlled_environment_pins_every_condition() -> None:
  cfg = unitree_g1_controlled_env_cfg()
  _assert_stock_policy_interface(cfg)

  assert cfg.scene.terrain is not None
  assert cfg.seed == CONTROLLED_SEED
  assert cfg.scene.terrain.terrain_type == "generator"
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  assert terrain.size == TREADMILL_SIZE
  assert terrain.num_rows == 1
  assert terrain.num_cols == 1
  assert tuple(terrain.sub_terrains) == ("treadmill",)
  assert "randomize_terrain" not in cfg.events

  assert cfg.events["foot_friction"].params["ranges"] == (
    CONTROLLED_FRICTION,
    CONTROLLED_FRICTION,
  )
  assert cfg.events["wind"].params["force_ranges"] == (
    CONTROLLED_WIND_FORCE_RANGES
  )
  assert "push_robot" not in cfg.events
  assert "encoder_bias" not in cfg.events
  assert cfg.observations["actor"].enable_corruption is False

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (
    CONTROLLED_FORWARD_SPEED,
    CONTROLLED_FORWARD_SPEED,
  )
  assert command.ranges.lin_vel_y == (-0.1, 0.1)
  assert command.ranges.ang_vel_z == (-0.1, 0.1)
  assert command.rel_forward_envs == 1.0
  assert command.heading_command is False
  assert command.ranges.heading is None


def test_showcase_environment_pins_the_requested_sequence() -> None:
  cfg = unitree_g1_randomized_env_cfg()
  _assert_stock_policy_interface(cfg)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "generator"
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  assert terrain.size == TREADMILL_SIZE
  assert terrain.num_rows == 1
  assert terrain.num_cols == 1
  assert tuple(terrain.sub_terrains) == ("treadmill",)
  assert "randomize_terrain" not in cfg.events
  assert "foot_friction" not in cfg.events
  assert "wind" not in cfg.events
  assert "push_robot" not in cfg.events

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (TREADMILL_FORWARD_SPEED, TREADMILL_FORWARD_SPEED)
  assert command.ranges.lin_vel_y == (-0.1, 0.1)
  assert command.ranges.ang_vel_z == (-0.1, 0.1)
  assert command.rel_forward_envs == 1.0
  assert command.heading_command is False
  assert set(cfg.events) == {
    "reset_base",
    "reset_robot_joints",
    "advance_infinite_trail",
    "orchestrated_policy_sequence",
  }
  sequencer = cfg.events["orchestrated_policy_sequence"]
  assert sequencer.mode == "step"
  assert sequencer.func is orchestrated_policy_sequence
  assert sequencer.params["normal_friction_range"] == (0.8, 0.8)
  assert sequencer.params["ice_friction_range"] == INCLINE_FRICTION_RANGE
  assert INCLINE_FRICTION_RANGE == (0.1, 0.3)
  assert INCLINE_ANGLE_RANGE_DEG == (10.0, 30.0)
  assert INCLINE_GRADIENT_RANGE == pytest.approx(
    (math.tan(math.radians(10.0)), math.tan(math.radians(30.0)))
  )
  assert "wind_ice_friction_range" not in sequencer.params
  assert sequencer.params["wind_force_ranges"] == HIGH_WIND_FORCE_RANGES
  assert HIGH_WIND_FORCE_RANGES == {
    "x": (0.0, 0.0),
    "y": (8.0, 20.0),
    "z": (0.0, 0.0),
  }
  assert sequencer.params["slope_gradient_range"] == INCLINE_GRADIENT_RANGE
  assert sequencer.params["active_slope_magnitude_range"] == INCLINE_GRADIENT_RANGE
  assert sequencer.params["slope_piece_count"] == SLOPE_PATCH_SEGMENT_COUNT
  assert sequencer.params["slope_outer_fraction_range"] == (
    RANDOM_SLOPE_OUTER_FRACTION_RANGE
  )
  assert sequencer.params["slope_inner_fraction_range"] == (
    RANDOM_SLOPE_INNER_FRACTION_RANGE
  )
  assert sequencer.params["event_patch_ahead_range_m"] == (2.0, 2.0)
  assert sequencer.params["rough_surface_height_range_m"] == (
    RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M
  )
  assert sequencer.params["rough_forward_range_m"] == RANDOM_ROUGH_FORWARD_RANGE_M
  assert sequencer.params["rough_lateral_range_m"] == RANDOM_ROUGH_LATERAL_RANGE_M
  assert sequencer.params["rock_height_range_m"] == RANDOM_ROCK_HEIGHT_RANGE_M
  assert sequencer.params["rock_count"] == TREADMILL_ROCK_COUNT
  assert sequencer.params["rough_rows"] == TREADMILL_ROUGH_ROWS
  assert sequencer.params["rough_cols"] == TREADMILL_ROUGH_COLS
  assert sequencer.params["event_duration_range_s"] == (
    STAGE_DURATION_SECONDS,
    STAGE_DURATION_SECONDS,
  )
  assert sequencer.params["policy_announcement_delay_s"] == (
    POLICY_ANNOUNCEMENT_DELAY_SECONDS
  )
  assert sequencer.params["max_delta_per_step"] == (
    TREADMILL_SLOPE_DELTA_PER_STEP
  )
  assert sequencer.params["rough_clear_delta_per_step_m"] == (
    TREADMILL_ROUGH_CLEAR_DELTA_PER_STEP_M
  )


def test_randomized_environment_stays_inside_router_envelope() -> None:
  assert RANDOM_ICE_FRICTION_RANGE[0] >= 0.005
  assert max(abs(value) for value in RANDOM_SLOPE_GRADIENT_RANGE) <= 0.20
  assert RANDOM_ACTIVE_SLOPE_MAGNITUDE_RANGE[1] <= 0.20
  assert RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M[1] <= 0.10
  # The local terrain probe can classify a box height jump as both step height
  # and roughness, so keep randomized rocks at or below the 0.10 m roughness
  # envelope as well as below the nominal 0.15 m step limit.
  assert RANDOM_ROCK_HEIGHT_RANGE_M[1] <= 0.10

  worst_case_wind_n = sum(
    max(abs(low), abs(high)) ** 2
    for low, high in RANDOM_WIND_FORCE_RANGES.values()
  ) ** 0.5
  assert worst_case_wind_n <= 18.0


def test_showcase_clock_announces_one_second_after_physical_stage() -> None:
  clock = ShowcaseClock(stage_duration_s=12.0, announcement_delay_s=1.0)
  assert clock.current.key == "normal"
  assert clock.requested_policy.key == "normal"
  assert clock.announcement_ready is False
  assert clock.advance(1.0) is False
  assert clock.announcement_ready is True
  assert clock.advance(11.0) is True
  assert clock.current.key == "incline"
  assert clock.requested_policy.key == "incline"
  assert clock.announcement_ready is False
  assert clock.advance(0.99) is False
  assert clock.announcement_ready is False
  assert clock.advance(0.01) is False
  assert clock.announcement_ready is True


def test_showcase_order_is_fixed_and_repeats() -> None:
  assert tuple(stage.key for stage in SHOWCASE_STAGES) == (
    "normal",
    "incline",
    "wind",
    "rough",
  )
  clock = ShowcaseClock(stage_duration_s=1.0, announcement_delay_s=0.1)
  observed = [clock.current.key]
  for _ in range(4):
    assert clock.advance(1.0) is True
    observed.append(clock.current.key)
  assert observed == ["normal", "incline", "wind", "rough", "normal"]


def test_showcase_clock_pauses_during_safety_recovery() -> None:
  clock = ShowcaseClock(stage_duration_s=12.0, announcement_delay_s=1.0)
  clock.advance(3.0)
  clock.pause()
  assert clock.advance(30.0) is False
  assert clock.current.key == "normal"
  assert clock.time_remaining_s == pytest.approx(9.0)
  clock.resume()
  assert clock.advance(9.0) is True
  assert clock.current.key == "incline"


def test_treadmill_play_mode_uses_one_long_flat_strip() -> None:
  cfg = unitree_g1_randomized_env_cfg(play=True)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "generator"
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  assert terrain.size == TREADMILL_SIZE
  assert terrain.num_rows == 1
  assert terrain.num_cols == 1
  assert tuple(terrain.sub_terrains) == ("treadmill",)
  assert "randomize_terrain" not in cfg.events
  assert cfg.rewards == {}
  assert cfg.metrics == {}
  assert cfg.terminations == {}
  assert cfg.events["reset_base"].params == {
    "pose_range": PLAYBACK_ROOT_POSE_RANGES,
    "velocity_range": PLAYBACK_ROOT_VELOCITY_RANGES,
  }
  assert cfg.events["reset_robot_joints"].params["position_range"] == (
    PLAYBACK_JOINT_POSITION_RANGE
  )
  assert cfg.events["reset_robot_joints"].params["velocity_range"] == (
    PLAYBACK_JOINT_VELOCITY_RANGE
  )

  # The strip must be a real movable body. A fixed world geom can have its
  # collision transform patched, but Viser will continue drawing it flat.
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  terrain_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "terrain"
  )
  assert terrain_body_id >= 0
  assert model.body_mocapid[terrain_body_id] >= 0
  marker_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    for geom_id in range(model.ngeom)
  }
  assert "treadmill_pitch_marker_0" in marker_names
  ground_material_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_MATERIAL,
    WINTER_GROUND_MATERIAL_NAME,
  )
  ground_texture_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_TEXTURE,
    WINTER_GROUND_TEXTURE_NAME,
  )
  assert ground_material_id >= 0
  assert ground_texture_id >= 0
  assert tuple(FUTURE_SNOW_EVENT_COLOR[:3]) != WINTER_GROUND_BASE_COLOR
  assert sum(
    (snow - ground) ** 2
    for snow, ground in zip(FUTURE_SNOW_EVENT_COLOR[:3], WINTER_GROUND_BASE_COLOR)
  ) ** 0.5 > 0.5

  hidden_visual_names = [
    name
    for name in marker_names
    if name is not None
    and (
      name.startswith("treadmill_trail_surface_")
      or name.startswith("treadmill_pitch_marker_")
      or name.startswith("treadmill_centerline_safety_")
    )
  ]
  assert hidden_visual_names
  for name in hidden_visual_names:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert tuple(model.geom_rgba[geom_id]) == pytest.approx(HIDDEN_TRAIL_RGBA)
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0
  for segment in TRAIL_SEGMENTS:
    for side in ("left", "right"):
      name = f"treadmill_centerline_safety_{side}_{segment.index:03d}"
      geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
      assert geom_id >= 0
      assert model.geom_contype[geom_id] == 0
      assert model.geom_conaffinity[geom_id] == 0

  ice_patch_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "treadmill_ice_patch"
  )
  assert ice_patch_id >= 0
  assert model.geom_type[ice_patch_id] == mujoco.mjtGeom.mjGEOM_BOX
  ice_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "treadmill_ice_patch_body"
  )
  assert ice_body_id >= 0
  assert model.body_mocapid[ice_body_id] >= 0
  assert model.body_pos[ice_body_id, 2] < -0.7
  assert model.geom_pos[ice_patch_id, 2] == pytest.approx(0.0)
  assert model.geom_rgba[ice_patch_id, 3] > 0.7
  assert model.geom_contype[ice_patch_id] == 1
  assert model.geom_conaffinity[ice_patch_id] == 1
  assert model.geom_priority[ice_patch_id] == 2
  assert model.geom_friction[ice_patch_id, 0] == pytest.approx(
    RANDOM_ICE_FRICTION_RANGE[1]
  )
  assert model.geom_friction[ice_patch_id, 1] <= 0.0005
  assert model.geom_friction[ice_patch_id, 2] <= 0.00001
  assert model.geom_size[ice_patch_id, 0] == pytest.approx(
    ICE_PATCH_LENGTH_M * 0.5
  )
  assert model.geom_size[ice_patch_id, 1] == pytest.approx(
    ICE_PATCH_HALF_WIDTH_M
  )
  assert ICE_PATCH_HALF_WIDTH_M > TRAIL_HALF_WIDTH_M
  assert HAZARD_WIDTH_MULTIPLIER == 8.0
  assert ICE_PATCH_HALF_WIDTH_M == pytest.approx(
    EVENT_PATCH_HALF_WIDTH_M * HAZARD_WIDTH_MULTIPLIER
  )

  for piece in range(SLOPE_PATCH_SEGMENT_COUNT):
    name = f"treadmill_slope_patch_{piece:02d}"
    slope_patch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert slope_patch_id >= 0
    assert model.geom_type[slope_patch_id] == mujoco.mjtGeom.mjGEOM_BOX
    slope_body_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_body"
    )
    assert slope_body_id >= 0
    assert model.body_mocapid[slope_body_id] >= 0
    assert model.body_pos[slope_body_id, 2] < -0.7
    assert model.geom_pos[slope_patch_id, 2] == pytest.approx(0.0)
    assert model.geom_rgba[slope_patch_id, 3] > 0.8
    assert model.geom_contype[slope_patch_id] == 1
    assert model.geom_conaffinity[slope_patch_id] == 1
    assert model.geom_size[slope_patch_id, 0] == pytest.approx(
      SLOPE_PATCH_LENGTH_M / (2.0 * SLOPE_PATCH_SEGMENT_COUNT)
    )
    assert model.geom_size[slope_patch_id, 1] == pytest.approx(
      SLOPE_PATCH_HALF_WIDTH_M
    )
    assert SLOPE_PATCH_HALF_WIDTH_M == pytest.approx(
      EVENT_PATCH_HALF_WIDTH_M * HAZARD_WIDTH_MULTIPLIER
    )

  assert not any(
    name is not None and name.startswith("treadmill_ice_overlay_")
    for name in marker_names
  )
  assert not any(
    name is not None and name.startswith("treadmill_slope_gradient_")
    for name in marker_names
  )
  mound_names = sorted(
    name
    for name in marker_names
    if name is not None and name.startswith("treadmill_rough_mound_")
  )
  rock_names = sorted(
    name
    for name in marker_names
    if name is not None and name.startswith("treadmill_rough_rock_")
  )
  assert len(mound_names) == TREADMILL_ROUGH_MOUND_COUNT
  assert len(rock_names) == TREADMILL_ROCK_COUNT
  assert RANDOM_ROUGH_FORWARD_RANGE_M == pytest.approx((-1.25, 1.25))
  assert RANDOM_ROUGH_LATERAL_RANGE_M == pytest.approx((-13.12, 13.12))
  assert TREADMILL_ROUGH_ROWS == 6
  assert TREADMILL_ROUGH_COLS == 80
  assert TREADMILL_ROUGH_MOUND_COUNT == 480
  assert TREADMILL_ROCK_COUNT == 128
  assert SNOW_SURFACE_TILE.is_file()
  assert MOUNTAIN_COUNT == 8
  assert any("front" in name for name in MOUNTAIN_RIDGE_GEOM_NAMES)
  assert any("rear" in name for name in MOUNTAIN_RIDGE_GEOM_NAMES)
  for name in MOUNTAIN_RIDGE_GEOM_NAMES:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0
    assert model.geom_group[geom_id] == 2
  rough_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "treadmill_rough_patch_body"
  )
  assert rough_body_id >= 0
  assert model.body_mocapid[rough_body_id] >= 0
  assert model.body_pos[rough_body_id, 2] < -0.7
  for name in mound_names:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    body_id = int(model.geom_bodyid[geom_id])
    assert model.body_mocapid[body_id] >= 0
    assert model.body_pos[body_id, 2] < -0.7
    assert model.geom_pos[geom_id, 2] == pytest.approx(0.0)
    assert model.geom_rgba[geom_id, 3] == pytest.approx(1.0)
    assert model.geom_contype[geom_id] == 1
    assert model.geom_conaffinity[geom_id] == 1
  for name in rock_names:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    body_id = int(model.geom_bodyid[geom_id])
    assert model.body_mocapid[body_id] >= 0
    assert model.body_pos[body_id, 2] < -0.7
    assert model.geom_pos[geom_id, 2] == pytest.approx(0.0)
    assert model.geom_rgba[geom_id, 3] == pytest.approx(1.0)
    assert model.geom_contype[geom_id] == 1
    assert model.geom_conaffinity[geom_id] == 1


def test_showcase_removes_unrelated_randomization() -> None:
  cfg = unitree_g1_randomized_env_cfg()

  assert cfg.rewards
  assert cfg.metrics
  assert cfg.terminations

  assert cfg.events["reset_base"].params == {
    "pose_range": PLAYBACK_ROOT_POSE_RANGES,
    "velocity_range": PLAYBACK_ROOT_VELOCITY_RANGES,
  }
  assert cfg.events["reset_robot_joints"].params["position_range"] == (
    PLAYBACK_JOINT_POSITION_RANGE
  )
  assert cfg.events["reset_robot_joints"].params["velocity_range"] == (
    PLAYBACK_JOINT_VELOCITY_RANGE
  )
  assert not any(
    name in cfg.events
    for name in (
      "encoder_bias",
      "inertial_properties",
      "joint_damping",
      "joint_friction",
      "joint_armature",
      "pd_gains",
      "push_robot",
    )
  )


def test_all_conditions_runners_use_separate_experiments() -> None:
  controlled = unitree_g1_controlled_ppo_runner_cfg()
  randomized = unitree_g1_randomized_ppo_runner_cfg()

  assert controlled.experiment_name == "g1_controlled"
  assert randomized.experiment_name == "g1_randomized"
  assert controlled.experiment_name != randomized.experiment_name
  assert randomized.max_iterations == 10_000
