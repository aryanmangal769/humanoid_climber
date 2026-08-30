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
  SLOPE_PATCH_SEGMENT_COUNT,
  TREADMILL_SIZE,
  TREADMILL_SLOPE_DELTA_PER_STEP,
  unitree_g1_controlled_env_cfg,
  unitree_g1_controlled_ppo_runner_cfg,
  unitree_g1_randomized_env_cfg,
  unitree_g1_randomized_ppo_runner_cfg,
)
from humanoid_climber.safety import CENTERLINE_MAX_OFFSET_M
from humanoid_climber.mdp import advance_infinite_trail, sequential_random_events
from humanoid_climber.trail import TRAIL_SEGMENTS


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


def test_randomized_environment_samples_all_scenario_conditions() -> None:
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
  sequencer = cfg.events["sequential_random_events"]
  assert sequencer.mode == "step"
  assert sequencer.params["normal_friction_range"] == NORMAL_FRICTION_RANGE
  assert sequencer.params["ice_friction_range"] == RANDOM_ICE_FRICTION_RANGE
  assert RANDOM_ICE_FRICTION_RANGE[1] <= 0.04
  assert sequencer.params["wind_force_ranges"] == RANDOM_WIND_FORCE_RANGES
  assert sequencer.params["slope_gradient_range"] == RANDOM_SLOPE_GRADIENT_RANGE
  assert sequencer.params["active_slope_magnitude_range"] == (
    RANDOM_ACTIVE_SLOPE_MAGNITUDE_RANGE
  )
  assert sequencer.params["slope_piece_count"] == SLOPE_PATCH_SEGMENT_COUNT
  assert sequencer.params["slope_outer_fraction_range"] == (
    RANDOM_SLOPE_OUTER_FRACTION_RANGE
  )
  assert sequencer.params["slope_inner_fraction_range"] == (
    RANDOM_SLOPE_INNER_FRACTION_RANGE
  )
  assert sequencer.params["event_patch_ahead_range_m"] == (
    RANDOM_EVENT_PATCH_AHEAD_RANGE_M
  )
  assert sequencer.params["rough_surface_height_range_m"] == (
    RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M
  )
  assert sequencer.params["rough_forward_range_m"] == RANDOM_ROUGH_FORWARD_RANGE_M
  assert sequencer.params["rough_lateral_range_m"] == RANDOM_ROUGH_LATERAL_RANGE_M
  assert sequencer.params["rock_height_range_m"] == RANDOM_ROCK_HEIGHT_RANGE_M
  assert sequencer.params["rock_count"] == TREADMILL_ROCK_COUNT
  assert sequencer.params["rough_rows"] == TREADMILL_ROUGH_ROWS
  assert sequencer.params["rough_cols"] == TREADMILL_ROUGH_COLS
  assert sequencer.params["event_duration_range_s"] == RANDOM_EVENT_DURATION_S
  assert sequencer.params["break_duration_range_s"] == (
    RANDOM_EVENT_BREAK_DURATION_S
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


def test_random_event_controller_reports_the_physical_active_condition() -> None:
  controller = object.__new__(sequential_random_events)
  controller._manual_mode = torch.tensor([False])
  controller._phase = torch.tensor([sequential_random_events.ICE])
  controller._manual_events = torch.zeros(
    (1, len(sequential_random_events.MANUAL_EVENT_NAMES)), dtype=torch.bool
  )
  assert controller.active_event_names(0) == ("ice",)

  controller._phase[0] = sequential_random_events.SLOPE_CLEARING
  assert controller.active_event_names(0) == ("slope",)

  controller._manual_mode[0] = True
  ice_idx = sequential_random_events.MANUAL_EVENT_NAMES.index("ice")
  slope_idx = sequential_random_events.MANUAL_EVENT_NAMES.index("slope")
  controller._manual_events[0, ice_idx] = True
  controller._manual_events[0, slope_idx] = True
  assert controller.active_event_names(0) == ("ice", "slope")


def test_random_slope_profile_uses_multiple_continuous_grades() -> None:
  controller = object.__new__(sequential_random_events)
  controller._env = type("Env", (), {"device": torch.device("cpu")})()
  controller._params = {
    "slope_outer_fraction_range": (0.55, 0.55),
    "slope_inner_fraction_range": (0.90, 0.90),
  }
  controller._slope_profile_factors = torch.zeros((1, 4))

  controller._sample_slope_profile(torch.tensor([0], dtype=torch.long))

  profile = controller._slope_profile_factors[0]
  assert profile.tolist() == pytest.approx([0.55, 0.90, -0.90, -0.55])
  assert abs(float(profile[0])) != pytest.approx(abs(float(profile[1])))
  assert float(profile.sum()) == pytest.approx(0.0, abs=1.0e-6)


def test_automatic_break_does_not_remove_spawned_terrain_patches() -> None:
  controller = object.__new__(sequential_random_events)
  controller._phase = torch.tensor([sequential_random_events.ICE])
  controller._params = {
    "normal_friction_range": (0.65, 1.0),
    "break_duration_range_s": (2.0, 5.0),
  }
  calls: list[str] = []
  controller._clear_wind = lambda env_ids: calls.append("wind")
  controller._set_friction = lambda env_ids, friction_range: calls.append("friction")
  controller._sample_duration = lambda env_ids, duration_range: calls.append("duration")
  controller._hide_event_overlays = lambda env_ids: calls.append("hide_overlays")
  controller._hide_rough_ground = lambda env_ids: calls.append("hide_rough")

  controller._set_break(torch.tensor([0]))

  assert int(controller._phase[0]) == sequential_random_events.BREAK
  assert calls == ["wind", "friction", "duration"]


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


def test_randomized_environment_samples_robot_and_sensor_parameters() -> None:
  cfg = unitree_g1_randomized_env_cfg()

  assert cfg.rewards
  assert cfg.metrics
  assert cfg.terminations

  assert cfg.events["reset_base"].params == {
    "pose_range": RANDOM_ROOT_POSE_RANGES,
    "velocity_range": RANDOM_ROOT_VELOCITY_RANGES,
  }
  assert cfg.events["reset_robot_joints"].params["position_range"] == (
    RANDOM_JOINT_POSITION_RANGE
  )
  assert cfg.events["reset_robot_joints"].params["velocity_range"] == (
    RANDOM_JOINT_VELOCITY_RANGE
  )
  assert cfg.events["encoder_bias"].params["bias_range"] == (
    RANDOM_ENCODER_BIAS_RANGE
  )
  assert cfg.events["inertial_properties"].params["alpha_range"] == (
    RANDOM_DENSITY_LOG_SCALE_RANGE
  )
  assert cfg.events["joint_damping"].params["ranges"] == (
    RANDOM_JOINT_DAMPING_SCALE_RANGE
  )
  assert cfg.events["joint_friction"].params["ranges"] == (
    RANDOM_JOINT_FRICTION_SCALE_RANGE
  )
  assert cfg.events["joint_armature"].params["ranges"] == (
    RANDOM_ARMATURE_SCALE_RANGE
  )
  assert cfg.events["pd_gains"].params["kp_range"] == (
    RANDOM_PD_GAIN_SCALE_RANGE
  )
  assert cfg.events["pd_gains"].params["kd_range"] == (
    RANDOM_PD_GAIN_SCALE_RANGE
  )

  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  for actuator in robot.articulation.actuators:
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (
      RANDOM_ACTION_DELAY_STEPS
    )


def test_all_conditions_runners_use_separate_experiments() -> None:
  controlled = unitree_g1_controlled_ppo_runner_cfg()
  randomized = unitree_g1_randomized_ppo_runner_cfg()

  assert controlled.experiment_name == "g1_controlled"
  assert randomized.experiment_name == "g1_randomized"
  assert controlled.experiment_name != randomized.experiment_name
  assert randomized.max_iterations == 10_000
