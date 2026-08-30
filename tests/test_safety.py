import math
from types import SimpleNamespace

import pytest
import torch

from humanoid_climber.safety import (
  CENTERLINE_MAX_OFFSET_M,
  IMBALANCE_CONFIRMATION_FRAMES,
  ImbalanceMonitor,
  outside_centerline,
  predict_imbalance,
)
from humanoid_climber.specialist_policies import RecoveryPolicyAdapter
from humanoid_climber.viewer import HumanoidClimberViserPlayViewer
from humanoid_climber.viewer import GROUND_STUCK_RESET_SECONDS
from humanoid_climber.viewer import SAFETY_REARM_CLEAR_SECONDS
from humanoid_climber.viewer import SPECIALIST_CHECKPOINT_NAME
from humanoid_climber.viewer import _advance_ground_stuck_timer
from humanoid_climber.viewer import _advance_safety_rearm_timer
from humanoid_climber.viewer import _adaptive_specialist_condition
from humanoid_climber.viewer import _apply_centering_command
from humanoid_climber.viewer import _apply_trail_following_command
from humanoid_climber.viewer import _fmt_balance
from humanoid_climber.viewer import _ensure_playback_observation_groups
from humanoid_climber.viewer import _measure_lateral_offset_m
from humanoid_climber.viewer import _recovery_should_release
from humanoid_climber.viewer import _robot_state_is_finite
from humanoid_climber.policy_router import (
  FOUNDATIONAL_POLICY_KEYS,
  SupervisorEventLog,
  TerrainContext,
  resolve_policy_execution,
  route_policy,
)
from humanoid_climber.trail import TrailFrame, nearest_trail_frame, trail_frame_ahead


def _up_vector(tilt_degrees: float) -> tuple[float, float, float]:
  tilt = math.radians(tilt_degrees)
  return math.sin(tilt), 0.0, math.cos(tilt)


def test_integrated_recovery_matches_standalone_start_frame() -> None:
  class Scene(dict):
    env_origins = torch.tensor([[10.0, -4.0, 0.0]])

  robot = SimpleNamespace(
    data=SimpleNamespace(body_link_pos_w=torch.tensor([[[12.0, -3.0, 0.4]]]))
  )
  adapter = object.__new__(RecoveryPolicyAdapter)
  adapter._env = SimpleNamespace(scene=Scene(robot=robot))
  adapter._joint_pos = torch.zeros((401, 29))
  adapter._body_pos = torch.tensor([[[0.5, 0.25, 0.1]]]).repeat(401, 1, 1)
  adapter._anchor_index = 0
  adapter._frame = 200
  adapter._xy_offset = None
  adapter._active = False

  adapter.start(0, torch.zeros((1, 99)))

  assert adapter._frame == 0
  assert torch.allclose(adapter._xy_offset, torch.tensor([1.5, 0.75]))
  assert adapter.active is True


def test_robot_state_finite_watchdog_detects_nan() -> None:
  data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[float("nan"), 0.0, 0.75]]),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    root_link_vel_w=torch.zeros((1, 6)),
    joint_pos=torch.zeros((1, 29)),
    joint_vel=torch.zeros((1, 29)),
  )
  env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}))
  assert _robot_state_is_finite(env, 0) is False

  data.root_link_pos_w[0, 0] = 0.0
  assert _robot_state_is_finite(env, 0) is True


def test_viewer_retains_policy_history_hook() -> None:
  assert callable(HumanoidClimberViserPlayViewer._record_policy_decision)


def test_unknown_combination_can_be_forwarded_to_future_policy_server() -> None:
  context = TerrainContext(
    slope_gradient=0.10,
    friction=0.4,
    roughness_m=0.08,
    step_height_m=0.10,
    wind_force_n=12.0,
    fallen=False,
    active_policy="flat",
  )
  decision = route_policy(context)
  execution = resolve_policy_execution(
    decision,
    available_policy_keys=FOUNDATIONAL_POLICY_KEYS,
    active_policy_key="flat",
  )
  log = SupervisorEventLog(stable_observations=1)
  assert log.observe(context, decision, execution, step=1)
  assert [entry.category for entry in log.entries] == ["TRAINING_REQUIRED"]

  viewer = object.__new__(HumanoidClimberViserPlayViewer)
  requests: list[dict[str, object]] = []
  viewer.env = SimpleNamespace(
    unwrapped=SimpleNamespace(summitos_policy_request_handler=requests.append)
  )
  viewer._active_specialist_key = "flat"

  assert viewer._submit_policy_training_request(context, decision) is True
  assert len(requests) == 1
  assert requests[0]["terrain_type"] == "rough terrain + wind"
  assert requests[0]["friction"] == pytest.approx(0.4)
  assert requests[0]["active_policy"] == "flat"


def test_ground_stuck_watchdog_requires_thirty_continuous_seconds() -> None:
  elapsed = 0.0
  should_reset = False
  for _ in range(1499):
    elapsed, should_reset = _advance_ground_stuck_timer(
      elapsed, fallen=True, step_dt=0.02
    )
  assert elapsed == pytest.approx(29.98)
  assert should_reset is False

  elapsed, should_reset = _advance_ground_stuck_timer(
    elapsed, fallen=True, step_dt=0.02
  )
  assert elapsed == pytest.approx(GROUND_STUCK_RESET_SECONDS)
  assert should_reset is True


def test_finished_getup_policy_stays_latched_until_robot_is_upright() -> None:
  assert _recovery_should_release(policy_finished=True, fallen=True) is False
  assert _recovery_should_release(policy_finished=True, fallen=None) is False
  assert _recovery_should_release(policy_finished=False, fallen=False) is False
  assert _recovery_should_release(policy_finished=True, fallen=False) is True


def test_safety_rearm_requires_one_continuous_clean_window() -> None:
  elapsed = 0.0
  rearmed = False
  steps = round(SAFETY_REARM_CLEAR_SECONDS / 0.02)
  for _ in range(steps - 1):
    elapsed, rearmed = _advance_safety_rearm_timer(
      elapsed,
      centerline_breached=False,
      imbalance_triggered=False,
      step_dt=0.02,
    )
  assert rearmed is False

  # Any recurrence belongs to the same incident and resets the clean window.
  elapsed, rearmed = _advance_safety_rearm_timer(
    elapsed,
    centerline_breached=False,
    imbalance_triggered=True,
    step_dt=0.02,
  )
  assert elapsed == 0.0
  assert rearmed is False

  for _ in range(steps):
    elapsed, rearmed = _advance_safety_rearm_timer(
      elapsed,
      centerline_breached=False,
      imbalance_triggered=False,
      step_dt=0.02,
    )
  assert elapsed == pytest.approx(SAFETY_REARM_CLEAR_SECONDS)
  assert rearmed is True


def test_ground_stuck_watchdog_clears_when_robot_recovers() -> None:
  elapsed, should_reset = _advance_ground_stuck_timer(
    7.5, fallen=False, step_dt=0.02
  )
  assert elapsed == 0.0
  assert should_reset is False

  elapsed, should_reset = _advance_ground_stuck_timer(
    7.5, fallen=None, step_dt=0.02
  )
  assert elapsed == 0.0
  assert should_reset is False


def test_external_auto_reset_clears_latched_four_point_safety_state() -> None:
  viewer = object.__new__(HumanoidClimberViserPlayViewer)
  viewer._imbalance_recovery_latched = True
  viewer._imbalance_risk = object()
  viewer._imbalance_monitor = ImbalanceMonitor()
  viewer._imbalance_monitor.candidate_frames = 3
  viewer._recovery_trigger_reason = "Predicted torso imbalance"
  viewer._recovery_steps_remaining = 50
  viewer._recovery_attack_steps_remaining = 3
  viewer._recovery_command_snapshot = (object(), object(), object())
  viewer._specialist_stage = "recovering"
  viewer._specialist_condition = "ice"
  viewer._specialist_wait_steps_remaining = 20
  viewer._specialist_centered_steps = 7
  viewer._specialist_missing_logged = True
  viewer._safety_rearm_required = True
  viewer._last_context = object()
  viewer._last_decision = object()
  viewer._last_execution = object()
  viewer._routing_context_refresh_steps_remaining = 4
  viewer._routing_context_checkpoint_name = "old.pt"
  viewer._routing_context_event_signature = ("ice",)
  viewer._force_model_field_sync = lambda: None

  viewer._clear_recovery_after_external_reset()

  assert viewer._imbalance_recovery_latched is False
  assert viewer._imbalance_risk is None
  assert viewer._imbalance_monitor.candidate_frames == 0
  assert viewer._recovery_trigger_reason == ""
  assert viewer._recovery_steps_remaining == 0
  assert viewer._recovery_attack_steps_remaining == 0
  assert viewer._recovery_command_snapshot is None
  assert viewer._specialist_stage == "idle"
  assert viewer._specialist_condition is None
  assert viewer._safety_rearm_required is False
  assert viewer._last_context is None


def test_four_point_safety_action_commands_arms_and_lower_limbs() -> None:
  names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
  )
  action_term = SimpleNamespace(
    target_names=names,
    target_ids=torch.arange(len(names)),
    scale=torch.ones(len(names)),
    offset=torch.zeros(len(names)),
  )
  robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros((1, len(names)))))
  base = SimpleNamespace(
    action_manager=SimpleNamespace(get_term=lambda name: action_term),
    scene={"robot": robot},
  )
  viewer = object.__new__(HumanoidClimberViserPlayViewer)
  viewer.env = SimpleNamespace(unwrapped=base)
  viewer._four_point_action = None
  viewer._fast_four_point_action = None

  action = viewer._four_point_safety_action()
  by_name = {name: float(action[0, i]) for i, name in enumerate(names)}

  assert by_name["left_hip_pitch_joint"] == pytest.approx(-0.65)
  assert by_name["left_hip_roll_joint"] == pytest.approx(0.15)
  assert by_name["right_hip_roll_joint"] == pytest.approx(-0.15)
  assert by_name["left_knee_joint"] == pytest.approx(1.15)
  assert by_name["waist_pitch_joint"] == pytest.approx(0.45)
  assert by_name["left_shoulder_pitch_joint"] == pytest.approx(-1.55)
  assert by_name["right_shoulder_pitch_joint"] == pytest.approx(-1.55)
  assert by_name["left_shoulder_roll_joint"] == pytest.approx(0.60)
  assert by_name["right_shoulder_roll_joint"] == pytest.approx(-0.60)
  assert abs(by_name["left_shoulder_roll_joint"]) == pytest.approx(
    abs(by_name["right_shoulder_roll_joint"])
  )
  assert abs(by_name["left_hip_roll_joint"]) == pytest.approx(
    abs(by_name["right_hip_roll_joint"])
  )
  assert by_name["left_elbow_joint"] == pytest.approx(0.65)
  assert by_name["right_elbow_joint"] == pytest.approx(0.65)


def test_playback_observation_guard_keeps_critic_group_intact() -> None:
  manager = SimpleNamespace(
    _group_obs_term_names={"actor": ["actor_term"], "critic": ["critic_term"]},
    cfg={
      "actor": SimpleNamespace(terms={"actor_term": object()}),
      "critic": SimpleNamespace(terms={"critic_term": object()}),
    },
    _obs_buffer={"actor": object(), "critic": object()},
  )
  env = SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager))
  original = dict(manager._group_obs_term_names)

  assert _ensure_playback_observation_groups(env) is True
  assert manager._group_obs_term_names == original
  assert manager._obs_buffer is not None


def test_playback_observation_guard_repairs_missing_critic_group() -> None:
  manager = SimpleNamespace(
    _group_obs_term_names={"actor": ["actor_term"]},
    cfg={
      "actor": SimpleNamespace(terms={"actor_term": object()}),
      "critic": SimpleNamespace(
        terms={"critic_a": object(), "critic_b": object()}
      ),
    },
    _obs_buffer={"actor": object()},
  )
  env = SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager))

  assert _ensure_playback_observation_groups(env) is True
  assert manager._group_obs_term_names["critic"] == ["critic_a", "critic_b"]
  assert manager._obs_buffer is None


def test_upright_robot_does_not_trigger() -> None:
  risk = predict_imbalance(_up_vector(2.0), (0.05, 0.05, 2.0), 2)
  assert risk.triggered is False
  assert risk.tipping_rate_rad_s < 0.1


def test_moderate_stable_tilt_does_not_trigger() -> None:
  risk = predict_imbalance(_up_vector(34.0), (0.1, 0.1, 0.0), 2)
  assert risk.triggered is False


def test_severe_tilt_triggers_without_high_rotation() -> None:
  risk = predict_imbalance(_up_vector(46.0), (0.1, 0.1, 0.0), 1)
  assert risk.triggered is True


def test_dynamic_roll_pitch_tip_triggers() -> None:
  risk = predict_imbalance(_up_vector(32.0), (0.8, 0.1, 4.0), 2)
  assert risk.triggered is True


def test_projected_tip_triggers_before_large_tilt() -> None:
  risk = predict_imbalance(_up_vector(12.0), (0.0, 1.4, 0.0), 2)
  assert risk.tilt_degrees < 18.0
  assert risk.projected_tilt_degrees >= 28.0
  assert risk.triggered is True


def test_fast_gait_motion_while_upright_does_not_predict_a_fall() -> None:
  risk = predict_imbalance(_up_vector(3.0), (0.0, 2.2, 0.0), 2)
  assert risk.projected_tilt_degrees >= 28.0
  assert risk.triggered is False


def test_yaw_rotation_is_not_mistaken_for_tipping() -> None:
  risk = predict_imbalance(_up_vector(32.0), (0.05, 0.05, 8.0), 2)
  assert risk.triggered is False


def test_unsupported_tilted_robot_triggers_at_lower_dynamic_threshold() -> None:
  risk = predict_imbalance(_up_vector(24.0), (0.55, 0.0, 0.0), 0)
  assert risk.triggered is True


def test_normal_double_support_loss_does_not_trigger() -> None:
  risk = predict_imbalance(_up_vector(5.0), (0.2, 0.1, 0.0), 0)
  assert risk.triggered is False


def test_centerline_limit_triggers_only_beyond_one_meter() -> None:
  assert CENTERLINE_MAX_OFFSET_M == 1.0
  assert outside_centerline(0.99) is False
  assert outside_centerline(1.0) is False
  assert outside_centerline(1.01) is True
  assert outside_centerline(-1.01) is True


def test_imbalance_must_persist_before_recovery() -> None:
  monitor = ImbalanceMonitor()
  unstable = predict_imbalance(_up_vector(50.0), (0.0, 0.0, 0.0), 1)
  for _ in range(IMBALANCE_CONFIRMATION_FRAMES - 1):
    assert monitor.observe(unstable) is False
  assert monitor.observe(unstable) is True


def test_stable_sample_resets_imbalance_persistence() -> None:
  monitor = ImbalanceMonitor()
  unstable = predict_imbalance(_up_vector(50.0), (0.0, 0.0, 0.0), 1)
  stable = predict_imbalance(_up_vector(0.0), (0.0, 0.0, 0.0), 2)
  for _ in range(IMBALANCE_CONFIRMATION_FRAMES - 1):
    monitor.observe(unstable)
  assert monitor.observe(stable) is False
  assert monitor.candidate_frames == 0


def test_live_balance_label_distinguishes_monitoring_and_recovery() -> None:
  stable = predict_imbalance(_up_vector(4.0), (0.0, 0.0, 0.0), 2)
  unstable = predict_imbalance(_up_vector(50.0), (0.0, 0.0, 0.0), 1)
  assert _fmt_balance(stable) == "OK 4.0 deg"
  assert _fmt_balance(unstable) == "UNSTABLE 50.0 deg"
  assert _fmt_balance(unstable, recovery_active=True) == "RECOVERY 50.0 deg"


def test_lateral_offset_is_relative_to_environment_winding_trail() -> None:
  class FakeScene(dict):
    env_origins = torch.tensor([[0.0, 5.0, 0.0]])

  robot = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[3.0, 6.2, 0.8]]))
  )
  base = SimpleNamespace(scene=FakeScene(robot=robot))
  env = SimpleNamespace(unwrapped=base)

  offset = _measure_lateral_offset_m(env, 0)
  assert offset is not None
  expected = nearest_trail_frame(3.0, 1.2).lateral_offset_m
  assert math.isclose(offset, expected, abs_tol=1.0e-6)


def test_trail_frame_ahead_places_patch_forward_on_centerline() -> None:
  frame = trail_frame_ahead(0.0, 0.0, 2.5)
  assert frame.center_x == pytest.approx(2.5, abs=0.08)
  assert frame.center_y == pytest.approx(0.0, abs=0.08)
  assert frame.tangent_x > 0.95
  assert abs(frame.tangent_y) < 0.2
  assert frame.lateral_offset_m == 0.0


def test_low_friction_and_slope_select_incline_specialist_condition() -> None:
  assert (
    _adaptive_specialist_condition(
      TerrainContext(friction=0.12, slope_gradient=0.0), ("ice",)
    )
    == "low-friction"
  )
  assert (
    _adaptive_specialist_condition(
      TerrainContext(friction=0.7, slope_gradient=0.12), ("slope",)
    )
    == "slope"
  )
  assert (
    _adaptive_specialist_condition(
      TerrainContext(friction=0.12, slope_gradient=0.12), ("ice", "slope")
    )
    == "low-friction + slope"
  )
  assert (
    _adaptive_specialist_condition(
      TerrainContext(friction=0.7, slope_gradient=0.18), ("bumps",)
    )
    is None
  )


def test_normal_trail_following_ignores_tiny_centerline_and_heading_errors() -> None:
  term = SimpleNamespace(
    is_standing_env=torch.ones(1, dtype=torch.bool),
    is_forward_env=torch.ones(1, dtype=torch.bool),
    is_world_env=torch.zeros(1, dtype=torch.bool),
    vel_command_w=torch.zeros((1, 3)),
    vel_command_b=torch.zeros((1, 3)),
  )
  command_manager = SimpleNamespace(get_term=lambda name: term)
  robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([math.radians(2.0)])))
  base = SimpleNamespace(command_manager=command_manager, scene={"robot": robot})
  env = SimpleNamespace(unwrapped=base)
  frame = TrailFrame(
    center_x=0.0, center_y=0.0, tangent_x=1.0, tangent_y=0.0,
    lateral_offset_m=0.05, distance_m=0.05, segment_index=0,
  )

  _apply_trail_following_command(env, 0, frame)
  assert math.isclose(float(term.vel_command_w[0, 1]), 0.0, abs_tol=1.0e-6)
  assert math.isclose(float(term.vel_command_w[0, 2]), 0.0, abs_tol=1.0e-6)


def test_centering_command_always_points_back_toward_trail_center() -> None:
  term = SimpleNamespace(
    is_standing_env=torch.ones(1, dtype=torch.bool),
    is_forward_env=torch.ones(1, dtype=torch.bool),
    is_world_env=torch.zeros(1, dtype=torch.bool),
    vel_command_w=torch.zeros((1, 3)),
    vel_command_b=torch.zeros((1, 3)),
  )
  command_manager = SimpleNamespace(get_term=lambda name: term)
  robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([0.0])))
  base = SimpleNamespace(command_manager=command_manager, scene={"robot": robot})
  env = SimpleNamespace(unwrapped=base)
  frame = TrailFrame(
    center_x=0.0,
    center_y=0.0,
    tangent_x=1.0,
    tangent_y=0.0,
    lateral_offset_m=1.2,
    distance_m=1.2,
    segment_index=0,
  )

  _apply_centering_command(env, 0, frame)
  assert term.vel_command_w[0, 0] > 0.0
  assert term.vel_command_w[0, 1] < 0.0
  assert abs(float(term.vel_command_w[0, 1])) <= 0.45
  assert bool(term.is_world_env[0]) is True


def test_specialist_checkpoint_hot_swap_requires_a_real_available_file() -> None:
  loaded: list[str] = []
  manager = SimpleNamespace(
    current_name="g1_velocity_model_final.pt",
    fetch_available=lambda: [
      ("g1_velocity_model_final.pt", "now"),
      (SPECIALIST_CHECKPOINT_NAME, "now"),
    ],
    load_checkpoint=lambda name: loaded.append(name) or (lambda obs: obs),
  )
  viewer = object.__new__(HumanoidClimberViserPlayViewer)
  viewer._ckpt_mgr = manager
  viewer.policy = None

  assert viewer._load_checkpoint_in_place(SPECIALIST_CHECKPOINT_NAME)
  assert manager.current_name == SPECIALIST_CHECKPOINT_NAME
  assert loaded == [SPECIALIST_CHECKPOINT_NAME]

  missing_manager = SimpleNamespace(
    current_name="g1_velocity_model_final.pt",
    fetch_available=lambda: [("g1_velocity_model_final.pt", "now")],
  )
  missing_viewer = object.__new__(HumanoidClimberViserPlayViewer)
  missing_viewer._ckpt_mgr = missing_manager
  assert not missing_viewer._checkpoint_is_available(SPECIALIST_CHECKPOINT_NAME)
