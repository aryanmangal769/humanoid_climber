"""Tests for the demo policy-routing heuristic."""

import math
from types import SimpleNamespace

import pytest
import torch

from humanoid_climber import cli
from humanoid_climber.policy_router import (
    RouterAction,
    SupervisorEventLog,
    TerrainContext,
    canonical_policy_key,
    resolve_policy_execution,
    route_policy,
)
from humanoid_climber.safety import ImbalanceRisk
from humanoid_climber.viewer import (
    _imbalance_recovery_decision,
    _motion_telemetry_html,
    _summitos_preview_message,
    _summitos_training_request_message,
    render_policy_overlay,
)


def test_motion_profile_separates_current_environment_and_command_velocity() -> None:
    command = SimpleNamespace(vel_command_b=torch.tensor([[0.5, -0.1, 0.2]]))
    controller = SimpleNamespace(
        stage=SimpleNamespace(key="incline", label="Low-friction incline"),
        upcoming_stage=SimpleNamespace(key="wind", label="High crosswind"),
        requested_policy_key="ice_incline",
        policy_announcement_ready=True,
        time_remaining_s=4.0,
        active_surface_friction=lambda _env_idx: 0.37,
    )
    base = SimpleNamespace(
        showcase_controller=controller,
        random_event_controller=controller,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        treadmill_slope_gradient=torch.tensor([0.126]),
    )

    html = _motion_telemetry_html(SimpleNamespace(unwrapped=base), 0)

    assert "CURRENT ENVIRONMENT" in html
    assert "UPCOMING ENVIRONMENT" not in html
    assert "Low-friction incline" in html
    assert "High crosswind" not in html
    assert "FRICTION" in html and "μ 0.37" in html
    assert "INCLINE" in html and "7.2°" in html
    assert "COMMAND VELOCITY" in html
    assert "+0.50 m/s" in html


def test_motion_profile_shows_sampled_crosswind_with_base_friction() -> None:
    command = SimpleNamespace(vel_command_b=torch.tensor([[0.5, 0.0, 0.0]]))
    controller = SimpleNamespace(
        stage=SimpleNamespace(key="wind", label="Variable crosswind"),
        upcoming_stage=SimpleNamespace(key="rough", label="Rough terrain"),
        requested_policy_key="wind",
        policy_announcement_ready=True,
        time_remaining_s=4.0,
        active_surface_friction=lambda _env_idx: None,
    )
    wrench = torch.zeros((1, 2, 6))
    wrench[0, 1, :3] = torch.tensor([-2.5, 11.75, 0.0])
    robot = SimpleNamespace(
        data=SimpleNamespace(body_external_wrench=wrench)
    )
    base = SimpleNamespace(
        showcase_controller=controller,
        random_event_controller=controller,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        scene={"robot": robot},
    )

    html = _motion_telemetry_html(SimpleNamespace(unwrapped=base), 0)

    assert "Variable crosswind" in html
    assert "X -2.5 · Y +11.8 N" in html
    assert "Normal" in html
    assert "Level" in html


def test_flat_nominal_selects_stock_walker() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.01,
            friction=0.8,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=0.0,
            fallen=False,
        )
    )
    assert decision.target_key == "flat"
    assert decision.action == RouterAction.USE_POLICY


def test_very_low_friction_ice_still_selects_ice_specialist() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.0,
            friction=0.02,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=0.0,
            fallen=False,
        )
    )
    assert decision.target_key == "ice_incline"
    assert decision.action == RouterAction.USE_POLICY


def test_off_path_recovery_log_does_not_report_imbalance() -> None:
    context = TerrainContext(
        slope_gradient=0.0,
        friction=0.8,
        roughness_m=0.0,
        step_height_m=0.0,
        wind_force_n=0.0,
        fallen=False,
    )
    baseline = route_policy(context)
    risk = ImbalanceRisk(
        tilt_degrees=2.0,
        tipping_rate_rad_s=0.0,
        projected_tilt_degrees=2.0,
        feet_in_contact=2,
        triggered=False,
    )
    reason = "Left the path. Lateral offset +1.20 m is outside the ±1.0 m boundary."
    decision = _imbalance_recovery_decision(baseline, risk, reason=reason)
    html = render_policy_overlay(
        context,
        decision,
        recovery_active=True,
        recovery_reason=reason,
        current_step=42,
    )

    assert decision.terrain_type == "off_path"
    assert "Left the path" in html
    assert "Imbalance detected" not in html
    assert "rgba(245,185,66,.16)" in html
    assert "border-left:4px solid #f5b942" in html


def test_active_safety_banner_stays_visible_above_existing_action_log() -> None:
    context = TerrainContext(
        slope_gradient=0.0,
        friction=0.8,
        wind_force_n=0.0,
        fallen=False,
    )
    decision = route_policy(context)
    log = SupervisorEventLog(stable_observations=1)
    execution = resolve_policy_execution(
        decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert log.observe(context, decision, execution, step=10) is True

    html = render_policy_overlay(
        context,
        decision,
        execution=execution,
        log_entries=tuple(log.entries),
        recovery_active=True,
        recovery_reason="Protective recovery active",
        current_step=11,
    )

    assert "SAFETY" in html
    assert "Protective recovery active" in html
    assert "rgba(245,185,66,.16)" in html
    assert "The terrain looks like flat / nominal terrain" in html


def test_low_friction_incline_selects_available_ice_policy() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.18,
            friction=0.15,
            roughness_m=0.01,
            step_height_m=0.0,
            wind_force_n=2.0,
            fallen=False,
            active_policy="g1_velocity_model_final.pt",
        )
    )
    assert decision.target_key == "ice_incline"
    assert decision.action == RouterAction.SWITCH_POLICY
    assert decision.training_request is None


def test_thirty_degree_incline_allows_small_sensor_roundoff() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=math.tan(math.radians(30.0)) + 0.0005,
            friction=0.8,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=0.0,
            fallen=False,
        )
    )
    assert decision.target_key == "ice_incline"
    assert decision.action == RouterAction.USE_POLICY
    assert decision.training_request is None


def test_flat_low_friction_is_identified_as_ice() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.0,
            friction=0.15,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=0.0,
            fallen=False,
        )
    )
    assert decision.terrain_type == "low-friction ice"
    assert decision.target_key == "ice_incline"


def test_wind_section_selects_foundational_wind_policy() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.01,
            friction=0.6,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=14.0,
            fallen=False,
        )
    )
    assert decision.target_key == "wind"
    assert decision.action == RouterAction.USE_POLICY
    assert decision.training_request is None


def test_flat_low_friction_wind_uses_wind_specialist_target() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.01,
            friction=0.15,
            roughness_m=0.0,
            step_height_m=0.0,
            wind_force_n=16.0,
            fallen=False,
        )
    )
    assert decision.target_key == "wind"
    assert decision.action == RouterAction.USE_POLICY


def test_rough_section_selects_foundational_rough_policy() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.10,
            friction=0.5,
            roughness_m=0.07,
            step_height_m=0.10,
            wind_force_n=1.0,
            fallen=False,
        )
    )
    assert decision.target_key == "rough"
    assert decision.action == RouterAction.USE_POLICY
    assert decision.training_request is None


def test_mixed_rough_wind_requests_new_combined_specialist() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.10,
            friction=0.4,
            roughness_m=0.06,
            step_height_m=0.08,
            wind_force_n=12.0,
            fallen=False,
        )
    )
    assert decision.target_key == "new_specialist"
    assert decision.action == RouterAction.FINE_TUNE_NEW_POLICY
    assert decision.training_request == "Sending sensor data to fine tune policy."


def test_fall_prioritizes_recovery_even_on_nominal_terrain() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=0.0,
            friction=0.8,
            wind_force_n=0.0,
            fallen=True,
        )
    )
    assert decision.target_key == "recovery"
    assert decision.action == RouterAction.USE_POLICY


def test_out_of_envelope_value_requests_training() -> None:
    decision = route_policy(
        TerrainContext(
            slope_gradient=math.tan(math.radians(35.0)),
            friction=0.08,
            roughness_m=0.01,
            step_height_m=0.0,
            wind_force_n=0.0,
            fallen=False,
        )
    )
    assert decision.target_key == "new_specialist"
    assert decision.action == RouterAction.FINE_TUNE_NEW_POLICY
    assert decision.training_request == "Sending sensor data to fine tune policy."


def test_context_mapping_accepts_scenario_aliases_and_wind_vector() -> None:
    context = TerrainContext.from_mapping(
        {
            "gradient": 0.12,
            "mu": 0.22,
            "roughness": 0.04,
            "step_height": 0.06,
            "wind": [3.0, 4.0, 0.0],
            "slip": 0.18,
            "terrain_type": "mixed test",
        }
    )
    assert context.slope_gradient == 0.12
    assert context.friction == 0.22
    assert context.wind_force_n == 5.0
    assert context.terrain_label == "mixed test"


def test_checkpoint_name_resolves_active_policy() -> None:
    assert canonical_policy_key("model_34400.pt") == "ice_incline"
    assert canonical_policy_key("g1_velocity_model_final.pt") == "flat"


def test_training_cli_is_hard_disabled() -> None:
    assert cli.TRAINING_ENABLED is False
    with pytest.raises(SystemExit, match="fine-tuning templates"):
        cli.train()


def test_foundational_incline_is_assumed_available_without_checkpoint_bookkeeping() -> None:
    context = TerrainContext(
        slope_gradient=0.18,
        friction=0.15,
        roughness_m=0.0,
        step_height_m=0.0,
        wind_force_n=0.0,
        fallen=False,
        active_policy="flat",
    )
    decision = route_policy(context)
    execution = resolve_policy_execution(
        decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert decision.target_key == "ice_incline"
    assert execution.executed_key == "ice_incline"
    assert execution.used_fallback is False


def test_known_incline_logs_one_conversational_handoff() -> None:
    context = TerrainContext(
        slope_gradient=0.12,
        friction=0.15,
        roughness_m=0.0,
        step_height_m=0.0,
        wind_force_n=0.0,
        fallen=False,
        active_policy="flat",
    )
    decision = route_policy(context)
    execution = resolve_policy_execution(
        decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    log = SupervisorEventLog(stable_observations=1)
    assert log.observe(context, decision, execution, step=1) is True
    assert decision.target_key == "ice_incline"
    assert execution.used_fallback is False
    assert len(log.entries) == 1
    entry = log.entries[0]
    assert entry.category == "HANDOFF"
    assert entry.message == (
        "The terrain looks like low-traction incline. I'll use the "
        "Low-friction incline, selected for conditions within its training envelope."
    )
    html = render_policy_overlay(
        context,
        decision,
        execution=execution,
        log_entries=tuple(log.entries),
        current_step=1,
    )
    assert "ACTION" not in html
    assert "UPCOMING" not in html
    assert "SummitOS" in html
    assert "font-size:15.5px" in html
    assert ">CURRENT<" in html
    assert "EXEC" not in html
    assert "not loaded" not in html
    assert "closest executable" not in html


def test_novel_combination_logs_one_concise_fine_tuning_action() -> None:
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
        decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    log = SupervisorEventLog(stable_observations=2)
    assert log.observe(context, decision, execution, step=1) is False
    assert log.observe(context, decision, execution, step=2) is True
    assert len(log.entries) == 1
    entry = log.entries[0]
    assert entry.category == "TRAINING_REQUIRED"
    assert entry.message == (
        "I don't have an existing policy trained for rough terrain + wind. "
        "I'll hold a stable stance with the Flat-ground walker and send a "
        "fine-tuning request to the policy server."
    )
    assert not hasattr(log, "sensor_buffer")
    log.observe(context, decision, execution, step=5)
    assert len(log.entries) == 1


def test_summitos_current_thought_is_larger_than_message_history() -> None:
    context = TerrainContext(slope_gradient=0.0, friction=0.8, wind_force_n=0.0)
    decision = route_policy(context)
    log = SupervisorEventLog()
    log.record_policy_lifecycle(
        step=10, category="HANDOFF", message="Older handoff message."
    )
    log.record_policy_lifecycle(
        step=20, category="THOUGHT", message="Current SummitOS thought."
    )

    html = render_policy_overlay(
        context, decision, log_entries=tuple(log.entries), current_step=20
    )

    assert html.index("Current SummitOS thought") < html.index("Older handoff message")
    assert html.count("font-size:15.5px") == 1
    assert html.count("font-size:9px") == 1
    assert "UPCOMING" not in html
    assert "ACTION" not in html


def test_summitos_explains_policy_training_and_unknown_condition_fallback() -> None:
    preview = _summitos_preview_message(
        "Low-friction incline", "ice_incline"
    )
    assert preview == (
        "It seems like there is an incline with low-friction footing ahead. "
        "Let me switch to the Low-friction incline walker."
    )
    assert "Low-friction incline walker" in preview
    assert "0.20" not in preview
    assert "0.15" not in preview

    unknown = TerrainContext(
        slope_gradient=0.28,
        friction=0.05,
        roughness_m=0.08,
        wind_force_n=12.0,
    )
    message = _summitos_training_request_message(
        unknown, "Normal-terrain walker"
    )
    assert "mu 0.05" in message
    assert "stable stance" in message
    assert "fine-tuning request" in message
    assert "policy server" in message


def test_route_state_reset_keeps_history_but_clears_stale_commit() -> None:
    context = TerrainContext(
        slope_gradient=0.0,
        friction=0.8,
        wind_force_n=0.0,
        fallen=False,
        active_policy="flat",
    )
    decision = route_policy(context)
    execution = resolve_policy_execution(
        decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    log = SupervisorEventLog(stable_observations=1)
    assert log.observe(context, decision, execution, step=1) is True
    entry_count = len(log.entries)

    log.reset_route_state()

    assert log.committed_decision is None
    assert log.committed_execution is None
    assert len(log.entries) == entry_count


def test_policy_library_contains_five_foundational_policy_classes() -> None:
    from humanoid_climber.policy_router import POLICIES

    assert {policy.key for policy in POLICIES} == {
        "flat",
        "ice_incline",
        "wind",
        "recovery",
        "rough",
    }


def test_small_terrain_sensor_jitter_does_not_emit_repeated_condition_events() -> None:
    first = TerrainContext(
        slope_gradient=0.01,
        friction=0.8,
        roughness_m=0.041,
        step_height_m=0.061,
        wind_force_n=12.0,
        fallen=False,
        active_policy="flat",
    )
    jittered = TerrainContext(
        slope_gradient=0.03,
        friction=0.81,
        roughness_m=0.049,
        step_height_m=0.071,
        wind_force_n=12.4,
        fallen=False,
        active_policy="flat",
    )
    log = SupervisorEventLog(stable_observations=1)
    first_decision = route_policy(first)
    first_execution = resolve_policy_execution(
        first_decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert log.observe(first, first_decision, first_execution, step=1) is True
    count = len(log.entries)
    jittered_decision = route_policy(jittered)
    jittered_execution = resolve_policy_execution(
        jittered_decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert log.observe(
        jittered, jittered_decision, jittered_execution, step=2
    ) is False
    assert len(log.entries) == count


def test_minimum_dwell_holds_committed_policy_through_boundary_chatter() -> None:
    flat = TerrainContext(
        slope_gradient=0.05,
        friction=0.8,
        wind_force_n=0.0,
        fallen=False,
        active_policy="flat",
    )
    incline = TerrainContext(
        slope_gradient=0.07,
        friction=0.8,
        wind_force_n=0.0,
        fallen=False,
        active_policy="flat",
    )
    log = SupervisorEventLog(
        stable_observations=1, min_steps_between_commits=50
    )
    flat_decision = route_policy(flat)
    flat_execution = resolve_policy_execution(
        flat_decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert log.observe(flat, flat_decision, flat_execution, step=1) is True
    incline_decision = route_policy(incline)
    incline_execution = resolve_policy_execution(
        incline_decision, available_policy_keys=("flat",), active_policy_key="flat"
    )
    assert log.observe(
        incline, incline_decision, incline_execution, step=20
    ) is False
    assert log.committed_decision is flat_decision
    assert log.observe(
        incline, incline_decision, incline_execution, step=51
    ) is True
    assert log.committed_decision is incline_decision


def test_safety_log_deduplicates_one_incident_and_banner_is_single() -> None:
    context = TerrainContext(
        slope_gradient=0.0,
        friction=0.8,
        wind_force_n=0.0,
        fallen=False,
    )
    decision = route_policy(context)
    reason = "Predicted torso imbalance exceeded the pre-fall safety envelope."
    message = f"{reason} Executing four-point safety recovery."
    log = SupervisorEventLog(stable_observations=1)
    log.record_safety_action(step=10, message=message)
    log.record_safety_action(step=11, message=message)

    safety_entries = [entry for entry in log.entries if entry.category == "SAFETY"]
    assert len(safety_entries) == 1

    html = render_policy_overlay(
        context,
        decision,
        log_entries=tuple(log.entries),
        recovery_active=True,
        recovery_reason=reason,
        current_step=11,
    )
    assert html.count(">SAFETY<") == 1
    assert html.count("Predicted torso imbalance") == 1
