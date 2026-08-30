"""Humanoid Climber extensions for MjLab's localhost Viser viewer."""

from __future__ import annotations

import html
import math
import traceback
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast, override

import viser
import torch

from mjlab.viewer.base import VerbosityLevel, ViewerAction
from mjlab.viewer.viser.viewer import ViserPlayViewer

from humanoid_climber.policy_router import (
    FOUNDATIONAL_POLICY_KEYS,
    PolicyExecution,
    PolicyReadiness,
    RouterAction,
    RoutingDecision,
    SupervisorEventLog,
    TerrainContext,
    canonical_policy_key,
    context_summary,
    resolve_policy_execution,
    route_policy,
)
from humanoid_climber.terrain_probe import estimate_local_terrain, sampled_foot_friction
from humanoid_climber.trail import TrailFrame, nearest_trail_frame
from humanoid_climber.safety import (
    CENTERLINE_MAX_OFFSET_M,
    ImbalanceMonitor,
    ImbalanceRisk,
    outside_centerline,
    predict_imbalance,
)


RECOVERY_HOLD_SECONDS = 2.0
RECOVERY_ATTACK_SECONDS = 0.12
SAFETY_REARM_CLEAR_SECONDS = 1.5
SPECIALIST_CHECKPOINT_NAME = "model_34400.pt"
SPECIALIST_RECOVERED_RELATIVE_NAME = "recovered/model_34400.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECIALIST_RECOVERED_PATH = (
    PROJECT_ROOT / "ckpt" / SPECIALIST_RECOVERED_RELATIVE_NAME
)
SPECIALIST_ADAPTATION_SECONDS = 3.0
CENTERING_FORWARD_SPEED_M_S = 0.30
CENTERING_LATERAL_GAIN = 0.55
CENTERING_MAX_LATERAL_SPEED_M_S = 0.45
CENTERING_TOLERANCE_M = 0.12
CENTERING_RELEASE_SECONDS = 2.0
TRAIL_FOLLOW_LATERAL_GAIN = 0.45
TRAIL_FOLLOW_MAX_LATERAL_SPEED_M_S = 0.25
TRAIL_FOLLOW_LATERAL_DEADBAND_M = 0.08
TRAIL_FOLLOW_HEADING_GAIN = 1.2
TRAIL_FOLLOW_MAX_YAW_RATE_RAD_S = 0.55
TRAIL_FOLLOW_HEADING_DEADBAND_RAD = math.radians(4.0)
# Full MuJoCo terrain ray sampling is dashboard/routing telemetry, not a safety
# control signal.  Safety posture and centerline measurements still run every
# control step; the heavier terrain context is refreshed at 5 Hz and immediately
# whenever the random-event phase or loaded checkpoint changes.
ROUTING_CONTEXT_REFRESH_SECONDS = 0.20
MODEL_FIELD_SYNC_SECONDS = 0.20
GROUND_STUCK_RESET_SECONDS = 10.0


def _advance_ground_stuck_timer(
    elapsed_s: float, *, fallen: bool | None, step_dt: float
) -> tuple[float, bool]:
    """Accumulate only uninterrupted time spent in a fallen/grounded posture."""
    if fallen is not True:
        return 0.0, False
    elapsed_s = max(0.0, float(elapsed_s)) + max(0.0, float(step_dt))
    return elapsed_s, elapsed_s + 1.0e-9 >= GROUND_STUCK_RESET_SECONDS


def _advance_safety_rearm_timer(
    clear_time_s: float,
    *,
    centerline_breached: bool,
    imbalance_triggered: bool,
    step_dt: float,
) -> tuple[float, bool]:
    """Require a continuous clean window before another safety incident can fire."""
    if centerline_breached or imbalance_triggered:
        return 0.0, False
    clear_time_s = max(0.0, clear_time_s) + max(0.0, step_dt)
    return clear_time_s, clear_time_s + 1.0e-9 >= SAFETY_REARM_CLEAR_SECONDS


def _robot_state_is_finite(env: Any, env_idx: int) -> bool:
    """Return False when the current robot state contains NaN/Inf values."""
    base = getattr(env, "unwrapped", env)
    try:
        robot = base.scene["robot"]
        data = robot.data
        fields_to_check = (
            "root_link_pos_w",
            "root_link_quat_w",
            "root_link_vel_w",
            "joint_pos",
            "joint_vel",
        )
        checked = False
        for field_name in fields_to_check:
            value = getattr(data, field_name, None)
            if not isinstance(value, torch.Tensor):
                continue
            checked = True
            sample = value[env_idx]
            if not bool(torch.isfinite(sample).all().item()):
                return False
        return True if checked else True
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
        # Missing telemetry is not itself evidence of numerical corruption.
        return True


def _ensure_playback_observation_groups(env: Any) -> bool:
    """Ensure playback exposes both observation groups expected by RSL-RL.

    The inference policy only consumes the actor set, but RSL-RL and parts of the
    viewer/runtime still assume the environment exposes both ``actor`` and
    ``critic``.  Do not mutate ObservationManager's private group dictionaries in
    playback: doing so can surface as a late ``KeyError('critic')`` inside the
    viewer loop, where the generic step exception handler pauses the simulation.

    Returns ``True`` when both groups are available.  The small repair path is
    intentionally limited to reconstructing a missing group's *name list* from
    the manager's immutable observation config; all term configs/buffers remain
    owned by MjLab.
    """
    base = getattr(env, "unwrapped", env)
    manager = getattr(base, "observation_manager", None)
    group_terms = getattr(manager, "_group_obs_term_names", None)
    if not isinstance(group_terms, dict):
        return False

    for group_name in ("actor", "critic"):
        if group_name in group_terms:
            continue
        cfg = getattr(manager, "cfg", None)
        try:
            group_cfg = cfg[group_name]
        except (KeyError, TypeError):
            return False
        terms = getattr(group_cfg, "terms", None)
        if not isinstance(terms, dict):
            return False
        group_terms[group_name] = list(terms)
        manager._obs_buffer = None

    return True


class HumanoidClimberViserPlayViewer(ViserPlayViewer):
    """MjLab viewer with a movable live policy-supervisor overlay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not _ensure_playback_observation_groups(self.env):
            raise RuntimeError(
                "Playback observation manager must expose actor and critic groups."
            )
        self._policy_panel = None
        self._policy_html = None
        self._policy_overlay_tick = 0
        self._policy_history: deque[str] = deque(maxlen=6)
        self._policy_history_signature: tuple[str, str, str] | None = None
        self._supervisor_log = SupervisorEventLog(
            max_entries=48,
            stable_observations=3,
            min_steps_between_commits=50,
        )
        self._last_context: TerrainContext | None = None
        self._last_decision: RoutingDecision | None = None
        self._last_execution: PolicyExecution | None = None
        self._routing_context_refresh_steps_remaining = 0
        self._routing_context_checkpoint_name: str | None = None
        self._routing_context_event_signature: tuple[str, ...] = ()
        self._model_field_sync_force = True
        self._model_field_sync_next_time = 0.0
        self._model_field_sync_original = None
        self._imbalance_recovery_latched = False
        self._imbalance_risk: ImbalanceRisk | None = None
        self._imbalance_monitor = ImbalanceMonitor()
        self._recovery_trigger_reason = ""
        self._recovery_steps_remaining = 0
        self._recovery_attack_steps_remaining = 0
        self._four_point_action: torch.Tensor | None = None
        self._fast_four_point_action: torch.Tensor | None = None
        self._recovery_command_snapshot: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._default_checkpoint_name = (
            self._ckpt_mgr.current_name if self._ckpt_mgr is not None else None
        )
        self._specialist_stage = "idle"
        self._specialist_condition: str | None = None
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0
        self._specialist_promoted = False
        self._specialist_missing_logged = False
        self._specialist_base_label: str | None = None
        self._fine_tuned_combinations: dict[str, str] = {}
        self._reset_after_safety = False
        self._safety_rearm_required = False
        self._safety_rearm_clear_time_s = 0.0
        self._ground_stuck_seconds = 0.0
        self._settings_folder = None
        self._reset_after_safety_checkbox = None
        self._auto_random_events_checkbox = None
        self._random_event_checkboxes: dict[str, Any] = {}
        self._syncing_settings_gui = False

    @override
    def setup(self) -> None:
        # The stock Viser chrome defaults to a bright theme that fights the
        # simulation canvas. Keep the controls familiar, but make the whole
        # localhost viewer feel like one coherent mission-control surface.
        self._server.gui.configure_theme(
            dark_mode=True,
            control_layout="floating",
            control_width="medium",
            show_logo=False,
            show_share_button=False,
            brand_color=(117, 191, 255),
        )
        super().setup()
        self._install_model_field_sync_gate()
        self._setup_settings_controls()
        self._policy_panel = self._server.gui.add_panel(order=-100.0)
        with self._policy_panel.add_tab("Policy Log"):
            self._policy_html = self._server.gui.add_html(_loading_html())
        # One compact message stream, pinned to the top-left of the canvas.
        self._policy_panel.float(x=16, y=16, width=340, height=410)
        self._update_policy_overlay()

    @override
    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._policy_overlay_tick += 1
        # 60 Hz simulation does not need 60 DOM updates.  ~4 Hz keeps telemetry
        # feeling live without competing with scene rendering.
        if self._policy_overlay_tick % 15 == 0:
            self._sync_random_event_controls()
            self._update_policy_overlay()

    @override
    def close(self) -> None:
        if self._settings_folder is not None:
            self._settings_folder.remove()
            self._settings_folder = None
        if self._policy_panel is not None:
            self._policy_panel.remove()
            self._policy_panel = None
        super().close()

    @override
    def reset_environment(self) -> None:
        self._imbalance_recovery_latched = False
        self._imbalance_risk = None
        self._imbalance_monitor.reset()
        self._recovery_trigger_reason = ""
        self._recovery_steps_remaining = 0
        self._recovery_attack_steps_remaining = 0
        self._four_point_action = None
        self._fast_four_point_action = None
        self._recovery_command_snapshot = None
        self._specialist_stage = "idle"
        self._specialist_condition = None
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0
        self._specialist_missing_logged = False
        self._specialist_base_label = None
        self._safety_rearm_required = False
        self._safety_rearm_clear_time_s = 0.0
        self._ground_stuck_seconds = 0.0
        self._last_context = None
        self._last_decision = None
        self._last_execution = None
        self._routing_context_refresh_steps_remaining = 0
        self._routing_context_checkpoint_name = None
        self._routing_context_event_signature = ()
        self._force_model_field_sync()
        self._policy_history_signature = None
        self._supervisor_log.reset_route_state()
        super().reset_environment()

    def _clear_recovery_after_external_reset(self) -> None:
        """Drop episode-local safety state after an env-owned auto-reset.

        MjLab can auto-reset inside ``env.step`` when a termination fires.  A
        viewer safety latch belongs to the episode that just ended and must not
        keep commanding the protective four-point pose in the freshly reset episode.
        Unlike ``_release_safety_recovery`` this deliberately does not restore
        the old locomotion command snapshot, which is also episode-local.
        """
        self._imbalance_recovery_latched = False
        self._imbalance_risk = None
        self._imbalance_monitor.reset()
        self._recovery_trigger_reason = ""
        self._recovery_steps_remaining = 0
        self._recovery_attack_steps_remaining = 0
        self._recovery_command_snapshot = None
        self._specialist_stage = "idle"
        self._specialist_condition = None
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0
        self._specialist_missing_logged = False
        self._specialist_base_label = None
        self._safety_rearm_required = False
        self._safety_rearm_clear_time_s = 0.0
        self._ground_stuck_seconds = 0.0
        self._last_context = None
        self._last_decision = None
        self._last_execution = None
        self._routing_context_refresh_steps_remaining = 0
        self._routing_context_checkpoint_name = None
        self._routing_context_event_signature = ()
        self._force_model_field_sync()

    def _release_safety_recovery(self) -> None:
        """Resume locomotion without resetting after the protective hold."""
        if self._recovery_command_snapshot is not None and hasattr(self, "_scene"):
            _restore_locomotion_command(
                self.env,
                int(self._scene.env_idx),
                self._recovery_command_snapshot,
            )
        self._recovery_command_snapshot = None
        self._imbalance_recovery_latched = False
        self._imbalance_monitor.reset()
        self._recovery_trigger_reason = ""
        self._recovery_steps_remaining = 0
        self._recovery_attack_steps_remaining = 0
        # Do not immediately retrigger on the exact same still-active hazard.
        # The guard rearms once the robot is back inside the path and the
        # imbalance monitor is clear.
        self._safety_rearm_required = True
        self._safety_rearm_clear_time_s = 0.0

    def _update_ground_stuck_watchdog(self, *, env_idx: int, step_dt: float) -> bool:
        """Reset only after ten continuous seconds spent down on the ground.

        This watchdog intentionally bypasses the dashboard's normal
        ``Reset robot after safety`` toggle. That toggle controls the short
        protective four-point recovery; this path is a last-resort escape from
        a robot that never gets back up.
        """
        base = getattr(self.env, "unwrapped", self.env)
        _, fallen = _posture_state(base, int(env_idx))
        self._ground_stuck_seconds, should_reset = _advance_ground_stuck_timer(
            self._ground_stuck_seconds,
            fallen=fallen,
            step_dt=step_dt,
        )
        if not should_reset:
            return False

        message = (
            f"Robot remained on the ground for {GROUND_STUCK_RESET_SECONDS:.0f}s; "
            "resetting the environment."
        )
        self._supervisor_log.record_safety_action(
            step=self.get_status().step_count,
            message=message,
        )
        print(f"[ROUTER] {message}")
        self.reset_environment()
        return True

    def _reset_nonfinite_robot_state(self, *, env_idx: int) -> bool:
        """Recover from an unrecoverable NaN/Inf physics state without pausing."""
        if _robot_state_is_finite(self.env, int(env_idx)):
            return False
        message = "Numerical instability detected in robot state; resetting the environment."
        self._supervisor_log.record_safety_action(
            step=self.get_status().step_count,
            message=message,
        )
        print(f"[ROUTER] {message}")
        self.reset_environment()
        return True

    def _routing_context_for_step(
        self,
        *,
        checkpoint_name: str | None,
        env_idx: int,
        step_dt: float,
        event_signature: tuple[str, ...],
    ) -> TerrainContext:
        """Return cached routing telemetry while keeping event transitions immediate."""
        refresh_steps = max(1, round(ROUTING_CONTEXT_REFRESH_SECONDS / step_dt))
        event_changed = event_signature != self._routing_context_event_signature
        needs_refresh = (
            self._last_context is None
            or self._routing_context_refresh_steps_remaining <= 0
            or checkpoint_name != self._routing_context_checkpoint_name
            or event_changed
        )
        if needs_refresh:
            context = extract_terrain_context(
                self.env, env_idx=env_idx, checkpoint_name=checkpoint_name
            )
            self._last_context = context
            self._routing_context_refresh_steps_remaining = refresh_steps - 1
            self._routing_context_checkpoint_name = checkpoint_name
            self._routing_context_event_signature = event_signature
            return context

        self._routing_context_refresh_steps_remaining -= 1
        context = self._last_context
        assert context is not None
        return context

    def _install_model_field_sync_gate(self) -> None:
        """Avoid redundant GPU->CPU geom model syncs between visual changes.

        Body transforms are still copied by Viser on every rendered frame. The
        expensive expanded model fields (geom position/size/quat/color) only
        need refreshing when an event changes terrain appearance/geometry, with
        a low-rate fallback for the rough-terrain clearing animation.
        """
        original = getattr(self._scene, "_sync_model_fields", None)
        if not callable(original):
            return
        self._model_field_sync_original = original

        def _gated_sync(env_idx: int) -> None:
            now = time.perf_counter()
            if not self._model_field_sync_force and now < self._model_field_sync_next_time:
                return
            self._model_field_sync_force = False
            self._model_field_sync_next_time = now + MODEL_FIELD_SYNC_SECONDS
            original(env_idx)

        self._scene._sync_model_fields = _gated_sync

    def _force_model_field_sync(self) -> None:
        self._model_field_sync_force = True
        self._model_field_sync_next_time = 0.0

    def _begin_specialist_adaptation(self, condition: str) -> None:
        """Remember that this edge recovery was caused by ice/slope conditions."""
        self._specialist_stage = "recovering"
        self._specialist_condition = condition
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0
        self._specialist_missing_logged = False

    def _begin_specialist_fine_tuning(self, step_dt: float) -> None:
        """Enter the short demo adaptation delay while holding a safe pose."""
        if self._specialist_condition is None:
            return
        self._specialist_stage = "fine_tuning"
        self._specialist_wait_steps_remaining = max(
            1, round(SPECIALIST_ADAPTATION_SECONDS / step_dt)
        )

    def _begin_combination_fine_tuning(
        self, decision: RoutingDecision, *, step_dt: float, env_idx: int
    ) -> None:
        """Hold safely while synthesizing a policy for a novel condition mix."""
        if self._specialist_stage != "idle":
            return
        condition = decision.terrain_type
        candidates = [
            evaluation
            for evaluation in decision.evaluations
            if evaluation.spec.key in FOUNDATIONAL_POLICY_KEYS
            and evaluation.spec.key != "recovery"
        ]
        self._specialist_condition = condition
        self._specialist_base_label = (
            candidates[0].spec.label if candidates else "Flat-ground walker"
        )
        self._begin_specialist_fine_tuning(step_dt)
        self._imbalance_recovery_latched = True
        self._recovery_trigger_reason = f"Fine tuning for {condition}."
        self._recovery_command_snapshot = _capture_locomotion_command(
            self.env, env_idx
        )
        self._recovery_attack_steps_remaining = max(
            1, round(RECOVERY_ATTACK_SECONDS / step_dt)
        )

    def _checkpoint_is_available(self, name: str) -> bool:
        manager = self._ckpt_mgr
        if manager is None:
            return False
        if manager.current_name == name:
            return True
        try:
            return name in {entry_name for entry_name, _ in manager.fetch_available()}
        except Exception:
            return False

    def _load_checkpoint_in_place(self, name: str) -> bool:
        """Hot-swap a compatible checkpoint without resetting robot state."""
        manager = self._ckpt_mgr
        if manager is None or not self._checkpoint_is_available(name):
            return False
        if manager.current_name != name:
            self.policy = manager.load_checkpoint(name)
            manager.current_name = name
            if hasattr(self, "_ckpt_dropdown"):
                try:
                    labels = tuple(self._ckpt_dropdown.options)
                    self._ckpt_dropdown.value = next(
                        (label for label in labels if label.startswith(name)), name
                    )
                except (AttributeError, RuntimeError, ValueError):
                    pass
        return True

    def _load_specialist_checkpoint(self) -> bool:
        """Load the trained incline policy from either supported local layout."""
        if self._load_checkpoint_in_place(SPECIALIST_CHECKPOINT_NAME):
            return True
        manager = self._ckpt_mgr
        if manager is None or not SPECIALIST_RECOVERED_PATH.is_file():
            return False
        self.policy = manager.load_checkpoint(SPECIALIST_RECOVERED_RELATIVE_NAME)
        manager.current_name = SPECIALIST_CHECKPOINT_NAME
        if hasattr(self, "_ckpt_dropdown"):
            try:
                self._ckpt_dropdown.value = SPECIALIST_CHECKPOINT_NAME
            except (AttributeError, RuntimeError, ValueError):
                pass
        return True

    def _try_activate_specialist(self, step_dt: float) -> bool:
        """Activate the demo's synthesized policy after the adaptation delay."""
        del step_dt
        if self._specialist_condition is None:
            return False
        condition = self._specialist_condition
        base_label = self._specialist_base_label or "Flat-ground walker"
        adapted_label = f"Fine-tuned {condition}"
        self._fine_tuned_combinations[condition] = adapted_label
        self._specialist_promoted = True
        self._specialist_stage = "idle"
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0
        self._supervisor_log.record_policy_lifecycle(
            step=self.get_status().step_count,
            category="ACTION",
            message=(
                f"Fine tuned {base_label} for {condition}; activating new "
                f"{adapted_label} policy."
            ),
        )
        self._specialist_condition = None
        self._specialist_base_label = None
        self._release_safety_recovery()
        return True

    def _restore_default_checkpoint(self) -> None:
        default_name = self._default_checkpoint_name
        if default_name and self._load_checkpoint_in_place(default_name):
            self._supervisor_log.record_policy_lifecycle(
                step=self.get_status().step_count,
                category="ACTION",
                message="Centered and condition cleared; restored the default walking policy.",
            )
        self._specialist_stage = "idle"
        self._specialist_condition = None
        self._specialist_wait_steps_remaining = 0
        self._specialist_centered_steps = 0

    def _update_specialist_centering(
        self,
        *,
        trail_frame: TrailFrame,
        lateral_offset_m: float,
        adaptive_condition: str | None,
        step_dt: float,
    ) -> None:
        """Steer the active incline specialist back to the winding centerline."""
        _apply_centering_command(self.env, int(self._scene.env_idx), trail_frame)
        centered = abs(lateral_offset_m) <= CENTERING_TOLERANCE_M
        if centered and adaptive_condition is None:
            self._specialist_centered_steps += 1
        else:
            self._specialist_centered_steps = 0
        required = max(1, round(CENTERING_RELEASE_SECONDS / step_dt))
        if self._specialist_centered_steps >= required:
            self._restore_default_checkpoint()

    def _setup_settings_controls(self) -> None:
        """Add climber-specific controls inside the stock Controls tab."""
        controls_tab = _find_controls_tab(self._server.gui)
        if controls_tab is not None:
            with controls_tab:
                self._create_settings_folder()
            print("[VIEWER] Settings controls attached to Controls tab.")
        else:
            # Viser internals can change between releases. Falling back to the
            # root control panel is preferable to silently losing the controls.
            self._create_settings_folder()
            print("[VIEWER] WARNING: Controls tab not found; Settings attached to root panel.")

    def _create_settings_folder(self) -> None:
        self._settings_folder = self._server.gui.add_folder(
            # Keep climber controls immediately visible in the Controls tab.
            # `order=100` placed this below the very large Scene/Debug Viz
            # subtree, which made the folder look missing unless the user
            # scrolled all the way to the bottom of the panel.
            "Settings", order=-10.0, expand_by_default=True
        )
        with self._settings_folder:
            self._reset_after_safety_checkbox = self._server.gui.add_checkbox(
                "Reset robot after safety",
                initial_value=False,
                hint=(
                    "When disabled, the robot exits the four-point safety pose and keeps "
                    "running instead of resetting the whole environment."
                ),
            )

            @self._reset_after_safety_checkbox.on_update
            def _(event) -> None:
                self._reset_after_safety = bool(event.target.value)

            controller = _random_event_controller(self.env)
            if controller is None or not hasattr(self, "_scene"):
                return
            env_idx = int(self._scene.env_idx)
            self._auto_random_events_checkbox = self._server.gui.add_checkbox(
                "Automatic random events",
                initial_value=bool(controller.automatic_mode(env_idx)),
                hint=(
                    "Disable automatic scheduling to pin wind, ice, slope, and "
                    "bumpy terrain manually."
                ),
            )

            @self._auto_random_events_checkbox.on_update
            def _(event) -> None:
                if self._syncing_settings_gui:
                    return
                self.request_action(
                    "CUSTOM",
                    (
                        "random_event_auto",
                        int(self._scene.env_idx),
                        bool(event.target.value),
                    ),
                )

            state = controller.manual_event_state(env_idx)
            labels = {
                "wind": "Wind",
                "ice": "Ice",
                "slope": "Slope",
                "bumps": "Bumpy terrain",
            }
            for event_name, label in labels.items():
                handle = self._server.gui.add_checkbox(
                    label,
                    initial_value=bool(state.get(event_name, False)),
                    hint="Immediately add or remove this condition from the viewed environment.",
                )
                self._random_event_checkboxes[event_name] = handle

                def _on_manual_event(event, *, name=event_name) -> None:
                    if self._syncing_settings_gui:
                        return
                    enabled = bool(event.target.value)
                    if (
                        enabled
                        and self._auto_random_events_checkbox is not None
                        and self._auto_random_events_checkbox.value
                    ):
                        self._auto_random_events_checkbox.value = False
                    self.request_action(
                        "CUSTOM",
                        (
                            "random_event_manual",
                            int(self._scene.env_idx),
                            name,
                            enabled,
                        ),
                    )

                handle.on_update(_on_manual_event)

    def _sync_random_event_controls(self) -> None:
        controller = _random_event_controller(self.env)
        if (
            controller is None
            or self._auto_random_events_checkbox is None
            or not hasattr(self, "_scene")
        ):
            return
        env_idx = int(self._scene.env_idx)
        state = controller.manual_event_state(env_idx)
        self._syncing_settings_gui = True
        try:
            self._auto_random_events_checkbox.value = bool(
                controller.automatic_mode(env_idx)
            )
            for event_name, handle in self._random_event_checkboxes.items():
                handle.value = bool(state.get(event_name, False))
        finally:
            self._syncing_settings_gui = False

    @override
    def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
        if action == ViewerAction.CUSTOM and isinstance(payload, tuple) and payload:
            controller = _random_event_controller(self.env)
            if payload[0] == "random_event_auto" and len(payload) == 3:
                if controller is not None:
                    _, env_idx, enabled = payload
                    controller.set_automatic_mode(int(env_idx), bool(enabled))
                    self._scene.request_update()
                    self._sync_random_event_controls()
                return True
            if payload[0] == "random_event_manual" and len(payload) == 4:
                if controller is not None:
                    _, env_idx, event_name, enabled = payload
                    controller.set_manual_event(
                        int(env_idx), str(event_name), bool(enabled)
                    )
                    self._scene.request_update()
                    self._sync_random_event_controls()
                return True
        return super()._handle_custom_action(action, payload)

    def _update_policy_overlay(self) -> None:
        if self._policy_html is None or not hasattr(self, "_scene"):
            return
        checkpoint_name = self._ckpt_mgr.current_name if self._ckpt_mgr is not None else None
        context = self._last_context or extract_terrain_context(
            self.env, env_idx=self._scene.env_idx, checkpoint_name=checkpoint_name
        )
        decision = self._last_decision or route_policy(context)
        execution = self._last_execution or resolve_policy_execution(
            decision, available_policy_keys=FOUNDATIONAL_POLICY_KEYS
        )
        self._record_policy_decision(context, decision)
        self._policy_html.content = render_policy_overlay(
            context,
            decision,
            execution=execution,
            history=tuple(self._policy_history),
            log_entries=tuple(self._supervisor_log.entries),
            imbalance_risk=self._imbalance_risk,
            recovery_active=self._imbalance_recovery_latched,
            recovery_reason=self._recovery_trigger_reason,
            current_step=self.get_status().step_count,
        )

    @override
    def _execute_step(self) -> bool:
        """Route and execute one action cycle from the same telemetry snapshot."""
        try:
            checkpoint_name = self._ckpt_mgr.current_name if self._ckpt_mgr is not None else None
            step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.02))
            env_idx = int(self._scene.env_idx)
            if self._reset_nonfinite_robot_state(env_idx=env_idx):
                return True
            active_events = _active_random_event_names(self.env, env_idx)
            context = self._routing_context_for_step(
                checkpoint_name=checkpoint_name,
                env_idx=env_idx,
                step_dt=step_dt,
                event_signature=active_events,
            )
            # Demo assumption: the five foundational specialists are the built-in
            # policy bank. Checkpoint bookkeeping is intentionally not surfaced in
            # the policy log; only novel combinations require adaptation.
            available = FOUNDATIONAL_POLICY_KEYS
            observed_decision = route_policy(context)
            observed_execution = resolve_policy_execution(
                observed_decision,
                available_policy_keys=available,
                active_policy_key=checkpoint_name,
            )
            adapted_label = self._fine_tuned_combinations.get(
                observed_decision.terrain_type
            )
            if adapted_label is not None:
                adapted_key = f"adapted:{observed_decision.terrain_type}"
                observed_decision = replace(
                    observed_decision,
                    action=RouterAction.USE_POLICY,
                    target_key=adapted_key,
                    target_label=adapted_label,
                    model="demo fine-tuned specialist",
                    readiness=PolicyReadiness.AVAILABLE,
                    headline=f"Use {adapted_label}",
                    detail="Previously fine tuned for this condition combination.",
                    training_request=None,
                )
                observed_execution = PolicyExecution(
                    requested_key=adapted_key,
                    executed_key=adapted_key,
                    executed_label=adapted_label,
                    used_fallback=False,
                    reason="Previously fine tuned for this condition combination.",
                )
            imbalance_risk = _measure_imbalance_risk(self.env, self._scene.env_idx)
            self._imbalance_risk = imbalance_risk
            imbalance_confirmed = self._imbalance_monitor.observe(imbalance_risk)
            adaptive_condition = _adaptive_specialist_condition(context, active_events)
            trail_frame = _measure_trail_frame(self.env, self._scene.env_idx)
            if trail_frame is None:
                raise RuntimeError(
                    "Centerline safety measurement unavailable; pausing instead of "
                    "continuing without the lateral boundary guard."
                )
            lateral_offset_m = trail_frame.lateral_offset_m
            centerline_breached = (
                outside_centerline(lateral_offset_m)
            )
            trigger_reason = ""
            if self._safety_rearm_required:
                # A completed recovery belongs to one incident. Do not launch
                # another four-point sequence while the same hazard is still
                # present or chattering around the threshold. Rearm only after
                # both centerline and raw IMU risk have stayed clear continuously.
                self._safety_rearm_clear_time_s, rearmed = _advance_safety_rearm_timer(
                    self._safety_rearm_clear_time_s,
                    centerline_breached=centerline_breached,
                    imbalance_triggered=bool(imbalance_risk.triggered),
                    step_dt=step_dt,
                )
                if rearmed:
                    self._safety_rearm_required = False
                    self._safety_rearm_clear_time_s = 0.0
                    self._imbalance_monitor.reset()
            elif centerline_breached:
                trigger_reason = (
                    f"Left the path. Lateral offset {lateral_offset_m:+.2f} m is "
                    f"outside the ±{CENTERLINE_MAX_OFFSET_M:.1f} m boundary."
                )
            elif imbalance_confirmed:
                trigger_reason = "Robot imbalance detected."

            if trigger_reason and not self._imbalance_recovery_latched:
                self._imbalance_recovery_latched = True
                self._recovery_trigger_reason = trigger_reason
                self._recovery_command_snapshot = _capture_locomotion_command(
                    self.env, int(self._scene.env_idx)
                )
                self._recovery_steps_remaining = max(
                    1, round(RECOVERY_HOLD_SECONDS / step_dt)
                )
                self._recovery_attack_steps_remaining = max(
                    1, round(RECOVERY_ATTACK_SECONDS / step_dt)
                )
                if trigger_reason == "Robot imbalance detected.":
                    message = (
                        "Robot imbalance detected. Activating safety recovery to prevent damage."
                    )
                else:
                    message = f"{trigger_reason} Activating safety recovery."
                self._supervisor_log.record_safety_action(
                    step=self.get_status().step_count,
                    message=message,
                )
                print(f"[ROUTER] {message}")

            if self._imbalance_recovery_latched:
                observed_decision = _imbalance_recovery_decision(
                    observed_decision,
                    imbalance_risk,
                    reason=self._recovery_trigger_reason,
                )
                observed_execution = PolicyExecution(
                    requested_key="imbalance_recovery",
                    executed_key="imbalance_recovery",
                    executed_label="Four-point safety pose",
                    used_fallback=False,
                    reason="The deterministic safety controller bypasses locomotion.",
                )
                condition_changed = False
                decision = observed_decision
                execution = observed_execution
            else:
                if adapted_label is not None:
                    condition_changed = False
                    decision = observed_decision
                    execution = observed_execution
                else:
                    condition_changed = self._supervisor_log.observe(
                        context,
                        observed_decision,
                        observed_execution,
                        step=self.get_status().step_count,
                    )
                    decision = self._supervisor_log.committed_decision or observed_decision
                    execution = self._supervisor_log.committed_execution or observed_execution
                if (
                    condition_changed
                    and observed_decision.action == RouterAction.FINE_TUNE_NEW_POLICY
                    and self._specialist_stage == "idle"
                ):
                    self._begin_combination_fine_tuning(
                        observed_decision, step_dt=step_dt, env_idx=env_idx
                    )
                elif condition_changed:
                    print(
                        f"[ROUTER] Detected {decision.terrain_type} environment, "
                        f"executing {execution.executed_label} policy."
                    )
            self._last_context = context
            self._last_decision = decision
            self._last_execution = execution
            with torch.no_grad():
                if (
                    self._specialist_stage == "centering"
                    and not self._imbalance_recovery_latched
                ):
                    self._update_specialist_centering(
                        trail_frame=trail_frame,
                        lateral_offset_m=lateral_offset_m,
                        adaptive_condition=adaptive_condition,
                        step_dt=step_dt,
                    )
                elif not self._imbalance_recovery_latched:
                    _apply_trail_following_command(
                        self.env,
                        int(self._scene.env_idx),
                        trail_frame,
                    )
                obs = self.env.get_observations()
                if self._imbalance_recovery_latched:
                    actions = self._four_point_safety_action(
                        aggressive=self._recovery_attack_steps_remaining > 0
                    )
                    _stop_locomotion(self.env, int(self._scene.env_idx))
                else:
                    actions = self.policy(obs)
                step_result = self.env.step(actions)
                if self._reset_nonfinite_robot_state(env_idx=int(self._scene.env_idx)):
                    return True
                # Defensive compatibility for tasks/configurations that still
                # auto-reset internally.  Our Humanoid Climber play configs
                # disable terminations, but a hot-reloaded or external config
                # must never carry the previous episode's four-point safety pose
                # across an environment-owned reset.
                if isinstance(step_result, tuple) and len(step_result) >= 3:
                    dones = step_result[2]
                    if isinstance(dones, torch.Tensor) and bool(torch.any(dones).item()):
                        self._clear_recovery_after_external_reset()
                if self._update_ground_stuck_watchdog(
                    env_idx=int(self._scene.env_idx), step_dt=step_dt
                ):
                    return True
                if self._imbalance_recovery_latched:
                    _stop_locomotion(self.env, int(self._scene.env_idx))
                    if self._specialist_stage == "recovering":
                        self._recovery_steps_remaining -= 1
                    elif self._specialist_stage in {
                        "fine_tuning",
                        "waiting_checkpoint",
                    }:
                        self._specialist_wait_steps_remaining -= 1
                    else:
                        self._recovery_steps_remaining -= 1
                    self._recovery_attack_steps_remaining = max(
                        0, self._recovery_attack_steps_remaining - 1
                    )
                self._step_count += 1
                self._stats_steps += 1
                if self._imbalance_recovery_latched:
                    if (
                        self._specialist_stage == "recovering"
                        and self._recovery_steps_remaining <= 0
                    ):
                        if self._specialist_promoted:
                            self._try_activate_specialist(step_dt)
                        else:
                            self._begin_specialist_fine_tuning(step_dt)
                    elif (
                        self._specialist_stage
                        in {"fine_tuning", "waiting_checkpoint"}
                        and self._specialist_wait_steps_remaining <= 0
                    ):
                        self._try_activate_specialist(step_dt)
                    elif (
                        self._specialist_stage == "idle"
                        and self._recovery_steps_remaining <= 0
                    ):
                        if self._reset_after_safety:
                            self.reset_environment()
                        else:
                            self._release_safety_recovery()
            return True
        except KeyError as exc:
            if exc.args == ("critic",) and _ensure_playback_observation_groups(self.env):
                self._last_error = traceback.format_exc()
                self.log(
                    "[WARN] Repaired missing critic observation group; retrying next frame.",
                    VerbosityLevel.INFO,
                )
                # Do not pause for this known, recoverable playback bookkeeping
                # issue. Returning False clears the current sim-time budget and
                # lets the next viewer tick retry from a clean observation read.
                return False
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during routed step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during routed step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False

    def _four_point_safety_action(self, *, aggressive: bool = False) -> torch.Tensor:
        """Drive into a stable hands-and-lower-limbs four-point safety pose."""
        cached = self._fast_four_point_action if aggressive else self._four_point_action
        if cached is not None:
            return cached
        base = getattr(self.env, "unwrapped", self.env)
        action_term = base.action_manager.get_term("joint_pos")
        names = action_term.target_names
        entity = base.scene["robot"]
        default = entity.data.default_joint_pos[:, action_term.target_ids]
        target = default.clone()
        for index, name in enumerate(names):
            if re.search(r"hip_pitch", name):
                target[:, index] = -0.75 if aggressive else -0.65
            elif re.search(r"knee", name):
                target[:, index] = 1.30 if aggressive else 1.15
            elif re.search(r"ankle_pitch", name):
                target[:, index] = -0.55 if aggressive else -0.50
            elif name == "left_hip_roll_joint":
                target[:, index] = 0.15
            elif name == "right_hip_roll_joint":
                target[:, index] = -0.15
            elif re.search(r"hip_yaw|ankle_roll", name):
                target[:, index] = 0.0
            elif name == "waist_pitch_joint":
                target[:, index] = 0.48 if aggressive else 0.45
            elif re.search(r"waist_roll|waist_yaw", name):
                target[:, index] = 0.0
            elif re.search(r"shoulder_pitch", name):
                target[:, index] = -1.55
            elif name == "left_shoulder_roll_joint":
                target[:, index] = 0.60
            elif name == "right_shoulder_roll_joint":
                target[:, index] = -0.60
            elif re.search(r"shoulder_yaw", name):
                target[:, index] = 0.0
            elif re.search(r"elbow", name):
                target[:, index] = 0.65
            elif re.search(r"wrist_pitch", name):
                target[:, index] = -0.45
            elif re.search(r"wrist_roll|wrist_yaw", name):
                target[:, index] = 0.0
        scale = action_term.scale
        offset = action_term.offset
        if not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale, device=target.device)
        if not isinstance(offset, torch.Tensor):
            offset = torch.as_tensor(offset, device=target.device)
        action = (target - offset) / scale.clamp_min(1.0e-6)
        if aggressive:
            self._fast_four_point_action = action
        else:
            self._four_point_action = action
        return action

    def _record_policy_decision(
            self, context: TerrainContext, decision: RoutingDecision
    ) -> None:
        signature = (
            decision.action.value,
            decision.target_key,
            decision.terrain_type,
        )
        if signature == self._policy_history_signature:
            return
        self._policy_history_signature = signature
        step_count = self.get_status().step_count
        action = {
            RouterAction.USE_POLICY: "SELECT",
            RouterAction.SWITCH_POLICY: "SWAP",
            RouterAction.WAIT_FOR_POLICY: "WAIT",
            RouterAction.FINE_TUNE_NEW_POLICY: "FINE TUNING NEW POLICY",
        }[decision.action]
        self._policy_history.appendleft(
            f"#{step_count:05d} {action} → {decision.target_label} · {context_summary(context)}"
        )


def _find_controls_tab(gui: Any) -> Any | None:
    """Locate the stock Viser Controls tab so settings stay in the existing UI."""
    registry = getattr(gui, "_container_handle_from_uuid", None)
    if not isinstance(registry, Mapping):
        return None
    for handle in registry.values():
        if isinstance(handle, viser.GuiTabHandle) and getattr(handle, "_label", None) == "Controls":
            return handle
    return None


def _random_event_controller(env: Any) -> Any | None:
    base = getattr(env, "unwrapped", env)
    controller = getattr(base, "random_event_controller", None)
    required = (
        "automatic_mode",
        "manual_event_state",
        "set_automatic_mode",
        "set_manual_event",
    )
    if controller is None or not all(callable(getattr(controller, name, None)) for name in required):
        return None
    return controller


def _active_random_event_names(env: Any, env_idx: int) -> tuple[str, ...]:
    """Read the physical event cause from the repository-owned sequencer."""
    controller = _random_event_controller(env)
    provider = getattr(controller, "active_event_names", None) if controller else None
    if callable(provider):
        try:
            return tuple(str(name) for name in provider(int(env_idx)))
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return ()
    return ()


def _adaptive_specialist_condition(
    context: TerrainContext, active_events: Sequence[str]
) -> str | None:
    """Name the low-traction condition eligible for the trained incline policy."""
    active = {str(name).lower() for name in active_events}
    if active:
        low_friction = "ice" in active
        slope = "slope" in active
    else:
        low_friction = context.friction is not None and context.friction < 0.30
        slope = abs(context.slope_gradient or 0.0) > 0.06
    if low_friction and slope:
        return "low-friction + slope"
    if low_friction:
        return "low-friction"
    if slope:
        return "slope"
    return None


def _apply_trail_command(
    env: Any,
    env_idx: int,
    trail_frame: TrailFrame,
    *,
    forward_speed_m_s: float,
    lateral_gain: float,
    max_lateral_speed_m_s: float,
) -> None:
    """Command forward motion along the local trail tangent with recentering."""
    base = getattr(env, "unwrapped", env)
    try:
        term = base.command_manager.get_term("twist")
        robot = base.scene["robot"]
        tangent_x = float(trail_frame.tangent_x)
        tangent_y = float(trail_frame.tangent_y)
        normal_x = -tangent_y
        normal_y = tangent_x
        lateral_error = float(trail_frame.lateral_offset_m)
        if abs(lateral_error) <= TRAIL_FOLLOW_LATERAL_DEADBAND_M:
            corrected_lateral_error = 0.0
        else:
            corrected_lateral_error = math.copysign(
                abs(lateral_error) - TRAIL_FOLLOW_LATERAL_DEADBAND_M,
                lateral_error,
            )
        lateral_speed = max(
            -max_lateral_speed_m_s,
            min(
                max_lateral_speed_m_s,
                -lateral_gain * corrected_lateral_error,
            ),
        )
        forward_w = forward_speed_m_s * tangent_x + lateral_speed * normal_x
        lateral_w = forward_speed_m_s * tangent_y + lateral_speed * normal_y
        env_idx = int(env_idx)
        term.is_standing_env[env_idx] = False
        term.is_forward_env[env_idx] = False
        term.is_world_env[env_idx] = True
        term.vel_command_w[env_idx, 0] = forward_w
        term.vel_command_w[env_idx, 1] = lateral_w

        # Observations are read before the environment's next command-manager
        # update, so compute the matching body-frame command immediately.
        heading = robot.data.heading_w[env_idx]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        term.vel_command_b[env_idx, 0] = cos_h * forward_w + sin_h * lateral_w
        term.vel_command_b[env_idx, 1] = -sin_h * forward_w + cos_h * lateral_w
        target_heading = math.atan2(tangent_y, tangent_x)
        heading_error = math.atan2(
            math.sin(target_heading - float(heading.item())),
            math.cos(target_heading - float(heading.item())),
        )
        if abs(heading_error) <= TRAIL_FOLLOW_HEADING_DEADBAND_RAD:
            corrected_heading_error = 0.0
        else:
            corrected_heading_error = math.copysign(
                abs(heading_error) - TRAIL_FOLLOW_HEADING_DEADBAND_RAD,
                heading_error,
            )
        yaw_rate = max(
            -TRAIL_FOLLOW_MAX_YAW_RATE_RAD_S,
            min(
                TRAIL_FOLLOW_MAX_YAW_RATE_RAD_S,
                TRAIL_FOLLOW_HEADING_GAIN * corrected_heading_error,
            ),
        )
        term.vel_command_w[env_idx, 2] = yaw_rate
        term.vel_command_b[env_idx, 2] = yaw_rate
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
        return


def _apply_centering_command(env: Any, env_idx: int, trail_frame: TrailFrame) -> None:
    """Drive the specialist conservatively toward and along the trail center."""
    _apply_trail_command(
        env,
        env_idx,
        trail_frame,
        forward_speed_m_s=CENTERING_FORWARD_SPEED_M_S,
        lateral_gain=CENTERING_LATERAL_GAIN,
        max_lateral_speed_m_s=CENTERING_MAX_LATERAL_SPEED_M_S,
    )


def _apply_trail_following_command(
    env: Any, env_idx: int, trail_frame: TrailFrame
) -> None:
    """Keep normal dashboard walking aligned with the winding route."""
    _apply_trail_command(
        env,
        env_idx,
        trail_frame,
        forward_speed_m_s=0.5,
        lateral_gain=TRAIL_FOLLOW_LATERAL_GAIN,
        max_lateral_speed_m_s=TRAIL_FOLLOW_MAX_LATERAL_SPEED_M_S,
    )


def _imbalance_recovery_decision(
    baseline: RoutingDecision,
    risk: ImbalanceRisk,
    *,
    reason: str = "",
) -> RoutingDecision:
    left_path = reason.lstrip().lower().startswith("left the path")
    return replace(
        baseline,
        action=RouterAction.SWITCH_POLICY,
        terrain_type="off_path" if left_path else "imbalance",
        target_key="imbalance_recovery",
        target_label="Four-point safety pose",
        model="deterministic safety controller",
        readiness=PolicyReadiness.AVAILABLE,
        confidence=0.99,
        headline=(
            "Left the path — execute four-point safety pose"
            if left_path
            else "Execute four-point safety pose"
        ),
        detail=(
            f"{reason} Locomotion is stopped while a hands-and-lower-limbs stance is commanded."
            if reason
            else "Safety recovery latched; locomotion is stopped while the four-point pose is commanded."
        ),
        training_request=None,
    )


def _measure_imbalance_risk(env: Any, env_idx: int) -> ImbalanceRisk:
    """Read only onboard-style IMU and foot-contact sensors."""
    base = getattr(env, "unwrapped", env)
    try:
        up_vector = base.scene["robot/imu_upvector"].data[env_idx]
        angular_velocity = base.scene["robot/imu_ang_vel"].data[env_idx]
        contact_data = base.scene["feet_ground_contact"].data.found
        feet_in_contact = (
            int(torch.count_nonzero(contact_data[env_idx] > 0).item())
            if contact_data is not None
            else None
        )
        return predict_imbalance(
            up_vector.detach().cpu().tolist(),
            angular_velocity.detach().cpu().tolist(),
            feet_in_contact,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        # Missing sensor data must fail open so it cannot freeze normal walking.
        return ImbalanceRisk(
            tilt_degrees=0.0,
            tipping_rate_rad_s=0.0,
            projected_tilt_degrees=0.0,
            feet_in_contact=None,
            triggered=False,
        )


def _measure_trail_frame(env: Any, env_idx: int) -> TrailFrame | None:
    """Measure the robot against the nearest point on the winding centerline."""
    base = getattr(env, "unwrapped", env)
    try:
        robot = base.scene["robot"]
        root_x = float(robot.data.root_link_pos_w[env_idx, 0].item())
        root_y = float(robot.data.root_link_pos_w[env_idx, 1].item())
        origin_x = float(base.scene.env_origins[env_idx, 0].item())
        origin_y = float(base.scene.env_origins[env_idx, 1].item())

        local_x = root_x - origin_x
        local_y = root_y - origin_y

        # Randomized slope events are compact ramp patches placed ahead of the
        # robot; the treadmill body itself remains flat. Therefore the winding
        # centerline stays in its authored XY coordinates during every event.
        return nearest_trail_frame(local_x, local_y)
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _measure_lateral_offset_m(env: Any, env_idx: int) -> float | None:
    """Compatibility wrapper returning signed distance from the curved trail."""
    frame = _measure_trail_frame(env, env_idx)
    return None if frame is None else frame.lateral_offset_m


def _capture_locomotion_command(
    env: Any, env_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Snapshot the current locomotion command before the safety controller zeros it."""
    base = getattr(env, "unwrapped", env)
    try:
        term = base.command_manager.get_term("twist")
        return (
            term.vel_command_b[env_idx].clone(),
            term.vel_command_w[env_idx].clone(),
            term.is_standing_env[env_idx].clone(),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, IndexError):
        return None


def _restore_locomotion_command(
    env: Any,
    env_idx: int,
    snapshot: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    base = getattr(env, "unwrapped", env)
    try:
        term = base.command_manager.get_term("twist")
        vel_b, vel_w, standing = snapshot
        term.vel_command_b[env_idx].copy_(vel_b)
        term.vel_command_w[env_idx].copy_(vel_w)
        term.is_standing_env[env_idx].copy_(standing)
    except (AttributeError, KeyError, RuntimeError, TypeError, IndexError):
        return


def _stop_locomotion(env: Any, env_idx: int) -> None:
    """Cancel planar travel while allowing gravity to settle the sit posture."""
    base = getattr(env, "unwrapped", env)
    try:
        robot = base.scene["robot"]
        velocity = robot.data.root_link_vel_w.clone()
        velocity[env_idx, :2] = 0.0
        robot.write_root_link_velocity_to_sim(velocity)
        term = base.command_manager.get_term("twist")
        term.vel_command_b[env_idx] = 0.0
        term.vel_command_w[env_idx] = 0.0
        term.is_standing_env[env_idx] = True
    except (AttributeError, KeyError, RuntimeError, TypeError, IndexError):
        return

def extract_terrain_context(
        env: Any, *, env_idx: int = 0, checkpoint_name: str | None = None
) -> TerrainContext:
    """Read scenario telemetry when available, then fill gaps from MjLab state.

    Other agents can integrate without importing the viewer by exposing one of:

    * ``env.unwrapped.get_policy_router_context(env_idx) -> Mapping``
    * ``env.unwrapped.policy_router_context`` (mapping or per-env sequence)
    * ``env.unwrapped.scenario_state`` / ``terrain_context`` with the same shape

    The mapping accepts aliases such as ``slope``, ``mu``, ``roughness``,
    ``step_height``, ``wind`` and ``slip``.
    """
    base = getattr(env, "unwrapped", env)
    published = _read_published_context(base, env_idx)
    inferred = _infer_context_from_mjlab(base, env_idx, checkpoint_name)
    if published is None:
        return inferred

    values: dict[str, Any] = {}
    for field in fields(TerrainContext):
        published_value = getattr(published, field.name)
        inferred_value = getattr(inferred, field.name)
        if field.name in {"source", "uncertainty"}:
            values[field.name] = published_value
        else:
            values[field.name] = published_value if published_value is not None else inferred_value
    return TerrainContext(**values)


def render_policy_overlay(
        context: TerrainContext,
        decision: RoutingDecision,
        *,
        history: Sequence[str] = (),
        execution: PolicyExecution | None = None,
        log_entries: Sequence[Any] = (),
        imbalance_risk: ImbalanceRisk | None = None,
        recovery_active: bool = False,
        recovery_reason: str = "",
        current_step: int | None = None,
) -> str:
    """Render the concise committed policy-action stream."""
    del context, history, imbalance_risk
    execution = execution or resolve_policy_execution(
        decision, available_policy_keys=FOUNDATIONAL_POLICY_KEYS
    )

    messages: list[str] = []

    safety_banner = ""
    if recovery_active:
        reason = recovery_reason.strip().rstrip(".")
        text = (
            "Robot imbalance detected. Activating safety recovery to prevent damage."
            if reason == "Robot imbalance detected"
            else f"{reason}. Activating safety recovery."
            if reason
            else "Activating safety recovery."
        )
        safety_banner = _safety_banner(
            text,
            f"step {current_step}" if isinstance(current_step, int) else "live",
        )

    for entry in tuple(log_entries)[:24]:
        category = str(getattr(entry, "category", "LOG"))
        if recovery_active and category == "SAFETY":
            # The active yellow banner is the single source of truth for the
            # current incident. Avoid showing the same safety event twice.
            continue
        message = str(getattr(entry, "message", entry))
        entry_step = getattr(entry, "step", None)
        color = {
            "ACTION": "#35d07f",
            "FINE TUNING": "#ff6b7a",
            "POLICY ADDED": "#35d07f",
            "CHECKPOINT MISSING": "#f5b942",
            "TRAINING_REQUIRED": "#ff6b7a",
            "SAFETY": "#f5b942",
        }.get(category, "#7d8797")
        messages.append(
            _log_message(
                category,
                message,
                f"step {entry_step}" if entry_step is not None else "log",
                color,
            )
        )

    if not messages and not recovery_active:
        if decision.action == RouterAction.FINE_TUNE_NEW_POLICY:
            label = "FINE TUNING"
            text = (
                f"Detected {decision.terrain_type}; waiting safely in recovery position "
                "and sending sensor data for fine tuning."
            )
        else:
            label = "ACTION"
            text = f"Detected {decision.terrain_type} environment, executing {execution.executed_label} policy."
        messages.append(
            _log_message(
                label,
                text,
                f"step {current_step}" if isinstance(current_step, int) else "live",
                "#f5b942" if label == "SAFETY" else "#ff6b7a" if label == "FINE TUNING" else "#35d07f",
            )
        )

    return (
        '<div style="font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
        'background:#0d1015;border:1px solid #252b35;border-radius:9px;padding:8px;">'
        + safety_banner
        + '<div style="max-height:335px;overflow-y:auto;padding-right:3px;">'
        + "".join(messages)
        + "</div></div>"
    )


def _log_message(label: str, text: str, meta: str, color: str) -> str:
    """One compact chat-style log message."""
    return (
        '<div style="padding:8px 9px;margin-bottom:6px;background:#12161c;'
        'border:1px solid #202630;border-left:2px solid ' + color + ';border-radius:7px;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
        f'<span style="font:700 8px ui-monospace,SFMono-Regular,monospace;color:{color};'
        f'letter-spacing:.07em;">{_esc(label)}</span>'
        f'<span style="font:500 8px ui-monospace,SFMono-Regular,monospace;color:#515b68;">{_esc(meta)}</span>'
        "</div>"
        f'<div style="font-size:10.5px;line-height:1.45;color:#c3cad4;margin-top:4px;">{_esc(text)}</div>'
        "</div>"
    )


def _safety_banner(text: str, meta: str) -> str:
    """Prominent active-safety banner kept separate from the normal action log."""
    return (
        '<div style="padding:9px 10px;margin-bottom:8px;background:rgba(245,185,66,.16);'
        'border:1px solid rgba(245,185,66,.55);border-left:4px solid #f5b942;'
        'border-radius:7px;box-shadow:0 0 0 1px rgba(245,185,66,.05) inset;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
        '<span style="font:800 8px ui-monospace,SFMono-Regular,monospace;color:#f5b942;'
        'letter-spacing:.08em;">SAFETY</span>'
        f'<span style="font:600 8px ui-monospace,SFMono-Regular,monospace;color:#9a8047;">{_esc(meta)}</span>'
        '</div>'
        f'<div style="font-size:10.5px;line-height:1.45;color:#f2d89b;margin-top:4px;font-weight:650;">{_esc(text)}</div>'
        '</div>'
    )

def render_supervisor_log(entries: Sequence[Any]) -> str:
    """Render the dedicated low-noise decision/execution event stream."""
    if not entries:
        rows = (
            '<div style="padding:18px 4px;color:#6f7987;font-size:10px;line-height:1.45;">'
            "Waiting for a stable terrain observation before committing a route.</div>"
        )
    else:
        rendered: list[str] = []
        for entry in entries:
            category = str(getattr(entry, "category", "TRACE"))
            message = _esc(getattr(entry, "message", entry))
            step = _esc(getattr(entry, "step", "—"))
            color = {
                "ACTION": "#35d07f",
                "FINE TUNING NEW POLICY": "#ff6b7a",
                "FINE TUNING POLICY": "#ff6b7a",
                "POLICY ADDED": "#35d07f",
                "CHECKPOINT MISSING": "#f5b942",
            }.get(category, "#818b99")
            rendered.append(
                '<div style="display:grid;grid-template-columns:7px 46px 1fr;gap:8px;padding:8px 0;'
                'border-bottom:1px solid #202630;align-items:start;">'
                f'<div style="width:6px;height:6px;border-radius:999px;background:{color};margin-top:4px;"></div>'
                f'<div style="font:650 8px/1.4 ui-monospace,SFMono-Regular,monospace;color:#626d7c;">#{step}</div>'
                '<div style="min-width:0;">'
                f'<div style="font:720 8px ui-monospace,SFMono-Regular,monospace;color:{color};letter-spacing:.06em;">{_esc(category)}</div>'
                f'<div style="font-size:10px;line-height:1.45;color:#aeb7c4;margin-top:3px;">{message}</div>'
                '</div></div>'
            )
        rows = "".join(rendered)
    return f"""
    <div style="font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#f4f7fb;background:#0d1015;padding:12px;border:1px solid #252b35;border-radius:10px;">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
        <div>
          <div style="font-size:12px;font-weight:720;color:#f7f8fa;">Decision log</div>
          <div style="font:600 8px ui-monospace,SFMono-Regular,monospace;color:#687383;letter-spacing:.09em;margin-top:2px;">SENSE → ROUTE → EXECUTE</div>
        </div>
        <div style="padding:3px 6px;border:1px solid #49232a;border-radius:5px;background:#201318;font:650 8px ui-monospace,SFMono-Regular,monospace;color:#e97b88;letter-spacing:.05em;">TRAINING OFF</div>
      </div>
      <div style="font-size:9px;color:#626d7c;line-height:1.45;margin-top:9px;padding-bottom:8px;border-bottom:1px solid #242a33;">
        Only committed state changes are logged. Training-required events are signals; this viewer never starts a training job or blends policies implicitly.
      </div>
      <div style="max-height:390px;overflow:auto;padding-right:2px;">{rows}</div>
    </div>
    """


def _read_published_context(base: Any, env_idx: int) -> TerrainContext | None:
    provider = getattr(base, "get_policy_router_context", None)
    if callable(provider):
        provider = cast(Callable[..., Any], provider)
        try:
            payload = provider(env_idx)
        except TypeError:
            payload = provider()
        parsed = _parse_published_payload(payload, env_idx, source="scenario adapter")
        if parsed is not None:
            return parsed

    for attr_name in ("policy_router_context", "scenario_state", "terrain_context"):
        payload = getattr(base, attr_name, None)
        parsed = _parse_published_payload(payload, env_idx, source=attr_name)
        if parsed is not None:
            return parsed
    return None


def _parse_published_payload(
        payload: Any, env_idx: int, *, source: str
) -> TerrainContext | None:
    if payload is None:
        return None
    if isinstance(payload, TerrainContext):
        return replace(payload, source=source)
    if isinstance(payload, Mapping):
        return TerrainContext.from_mapping(payload, source=source)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if 0 <= env_idx < len(payload):
            return _parse_published_payload(payload[env_idx], env_idx, source=source)
    if hasattr(payload, "__dict__"):
        return TerrainContext.from_mapping(vars(payload), source=source)
    return None


def _infer_context_from_mjlab(
        base: Any, env_idx: int, checkpoint_name: str | None
) -> TerrainContext:
    friction = _active_event_surface_friction(base, env_idx)
    if friction is None:
        friction = sampled_foot_friction(base, env_idx)
    if friction is None:
        friction = _fixed_event_value(base, "foot_friction", "ranges")

    local_terrain = estimate_local_terrain(base, env_idx)
    slope = _actual_treadmill_slope(base, env_idx)
    if slope is None:
        slope = _fixed_terrain_slope(base)
    if slope is None:
        slope = local_terrain.slope_gradient
    wind = _actual_wind_force(base, env_idx)
    if wind is None:
        wind = _fixed_wind_force(base)
    torso_height, fallen = _posture_state(base, env_idx)
    active_policy = canonical_policy_key(checkpoint_name)
    terrain_label = local_terrain.terrain_label
    if slope is not None and slope > 0.06 and terrain_label == "flat":
        terrain_label = "incline"
    has_geometry = local_terrain.sample_count >= 5
    return TerrainContext(
        slope_gradient=slope,
        friction=friction,
        roughness_m=local_terrain.roughness_m,
        step_height_m=local_terrain.step_height_m,
        wind_force_n=wind,
        fallen=fallen,
        torso_height_m=torso_height,
        active_policy=active_policy,
        terrain_label=terrain_label,
        source=(
            "MuJoCo local terrain + MjLab runtime state"
            if has_geometry
            else "MjLab state + task config"
        ),
        uncertainty=0.14 if has_geometry else 0.22,
    )


def _actual_treadmill_slope(base: Any, env_idx: int) -> float | None:
    slope = getattr(base, "treadmill_slope_gradient", None)
    if slope is None:
        return None
    try:
        value = slope[env_idx]
        return float(value.item() if hasattr(value, "item") else value)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return None


def _active_event_surface_friction(base: Any, env_idx: int) -> float | None:
    """Return an upcoming localized event surface's friction when published."""
    controller = getattr(base, "random_event_controller", None)
    provider = getattr(controller, "active_surface_friction", None)
    if not callable(provider):
        return None
    try:
        value = provider(int(env_idx))
        return None if value is None else float(value)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return None


def _fixed_event_value(base: Any, event_name: str, param_name: str) -> float | None:
    cfg = getattr(base, "cfg", None)
    events = getattr(cfg, "events", None)
    term = events.get(event_name) if isinstance(events, Mapping) else None
    params = getattr(term, "params", None)
    value = params.get(param_name) if isinstance(params, Mapping) else None
    return _fixed_scalar_range(value)


def _fixed_terrain_slope(base: Any) -> float | None:
    cfg = getattr(base, "cfg", None)
    scene_cfg = getattr(cfg, "scene", None)
    terrain_cfg = getattr(scene_cfg, "terrain", None)
    generator = getattr(terrain_cfg, "terrain_generator", None)
    sub_terrains = getattr(generator, "sub_terrains", None)
    if not isinstance(sub_terrains, Mapping):
        return None
    for sub_terrain in sub_terrains.values():
        slope = _fixed_scalar_range(getattr(sub_terrain, "slope_range", None))
        if slope is not None:
            return abs(slope)
    return None


def _actual_wind_force(base: Any, env_idx: int) -> float | None:
    scene = getattr(base, "scene", None)
    if scene is None:
        return None
    try:
        robot = scene["robot"]
        wrench = robot.data.body_external_wrench[env_idx, :, :3]
        return float(wrench.square().sum(dim=-1).sqrt().max().item())
    except (AttributeError, KeyError, IndexError, TypeError, RuntimeError):
        return None


def _fixed_wind_force(base: Any) -> float | None:
    cfg = getattr(base, "cfg", None)
    events = getattr(cfg, "events", None)
    term = events.get("wind") if isinstance(events, Mapping) else None
    params = getattr(term, "params", None)
    ranges = params.get("force_ranges") if isinstance(params, Mapping) else None
    if not isinstance(ranges, Mapping):
        return None
    components: list[float] = []
    for axis in ("x", "y", "z"):
        value = _fixed_scalar_range(ranges.get(axis))
        if value is None:
            return None
        components.append(value)
    return sum(component * component for component in components) ** 0.5


def _posture_state(base: Any, env_idx: int) -> tuple[float | None, bool | None]:
    scene = getattr(base, "scene", None)
    if scene is None:
        return None, None
    try:
        robot = scene["robot"]
        pos = robot.data.root_link_pos_w[env_idx]
        quat = robot.data.root_link_quat_w[env_idx]
        height = float(pos[2].item())
        # MjLab/MuJoCo uses scalar-first quaternions: w, x, y, z.
        x = float(quat[1].item())
        y = float(quat[2].item())
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return height, bool(height < 0.55 or up_z < 0.45)
    except (AttributeError, KeyError, IndexError, TypeError, RuntimeError):
        return None, None


def _fixed_scalar_range(value: Any) -> float | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError):
        return None
    if abs(lo - hi) > 1.0e-6:
        return None
    return (lo + hi) * 0.5


def _action_presentation(action: RouterAction) -> tuple[str, str]:
    return {
        RouterAction.USE_POLICY: ("#35d07f", "ROUTE STABLE"),
        RouterAction.SWITCH_POLICY: ("#75bfff", "ROUTE CHANGE"),
        RouterAction.WAIT_FOR_POLICY: ("#f5b942", "CHECKPOINT MISSING"),
        RouterAction.FINE_TUNE_NEW_POLICY: ("#ff6b7a", "FINE TUNING NEW POLICY"),
    }[action]


def _flow_cell(label: str, value: str, color: str) -> str:
    return (
        '<div style="min-width:0;">'
        f'<div style="font:650 7px ui-monospace,SFMono-Regular,monospace;color:#596473;letter-spacing:.09em;">{_esc(label)}</div>'
        f'<div style="font-size:9px;font-weight:680;color:{color};margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_esc(value)}</div>'
        '</div>'
    )


def _telemetry_cell(label: str, value: str) -> str:
    return (
        '<div style="min-width:0;padding:6px 6px 5px;background:#11151b;border:1px solid #202630;border-radius:6px;">'
        f'<div style="font:650 7px ui-monospace,SFMono-Regular,monospace;color:#566170;letter-spacing:.08em;">{_esc(label)}</div>'
        f'<div style="font:650 9px ui-monospace,SFMono-Regular,monospace;color:#b7c0cc;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_esc(value)}</div>'
        '</div>'
    )


def _reasoning_row(label: str, text: str, color: str) -> str:
    return (
        '<div style="position:relative;display:flex;justify-content:center;">'
        f'<span style="width:6px;height:6px;border-radius:999px;background:{color};display:block;margin-top:5px;"></span>'
        '</div>'
        '<div style="padding:0 0 8px 0;min-width:0;">'
        f'<span style="font:700 8px ui-monospace,SFMono-Regular,monospace;color:{color};letter-spacing:.07em;margin-right:6px;">{_esc(label)}</span>'
        f'<span style="font-size:10px;line-height:1.42;color:#929dab;">{_esc(text)}</span>'
        '</div>'
    )


def _comparison_trace(decision: RoutingDecision) -> str:
    if decision.target_key == "recovery":
        ranked = list(decision.evaluations[:2])
    else:
        ranked = [
            evaluation
            for evaluation in decision.evaluations
            if evaluation.spec.key != "recovery"
        ][:2]
    if not ranked:
        return "No policy candidates available."
    parts = [f"{evaluation.spec.label} {evaluation.score * 100:.0f}% fit" for evaluation in ranked]
    return "; ".join(parts) + "."


def _fmt_slope(context: TerrainContext) -> str:
    slope_degrees = context.slope_degrees
    return f"{slope_degrees:.1f} deg" if slope_degrees is not None else "—"


def _fmt_friction(context: TerrainContext) -> str:
    return f"mu {context.friction:.2f}" if context.friction is not None else "—"


def _fmt_relief(context: TerrainContext) -> str:
    rough = context.roughness_m
    step = context.step_height_m
    if rough is None and step is None:
        return "—"
    rough_cm = (rough or 0.0) * 100.0
    step_cm = (step or 0.0) * 100.0
    if step_cm >= max(rough_cm, 0.5):
        return f"step {step_cm:.1f} cm"
    return f"rough {rough_cm:.1f} cm"


def _fmt_force(value: float | None) -> str:
    return f"{value:.0f} N" if value is not None else "—"


def _fmt_balance(risk: ImbalanceRisk | None, recovery_active: bool = False) -> str:
    if risk is None:
        return "—"
    state = "RECOVERY" if recovery_active else "UNSTABLE" if risk.triggered else "OK"
    return f"{state} {risk.tilt_degrees:.1f} deg"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _loading_html() -> str:
    return (
        '<div style="font-family:ui-monospace,monospace;padding:12px;color:#94a3b8;">'
        "Connecting terrain telemetry…</div>"
    )


def _loading_log_html() -> str:
    return (
        '<div style="font-family:ui-monospace,monospace;padding:12px;color:#94a3b8;">'
        "Waiting for the first routed action…</div>"
    )
