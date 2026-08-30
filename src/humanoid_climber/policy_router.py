"""Heuristic terrain context and policy routing for the live demo UI.

This module intentionally contains no model loading or action blending.  It is a
small, inspectable supervisor used by the viewer while the real policy registry
and scenario runners are being built by other agents.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any


class PolicyReadiness(str, Enum):
    """Current deployment state of a specialist policy."""

    AVAILABLE = "available"
    TRAINING_DISABLED = "training_disabled"


class RouterAction(str, Enum):
    """UI-level action recommended by the heuristic supervisor."""

    USE_POLICY = "use_policy"
    SWITCH_POLICY = "switch_policy"
    WAIT_FOR_POLICY = "wait_for_policy"
    FINE_TUNE_NEW_POLICY = "fine_tuning_new_policy"


@dataclass(frozen=True)
class TerrainContext:
    """Small telemetry contract shared between scenarios and the UI.

    Scenario code may publish these values through ``get_policy_router_context``
    or a ``policy_router_context`` mapping on the unwrapped MjLab environment.
    Unknown values should remain ``None`` rather than being guessed.
    """

    slope_gradient: float | None = None
    friction: float | None = None
    roughness_m: float | None = None
    step_height_m: float | None = None
    wind_force_n: float | None = None
    slip_ratio: float | None = None
    fallen: bool | None = None
    torso_height_m: float | None = None
    active_policy: str | None = None
    terrain_label: str | None = None
    source: str = "viewer inference"
    uncertainty: float = 0.15

    @classmethod
    def from_mapping(
            cls, payload: Mapping[str, Any], *, source: str | None = None
    ) -> "TerrainContext":
        """Normalize common scenario field names into the viewer contract."""
        aliases = {
            "slope": "slope_gradient",
            "gradient": "slope_gradient",
            "terrain_gradient": "slope_gradient",
            "mu": "friction",
            "estimated_friction": "friction",
            "roughness": "roughness_m",
            "terrain_roughness": "roughness_m",
            "step_height": "step_height_m",
            "wind": "wind_force_n",
            "wind_n": "wind_force_n",
            "slip": "slip_ratio",
            "is_fallen": "fallen",
            "height": "torso_height_m",
            "policy": "active_policy",
            "terrain_type": "terrain_label",
        }
        valid = {field.name for field in fields(cls)}
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = aliases.get(key, key)
            if normalized_key not in valid:
                continue
            if normalized_key == "wind_force_n":
                value = _vector_magnitude(value)
            normalized[normalized_key] = value
        if source is not None:
            normalized["source"] = source
        return cls(**normalized)

    @property
    def slope_degrees(self) -> float | None:
        if self.slope_gradient is None:
            return None
        return math.degrees(math.atan(self.slope_gradient))


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    model: str
    readiness: PolicyReadiness
    limits: Mapping[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class PolicyEvaluation:
    spec: PolicySpec
    score: float
    blockers: tuple[str, ...]
    checked: int


@dataclass(frozen=True)
class RoutingDecision:
    action: RouterAction
    terrain_type: str
    target_key: str
    target_label: str
    model: str
    readiness: PolicyReadiness | None
    confidence: float
    headline: str
    detail: str
    evaluations: tuple[PolicyEvaluation, ...]
    training_request: str | None = None


@dataclass(frozen=True)
class PolicyExecution:
    """Logical policy selected by the demo supervisor for the next action cycle."""

    requested_key: str
    executed_key: str
    executed_label: str
    used_fallback: bool
    reason: str


@dataclass(frozen=True)
class SupervisorLogEntry:
    step: int
    category: str
    message: str


class SupervisorEventLog:
    """Debounced, low-noise decision trace for the demo supervisor."""

    def __init__(
        self,
        *,
        max_entries: int = 40,
        stable_observations: int = 3,
        min_steps_between_commits: int = 50,
    ) -> None:
        self.entries: deque[SupervisorLogEntry] = deque(maxlen=max_entries)
        self._stable_observations = stable_observations
        self._min_steps_between_commits = min_steps_between_commits
        self._committed_signature: tuple[Any, ...] | None = None
        self._committed_context: TerrainContext | None = None
        self.committed_decision: RoutingDecision | None = None
        self.committed_execution: PolicyExecution | None = None
        self._last_commit_step: int | None = None
        self._candidate_signature: tuple[Any, ...] | None = None
        self._candidate_count = 0

    def observe(
        self,
        context: TerrainContext,
        decision: RoutingDecision,
        execution: PolicyExecution,
        *,
        step: int,
    ) -> bool:
        """Record a stable condition transition and its real action policy.

        Returns ``True`` when a new condition was committed.  The same method is
        called immediately before the selected policy produces an action, so the
        trace and actuator path share one decision.
        """
        signature = condition_signature(context, decision)
        changed = False
        if signature == self._committed_signature:
            self._candidate_signature = None
            self._candidate_count = 0
        else:
            if signature != self._candidate_signature:
                self._candidate_signature = signature
                self._candidate_count = 1
            else:
                self._candidate_count += 1
            dwell_satisfied = (
                self._last_commit_step is None
                or step - self._last_commit_step >= self._min_steps_between_commits
            )
            if self._candidate_count >= self._stable_observations and dwell_satisfied:
                self._commit(context, decision, execution, step=step)
                changed = True

        return changed

    def _commit(
        self,
        context: TerrainContext,
        decision: RoutingDecision,
        execution: PolicyExecution,
        *,
        step: int,
    ) -> None:
        self._committed_signature = self._candidate_signature
        self._committed_context = context
        self.committed_decision = decision
        self.committed_execution = execution
        self._last_commit_step = step
        self._candidate_signature = None
        self._candidate_count = 0

        if decision.action == RouterAction.FINE_TUNE_NEW_POLICY:
            self._append(
                step,
                "FINE TUNING",
                f"Detected {decision.terrain_type}; waiting safely in recovery "
                "position and sending sensor data for fine tuning.",
            )
        else:
            self._append(
                step,
                "ACTION",
                f"Detected {decision.terrain_type} environment, executing "
                f"{execution.executed_label} policy.",
            )

    def _append(self, step: int, category: str, message: str) -> None:
        self.entries.appendleft(
            SupervisorLogEntry(step=step, category=category, message=message)
        )

    def record_safety_action(self, *, step: int, message: str) -> None:
        """Record an immediate safety override without waiting for route debounce."""
        if (
            self.entries
            and self.entries[0].category == "SAFETY"
            and self.entries[0].message == message
        ):
            return
        self._append(step, "SAFETY", message)

    def record_policy_lifecycle(
        self, *, step: int, category: str, message: str
    ) -> None:
        """Record an explicit demo policy handoff/promotion lifecycle event."""
        self._append(step, category, message)

    def reset_route_state(self) -> None:
        """Forget the prior episode's committed route while retaining its log."""
        self._committed_signature = None
        self._committed_context = None
        self.committed_decision = None
        self.committed_execution = None
        self._last_commit_step = None
        self._candidate_signature = None
        self._candidate_count = 0


POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(
        key="flat",
        label="Flat-ground walker",
        model="g1_velocity_model_final.pt",
        readiness=PolicyReadiness.AVAILABLE,
        limits={
            "slope_gradient": (None, 0.06),
            "friction": (0.30, None),
            "roughness_m": (None, 0.025),
            "step_height_m": (None, 0.04),
            "wind_force_n": (None, 6.0),
        },
    ),
    PolicySpec(
        key="ice_incline",
        label="Low-friction incline",
        model="model_34400.pt",
        readiness=PolicyReadiness.AVAILABLE,
        limits={
            "slope_gradient": (None, 0.20),
            "friction": (0.005, None),
            "roughness_m": (None, 0.03),
            "step_height_m": (None, 0.04),
            "wind_force_n": (None, 6.0),
        },
    ),
    PolicySpec(
        key="wind",
        label="Flat-ground wind walker",
        model="flat_wind candidate",
        readiness=PolicyReadiness.AVAILABLE,
        limits={
            "slope_gradient": (None, 0.06),
            "friction": (0.15, None),
            "roughness_m": (None, 0.025),
            "step_height_m": (None, 0.04),
            "wind_force_n": (None, 18.0),
        },
    ),
    PolicySpec(
        key="recovery",
        label="Supine recovery",
        model="recovery candidate",
        readiness=PolicyReadiness.AVAILABLE,
        limits={},
    ),
    PolicySpec(
        key="rough",
        label="Rough-terrain walker",
        model="rough terrain specialist",
        readiness=PolicyReadiness.AVAILABLE,
        limits={
            "slope_gradient": (None, 0.20),
            "friction": (0.20, None),
            "roughness_m": (None, 0.10),
            "step_height_m": (None, 0.15),
            "wind_force_n": (None, 6.0),
        },
    ),
)

_POLICY_BY_KEY = {policy.key: policy for policy in POLICIES}
FOUNDATIONAL_POLICY_KEYS: tuple[str, ...] = tuple(policy.key for policy in POLICIES)


def route_policy(context: TerrainContext) -> RoutingDecision:
    """Route a terrain snapshot using transparent safety-first heuristics."""
    evaluations = tuple(sorted((_evaluate(p, context) for p in POLICIES), key=lambda e: e.score, reverse=True))
    terrain_type, preferred_key, request = _classify(context)

    if preferred_key is None:
        confidence = _context_confidence(context, 0.44)
        return RoutingDecision(
            action=RouterAction.FINE_TUNE_NEW_POLICY,
            terrain_type=terrain_type,
            target_key="new_specialist",
            target_label="New terrain specialist",
            model="no checkpoint; training execution disabled",
            readiness=None,
            confidence=confidence,
            headline="No existing specialist covers this combination",
            detail=request or "Observed conditions exceed the current specialist envelopes.",
            evaluations=evaluations,
            training_request="Sending sensor data to fine tune policy.",
        )

    target = _POLICY_BY_KEY[preferred_key]
    target_eval = next(e for e in evaluations if e.spec.key == preferred_key)
    confidence = _context_confidence(context, target_eval.score)

    if target_eval.blockers:
        blocker_text = "; ".join(target_eval.blockers[:2])
        request_text = f"{target.label} extension for {terrain_type}: {blocker_text}"
        return RoutingDecision(
            action=RouterAction.FINE_TUNE_NEW_POLICY,
            terrain_type=terrain_type,
            target_key=target.key,
            target_label=target.label,
            model=target.model,
            readiness=target.readiness,
            confidence=confidence,
            headline=f"{target.label} is the closest match, but outside its envelope",
            detail=blocker_text,
            evaluations=evaluations,
            training_request="Sending sensor data to fine tune policy.",
        )

    if target.readiness == PolicyReadiness.TRAINING_DISABLED:
        return RoutingDecision(
            action=RouterAction.FINE_TUNE_NEW_POLICY,
            terrain_type=terrain_type,
            target_key=target.key,
            target_label=target.label,
            model=target.model,
            readiness=target.readiness,
            confidence=confidence,
            headline=f"Fine tuning new {target.label} policy",
            detail=(
                "No deployable checkpoint exists. The decision log emits a "
                "signal, but training execution is disabled."
            ),
            evaluations=evaluations,
            training_request="Sending sensor data to fine tune policy.",
        )

    active = canonical_policy_key(context.active_policy)
    action = RouterAction.USE_POLICY if active in (None, target.key) else RouterAction.SWITCH_POLICY
    verb = "Keep" if action == RouterAction.USE_POLICY and active == target.key else "Select"
    if action == RouterAction.SWITCH_POLICY:
        verb = "Switch to"
    return RoutingDecision(
        action=action,
        terrain_type=terrain_type,
        target_key=target.key,
        target_label=target.label,
        model=target.model,
        readiness=target.readiness,
        confidence=confidence,
        headline=f"{verb} {target.label}",
        detail=_selection_detail(target.key, context),
        evaluations=evaluations,
        training_request=None,
    )


def resolve_policy_execution(
    decision: RoutingDecision,
    *,
    available_policy_keys: Sequence[str],
    active_policy_key: str | None = None,
) -> PolicyExecution:
    """Resolve against the demo's assumed foundational specialist bank."""
    available = set(_POLICY_BY_KEY)
    available.update({
        key
        for value in available_policy_keys
        if (key := canonical_policy_key(value)) is not None
    })
    active = canonical_policy_key(active_policy_key)
    if decision.target_key in available:
        target = _POLICY_BY_KEY[decision.target_key]
        return PolicyExecution(
            requested_key=decision.target_key,
            executed_key=decision.target_key,
            executed_label=target.label,
            used_fallback=False,
            reason="The specialist is part of the assumed foundational policy bank.",
        )

    ranked_available = [
        evaluation
        for evaluation in decision.evaluations
        if evaluation.spec.key in available
    ]
    fallback_key = (
        active
        if active in available
        else ranked_available[0].spec.key
        if ranked_available
        else "flat"
    )
    fallback = _POLICY_BY_KEY.get(fallback_key, _POLICY_BY_KEY["flat"])
    requested = (
        decision.target_label
        if decision.target_key != "new_specialist"
        else "a new specialist"
    )
    return PolicyExecution(
        requested_key=decision.target_key,
        executed_key=fallback.key,
        executed_label=fallback.label,
        used_fallback=True,
        reason=(
            f"{requested} is not loaded; this is the closest executable policy. "
            "The log reports the fallback instead of pretending the requested checkpoint ran."
        ),
    )


def canonical_policy_key(value: str | None) -> str | None:
    """Resolve checkpoint names and human labels to a router policy key."""
    if not value:
        return None
    lowered = value.lower()
    if "34400" in lowered or "ice" in lowered or "incline" in lowered:
        return "ice_incline"
    if "wind" in lowered:
        return "wind"
    if "recover" in lowered or "supine" in lowered or "getup" in lowered:
        return "recovery"
    if "rough" in lowered:
        return "rough"
    if "velocity_model_final" in lowered or "flat" in lowered or "stock" in lowered:
        return "flat"
    if lowered in _POLICY_BY_KEY:
        return lowered
    return None


def context_summary(context: TerrainContext) -> str:
    """Compact one-line observation summary used by the agent-style UI trace."""
    chunks: list[str] = []
    if context.terrain_label:
        chunks.append(context.terrain_label)
    if context.friction is not None:
        chunks.append(f"mu {context.friction:.2f}")
    slope_degrees = context.slope_degrees
    if slope_degrees is not None:
        chunks.append(f"slope {slope_degrees:.1f} deg")
    if context.roughness_m is not None:
        chunks.append(f"rough {context.roughness_m * 100:.1f} cm")
    if context.step_height_m is not None:
        chunks.append(f"step {context.step_height_m * 100:.1f} cm")
    if context.wind_force_n is not None:
        chunks.append(f"wind {context.wind_force_n:.0f} N")
    if context.slip_ratio is not None:
        chunks.append(f"slip {context.slip_ratio:.2f}")
    if context.fallen:
        chunks.append("fallen")
    return " | ".join(chunks) if chunks else "terrain telemetry pending"


def condition_signature(
    context: TerrainContext, decision: RoutingDecision
) -> tuple[Any, ...]:
    """Identify the stable routing state, independent of raw sensor jitter."""
    return (
        decision.terrain_type,
        decision.target_key,
        bool(context.fallen),
    )


def describe_condition_change(
    previous: TerrainContext | None, current: TerrainContext
) -> str:
    """Describe only the dimensions that changed enough to affect routing."""
    if previous is None:
        return context_summary(current)
    changes: list[str] = []
    fields_and_labels = (
        ("slope_gradient", "slope", 0.01),
        ("friction", "friction", 0.02),
        ("roughness_m", "roughness", 0.01),
        ("step_height_m", "step", 0.01),
        ("wind_force_n", "wind", 1.0),
    )
    for field_name, label, threshold in fields_and_labels:
        old = getattr(previous, field_name)
        new = getattr(current, field_name)
        if old is None and new is None:
            continue
        if old is None or new is None or abs(float(new) - float(old)) >= threshold:
            changes.append(f"{label} {_format_change(old, new, field_name)}")
    if previous.fallen != current.fallen:
        changes.append("fall state changed")
    return ", ".join(changes) if changes else context_summary(current)


def _format_change(old: float | None, new: float | None, field_name: str) -> str:
    if old is None:
        return f"unknown → {float(new):.2f}"
    if new is None:
        return f"{float(old):.2f} → unknown"
    suffix = " N" if field_name == "wind_force_n" else ""
    return f"{float(old):.2f} → {float(new):.2f}{suffix}"


def _classify(context: TerrainContext) -> tuple[str, str | None, str | None]:
    slope = abs(context.slope_gradient or 0.0)
    friction = context.friction
    rough = context.roughness_m or 0.0
    step = context.step_height_m or 0.0
    wind = context.wind_force_n or 0.0
    slip = context.slip_ratio or 0.0

    if context.fallen:
        return "fallen / recovery", "recovery", None

    extreme_reasons: list[str] = []
    if slope > 0.20:
        extreme_reasons.append(f"gradient {slope:.2f} > 0.20")
    if friction is not None and friction < 0.005:
        extreme_reasons.append(f"friction {friction:.3f} < 0.005")
    if rough > 0.10:
        extreme_reasons.append(f"roughness {rough:.2f} m > 0.10 m")
    if step > 0.15:
        extreme_reasons.append(f"step {step:.2f} m > 0.15 m")
    if wind > 18.0:
        extreme_reasons.append(f"wind {wind:.0f} N > 18 N")
    if extreme_reasons:
        return "out-of-envelope terrain", None, "; ".join(extreme_reasons)

    rough_hazard = rough > 0.03 or step > 0.05
    wind_hazard = wind > 6.0
    slippery_hazard = (friction is not None and friction < 0.30) or slip > 0.10
    incline_hazard = slope > 0.06

    if rough_hazard and wind_hazard:
        return (
            "rough terrain + wind",
            None,
            "Combined roughness and wind are not covered by a deployable specialist.",
        )
    if rough_hazard:
        return "rough / stepped terrain", "rough", None
    # The flat-wind specialist is explicitly trained down to friction 0.15, so
    # low friction alone does not make wind a mixed/unsupported condition.
    wind_traction_outside_target = friction is not None and friction < 0.15
    if wind_hazard and (incline_hazard or wind_traction_outside_target):
        return (
            "wind + low-traction incline",
            None,
            "Wind combined with slope or traction below the wind target needs a combined specialist.",
        )
    if wind_hazard:
        return "wind-exposed flat terrain", "wind", None
    if incline_hazard:
        return "low-traction incline", "ice_incline", None
    if slippery_hazard:
        return "low-friction ice", "ice_incline", None
    return "flat / nominal terrain", "flat", None


def _evaluate(policy: PolicySpec, context: TerrainContext) -> PolicyEvaluation:
    if policy.key == "recovery":
        score = 0.98 if context.fallen else 0.05
        blockers = () if context.fallen else ("robot is upright",)
        return PolicyEvaluation(policy, score, blockers, 1)

    blockers: list[str] = []
    penalties = 0.0
    checked = 0
    for field_name, (minimum, maximum) in policy.limits.items():
        value = getattr(context, field_name)
        if value is None:
            continue
        checked += 1
        value = abs(float(value)) if field_name == "slope_gradient" else float(value)
        if minimum is not None and value < minimum:
            blockers.append(f"{_pretty_field(field_name)} {value:.2f} < {minimum:.2f}")
            penalties += 0.55 + min(0.35, (minimum - value) / max(abs(minimum), 0.1) * 0.25)
        if maximum is not None and value > maximum:
            blockers.append(f"{_pretty_field(field_name)} {value:.2f} > {maximum:.2f}")
            penalties += 0.55 + min(0.35, (value - maximum) / max(abs(maximum), 0.1) * 0.25)

    score = 0.88 - penalties / max(checked, 1)
    score += _affinity_bonus(policy.key, context)
    if checked < 3:
        score -= (3 - checked) * 0.04
    return PolicyEvaluation(policy, max(0.01, min(0.99, score)), tuple(blockers), checked)


def _affinity_bonus(policy_key: str, context: TerrainContext) -> float:
    slope = abs(context.slope_gradient or 0.0)
    friction = context.friction if context.friction is not None else 0.8
    rough = context.roughness_m or 0.0
    step = context.step_height_m or 0.0
    wind = context.wind_force_n or 0.0
    if policy_key == "flat" and slope <= 0.04 and friction >= 0.30 and wind <= 4.0:
        return 0.08
    if policy_key == "ice_incline" and (slope > 0.06 or friction < 0.30):
        return 0.09
    if policy_key == "wind" and wind > 6.0:
        return 0.09
    if policy_key == "rough" and (rough > 0.03 or step > 0.05):
        return 0.09
    return 0.0


def _selection_detail(policy_key: str, context: TerrainContext) -> str:
    if policy_key == "ice_incline":
        return "Slope / traction crossed the flat-walker threshold while remaining inside the incline specialist target."
    if policy_key == "flat":
        return "No terrain hazard currently exceeds the nominal flat-walker thresholds."
    return f"{context_summary(context)} fits the selected specialist envelope."


def _context_confidence(context: TerrainContext, fit_score: float) -> float:
    known = sum(
        value is not None
        for value in (
            context.slope_gradient,
            context.friction,
            context.roughness_m,
            context.step_height_m,
            context.wind_force_n,
            context.slip_ratio,
            context.fallen,
        )
    )
    coverage = min(1.0, known / 5.0)
    uncertainty = max(0.0, min(1.0, context.uncertainty))
    score = fit_score * (0.72 + 0.28 * coverage) * (1.0 - 0.35 * uncertainty)
    return max(0.05, min(0.99, score))


def _pretty_field(name: str) -> str:
    return {
        "slope_gradient": "gradient",
        "friction": "friction",
        "roughness_m": "roughness",
        "step_height_m": "step",
        "wind_force_n": "wind",
    }.get(name, name)


def _vector_magnitude(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return math.sqrt(sum(float(component) ** 2 for component in value))
        except (TypeError, ValueError):
            return value
    return value
