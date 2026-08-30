"""Deterministic failure detection and policy/retraining demo lifecycle.

The thresholds and three-frame confirmation mirror humanoid_climber's
``safety.py`` and its policy categories.  This module owns no actuator path;
the MuJoCo engine records the same decision it actually executes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Any


DYNAMIC_TILT_DEGREES = 18.0
DYNAMIC_TIPPING_RATE_RAD_S = 0.35
SEVERE_TILT_DEGREES = 40.0
UNSUPPORTED_TILT_DEGREES = 12.0
UNSUPPORTED_TIPPING_RATE_RAD_S = 0.25
PREDICTION_HORIZON_S = 0.20
PREDICTED_TILT_DEGREES = 28.0
PREDICTED_MIN_TILT_DEGREES = 10.0
PREDICTED_MIN_TIPPING_RATE_RAD_S = 0.25
IMBALANCE_CONFIRMATION_FRAMES = 3


@dataclass(frozen=True)
class FailureRisk:
    tilt_degrees: float
    tipping_rate_rad_s: float
    projected_tilt_degrees: float
    feet_in_contact: int
    triggered: bool


@dataclass(frozen=True)
class PolicyRoute:
    terrain_type: str
    requested_key: str
    requested_label: str
    confidence: float
    needs_training: bool
    reason: str


def predict_imbalance(
    up_vector_xyz: tuple[float, float, float],
    angular_velocity_xyz_rad_s: tuple[float, float, float],
    feet_in_contact: int,
) -> FailureRisk:
    up_x, up_y, up_z = (float(value) for value in up_vector_xyz)
    norm = math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)
    if not math.isfinite(norm) or norm <= 1.0e-8:
        return FailureRisk(0.0, 0.0, 0.0, feet_in_contact, False)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z / norm))))
    rate = math.hypot(float(angular_velocity_xyz_rad_s[0]), float(angular_velocity_xyz_rad_s[1]))
    projected = tilt + math.degrees(rate * PREDICTION_HORIZON_S)
    triggered = (
        tilt >= SEVERE_TILT_DEGREES
        or (tilt >= DYNAMIC_TILT_DEGREES and rate >= DYNAMIC_TIPPING_RATE_RAD_S)
        or (
            feet_in_contact == 0
            and tilt >= UNSUPPORTED_TILT_DEGREES
            and rate >= UNSUPPORTED_TIPPING_RATE_RAD_S
        )
        or (
            tilt >= PREDICTED_MIN_TILT_DEGREES
            and rate >= PREDICTED_MIN_TIPPING_RATE_RAD_S
            and projected >= PREDICTED_TILT_DEGREES
        )
    )
    return FailureRisk(tilt, rate, projected, feet_in_contact, triggered)


def route_policy(context: dict[str, Any]) -> PolicyRoute:
    """Port the stable routing classes from humanoid_climber/policy_router.py."""
    slope = abs(float(context.get("slope_gradient") or 0.0))
    friction = float(context.get("friction") if context.get("friction") is not None else 0.8)
    roughness = float(context.get("roughness_m") or 0.0)
    wind = float(context.get("wind_force_n") or 0.0)
    fallen = bool(context.get("fallen", False))
    if fallen:
        return PolicyRoute("fallen / recovery", "recovery", "Supine recovery", 0.98, True, "Failure detector confirmed a fall.")
    if slope > 0.20 or friction < 0.10 or roughness > 0.10 or wind > 18.0:
        return PolicyRoute("out-of-envelope terrain", "new_specialist", "New terrain specialist", 0.88, True, "Observed conditions exceed all validated envelopes.")
    if roughness > 0.03:
        return PolicyRoute("rough / stepped terrain", "rough", "Rough-terrain walker", 0.82, True, "No compatible rough-terrain checkpoint is loaded.")
    if wind > 6.0 and (slope > 0.06 or friction < 0.15):
        return PolicyRoute("wind + low-traction incline", "new_specialist", "Combined-condition specialist", 0.84, True, "Wind and low traction require a combined specialist.")
    if wind > 6.0:
        return PolicyRoute("wind-exposed terrain", "wind", "Wind walker", 0.86, True, "Wind specialist requested.")
    if slope > 0.06 or friction < 0.30:
        return PolicyRoute("low-traction incline", "ice_incline", "Low-friction incline", 0.90, True, "Slope or traction crossed the flat-walker envelope.")
    return PolicyRoute("flat / nominal terrain", "flat", "Flat-ground walker", 0.94, False, "Nominal conditions fit the stock policy envelope.")


class PolicySupervisor:
    def __init__(self) -> None:
        self.failure_candidate_frames = 0
        self.failure_latched = False
        self.risk = FailureRisk(0.0, 0.0, 0.0, 2, False)
        self.route = route_policy({})
        self.stage = "monitoring"
        self.request_id: str | None = None
        self.request_manifest: str | None = None
        self.requested_at: float | None = None
        self.active_policy_key = "flat"
        self.active_policy_label = "Flat-ground walker"
        self.executed_checkpoint = ""
        self.demo_pretrained = False
        self.entries: deque[dict[str, Any]] = deque(maxlen=48)
        self._last_signature: tuple[str, str] | None = None

    def log(self, category: str, message: str, *, sim_time: float) -> None:
        self.entries.appendleft({
            "time": time.time(),
            "sim_time": float(sim_time),
            "category": category,
            "message": message,
        })

    def observe(self, risk: FailureRisk, context: dict[str, Any], *, sim_time: float) -> bool:
        self.risk = risk
        self.route = route_policy({**context, "fallen": risk.tilt_degrees >= SEVERE_TILT_DEGREES})
        signature = (self.route.terrain_type, self.route.requested_key)
        if signature != self._last_signature:
            self._last_signature = signature
            self.log(
                "ROUTE",
                f"Detected {self.route.terrain_type}; requested {self.route.requested_label}.",
                sim_time=sim_time,
            )
        self.failure_candidate_frames = self.failure_candidate_frames + 1 if risk.triggered else 0
        confirmed = self.failure_candidate_frames >= IMBALANCE_CONFIRMATION_FRAMES
        if confirmed and not self.failure_latched:
            self.failure_latched = True
            self.stage = "failure_detected"
            self.log(
                "FAILURE DETECTED",
                f"Tilt {risk.tilt_degrees:.1f} deg, projected {risk.projected_tilt_degrees:.1f} deg, feet {risk.feet_in_contact}.",
                sim_time=sim_time,
            )
            return True
        return False

    def request_training(self, request_id: str, manifest: str, *, sim_time: float) -> None:
        self.stage = "waiting_checkpoint"
        self.request_id = request_id
        self.request_manifest = manifest
        self.requested_at = time.time()
        self.log(
            "RETRAIN REQUEST",
            f"Captured Newton subset {request_id}; active safety posture remains engaged while awaiting a checkpoint.",
            sim_time=sim_time,
        )

    def activate_policy(
        self,
        key: str,
        label: str,
        checkpoint: str,
        *,
        sim_time: float,
        demo_pretrained: bool = False,
    ) -> None:
        self.active_policy_key = key
        self.active_policy_label = label
        self.executed_checkpoint = checkpoint
        self.demo_pretrained = demo_pretrained
        self.stage = "policy_active"
        self.failure_latched = False
        self.failure_candidate_frames = 0
        suffix = " (demo-pretrained surrogate)" if demo_pretrained else ""
        self.log("POLICY ACTIVE", f"Executing {label}{suffix}: {checkpoint}", sim_time=sim_time)

    def manifest(self) -> dict[str, Any]:
        return {
            "detector": {
                "kind": "deterministic_imu_contact",
                "source": "humanoid_climber/safety.py",
                "confirmation_frames": IMBALANCE_CONFIRMATION_FRAMES,
                "risk": asdict(self.risk),
            },
            "route": asdict(self.route),
            "stage": self.stage,
            "request_id": self.request_id,
            "request_manifest": self.request_manifest,
            "requested_at": self.requested_at,
            "active_policy_key": self.active_policy_key,
            "active_policy_label": self.active_policy_label,
            "executed_checkpoint": self.executed_checkpoint,
            "demo_pretrained": self.demo_pretrained,
            "decision_log": list(self.entries),
        }
