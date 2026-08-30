"""Small, deterministic safety checks for the live treadmill demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# These thresholds describe body imbalance, not treadmill geometry.  They are
# intentionally *pre-fall* thresholds: waiting for the legacy 45-degree/low-
# torso fall detector leaves too little time for a protective posture.  Normal
# G1 walking keeps the torso well inside this envelope, while a fast-growing
# roll/pitch excursion is caught roughly a few tenths of a second earlier.
DYNAMIC_TILT_DEGREES = 18.0
DYNAMIC_TIPPING_RATE_RAD_S = 0.35
SEVERE_TILT_DEGREES = 40.0
UNSUPPORTED_TILT_DEGREES = 12.0
UNSUPPORTED_TIPPING_RATE_RAD_S = 0.25
PREDICTION_HORIZON_S = 0.20
PREDICTED_TILT_DEGREES = 28.0
# Do not extrapolate ordinary gait angular velocity from an almost-upright pose
# into a fake future fall. The predictive branch only arms once there is already
# a meaningful torso lean.
PREDICTED_MIN_TILT_DEGREES = 10.0
PREDICTED_MIN_TIPPING_RATE_RAD_S = 0.25
CENTERLINE_MAX_OFFSET_M = 1.0

# Three policy frames (~60 ms) reject brief gait oscillations while still
# reacting far ahead of the legacy fall detector.
IMBALANCE_CONFIRMATION_FRAMES = 3

# The four-point safety pose is an emergency posture, not the eight-second
# learned get-up motion. Reach the commanded hands-and-lower-limbs stance in
# four policy frames and keep it latched briefly so the walking controller
# cannot immediately fight it.
PROTECTIVE_SIT_ATTACK_FRAMES = 4
PROTECTIVE_SIT_HOLD_FRAMES = 20


@dataclass(frozen=True)
class ImbalanceRisk:
  """One sensor-only estimate of whether the robot is tipping over."""

  tilt_degrees: float
  tipping_rate_rad_s: float
  projected_tilt_degrees: float
  feet_in_contact: int | None
  triggered: bool


@dataclass
class ImbalanceMonitor:
  """Debounce transient gait motion before confirming an imbalance."""

  confirmation_frames: int = IMBALANCE_CONFIRMATION_FRAMES
  candidate_frames: int = 0

  def observe(self, risk: ImbalanceRisk) -> bool:
    if risk.triggered:
      self.candidate_frames += 1
    else:
      self.candidate_frames = 0
    return self.candidate_frames >= self.confirmation_frames

  def reset(self) -> None:
    self.candidate_frames = 0


def outside_centerline(
  lateral_position_m: float,
  max_offset_m: float = CENTERLINE_MAX_OFFSET_M,
) -> bool:
  """Return whether the robot has left the treadmill's lateral safety corridor.

  The treadmill runs along world X, so its centerline is world ``y == 0``.
  Crossing exactly 1 m is still allowed; the emergency sit starts only once
  the absolute lateral displacement is greater than the configured limit.
  """
  lateral_position_m = float(lateral_position_m)
  max_offset_m = float(max_offset_m)
  return (
    math.isfinite(lateral_position_m)
    and math.isfinite(max_offset_m)
    and max_offset_m >= 0.0
    and abs(lateral_position_m) > max_offset_m
  )


def predict_imbalance(
  up_vector_xyz: Sequence[float],
  angular_velocity_xyz_rad_s: Sequence[float],
  feet_in_contact: int | None,
) -> ImbalanceRisk:
  """Estimate tipping risk using only IMU and foot-contact sensor readings.

  ``up_vector_xyz`` is the world-up direction expressed in the IMU frame.  A
  level robot therefore reads approximately ``(0, 0, 1)``.  Yaw rate is
  deliberately ignored because spinning about a vertical axis is not tipping.
  """
  if len(up_vector_xyz) < 3 or len(angular_velocity_xyz_rad_s) < 2:
    raise ValueError("Imbalance prediction requires 3D IMU sensor readings")

  up_x, up_y, up_z = (float(value) for value in up_vector_xyz[:3])
  up_norm = math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)
  if not math.isfinite(up_norm) or up_norm <= 1.0e-8:
    return ImbalanceRisk(0.0, 0.0, 0.0, feet_in_contact, False)
  upright_cosine = max(-1.0, min(1.0, up_z / up_norm))
  tilt_degrees = math.degrees(math.acos(upright_cosine))

  roll_rate = float(angular_velocity_xyz_rad_s[0])
  pitch_rate = float(angular_velocity_xyz_rad_s[1])
  tipping_rate = math.hypot(roll_rate, pitch_rate)
  if not math.isfinite(tipping_rate):
    return ImbalanceRisk(
      tilt_degrees, 0.0, tilt_degrees, feet_in_contact, False
    )

  projected_tilt = tilt_degrees + math.degrees(
    tipping_rate * PREDICTION_HORIZON_S
  )

  severe_tilt = tilt_degrees >= SEVERE_TILT_DEGREES
  dynamic_tip = (
    tilt_degrees >= DYNAMIC_TILT_DEGREES
    and tipping_rate >= DYNAMIC_TIPPING_RATE_RAD_S
  )
  unsupported_tip = (
    feet_in_contact == 0
    and tilt_degrees >= UNSUPPORTED_TILT_DEGREES
    and tipping_rate >= UNSUPPORTED_TIPPING_RATE_RAD_S
  )
  predicted_tip = (
    tilt_degrees >= PREDICTED_MIN_TILT_DEGREES
    and tipping_rate >= PREDICTED_MIN_TIPPING_RATE_RAD_S
    and projected_tilt >= PREDICTED_TILT_DEGREES
  )
  return ImbalanceRisk(
    tilt_degrees=tilt_degrees,
    tipping_rate_rad_s=tipping_rate,
    projected_tilt_degrees=projected_tilt,
    feet_in_contact=feet_in_contact,
    triggered=severe_tilt or dynamic_tip or unsupported_tip or predicted_tip,
  )
