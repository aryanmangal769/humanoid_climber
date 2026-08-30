"""Deterministic specialist-policy showcase sequence."""

from __future__ import annotations

from dataclasses import dataclass
import math


STAGE_DURATION_SECONDS = 12.0
POLICY_ANNOUNCEMENT_DELAY_SECONDS = 1.0
INCLINE_GRADIENT = 0.20
INCLINE_FRICTION = 0.15
INCLINE_ANGLE_RANGE_DEG = (10.0, 30.0)
INCLINE_GRADIENT_RANGE = tuple(
  math.tan(math.radians(angle)) for angle in INCLINE_ANGLE_RANGE_DEG
)
INCLINE_FRICTION_RANGE = (0.1, 0.3)
HIGH_WIND_FORCE_RANGES = {
  "x": (0.0, 0.0),
  "y": (8.0, 20.0),
  "z": (0.0, 0.0),
}


@dataclass(frozen=True)
class ShowcaseStage:
  key: str
  label: str
  policy_key: str
  event_name: str | None


SHOWCASE_STAGES = (
  ShowcaseStage("normal", "Normal terrain", "flat", None),
  ShowcaseStage("incline", "Low-friction incline", "ice_incline", "incline"),
  ShowcaseStage("wind", "Variable crosswind", "wind", "wind"),
  ShowcaseStage("rough", "Rough terrain", "rough", "bumps"),
)


class ShowcaseClock:
  """Advance fixed stages and expose a delayed SummitOS announcement gate."""

  def __init__(
    self,
    *,
    stage_duration_s: float = STAGE_DURATION_SECONDS,
    announcement_delay_s: float = POLICY_ANNOUNCEMENT_DELAY_SECONDS,
  ) -> None:
    if stage_duration_s <= 0.0:
      raise ValueError("stage_duration_s must be positive")
    if not 0.0 <= announcement_delay_s < stage_duration_s:
      raise ValueError("announcement_delay_s must be within the stage duration")
    self.stage_duration_s = float(stage_duration_s)
    self.announcement_delay_s = float(announcement_delay_s)
    self.stage_index = 0
    self.time_remaining_s = self.stage_duration_s
    self.paused = False

  @property
  def current(self) -> ShowcaseStage:
    return SHOWCASE_STAGES[self.stage_index]

  @property
  def upcoming(self) -> ShowcaseStage:
    return SHOWCASE_STAGES[(self.stage_index + 1) % len(SHOWCASE_STAGES)]

  @property
  def requested_policy(self) -> ShowcaseStage:
    """Return the specialist for the physical condition already on screen."""
    return self.current

  @property
  def announcement_ready(self) -> bool:
    """Wait until the physical condition has been visible for one second."""
    elapsed_s = self.stage_duration_s - self.time_remaining_s
    return elapsed_s + 1.0e-9 >= self.announcement_delay_s

  def reset(self) -> None:
    self.stage_index = 0
    self.time_remaining_s = self.stage_duration_s
    self.paused = False

  def pause(self) -> None:
    self.paused = True

  def resume(self) -> None:
    self.paused = False

  def advance(self, step_dt: float) -> bool:
    """Advance time and return True exactly when the physical stage changes."""
    if self.paused:
      return False
    self.time_remaining_s -= max(0.0, float(step_dt))
    if self.time_remaining_s > 1.0e-9:
      return False
    self.stage_index = (self.stage_index + 1) % len(SHOWCASE_STAGES)
    self.time_remaining_s = self.stage_duration_s
    return True
