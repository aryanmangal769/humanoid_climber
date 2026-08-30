"""Procedural infinite winding trail geometry for the dashboard simulation.

The route advances monotonically in world +X and never closes back on itself.
Only a fixed-size window of short visual segments is materialized by MuJoCo;
the mathematical centerline itself is defined for arbitrary X so steering,
safety checks, and terrain-event placement continue indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TRAIL_HALF_WIDTH_M = 1.20
TRAIL_SEGMENT_OVERLAP_M = 0.10
TRAIL_SEGMENT_X_SPACING_M = 1.35
TRAIL_WINDOW_BEHIND_M = 16.0
TRAIL_WINDOW_AHEAD_M = 92.0
TRAIL_MARKER_STRIDE = 9

# The route is a sum of several very long, incommensurate waves.  Using
# ``1-cos`` terms makes y(0)=0 and dy/dx(0)=0, so the stock G1 spawn enters on a
# straight section, while the mixed wavelengths prevent the trail from reading
# like a simple repeating sine wave farther forward.
_WINDING_TERMS: tuple[tuple[float, float], ...] = (
    (5.2, 30.0),
    (-3.7, 17.0),
    (2.2, 9.5),
)


@dataclass(frozen=True)
class TrailFrame:
    """Nearest local trail frame for one XY position."""

    center_x: float
    center_y: float
    tangent_x: float
    tangent_y: float
    lateral_offset_m: float
    distance_m: float
    segment_index: int


@dataclass(frozen=True)
class TrailSegment:
    """One short box-friendly segment sampled from the procedural centerline."""

    index: int
    start: tuple[float, float]
    end: tuple[float, float]
    center: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    length: float
    yaw: float


def trail_center_y(x: float) -> float:
    """Return the centerline Y coordinate for arbitrary forward coordinate X."""
    x = float(x)
    return sum(amplitude * (1.0 - math.cos(x / wavelength)) for amplitude, wavelength in _WINDING_TERMS)


def trail_center_slope(x: float) -> float:
    """Return dy/dx of the procedural centerline."""
    x = float(x)
    return sum((amplitude / wavelength) * math.sin(x / wavelength) for amplitude, wavelength in _WINDING_TERMS)


def _trail_center_second_derivative(x: float) -> float:
    x = float(x)
    return sum(
        (amplitude / (wavelength * wavelength)) * math.cos(x / wavelength)
        for amplitude, wavelength in _WINDING_TERMS
    )


def _segment_from_x(index: int, start_x: float, end_x: float) -> TrailSegment:
    start = (start_x, trail_center_y(start_x))
    end = (end_x, trail_center_y(end_x))
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    tangent = (dx / length, dy / length)
    normal = (-tangent[1], tangent[0])
    return TrailSegment(
        index=index,
        start=start,
        end=end,
        center=((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5),
        tangent=tangent,
        normal=normal,
        length=length,
        yaw=math.atan2(dy, dx),
    )


def trail_window_segments(center_x: float) -> tuple[TrailSegment, ...]:
    """Return a fixed-size recycled geometry window around ``center_x``.

    Segment indices are *pool slots*, not global path IDs.  That keeps the
    MuJoCo geom names stable while their positions are recycled farther forward.
    """
    first_x = math.floor(
        (float(center_x) - TRAIL_WINDOW_BEHIND_M) / TRAIL_SEGMENT_X_SPACING_M
    ) * TRAIL_SEGMENT_X_SPACING_M
    span = TRAIL_WINDOW_BEHIND_M + TRAIL_WINDOW_AHEAD_M
    count = math.ceil(span / TRAIL_SEGMENT_X_SPACING_M) + 2
    return tuple(
        _segment_from_x(
            slot,
            first_x + slot * TRAIL_SEGMENT_X_SPACING_M,
            first_x + (slot + 1) * TRAIL_SEGMENT_X_SPACING_M,
        )
        for slot in range(count)
    )


# Compile-time geometry pool. Runtime code repositions these exact slots as the
# robot advances, so the number of MuJoCo geoms never grows.
TRAIL_SEGMENTS: tuple[TrailSegment, ...] = trail_window_segments(0.0)


def nearest_trail_frame(x: float, y: float) -> TrailFrame:
    """Project ``(x, y)`` onto the infinite procedural centerline.

    Because X is monotonic, the nearest curve parameter is close to the query X.
    A few bounded Newton iterations solve the exact perpendicular projection
    without scanning a growing list of generated segments.
    """
    x = float(x)
    y = float(y)
    parameter_x = x
    for _ in range(6):
        center_y = trail_center_y(parameter_x)
        slope = trail_center_slope(parameter_x)
        second = _trail_center_second_derivative(parameter_x)
        residual = (parameter_x - x) + (center_y - y) * slope
        derivative = 1.0 + slope * slope + (center_y - y) * second
        if abs(derivative) <= 1.0e-8:
            break
        step = max(-3.0, min(3.0, residual / derivative))
        parameter_x -= step
        if abs(step) <= 1.0e-7:
            break

    center_y = trail_center_y(parameter_x)
    slope = trail_center_slope(parameter_x)
    norm = math.hypot(1.0, slope)
    tangent_x = 1.0 / norm
    tangent_y = slope / norm
    normal_x = -tangent_y
    normal_y = tangent_x
    error_x = x - parameter_x
    error_y = y - center_y
    lateral = error_x * normal_x + error_y * normal_y
    return TrailFrame(
        center_x=parameter_x,
        center_y=center_y,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        lateral_offset_m=lateral,
        distance_m=math.hypot(error_x, error_y),
        segment_index=math.floor(parameter_x / TRAIL_SEGMENT_X_SPACING_M),
    )


def trail_frame_ahead(x: float, y: float, distance_m: float) -> TrailFrame:
    """Return a centerline frame ``distance_m`` forward along the open trail."""
    nearest = nearest_trail_frame(float(x), float(y))
    remaining = max(0.0, float(distance_m))
    parameter_x = nearest.center_x
    # Arc-length integration with short steps is more than sufficient for the
    # short event lookahead and keeps this helper valid for arbitrary distances.
    while remaining > 1.0e-8:
        slope = trail_center_slope(parameter_x)
        arc_per_x = math.hypot(1.0, slope)
        arc_step = min(0.45, remaining)
        parameter_x += arc_step / arc_per_x
        remaining -= arc_step
    slope = trail_center_slope(parameter_x)
    norm = math.hypot(1.0, slope)
    return TrailFrame(
        center_x=parameter_x,
        center_y=trail_center_y(parameter_x),
        tangent_x=1.0 / norm,
        tangent_y=slope / norm,
        lateral_offset_m=0.0,
        distance_m=0.0,
        segment_index=math.floor(parameter_x / TRAIL_SEGMENT_X_SPACING_M),
    )
