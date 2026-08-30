"""Packaged procedural MuJoCo mountain-ridge generator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import mujoco


MOUNTAIN_RANGE_SPECS: tuple[tuple[float, float, float], ...] = (
  (46.0, 0.58, 0.30),
  (64.0, 0.42, 2.10),
)
MOUNTAIN_SIDE_CLEARANCE_M = (31.0, 49.0)
MOUNTAIN_SIDE_HEIGHT_MULTIPLIER = 0.65
MOUNTAIN_SIDE_X_RANGE_M = (-80.0, 82.0)
MOUNTAIN_DEPTH_OFFSETS_M = (0.0, 3.0, 8.0, 15.0, 24.0, 34.0)
MOUNTAIN_DEPTH_SCALES = (0.03, 0.22, 0.58, 1.00, 0.55, 0.08)
PROFILE_MIN_M = -210.0
PROFILE_MAX_M = 210.0


def _mountain_height(coord: float, *, height_scale: float, phase: float) -> float:
  peaks = (
    (-176.0, 28.0, 18.0),
    (-142.0, 23.0, 27.0),
    (-108.0, 24.0, 22.0),
    (-76.0, 20.0, 35.0),
    (-43.0, 28.0, 29.0),
    (-9.0, 22.0, 46.0),
    (25.0, 31.0, 34.0),
    (62.0, 24.0, 42.0),
    (99.0, 29.0, 31.0),
    (132.0, 21.0, 24.0),
    (164.0, 23.0, 36.0),
    (196.0, 20.0, 25.0),
  )
  height = 3.5
  for center, width, amplitude in peaks:
    shifted = center + 7.0 * math.sin(phase + center * 0.017)
    u = (coord - shifted) / width
    height += amplitude * math.exp(-0.5 * u * u)
  height += 3.4 * math.sin(0.115 * coord + phase)
  height += 1.9 * math.sin(0.255 * coord + 1.7 * phase)

  edge_distance = min(coord - PROFILE_MIN_M, PROFILE_MAX_M - coord)
  edge_t = max(0.0, min(1.0, edge_distance / 30.0))
  smooth = edge_t * edge_t * (3.0 - 2.0 * edge_t)
  edge_scale = 0.42 + 0.58 * smooth
  return max(0.0, height * edge_scale * height_scale)


def _add_horizon_range(
  spec: mujoco.MjSpec,
  *,
  name: str,
  x_start: float,
  depth_sign: float,
  height_scale: float,
  phase: float,
  rgba: tuple[float, float, float, float],
) -> None:
  lateral_samples = 65
  vertices: list[float] = []
  for layer, (x_offset, layer_scale) in enumerate(
    zip(MOUNTAIN_DEPTH_OFFSETS_M, MOUNTAIN_DEPTH_SCALES, strict=True)
  ):
    for sample in range(lateral_samples):
      t = sample / (lateral_samples - 1)
      y = PROFILE_MIN_M + t * (PROFILE_MAX_M - PROFILE_MIN_M)
      y_warp = 2.8 * math.sin(0.071 * y + phase + layer * 0.61)
      z = _mountain_height(y + y_warp, height_scale=height_scale, phase=phase)
      z *= layer_scale
      z += 1.1 * height_scale * layer_scale * math.sin(
        0.31 * y + layer * 1.13 + phase
      )
      vertices.extend((depth_sign * x_offset, y, max(0.0, z)))

  faces: list[int] = []
  for layer in range(len(MOUNTAIN_DEPTH_OFFSETS_M) - 1):
    row0 = layer * lateral_samples
    row1 = (layer + 1) * lateral_samples
    for sample in range(lateral_samples - 1):
      a = row0 + sample
      b = a + 1
      c = row1 + sample
      d = c + 1
      if depth_sign > 0.0:
        faces.extend((a, c, b, b, c, d))
      else:
        faces.extend((a, b, c, b, d, c))

  mesh_name = f"{name}_mesh"
  spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
  spec.worldbody.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    pos=(x_start, 0.0, -0.5),
    rgba=rgba,
    contype=0,
    conaffinity=0,
    group=2,
  )


def _add_side_range(
  spec: mujoco.MjSpec,
  *,
  name: str,
  side_sign: float,
  side_y: float,
  corner_x: float,
  height_scale: float,
  phase: float,
  rgba: tuple[float, float, float, float],
  centerline_y: Callable[[float], float],
) -> None:
  longitudinal_samples = 65
  x_min, x_max = MOUNTAIN_SIDE_X_RANGE_M
  corner_profile_coord = centerline_y(corner_x) + side_sign * side_y

  vertices: list[float] = []
  for layer, (y_offset, layer_scale) in enumerate(
    zip(MOUNTAIN_DEPTH_OFFSETS_M, MOUNTAIN_DEPTH_SCALES, strict=True)
  ):
    for sample in range(longitudinal_samples):
      t = sample / (longitudinal_samples - 1)
      x = x_min + t * (x_max - x_min)
      profile_coord = corner_profile_coord + (x - corner_x)
      coord_warp = 2.8 * math.sin(0.071 * profile_coord + phase + layer * 0.61)
      z = _mountain_height(
        profile_coord + coord_warp,
        height_scale=height_scale,
        phase=phase,
      ) * layer_scale
      z += 1.1 * height_scale * layer_scale * math.sin(
        0.31 * profile_coord + layer * 1.13 + phase
      )
      rear_t = max(0.0, min(1.0, (x - x_min) / 55.0))
      rear_smooth = rear_t * rear_t * (3.0 - 2.0 * rear_t)
      z *= 0.35 + 0.65 * rear_smooth
      y = centerline_y(x) + side_sign * (side_y + y_offset)
      vertices.extend((x, y, max(0.0, z)))

  faces: list[int] = []
  for layer in range(len(MOUNTAIN_DEPTH_OFFSETS_M) - 1):
    row0 = layer * longitudinal_samples
    row1 = (layer + 1) * longitudinal_samples
    for sample in range(longitudinal_samples - 1):
      a = row0 + sample
      b = a + 1
      c = row1 + sample
      d = c + 1
      if side_sign > 0.0:
        faces.extend((a, b, c, b, d, c))
      else:
        faces.extend((a, c, b, b, c, d))

  mesh_name = f"{name}_mesh"
  spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
  spec.worldbody.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    rgba=rgba,
    contype=0,
    conaffinity=0,
    group=2,
  )


def add_mountain_enclosure(
  spec: mujoco.MjSpec,
  *,
  centerline_y: Callable[[float], float] | None = None,
  range_specs: Sequence[tuple[float, float, float]] = MOUNTAIN_RANGE_SPECS,
) -> None:
  """Add front/rear horizon ridges plus matching visual-only side ridges."""
  if centerline_y is None:
    centerline_y = lambda _x: 0.0

  if len(range_specs) != len(MOUNTAIN_SIDE_CLEARANCE_M):
    raise ValueError("range_specs must match MOUNTAIN_SIDE_CLEARANCE_M")

  for index, (x_start, height_scale, phase) in enumerate(range_specs):
    color = (
      0.31 + 0.06 * index,
      0.34 + 0.05 * index,
      0.38 + 0.05 * index,
      1.0,
    )
    for end_name, depth_sign in (("front", 1.0), ("rear", -1.0)):
      _add_horizon_range(
        spec,
        name=f"mountain_range_{end_name}_{index:02d}",
        x_start=depth_sign * x_start,
        depth_sign=depth_sign,
        height_scale=height_scale,
        phase=phase,
        rgba=color,
      )
    for side_name, side_sign in (("left", -1.0), ("right", 1.0)):
      _add_side_range(
        spec,
        name=f"mountain_side_{side_name}_{index:02d}",
        side_sign=side_sign,
        side_y=MOUNTAIN_SIDE_CLEARANCE_M[index],
        corner_x=x_start,
        height_scale=height_scale * MOUNTAIN_SIDE_HEIGHT_MULTIPLIER,
        phase=phase,
        rgba=color,
        centerline_y=centerline_y,
      )
