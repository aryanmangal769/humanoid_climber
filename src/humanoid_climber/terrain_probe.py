"""Lightweight runtime terrain sensing for the demo policy supervisor.

The scenario implementations are free to publish exact sampled telemetry.  When
they do not, this module derives a local terrain estimate directly from MuJoCo's
compiled static geometry and reads per-environment randomized foot friction from
MjLab's model bridge.  It deliberately stays outside the scenario task modules
so concurrent scenario work does not need to depend on the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class LocalTerrainEstimate:
    """Geometric properties estimated from a small patch around the robot."""

    slope_gradient: float | None = None
    roughness_m: float | None = None
    step_height_m: float | None = None
    terrain_label: str | None = None
    sample_count: int = 0


def estimate_local_terrain(
        base: Any,
        env_idx: int,
        *,
        radius_m: float = 0.8,
        grid_size: int = 9,
) -> LocalTerrainEstimate:
    """Estimate slope/relief/steps from static terrain under and around the G1.

    This is a UI/demo heuristic, not a replacement for the eventual onboard
    terrain estimator.  Exact scenario telemetry, when supplied, overrides it.
    """
    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer >= 3")

    sim = getattr(base, "sim", None)
    scene = getattr(base, "scene", None)
    if sim is None or scene is None:
        return LocalTerrainEstimate()

    try:
        robot = scene["robot"]
        center = robot.data.root_link_pos_w[env_idx, :2].detach().cpu().numpy()
        model = sim.mj_model
        data = sim.mj_data
    except (AttributeError, KeyError, IndexError, TypeError, RuntimeError):
        return LocalTerrainEstimate()

    terrain_geom_ids = _terrain_geom_ids(model)
    if terrain_geom_ids.size == 0:
        return LocalTerrainEstimate()

    candidates = _nearby_terrain_geoms(
        model,
        data,
        terrain_geom_ids,
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius_m=radius_m,
    )
    if candidates.size == 0:
        return LocalTerrainEstimate()

    positions = data.geom_xpos[candidates]
    sizes = model.geom_size[candidates]
    z_start = max(
        float(robot.data.root_link_pos_w[env_idx, 2].item()) + 1.0,
        float(np.max(positions[:, 2] + np.maximum(sizes[:, 2], 0.05))) + 2.0,
    )

    offsets = np.linspace(-radius_m, radius_m, grid_size, dtype=np.float64)
    heights = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    for row, dy in enumerate(offsets):
        for col, dx in enumerate(offsets):
            heights[row, col] = _terrain_height_at(
                model,
                data,
                candidates,
                float(center[0] + dx),
                float(center[1] + dy),
                z_start,
            )

    has_box = bool(np.any(model.geom_type[candidates] == mujoco.mjtGeom.mjGEOM_BOX))
    return terrain_metrics_from_height_grid(
        heights,
        spacing_m=(2.0 * radius_m) / (grid_size - 1),
        box_dominated=has_box,
    )


def sampled_foot_friction(base: Any, env_idx: int) -> float | None:
    """Read the current per-environment sliding friction on the G1 foot geoms."""
    sim = getattr(base, "sim", None)
    scene = getattr(base, "scene", None)
    if sim is None or scene is None:
        return None
    try:
        robot = scene["robot"]
        local_ids, _ = robot.find_geoms((".*foot.*",))
        if not local_ids:
            return None
        global_ids = robot.indexing.geom_ids[local_ids]
        friction = sim.model.geom_friction
        values = friction[env_idx] if len(friction.shape) == 3 else friction
        sliding = values[global_ids, 0]
        return float(sliding.median().item())
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def terrain_metrics_from_height_grid(
        heights: np.ndarray,
        *,
        spacing_m: float,
        box_dominated: bool = False,
) -> LocalTerrainEstimate:
    """Convert a sampled height grid into stable heuristic terrain metrics."""
    heights = np.asarray(heights, dtype=np.float64)
    if heights.ndim != 2:
        raise ValueError("heights must be a 2D array")
    valid = np.isfinite(heights)
    sample_count = int(valid.sum())
    if sample_count < 5:
        return LocalTerrainEstimate(sample_count=sample_count)

    rows, cols = np.indices(heights.shape)
    x = (cols - (heights.shape[1] - 1) / 2.0) * spacing_m
    y = (rows - (heights.shape[0] - 1) / 2.0) * spacing_m
    design = np.column_stack((x[valid], y[valid], np.ones(sample_count)))
    coeff, *_ = np.linalg.lstsq(design, heights[valid], rcond=None)
    plane_slope = float(np.hypot(coeff[0], coeff[1]))

    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    residual = heights - plane
    residual_values = residual[valid]
    relief = float(np.percentile(residual_values, 95) - np.percentile(residual_values, 5))

    jumps: list[float] = []
    for axis in (0, 1):
        delta = np.diff(residual, axis=axis)
        delta = np.abs(delta[np.isfinite(delta)])
        if delta.size:
            jumps.append(float(np.percentile(delta, 95)))
    step_height = max(jumps, default=0.0) if box_dominated else 0.0

    slope_gradient = plane_slope
    if not box_dominated and bool(valid.all()):
        grad_y, grad_x = np.gradient(heights, spacing_m)
        local_gradient = np.hypot(grad_x, grad_y)
        slope_gradient = max(
            plane_slope, float(np.percentile(local_gradient, 75))
        )

        curvature: list[np.ndarray] = []
        for axis in (0, 1):
            if heights.shape[axis] >= 3:
                curvature.append(np.abs(np.diff(heights, n=2, axis=axis)).ravel())
        curvature_values = (
            np.concatenate(curvature) if curvature else np.empty(0, dtype=np.float64)
        )
        roughness = (
            float(np.percentile(curvature_values, 50))
            if curvature_values.size
            else 0.0
        )
    else:
        # Box terrain tends to represent stairs/obstacles, where the discrete
        # jump is the more useful signal.
        roughness = min(relief, step_height)
    label = _terrain_label(slope_gradient, roughness, step_height)
    return LocalTerrainEstimate(
        slope_gradient=slope_gradient,
        roughness_m=roughness,
        step_height_m=step_height,
        terrain_label=label,
        sample_count=sample_count,
    )


def _terrain_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    terrain_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "terrain"
    )
    event_prefixes = (
        "treadmill_ice_patch",
        "treadmill_slope_patch_",
        "treadmill_rough_mound_",
        "treadmill_rough_rock_",
    )
    selected: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        on_base_terrain = model.geom_bodyid[geom_id] == terrain_body_id
        is_event_terrain = bool(
            name is not None and name.startswith(event_prefixes)
        )
        if on_base_terrain or is_event_terrain:
            selected.append(geom_id)
    if not selected:
        return np.empty(0, dtype=np.int32)
    ids = np.asarray(selected, dtype=np.int32)
    supported = np.isin(
        model.geom_type[ids],
        (
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            mujoco.mjtGeom.mjGEOM_HFIELD,
            mujoco.mjtGeom.mjGEOM_PLANE,
        ),
    )
    # Route/safety visual boxes are non-colliding and should never be considered
    # physical terrain. Dedicated event mocap bodies are colliding, so they stay
    # in this list and are sensed once the local patch approaches the robot.
    collidable = (model.geom_contype[ids] != 0) | (model.geom_conaffinity[ids] != 0)
    return ids[supported & collidable].astype(np.int32, copy=False)


def _nearby_terrain_geoms(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_ids: np.ndarray,
        *,
        center_x: float,
        center_y: float,
        radius_m: float,
) -> np.ndarray:
    positions = data.geom_xpos[geom_ids]
    sizes = model.geom_size[geom_ids]
    geom_types = model.geom_type[geom_ids]
    # Boxes and generated heightfields expose their XY half extents in geom_size.
    # Planes are unbounded and are always retained.
    overlap_x = np.abs(positions[:, 0] - center_x) <= sizes[:, 0] + radius_m
    overlap_y = np.abs(positions[:, 1] - center_y) <= sizes[:, 1] + radius_m
    planes = geom_types == mujoco.mjtGeom.mjGEOM_PLANE
    nearby = geom_ids[(overlap_x & overlap_y) | planes]
    if nearby.size <= 1:
        return nearby

    # Test the highest possible surfaces first. Once a real surface has been
    # hit, _terrain_height_at can cheaply reject buried rough-terrain geoms by
    # their conservative bounding radius instead of running an ellipsoid/box
    # intersection for every single grid sample.
    top_bounds = data.geom_xpos[nearby, 2] + model.geom_rbound[nearby]
    return nearby[np.argsort(top_bounds)[::-1]]


def _terrain_height_at(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_ids: np.ndarray,
        x: float,
        y: float,
        z_start: float,
) -> float:
    best = -np.inf
    point = np.array([x, y, z_start], dtype=np.float64)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for geom_id in geom_ids:
        geom_type = model.geom_type[geom_id]
        if geom_type not in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_PLANE):
            # geom_rbound is rotation-independent and therefore gives a safe
            # world-Z upper bound. Hidden rough mounds live well below the base
            # treadmill, so after the base surface is known they can be skipped
            # without changing the resulting height sample.
            top_bound = float(data.geom_xpos[geom_id, 2] + model.geom_rbound[geom_id])
            if np.isfinite(best) and top_bound <= best:
                continue
        if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
            distance = mujoco.mj_rayHfield(
                model, data, int(geom_id), point, direction, None
            )
            if distance >= 0.0:
                best = max(best, z_start - float(distance))
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            hit = _ray_box_height(model, data, int(geom_id), point, direction)
            if hit is not None:
                best = max(best, hit)
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            hit = _ray_ellipsoid_height(model, data, int(geom_id), point, direction)
            if hit is not None:
                best = max(best, hit)
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            hit = _ray_plane_height(data, int(geom_id), point, direction)
            if hit is not None:
                best = max(best, hit)
    return best if np.isfinite(best) else np.nan


def _ray_box_height(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_id: int,
        point: np.ndarray,
        direction: np.ndarray,
) -> float | None:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local_point = rotation.T @ (point - data.geom_xpos[geom_id])
    local_direction = rotation.T @ direction
    half_size = model.geom_size[geom_id]
    t_min = -np.inf
    t_max = np.inf
    for axis in range(3):
        velocity = local_direction[axis]
        if abs(velocity) < 1.0e-10:
            if local_point[axis] < -half_size[axis] or local_point[axis] > half_size[axis]:
                return None
            continue
        t1 = (-half_size[axis] - local_point[axis]) / velocity
        t2 = (half_size[axis] - local_point[axis]) / velocity
        near, far = sorted((t1, t2))
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    distance = t_min if t_min >= 0.0 else t_max
    return float((point + distance * direction)[2])


def _ray_ellipsoid_height(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_id: int,
        point: np.ndarray,
        direction: np.ndarray,
) -> float | None:
    """Intersect a ray with an arbitrarily oriented MuJoCo ellipsoid."""
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local_point = rotation.T @ (point - data.geom_xpos[geom_id])
    local_direction = rotation.T @ direction
    radii = np.maximum(model.geom_size[geom_id], 1.0e-8)
    inv_r2 = 1.0 / np.square(radii)
    a = float(np.sum(np.square(local_direction) * inv_r2))
    b = float(2.0 * np.sum(local_point * local_direction * inv_r2))
    c = float(np.sum(np.square(local_point) * inv_r2) - 1.0)
    if a <= 1.0e-12:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(max(discriminant, 0.0)))
    roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    distances = [value for value in roots if value >= 0.0]
    if not distances:
        return None
    distance = min(distances)
    return float((point + distance * direction)[2])


def _ray_plane_height(
        data: mujoco.MjData,
        geom_id: int,
        point: np.ndarray,
        direction: np.ndarray,
) -> float | None:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    normal = rotation[:, 2]
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1.0e-10:
        return None
    distance = float(np.dot(normal, data.geom_xpos[geom_id] - point) / denominator)
    if distance < 0.0:
        return None
    return float((point + distance * direction)[2])


def _terrain_label(slope: float, roughness: float, step_height: float) -> str:
    if step_height > 0.05:
        return "stairs / obstacles"
    if roughness > 0.03:
        return "rough terrain"
    if slope > 0.06:
        return "incline"
    return "flat"
