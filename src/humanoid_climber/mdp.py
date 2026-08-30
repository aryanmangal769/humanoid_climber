"""Project-specific MDP terms for Hum Climber."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from humanoid_climber.trail import (
  TRAIL_MARKER_STRIDE,
  TRAIL_SEGMENT_OVERLAP_M,
  TRAIL_SEGMENT_X_SPACING_M,
  TRAIL_SEGMENTS,
  TRAIL_WINDOW_BEHIND_M,
  nearest_trail_frame,
  trail_frame_ahead,
  trail_window_segments,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_FORCE_AXES = ("x", "y", "z")


def _mocap_id(env: ManagerBasedRlEnv, entity) -> int:
  root_body_id = int(entity.indexing.body_ids[0].item())
  mocap_id = int(env.sim.mj_model.body_mocapid[root_body_id])
  if mocap_id < 0:
    raise ValueError(
      "The treadmill terrain must be a mocap body so its rendered and "
      "physical transforms stay synchronized."
    )
  return mocap_id


def _write_treadmill_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  mocap_id: int,
  gradients: torch.Tensor,
) -> None:
  # Positive gradient means the surface rises along world +X. A Y-axis body
  # pitch has the opposite plane-gradient sign, hence the leading minus.
  #
  # The old straight treadmill could simply translate its 200 m strip so the
  # pitch pivot sat below the robot. A finite winding trail cannot: translating
  # the body would move every turn and safety boundary out from under the
  # robot. Instead use T = P - R*P so the rigid terrain rotates around the
  # stored world-X pivot without relocating the trail in the horizontal plane.
  pitch = -torch.atan(gradients)
  half_pitch = 0.5 * pitch
  cos_pitch = torch.cos(pitch)
  sin_pitch = torch.sin(pitch)
  pivot_x = env.treadmill_pivot_xy[env_ids, 0]
  pose = torch.zeros((len(env_ids), 7), device=env.device)
  pose[:, 0] = (1.0 - cos_pitch) * pivot_x
  pose[:, 2] = sin_pitch * pivot_x
  pose[:, 3] = torch.cos(half_pitch)
  pose[:, 5] = torch.sin(half_pitch)
  env.sim.data.mocap_pos[env_ids, mocap_id] = pose[:, :3]
  env.sim.data.mocap_quat[env_ids, mocap_id] = pose[:, 3:]


class randomize_treadmill_slope:
  """Choose the next slope target and pivot it beneath the robot."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._terrain = env.scene[cfg.params["asset_cfg"].name]
    self._robot = env.scene[cfg.params["robot_cfg"].name]
    self._mocap_id = _mocap_id(env, self._terrain)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    gradient_range: tuple[float, float],
    occurrence_probability: float,
    active_magnitude_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
  ) -> None:
    del robot_cfg
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
      env_ids = env_ids.to(env.device, dtype=torch.long)

    del asset_cfg
    if not 0.0 <= occurrence_probability <= 1.0:
      raise ValueError("occurrence_probability must be between 0 and 1.")
    minimum_magnitude, maximum_magnitude = active_magnitude_range
    maximum_allowed = max(abs(gradient_range[0]), abs(gradient_range[1]))
    if not 0.0 <= minimum_magnitude <= maximum_magnitude <= maximum_allowed:
      raise ValueError(
        "active_magnitude_range must be non-negative and remain inside "
        "gradient_range."
      )
    active = torch.rand(len(env_ids), device=env.device) < occurrence_probability
    magnitudes = torch.empty(len(env_ids), device=env.device).uniform_(
      minimum_magnitude, maximum_magnitude
    )
    signs = torch.where(
      torch.rand(len(env_ids), device=env.device) < 0.5,
      -torch.ones_like(magnitudes),
      torch.ones_like(magnitudes),
    )
    targets = torch.where(active, magnitudes * signs, torch.zeros_like(magnitudes))
    slope_state = getattr(env, "treadmill_slope_gradient", None)
    if slope_state is None:
      slope_state = torch.zeros(env.num_envs, device=env.device)
      env.treadmill_slope_gradient = slope_state
    target_state = getattr(env, "treadmill_slope_target_gradient", None)
    if target_state is None:
      target_state = torch.zeros(env.num_envs, device=env.device)
      env.treadmill_slope_target_gradient = target_state
    pivot_state = getattr(env, "treadmill_pivot_xy", None)
    if pivot_state is None:
      pivot_state = torch.zeros((env.num_envs, 2), device=env.device)
      env.treadmill_pivot_xy = pivot_state
    target_state[env_ids] = targets
    # Store the robot's current X as the rigid-body pitch pivot. The terrain
    # transform rotates around this point; it no longer translates the route.
    pivot_state[env_ids, 0] = self._robot.data.root_link_pos_w[env_ids, 0]
    # Recenter immediately at the current angle, then let the step event move
    # smoothly toward the new target. This avoids teleporting the floor through
    # the feet when an interval changes.
    _write_treadmill_pose(env, env_ids, self._mocap_id, slope_state[env_ids])


class advance_treadmill_slope:
  """Move the physical and rendered treadmill smoothly toward its target."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._terrain = env.scene[cfg.params["asset_cfg"].name]
    self._mocap_id = _mocap_id(env, self._terrain)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_delta_per_step: float,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
  ) -> None:
    del asset_cfg, robot_cfg
    if not hasattr(env, "treadmill_slope_target_gradient"):
      return
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
      env_ids = env_ids.to(env.device, dtype=torch.long)
    current = env.treadmill_slope_gradient[env_ids]
    target = env.treadmill_slope_target_gradient[env_ids]
    delta = torch.clamp(
      target - current,
      min=-max_delta_per_step,
      max=max_delta_per_step,
    )
    current = current + delta
    env.treadmill_slope_gradient[env_ids] = current
    _write_treadmill_pose(env, env_ids, self._mocap_id, current)


@requires_model_fields("geom_pos", "geom_size", "geom_quat")
class advance_infinite_trail:
  """Recycle the fixed trail-geometry pool as the robot walks forward.

  The centerline is procedural and mathematically unbounded. Only the rendered
  strip, pitch markers, and safety lines use MuJoCo geoms, so this event moves a
  constant-size pool when the robot crosses the next X sampling interval.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._terrain = env.scene[cfg.params["terrain_cfg"].name]
    self._robot = env.scene[cfg.params["robot_cfg"].name]

    surface_local_ids, _ = self._terrain.find_geoms(
      ("treadmill_trail_surface_.*",), preserve_order=True
    )
    left_local_ids, _ = self._terrain.find_geoms(
      ("treadmill_centerline_safety_left_.*",), preserve_order=True
    )
    right_local_ids, _ = self._terrain.find_geoms(
      ("treadmill_centerline_safety_right_.*",), preserve_order=True
    )
    marker_local_ids, _ = self._terrain.find_geoms(
      ("treadmill_pitch_marker_.*",), preserve_order=True
    )
    self._surface_geom_ids = self._terrain.indexing.geom_ids[surface_local_ids].to(
      env.device, dtype=torch.long
    )
    self._left_geom_ids = self._terrain.indexing.geom_ids[left_local_ids].to(
      env.device, dtype=torch.long
    )
    self._right_geom_ids = self._terrain.indexing.geom_ids[right_local_ids].to(
      env.device, dtype=torch.long
    )
    self._marker_geom_ids = self._terrain.indexing.geom_ids[marker_local_ids].to(
      env.device, dtype=torch.long
    )
    expected_markers = len(TRAIL_SEGMENTS[::TRAIL_MARKER_STRIDE])
    if not (
      len(self._surface_geom_ids) == len(TRAIL_SEGMENTS)
      and len(self._left_geom_ids) == len(TRAIL_SEGMENTS)
      and len(self._right_geom_ids) == len(TRAIL_SEGMENTS)
      and len(self._marker_geom_ids) == expected_markers
    ):
      raise ValueError("The recycled infinite-trail geometry pool is incomplete.")

    self._first_chunk = torch.full(
      (env.num_envs,),
      -(2**30),
      dtype=torch.long,
      device=env.device,
    )

  def _unpitched_local_x(self, env_id: int) -> float | None:
    root_x = float(self._robot.data.root_link_pos_w[env_id, 0].item())
    origin_x = float(self._env.scene.env_origins[env_id, 0].item())
    if not math.isfinite(root_x) or not math.isfinite(origin_x):
      return None
    local_x = root_x - origin_x
    slopes = getattr(self._env, "treadmill_slope_gradient", None)
    pivot = getattr(self._env, "treadmill_pivot_xy", None)
    if slopes is not None and pivot is not None:
      gradient = float(slopes[env_id].item())
      pivot_x = float(pivot[env_id, 0].item()) - origin_x
      if not math.isfinite(gradient) or not math.isfinite(pivot_x):
        return None
      cos_pitch = 1.0 / (1.0 + gradient * gradient) ** 0.5
      translation_x = (1.0 - cos_pitch) * pivot_x
      local_x = (local_x - translation_x) / max(cos_pitch, 1.0e-6)
    return local_x if math.isfinite(local_x) else None

  def _write_window(self, env_id: int, center_x: float, boundary_offset_m: float) -> None:
    segments = trail_window_segments(center_x)
    model = self._env.sim.model
    for slot, segment in enumerate(segments):
      half_yaw = 0.5 * segment.yaw
      quat = torch.tensor(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        device=self._env.device,
        dtype=model.geom_quat.dtype,
      )
      surface_id = int(self._surface_geom_ids[slot].item())
      model.geom_pos[env_id, surface_id, 0] = segment.center[0]
      model.geom_pos[env_id, surface_id, 1] = segment.center[1]
      model.geom_size[env_id, surface_id, 0] = (
        segment.length * 0.5 + TRAIL_SEGMENT_OVERLAP_M
      )
      model.geom_quat[env_id, surface_id] = quat

      left_id = int(self._left_geom_ids[slot].item())
      right_id = int(self._right_geom_ids[slot].item())
      for geom_id, sign in ((left_id, 1.0), (right_id, -1.0)):
        model.geom_pos[env_id, geom_id, 0] = (
          segment.center[0] + sign * boundary_offset_m * segment.normal[0]
        )
        model.geom_pos[env_id, geom_id, 1] = (
          segment.center[1] + sign * boundary_offset_m * segment.normal[1]
        )
        model.geom_size[env_id, geom_id, 0] = (
          segment.length * 0.5 + TRAIL_SEGMENT_OVERLAP_M
        )
        model.geom_quat[env_id, geom_id] = quat

    marker_segments = segments[::TRAIL_MARKER_STRIDE]
    for marker_slot, segment in enumerate(marker_segments):
      marker_id = int(self._marker_geom_ids[marker_slot].item())
      half_yaw = 0.5 * segment.yaw
      model.geom_pos[env_id, marker_id, 0] = segment.center[0]
      model.geom_pos[env_id, marker_id, 1] = segment.center[1]
      model.geom_quat[env_id, marker_id, 0] = math.cos(half_yaw)
      model.geom_quat[env_id, marker_id, 1:3] = 0.0
      model.geom_quat[env_id, marker_id, 3] = math.sin(half_yaw)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._first_chunk[:] = -(2**30)
    else:
      self._first_chunk[env_ids.to(self._env.device, dtype=torch.long)] = -(2**30)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    centerline_max_offset_m: float,
  ) -> None:
    del env, terrain_cfg, robot_cfg
    if env_ids is None:
      env_ids = torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )
    else:
      env_ids = env_ids.to(self._env.device, dtype=torch.long)
    for env_id in env_ids.tolist():
      local_x = self._unpitched_local_x(int(env_id))
      # A physics blow-up can briefly leave the root state non-finite. Geometry
      # recycling is purely visual/support bookkeeping, so skip that world until
      # the viewer's numerical-state watchdog resets it instead of crashing on
      # math.floor(NaN).
      if local_x is None:
        continue
      first_chunk = math.floor(
        (local_x - TRAIL_WINDOW_BEHIND_M) / TRAIL_SEGMENT_X_SPACING_M
      )
      if int(self._first_chunk[env_id].item()) == first_chunk:
        continue
      self._write_window(int(env_id), local_x, float(centerline_max_offset_m))
      self._first_chunk[env_id] = first_chunk


class apply_wind_force:
  """Apply wind and visualize its world-frame direction at the robot torso."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._wind_labels = {}

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    """Sample and apply an episode-long wind force to selected bodies."""
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    num_bodies = (
      len(asset_cfg.body_ids)
      if isinstance(asset_cfg.body_ids, list)
      else self._asset.num_bodies
    )
    ranges = torch.tensor(
      [force_ranges.get(axis, (0.0, 0.0)) for axis in _FORCE_AXES],
      device=env.device,
    )
    shape = (len(env_ids), num_bodies, 3)
    forces = torch.rand(shape, device=env.device)
    forces = forces * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=asset_cfg.body_ids,
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw the active wind vector as a cyan arrow attached to each target body."""
    wrench = self._asset.data.body_external_wrench
    body_pos = self._asset.data.body_com_pos_w
    body_ids = (
      self._body_ids
      if isinstance(self._body_ids, list)
      else range(self._asset.num_bodies)
    )
    for env_idx in visualizer.get_env_indices(self._num_envs):
      for body_id in body_ids:
        force = wrench[env_idx, body_id, :3]
        if force.square().sum().item() < 1.0e-6:
          continue
        start = body_pos[env_idx, body_id].detach().cpu().numpy().copy()
        start[2] += 0.65
        end = start + force.detach().cpu().numpy() * 0.04
        magnitude = force.norm().item()
        axis_index = int(force.abs().argmax().item())
        axis = _FORCE_AXES[axis_index].upper()
        sign = "+" if force[axis_index].item() >= 0.0 else "-"
        text = f"WIND  {sign}{axis}  |  {magnitude:.0f} N"
        visualizer.add_arrow(
          start=start,
          end=end,
          color=(0.0, 0.85, 1.0, 0.95),
          width=0.025,
          label=text,
        )
        server = getattr(visualizer, "server", None)
        if server is not None:
          key = (env_idx, body_id)
          label_pos = start.copy()
          label_pos[2] += 0.15
          label_pos += getattr(visualizer, "_scene_offset", 0.0)
          html = (
            '<div style="font-weight:900;color:#000;background:rgba(255,255,255,.85);'
            f'padding:4px 8px;border-radius:5px;white-space:nowrap">{text}</div>'
          )
          if key in self._wind_labels:
            container, label = self._wind_labels[key]
            try:
              container.position = label_pos
              label.content = html
            except RuntimeError:
              # Viser invalidates 3D GUI handles when a client/scene is rebuilt.
              # Drop the stale pair and recreate it below on this same frame.
              del self._wind_labels[key]
          if key not in self._wind_labels:
            container = server.scene.add_3d_gui_container(
              name=f"/wind/status/{env_idx}/{body_id}",
              position=label_pos,
            )
            with container:
              label = server.gui.add_html(html)
            self._wind_labels[key] = (container, label)
  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Participate in manager lifecycle so debug visualization is discovered."""
    del env_ids


@requires_model_fields("geom_friction")
class sequential_random_events(apply_wind_force):
  """Alternate wind, ice, slope, and organic rough ground with quiet breaks."""

  BREAK = 0
  WIND = 1
  ICE = 2
  SLOPE = 3
  BUMPS = 4
  SLOPE_CLEARING = 5
  BUMPS_CLEARING = 6

  MANUAL_EVENT_NAMES = ("wind", "ice", "slope", "bumps")

  _SLOPE_COLORS = (
    (0.12, 0.42, 1.00, 0.78),
    (0.05, 0.72, 1.00, 0.78),
    (0.10, 0.88, 0.68, 0.78),
    (0.92, 0.90, 0.20, 0.78),
    (1.00, 0.66, 0.12, 0.78),
    (1.00, 0.36, 0.10, 0.78),
    (0.92, 0.12, 0.16, 0.78),
  )
  _ICE_COLOR = (0.42, 0.82, 1.00, 0.72)
  _HIDDEN_COLOR = (0.0, 0.0, 0.0, 0.0)
  _ROUGH_LOW_COLOR = (0.28, 0.27, 0.25, 1.0)
  _ROUGH_HIGH_COLOR = (0.58, 0.48, 0.34, 1.0)
  _ROCK_LOW_COLOR = (0.24, 0.24, 0.23, 1.0)
  _ROCK_HIGH_COLOR = (0.48, 0.45, 0.40, 1.0)
  _ROUGH_BURIED_Z = -0.55
  # Sink the supporting corner very slightly into the rendered/collision
  # surface.  A positive air gap is visually obvious on the faceted rocks,
  # while a 1.5 mm embed is below the scale of the terrain and prevents a
  # rasterized hairline from making a correctly seated rock look like it floats.
  _ROCK_SEATING_EMBED_M = 0.0015

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    self._params = cfg.params
    self._friction_asset = env.scene[cfg.params["friction_asset_cfg"].name]
    friction_local_ids = cfg.params["friction_asset_cfg"].geom_ids
    if isinstance(friction_local_ids, slice):
      friction_local_ids = list(range(self._friction_asset.num_geoms))
    self._friction_geom_ids = self._friction_asset.indexing.geom_ids[
      friction_local_ids
    ].to(env.device, dtype=torch.long)
    self._robot = env.scene[cfg.params["robot_cfg"].name]
    model = env.sim.mj_model

    def geom_id(name: str) -> int:
      value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
      if value < 0:
        raise ValueError(f"Missing random-event geom: {name}")
      return value

    def mocap_id(name: str) -> int:
      body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
      if body_id < 0:
        raise ValueError(f"Missing random-event body: {name}")
      value = int(model.body_mocapid[body_id])
      if value < 0:
        raise ValueError(f"Random-event body is not mocap-enabled: {name}")
      return value

    self._ice_geom_ids = torch.tensor(
      [geom_id("treadmill_ice_patch")], device=env.device, dtype=torch.long
    )
    slope_piece_count = int(cfg.params["slope_piece_count"])
    rock_count = int(cfg.params["rock_count"])
    self._slope_patch_geom_ids = torch.tensor(
      [geom_id(f"treadmill_slope_patch_{piece:02d}") for piece in range(slope_piece_count)],
      device=env.device,
      dtype=torch.long,
    )
    self._rough_mound_geom_ids = torch.tensor(
      [geom_id(f"treadmill_rough_mound_{index:03d}") for index in range(int(cfg.params["rough_rows"]) * int(cfg.params["rough_cols"]))],
      device=env.device,
      dtype=torch.long,
    )
    self._rough_rock_geom_ids = torch.tensor(
      [geom_id(f"treadmill_rough_rock_{index:02d}") for index in range(rock_count)],
      device=env.device,
      dtype=torch.long,
    )
    self._rough_geom_ids = torch.cat(
      (self._rough_mound_geom_ids, self._rough_rock_geom_ids)
    )
    self._ice_mocap_id = mocap_id("treadmill_ice_patch_body")
    self._slope_mocap_ids = tuple(
      mocap_id(f"treadmill_slope_patch_{piece:02d}_body")
      for piece in range(slope_piece_count)
    )
    self._rough_mocap_id = mocap_id("treadmill_rough_patch_body")
    self._rough_mound_mocap_ids = tuple(
      mocap_id(f"treadmill_rough_mound_{index:03d}_body")
      for index in range(len(self._rough_mound_geom_ids))
    )
    self._rough_rock_mocap_ids = tuple(
      mocap_id(f"treadmill_rough_rock_{index:02d}_body")
      for index in range(rock_count)
    )
    self._rough_primitive_mocap_ids = torch.tensor(
      (*self._rough_mound_mocap_ids, *self._rough_rock_mocap_ids),
      device=env.device,
      dtype=torch.long,
    )
    if (
      len(self._ice_geom_ids) != 1
      or len(self._slope_patch_geom_ids) != slope_piece_count
      or len(self._rough_mound_geom_ids) == 0
      or len(self._rough_rock_geom_ids) != rock_count
    ):
      raise ValueError(
        "The treadmill event-patch or rough-terrain geoms are incomplete."
      )

    self._phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._time_remaining = torch.zeros(env.num_envs, device=env.device)
    self._slope_patch_center_xy = torch.zeros(
      (env.num_envs, 2), device=env.device
    )
    self._slope_patch_tangent_xy = torch.zeros(
      (env.num_envs, 2), device=env.device
    )
    self._slope_patch_tangent_xy[:, 0] = 1.0
    self._slope_profile_factors = torch.zeros(
      (env.num_envs, slope_piece_count), device=env.device
    )
    if slope_piece_count != 4:
      raise ValueError("Variable slope profile currently requires four pieces.")
    self._slope_profile_factors[:, 0] = 0.6
    self._slope_profile_factors[:, 1] = 1.0
    self._slope_profile_factors[:, 2] = -1.0
    self._slope_profile_factors[:, 3] = -0.6
    self._rough_patch_center_xy = torch.zeros(
      (env.num_envs, 2), device=env.device
    )
    self._rough_patch_tangent_xy = torch.zeros(
      (env.num_envs, 2), device=env.device
    )
    self._rough_patch_tangent_xy[:, 0] = 1.0
    self._rough_patch_z = torch.full(
      (env.num_envs,), self._ROUGH_BURIED_Z, device=env.device
    )
    mound_count = len(self._rough_mound_geom_ids)
    self._rough_mound_center_xy = torch.zeros(
      (env.num_envs, mound_count, 2), device=env.device
    )
    self._rough_mound_center_z = torch.zeros(
      (env.num_envs, mound_count), device=env.device
    )
    self._rough_mound_quat = torch.zeros(
      (env.num_envs, mound_count, 4), device=env.device
    )
    self._rough_mound_quat[:, :, 0] = 1.0
    self._rough_rock_center_xy = torch.zeros(
      (env.num_envs, rock_count, 2), device=env.device
    )
    self._rough_rock_center_z = torch.zeros(
      (env.num_envs, rock_count), device=env.device
    )
    self._rough_rock_quat = torch.zeros(
      (env.num_envs, rock_count, 4), device=env.device
    )
    self._rough_rock_quat[:, :, 0] = 1.0
    self._manual_mode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._manual_events = torch.zeros(
      (env.num_envs, len(self.MANUAL_EVENT_NAMES)),
      dtype=torch.bool,
      device=env.device,
    )
    env.random_event_kind = self._phase
    # Publish the live controller so the dashboard can apply manual event
    # overrides on the simulation thread without reaching into EventManager
    # internals.
    env.random_event_controller = self

  def _ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
      return torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )
    return env_ids.to(self._env.device, dtype=torch.long)

  def _robot_local_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Return robot XY in the terrain body's local trail coordinates."""
    return (
      self._robot.data.root_link_pos_w[env_ids, :2]
      - self._env.scene.env_origins[env_ids, :2]
    )

  def _sample_patch_frames(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample compact obstacle centers a short distance ahead on the trail."""
    low, high = self._params["event_patch_ahead_range_m"]
    ahead = torch.empty(len(env_ids), device=self._env.device).uniform_(low, high)
    robot_xy = self._robot_local_xy(env_ids)
    centers = torch.empty((len(env_ids), 2), device=self._env.device)
    tangents = torch.empty((len(env_ids), 2), device=self._env.device)
    for row in range(len(env_ids)):
      frame = trail_frame_ahead(
        float(robot_xy[row, 0].item()),
        float(robot_xy[row, 1].item()),
        float(ahead[row].item()),
      )
      centers[row] = torch.tensor(
        (frame.center_x, frame.center_y), device=self._env.device
      )
      tangents[row] = torch.tensor(
        (frame.tangent_x, frame.tangent_y), device=self._env.device
      )
    return centers, tangents

  def _sample_slope_profile(self, env_ids: torch.Tensor) -> None:
    """Sample a continuous four-piece hill with non-constant local grades."""
    if len(env_ids) == 0:
      return
    outer_low, outer_high = self._params["slope_outer_fraction_range"]
    inner_low, inner_high = self._params["slope_inner_fraction_range"]
    outer = torch.empty(len(env_ids), device=self._env.device).uniform_(
      outer_low, outer_high
    )
    inner = torch.empty(len(env_ids), device=self._env.device).uniform_(
      inner_low, inner_high
    )
    # Paired signed factors make the integrated rise exactly cancel on the way
    # down while still giving each event two distinct ascent/descent grades.
    self._slope_profile_factors[env_ids, 0] = outer
    self._slope_profile_factors[env_ids, 1] = inner
    self._slope_profile_factors[env_ids, 2] = -inner
    self._slope_profile_factors[env_ids, 3] = -outer

  @staticmethod
  def _yaw_pitch_quat(
    tangents: torch.Tensor, pitches: torch.Tensor
  ) -> torch.Tensor:
    """Build scalar-first quaternions for trail yaw followed by local pitch."""
    yaw = torch.atan2(tangents[:, 1], tangents[:, 0])
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)
    cp = torch.cos(0.5 * pitches)
    sp = torch.sin(0.5 * pitches)
    quat = torch.empty((len(tangents), 4), device=tangents.device, dtype=tangents.dtype)
    quat[:, 0] = cy * cp
    quat[:, 1] = -sy * sp
    quat[:, 2] = cy * sp
    quat[:, 3] = sy * cp
    return quat

  def _set_geom_friction(
    self,
    env_ids: torch.Tensor,
    geom_ids: torch.Tensor,
    friction_range: tuple[float, float],
  ) -> None:
    if len(env_ids) == 0 or len(geom_ids) == 0:
      return
    values = torch.empty(len(env_ids), device=self._env.device).uniform_(
      *friction_range
    )
    friction = self._env.sim.model.geom_friction
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    friction[env_grid, geom_grid, 0] = values[:, None]

  def _write_mocap_pose(
    self,
    env_ids: torch.Tensor,
    mocap_id: int,
    local_pos: torch.Tensor,
    quat: torch.Tensor,
  ) -> None:
    """Write a per-world patch pose without touching baked geom model fields."""
    if len(env_ids) == 0:
      return
    world_pos = local_pos + self._env.scene.env_origins[env_ids]
    self._env.sim.data.mocap_pos[env_ids, mocap_id] = world_pos
    self._env.sim.data.mocap_quat[env_ids, mocap_id] = quat

  def _hide_mocap_body(self, env_ids: torch.Tensor, mocap_id: int) -> None:
    local_pos = torch.zeros((len(env_ids), 3), device=self._env.device)
    local_pos[:, 2] = self._ROUGH_BURIED_Z
    quat = torch.zeros((len(env_ids), 4), device=self._env.device)
    quat[:, 0] = 1.0
    self._write_mocap_pose(env_ids, mocap_id, local_pos, quat)

  def automatic_mode(self, env_id: int) -> bool:
    """Return whether the sequential random-event scheduler owns one env."""
    return not bool(self._manual_mode[int(env_id)].item())

  def manual_event_state(self, env_id: int) -> dict[str, bool]:
    """Return the dashboard-visible manual event state for one environment."""
    values = self._manual_events[int(env_id)]
    return {
      name: bool(values[index].item())
      for index, name in enumerate(self.MANUAL_EVENT_NAMES)
    }

  def active_event_names(self, env_id: int) -> tuple[str, ...]:
    """Return the physical random conditions currently affecting one env."""
    env_id = int(env_id)
    if bool(self._manual_mode[env_id].item()):
      state = self.manual_event_state(env_id)
      return tuple(name for name in self.MANUAL_EVENT_NAMES if state[name])

    phase = int(self._phase[env_id].item())
    phase_name = {
      self.WIND: "wind",
      self.ICE: "ice",
      self.SLOPE: "slope",
      self.SLOPE_CLEARING: "slope",
      self.BUMPS: "bumps",
      self.BUMPS_CLEARING: "bumps",
    }.get(phase)
    return (phase_name,) if phase_name is not None else ()

  def active_surface_friction(self, env_id: int) -> float | None:
    """Return the upcoming local patch friction when an ice event is active."""
    if "ice" not in self.active_event_names(int(env_id)):
      return None
    friction = self._env.sim.model.geom_friction
    geom_id = int(self._ice_geom_ids[0].item())
    value = friction[int(env_id), geom_id, 0] if friction.ndim == 3 else friction[geom_id, 0]
    return float(value.item())

  def _manual_event_index(self, event_name: str) -> int:
    try:
      return self.MANUAL_EVENT_NAMES.index(event_name)
    except ValueError as exc:
      raise ValueError(f"Unknown manual random event: {event_name!r}") from exc

  def _clear_slope_immediately(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._ensure_slope_state()
    self._env.treadmill_slope_gradient[env_ids] = 0.0
    self._env.treadmill_slope_target_gradient[env_ids] = 0.0
    self._hide_slope_patch(env_ids)

  def _clear_all_event_effects(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._clear_wind(env_ids)
    self._set_friction(env_ids, self._params["normal_friction_range"])
    self._hide_ice_patch(env_ids)
    self._clear_slope_immediately(env_ids)
    self._hide_rough_ground(env_ids)

  def _apply_manual_event(self, env_ids: torch.Tensor, event_name: str) -> None:
    if len(env_ids) == 0:
      return
    if event_name == "wind":
      super().__call__(
        self._env,
        env_ids,
        self._params["wind_force_ranges"],
        self._params["asset_cfg"],
      )
      return
    if event_name == "ice":
      self._set_friction(env_ids, self._params["normal_friction_range"])
      self._activate_ice_patch(env_ids)
      return
    if event_name == "slope":
      self._ensure_slope_state()
      low, high = self._params["active_slope_magnitude_range"]
      targets = torch.empty(len(env_ids), device=self._env.device).uniform_(low, high)
      self._sample_slope_profile(env_ids)
      self._env.treadmill_slope_gradient[env_ids] = targets
      self._env.treadmill_slope_target_gradient[env_ids] = targets
      centers, tangents = self._sample_patch_frames(env_ids)
      self._slope_patch_center_xy[env_ids] = centers
      self._slope_patch_tangent_xy[env_ids] = tangents
      self._update_slope_patch_geometry(env_ids, targets)
      return
    if event_name == "bumps":
      self._activate_rough_ground(env_ids)
      return
    raise ValueError(f"Unknown manual random event: {event_name!r}")

  def set_automatic_mode(self, env_id: int, enabled: bool) -> None:
    """Switch one environment between automatic and dashboard-manual events."""
    env_ids = torch.tensor([int(env_id)], device=self._env.device, dtype=torch.long)
    if enabled:
      if not bool(self._manual_mode[env_ids].any().item()):
        return
      self._clear_all_event_effects(env_ids)
      self._manual_events[env_ids] = False
      self._manual_mode[env_ids] = False
      self._set_break(env_ids)
      return

    if bool(self._manual_mode[env_ids].all().item()):
      return
    self._manual_mode[env_ids] = True
    self._manual_events[env_ids] = False
    self._phase[env_ids] = self.BREAK
    self._time_remaining[env_ids] = 0.0
    self._clear_all_event_effects(env_ids)

  def set_manual_event(self, env_id: int, event_name: str, enabled: bool) -> None:
    """Immediately add or remove one physical random-event effect."""
    event_index = self._manual_event_index(event_name)
    env_ids = torch.tensor([int(env_id)], device=self._env.device, dtype=torch.long)
    if not bool(self._manual_mode[env_ids].all().item()):
      self.set_automatic_mode(int(env_id), False)

    current = bool(self._manual_events[int(env_id), event_index].item())
    if current == bool(enabled):
      return
    self._manual_events[int(env_id), event_index] = bool(enabled)

    if enabled:
      self._apply_manual_event(env_ids, event_name)
    elif event_name == "wind":
      self._clear_wind(env_ids)
    elif event_name == "ice":
      self._set_friction(env_ids, self._params["normal_friction_range"])
      self._hide_ice_patch(env_ids)
    elif event_name == "slope":
      self._clear_slope_immediately(env_ids)
    elif event_name == "bumps":
      self._hide_rough_ground(env_ids)

  def _sample_duration(
    self, env_ids: torch.Tensor, duration_range: tuple[float, float]
  ) -> None:
    low, high = duration_range
    self._time_remaining[env_ids] = (
      torch.rand(len(env_ids), device=self._env.device) * (high - low) + low
    )

  def _set_friction(
    self, env_ids: torch.Tensor, friction_range: tuple[float, float]
  ) -> None:
    values = torch.empty(len(env_ids), device=self._env.device).uniform_(
      *friction_range
    )
    friction = self._env.sim.model.geom_friction
    env_grid, geom_grid = torch.meshgrid(
      env_ids, self._friction_geom_ids, indexing="ij"
    )
    friction[env_grid, geom_grid, 0] = values[:, None]

  def _clear_wind(self, env_ids: torch.Tensor) -> None:
    body_ids = self._params["asset_cfg"].body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else self._asset.num_bodies
    zeros = torch.zeros(
      (len(env_ids), num_bodies, 3), device=self._env.device
    )
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=body_ids
    )

  def _set_overlay_rgba(
    self, env_ids: torch.Tensor, geom_ids: torch.Tensor, rgba
  ) -> None:
    colors = self._env.sim.model.geom_rgba
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    color = torch.as_tensor(rgba, device=self._env.device, dtype=colors.dtype)
    if color.ndim == 1:
      color = color.expand(len(geom_ids), 4)
    colors[env_grid, geom_grid] = color[None, :, :]

  def _hide_event_overlays(self, env_ids: torch.Tensor) -> None:
    self._hide_ice_patch(env_ids)
    self._hide_slope_patch(env_ids)

  def _hide_ice_patch(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._hide_mocap_body(env_ids, self._ice_mocap_id)

  def _activate_ice_patch(self, env_ids: torch.Tensor) -> None:
    """Place a short low-friction strip a few metres ahead on the trail."""
    if len(env_ids) == 0:
      return
    centers, tangents = self._sample_patch_frames(env_ids)
    local_pos = torch.zeros((len(env_ids), 3), device=self._env.device)
    local_pos[:, :2] = centers
    # Slightly proud of the base treadmill so the high-priority ice geom wins
    # the contact without introducing a perceptible gait step.
    local_pos[:, 2] = 0.0055
    zero_pitch = torch.zeros(len(env_ids), device=self._env.device)
    quat = self._yaw_pitch_quat(tangents, zero_pitch)
    self._write_mocap_pose(env_ids, self._ice_mocap_id, local_pos, quat)
    self._set_geom_friction(
      env_ids, self._ice_geom_ids, self._params["ice_friction_range"]
    )

  def _hide_slope_patch(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    for mocap_id in self._slope_mocap_ids:
      self._hide_mocap_body(env_ids, mocap_id)

  def _write_rough_patch_pose(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    centers = self._rough_patch_center_xy[env_ids]
    tangents = self._rough_patch_tangent_xy[env_ids]
    local_pos = torch.zeros((len(env_ids), 3), device=self._env.device)
    local_pos[:, :2] = centers
    local_pos[:, 2] = self._rough_patch_z[env_ids]
    zero_pitch = torch.zeros(len(env_ids), device=self._env.device)
    quat = self._yaw_pitch_quat(tangents, zero_pitch)
    self._write_mocap_pose(env_ids, self._rough_mocap_id, local_pos, quat)
    mound_pos = torch.zeros(
      (len(env_ids), len(self._rough_mound_mocap_ids), 3),
      device=self._env.device,
    )
    mound_pos[:, :, :2] = self._rough_mound_center_xy[env_ids]
    mound_pos[:, :, 2] = (
      self._rough_mound_center_z[env_ids] + self._rough_patch_z[env_ids, None]
    )
    rock_pos = torch.zeros(
      (len(env_ids), len(self._rough_rock_mocap_ids), 3),
      device=self._env.device,
    )
    rock_pos[:, :, :2] = self._rough_rock_center_xy[env_ids]
    rock_pos[:, :, 2] = (
      self._rough_rock_center_z[env_ids] + self._rough_patch_z[env_ids, None]
    )
    primitive_pos = torch.cat((mound_pos, rock_pos), dim=1)
    primitive_pos = primitive_pos + self._env.scene.env_origins[env_ids, None, :]
    primitive_quat = torch.cat(
      (self._rough_mound_quat[env_ids], self._rough_rock_quat[env_ids]), dim=1
    )
    env_grid, mocap_grid = torch.meshgrid(
      env_ids, self._rough_primitive_mocap_ids, indexing="ij"
    )
    self._env.sim.data.mocap_pos[env_grid, mocap_grid] = primitive_pos
    self._env.sim.data.mocap_quat[env_grid, mocap_grid] = primitive_quat

  def _update_slope_patch_geometry(
    self, env_ids: torch.Tensor, gradients: torch.Tensor
  ) -> None:
    """Update a continuous multi-piece hill with varying local gradients."""
    if len(env_ids) == 0:
      return
    centers = self._slope_patch_center_xy[env_ids]
    tangents = self._slope_patch_tangent_xy[env_ids]
    piece_count = len(self._slope_patch_geom_ids)
    if piece_count != self._slope_profile_factors.shape[1]:
      raise ValueError("Slope patch geometry/profile piece counts do not match.")

    half_length = torch.full(
      (len(env_ids),),
      float(self._env.sim.mj_model.geom_size[int(self._slope_patch_geom_ids[0].item()), 0]),
      device=self._env.device,
    )
    half_height = torch.full(
      (len(env_ids),),
      float(self._env.sim.mj_model.geom_size[int(self._slope_patch_geom_ids[0].item()), 2]),
      device=self._env.device,
    )
    local_gradients = gradients[:, None] * self._slope_profile_factors[env_ids]
    theta = torch.atan(local_gradients)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    full_horizontal = 2.0 * half_length[:, None] * cos_theta
    full_rise = 2.0 * half_length[:, None] * sin_theta
    total_horizontal = full_horizontal.sum(dim=1)
    horizontal_cursor = -0.5 * total_horizontal
    vertical_cursor = torch.full(
      (len(env_ids),), 0.0015, device=self._env.device
    )

    # Construct the pieces from left to right. Each next top face begins exactly
    # where the previous one ends, so changing grade does not introduce steps.
    for piece in range(piece_count):
      piece_cos = cos_theta[:, piece]
      piece_sin = sin_theta[:, piece]
      # A tilted box's top face shifts horizontally by h*sin(theta) relative
      # to its body origin. Compensate for that so adjacent *top faces*, not
      # merely body centerlines, meet at the same point when grade changes.
      center_offset = (
        horizontal_cursor
        + half_length * piece_cos
        + half_height * piece_sin
      )
      local_pos = torch.zeros((len(env_ids), 3), device=self._env.device)
      local_pos[:, :2] = centers + center_offset[:, None] * tangents
      top_center_z = vertical_cursor + half_length * piece_sin
      local_pos[:, 2] = top_center_z - half_height * piece_cos
      # Positive terrain grade requires negative MuJoCo Y pitch in this frame.
      quat = self._yaw_pitch_quat(tangents, -theta[:, piece])
      self._write_mocap_pose(
        env_ids, self._slope_mocap_ids[piece], local_pos, quat
      )
      horizontal_cursor = horizontal_cursor + full_horizontal[:, piece]
      vertical_cursor = vertical_cursor + full_rise[:, piece]

  def _hide_rough_ground(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._rough_patch_z[env_ids] = self._ROUGH_BURIED_Z
    self._write_rough_patch_pose(env_ids)

  def _sync_rough_collision_bounds(self, env_ids: torch.Tensor) -> None:
    """Keep MuJoCo-Warp broad-phase bounds in sync with resized primitives.

    ``geom_size`` is changed per-world at runtime for the rough terrain. Warp's
    broad phase does not derive ``geom_aabb``/``geom_rbound`` from that field on
    every step, so leaving the compile-time bounds in place can cause a resized
    rock or mound to miss contacts even though Viser draws the new dimensions.
    """
    if len(env_ids) == 0:
      return
    model = self._env.sim.model

    mound_env_grid, mound_geom_grid = torch.meshgrid(
      env_ids, self._rough_mound_geom_ids, indexing="ij"
    )
    mound_sizes = model.geom_size[mound_env_grid, mound_geom_grid]
    model.geom_aabb[mound_env_grid, mound_geom_grid, 0] = 0.0
    model.geom_aabb[mound_env_grid, mound_geom_grid, 1] = mound_sizes
    model.geom_rbound[mound_env_grid, mound_geom_grid] = mound_sizes.amax(dim=-1)

    rock_env_grid, rock_geom_grid = torch.meshgrid(
      env_ids, self._rough_rock_geom_ids, indexing="ij"
    )
    rock_sizes = model.geom_size[rock_env_grid, rock_geom_grid]
    model.geom_aabb[rock_env_grid, rock_geom_grid, 0] = 0.0
    model.geom_aabb[rock_env_grid, rock_geom_grid, 1] = rock_sizes
    model.geom_rbound[rock_env_grid, rock_geom_grid] = torch.linalg.vector_norm(
      rock_sizes, dim=-1
    )

  def _rock_vertical_half_extents(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Return world-Z half extents for the currently rotated box rocks."""
    model = self._env.sim.model
    env_grid, geom_grid = torch.meshgrid(
      env_ids, self._rough_rock_geom_ids, indexing="ij"
    )
    sizes = model.geom_size[env_grid, geom_grid]
    quat = model.geom_quat[env_grid, geom_grid]
    w, x, y, z = quat.unbind(dim=-1)
    # Third row of the scalar-first quaternion rotation matrix. A box's
    # vertical support radius is |R_zx|*sx + |R_zy|*sy + |R_zz|*sz.
    rz_x = 2.0 * (x * z - w * y)
    rz_y = 2.0 * (y * z + w * x)
    rz_z = 1.0 - 2.0 * (x * x + y * y)
    return (
      rz_x.abs() * sizes[..., 0]
      + rz_y.abs() * sizes[..., 1]
      + rz_z.abs() * sizes[..., 2]
    )

  def _rock_corner_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Return all eight rotated box-corner offsets in terrain-local XYZ.

    Seating a tilted box from a single height sample at its center is not
    sufficient on an ellipsoid: the lowest corner is horizontally displaced
    from the center and can therefore sit over a much lower part of the mound.
    Sampling the actual corner footprint guarantees that at least one physical
    rock corner is grounded on the visible collision surface.
    """
    model = self._env.sim.model
    env_grid, geom_grid = torch.meshgrid(
      env_ids, self._rough_rock_geom_ids, indexing="ij"
    )
    sizes = model.geom_size[env_grid, geom_grid]
    quat = model.geom_quat[env_grid, geom_grid]
    w, x, y, z = quat.unbind(dim=-1)

    rotation = torch.empty(
      (*quat.shape[:-1], 3, 3), device=quat.device, dtype=quat.dtype
    )
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - w * z)
    rotation[..., 0, 2] = 2.0 * (x * z + w * y)
    rotation[..., 1, 0] = 2.0 * (x * y + w * z)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - w * x)
    rotation[..., 2, 0] = 2.0 * (x * z - w * y)
    rotation[..., 2, 1] = 2.0 * (y * z + w * x)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    signs = torch.tensor(
      (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
      ),
      device=sizes.device,
      dtype=sizes.dtype,
    )
    local_corners = sizes[:, :, None, :] * signs[None, None, :, :]
    return torch.matmul(local_corners, rotation.transpose(-1, -2))

  def _mound_surface_at(
    self,
    mound_xy: torch.Tensor,
    mound_sizes: torch.Tensor,
    mound_top_heights: torch.Tensor,
    query_xy: torch.Tensor,
  ) -> torch.Tensor:
    """Sample the top of the overlapping ellipsoid collision surface."""
    # Shapes: mound_xy (E,M,2), mound_sizes (E,M,3), top (E,M), query (E,Q,2).
    delta = query_xy[:, :, None, :] - mound_xy[:, None, :, :]
    normalized_sq = (
      (delta[..., 0] / mound_sizes[:, None, :, 0]).square()
      + (delta[..., 1] / mound_sizes[:, None, :, 1]).square()
    )
    inside = normalized_sq < 1.0
    center_z = mound_top_heights - mound_sizes[..., 2]
    surface = center_z[:, None, :] + mound_sizes[:, None, :, 2] * torch.sqrt(
      torch.clamp(1.0 - normalized_sq, min=0.0)
    )
    # The base treadmill is z=0, so an uncovered query should seat there.
    surface = torch.where(inside, surface, torch.zeros_like(surface))
    return surface.amax(dim=-1).clamp_min(0.0)

  def _write_rough_heights(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    positions = self._env.sim.model.geom_pos
    sizes = self._env.sim.model.geom_size
    mound_count = len(self._rough_mound_geom_ids)

    mound_env_grid, mound_geom_grid = torch.meshgrid(
      env_ids, self._rough_mound_geom_ids, indexing="ij"
    )
    positions[mound_env_grid, mound_geom_grid, 2] = (
      self._rough_heights[env_ids, :mound_count]
      - sizes[mound_env_grid, mound_geom_grid, 2]
    )

    rock_env_grid, rock_geom_grid = torch.meshgrid(
      env_ids, self._rough_rock_geom_ids, indexing="ij"
    )
    positions[rock_env_grid, rock_geom_grid, 2] = (
      self._rough_heights[env_ids, mound_count:]
      - self._rock_vertical_half_extents(env_ids)
    )

  def _activate_rough_ground(self, env_ids: torch.Tensor) -> None:
    """Spawn a fully randomized formation of rounded mounds and angular rocks."""
    if len(env_ids) == 0:
      return
    centers, tangents = self._sample_patch_frames(env_ids)
    self._rough_patch_center_xy[env_ids] = centers
    self._rough_patch_tangent_xy[env_ids] = tangents
    normals = torch.stack((-tangents[:, 1], tangents[:, 0]), dim=1)

    forward_min, forward_max = self._params["rough_forward_range_m"]
    lateral_min, lateral_max = self._params["rough_lateral_range_m"]
    mound_height_min, mound_height_max = self._params[
      "rough_surface_height_range_m"
    ]

    # Every rounded mound also moves independently. Start from a stratified
    # rows/cols footprint so the patch remains well covered, shuffle which
    # differently-sized mound occupies each slot, then add fresh XY jitter.
    rows = int(self._params["rough_rows"])
    cols = int(self._params["rough_cols"])
    mound_count = len(self._rough_mound_geom_ids)
    if rows * cols != mound_count:
      raise ValueError("Rough-terrain mound layout does not match rows/cols.")
    forward_slots = torch.linspace(
      forward_min, forward_max, rows, device=self._env.device
    ).repeat_interleave(cols)
    lateral_slots = torch.linspace(
      lateral_min, lateral_max, cols, device=self._env.device
    ).repeat(rows)
    slot_xy = torch.stack((forward_slots, lateral_slots), dim=1)
    mound_permutations = torch.argsort(
      torch.rand((len(env_ids), mound_count), device=self._env.device), dim=1
    )
    mound_slots = slot_xy[mound_permutations]
    mound_forward = mound_slots[:, :, 0] + (
      torch.rand((len(env_ids), mound_count), device=self._env.device) - 0.5
    ) * 0.20
    mound_lateral = mound_slots[:, :, 1] + (
      torch.rand((len(env_ids), mound_count), device=self._env.device) - 0.5
    ) * 0.14
    mound_forward.clamp_(forward_min, forward_max)
    mound_lateral.clamp_(lateral_min, lateral_max)
    self._rough_mound_center_xy[env_ids] = (
      centers[:, None, :]
      + mound_forward[:, :, None] * tangents[:, None, :]
      + mound_lateral[:, :, None] * normals[:, None, :]
    )

    mound_exposed_heights = torch.empty(
      (len(env_ids), mound_count), device=self._env.device
    ).uniform_(mound_height_min, mound_height_max)
    mound_radius_z = torch.tensor(
      [
        float(self._env.sim.mj_model.geom_size[int(geom_id.item()), 2])
        for geom_id in self._rough_mound_geom_ids
      ],
      device=self._env.device,
    )
    self._rough_mound_center_z[env_ids] = (
      mound_exposed_heights - mound_radius_z[None, :]
    )
    base_yaw = torch.atan2(tangents[:, 1], tangents[:, 0])[:, None]
    mound_yaw = base_yaw + (
      torch.rand((len(env_ids), mound_count), device=self._env.device) - 0.5
    ) * (2.0 * math.pi)
    self._rough_mound_quat[env_ids, :, 0] = torch.cos(0.5 * mound_yaw)
    self._rough_mound_quat[env_ids, :, 1] = 0.0
    self._rough_mound_quat[env_ids, :, 2] = 0.0
    self._rough_mound_quat[env_ids, :, 3] = torch.sin(0.5 * mound_yaw)

    rock_count = len(self._rough_rock_geom_ids)
    height_min, height_max = self._params["rock_height_range_m"]

    # Stratify the forward axis so rocks do not all clump into one small area,
    # then jitter every slot and independently randomize lateral position.
    slot_centers = torch.linspace(
      forward_min + 0.10,
      forward_max - 0.10,
      rock_count,
      device=self._env.device,
    )
    slot_spacing = (forward_max - forward_min) / max(rock_count, 1)
    permutations = torch.argsort(
      torch.rand((len(env_ids), rock_count), device=self._env.device), dim=1
    )
    rock_forward = slot_centers[permutations]
    rock_forward += (
      torch.rand((len(env_ids), rock_count), device=self._env.device) - 0.5
    ) * (0.75 * slot_spacing)
    rock_forward.clamp_(forward_min + 0.05, forward_max - 0.05)
    rock_lateral = torch.empty(
      (len(env_ids), rock_count), device=self._env.device
    ).uniform_(lateral_min + 0.06, lateral_max - 0.06)
    self._rough_rock_center_xy[env_ids] = (
      centers[:, None, :]
      + rock_forward[:, :, None] * tangents[:, None, :]
      + rock_lateral[:, :, None] * normals[:, None, :]
    )

    exposed_heights = torch.empty(
      (len(env_ids), rock_count), device=self._env.device
    ).uniform_(height_min, height_max)
    rock_half_heights = torch.tensor(
      [
        float(self._env.sim.mj_model.geom_size[int(geom_id.item()), 2])
        for geom_id in self._rough_rock_geom_ids
      ],
      device=self._env.device,
    )
    # The box stays fixed-size; burying more or less of it changes the exposed
    # collision height without modifying geom_size and triggering a Viser rebuild.
    self._rough_rock_center_z[env_ids] = (
      exposed_heights - rock_half_heights[None, :] - self._ROCK_SEATING_EMBED_M
    )

    yaw = base_yaw + (
      torch.rand((len(env_ids), rock_count), device=self._env.device) - 0.5
    ) * (2.0 * math.pi)
    self._rough_rock_quat[env_ids, :, 0] = torch.cos(0.5 * yaw)
    self._rough_rock_quat[env_ids, :, 1] = 0.0
    self._rough_rock_quat[env_ids, :, 2] = 0.0
    self._rough_rock_quat[env_ids, :, 3] = torch.sin(0.5 * yaw)

    self._rough_patch_z[env_ids] = 0.0
    self._write_rough_patch_pose(env_ids)

  def _lower_rough_ground(self, env_ids: torch.Tensor) -> torch.Tensor:
    if len(env_ids) == 0:
      return env_ids
    delta = self._params["rough_clear_delta_per_step_m"]
    self._rough_patch_z[env_ids] = torch.clamp(
      self._rough_patch_z[env_ids] - delta,
      min=self._ROUGH_BURIED_Z,
      max=0.0,
    )
    self._write_rough_patch_pose(env_ids)
    cleared = env_ids[self._rough_patch_z[env_ids] <= self._ROUGH_BURIED_Z + 1.0e-6]
    self._hide_rough_ground(cleared)
    return cleared

  def _ensure_slope_state(self) -> None:
    env = self._env
    if not hasattr(env, "treadmill_slope_gradient"):
      env.treadmill_slope_gradient = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "treadmill_slope_target_gradient"):
      env.treadmill_slope_target_gradient = torch.zeros(
        env.num_envs, device=env.device
      )

  def _set_break(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._phase[env_ids] = self.BREAK
    self._clear_wind(env_ids)
    self._set_friction(env_ids, self._params["normal_friction_range"])
    # Terrain obstacles are persistent once spawned. A break only ends the
    # scheduler's active-event phase; it must not remove the physical patch the
    # robot just encountered. Explicit manual removal and environment reset can
    # still hide patches through the dedicated clear helpers.
    self._sample_duration(env_ids, self._params["break_duration_range_s"])

  def _activate_random_event(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._clear_wind(env_ids)
    self._set_friction(env_ids, self._params["normal_friction_range"])
    choices = torch.randint(
      self.WIND, self.BUMPS + 1, (len(env_ids),), device=self._env.device
    )
    self._phase[env_ids] = choices
    self._sample_duration(env_ids, self._params["event_duration_range_s"])

    wind_ids = env_ids[choices == self.WIND]
    if len(wind_ids) > 0:
      super().__call__(
        self._env,
        wind_ids,
        self._params["wind_force_ranges"],
        self._params["asset_cfg"],
      )

    ice_ids = env_ids[choices == self.ICE]
    if len(ice_ids) > 0:
      self._set_friction(ice_ids, self._params["normal_friction_range"])
      self._activate_ice_patch(ice_ids)

    slope_ids = env_ids[choices == self.SLOPE]
    if len(slope_ids) > 0:
      low, high = self._params["active_slope_magnitude_range"]
      targets = torch.empty(len(slope_ids), device=self._env.device).uniform_(low, high)
      self._sample_slope_profile(slope_ids)
      centers, tangents = self._sample_patch_frames(slope_ids)
      self._slope_patch_center_xy[slope_ids] = centers
      self._slope_patch_tangent_xy[slope_ids] = tangents
      self._env.treadmill_slope_gradient[slope_ids] = 0.0
      self._env.treadmill_slope_target_gradient[slope_ids] = targets
      self._update_slope_patch_geometry(
        slope_ids, self._env.treadmill_slope_gradient[slope_ids]
      )

    bump_ids = env_ids[choices == self.BUMPS]
    self._activate_rough_ground(bump_ids)

  def _advance_slope(self, env_ids: torch.Tensor) -> None:
    active = env_ids[
      (self._phase[env_ids] == self.SLOPE)
      | (self._phase[env_ids] == self.SLOPE_CLEARING)
    ]
    if len(active) == 0:
      return
    current = self._env.treadmill_slope_gradient[active]
    target = self._env.treadmill_slope_target_gradient[active]
    maximum_delta = self._params["max_delta_per_step"]
    current = current + torch.clamp(
      target - current, min=-maximum_delta, max=maximum_delta
    )
    self._env.treadmill_slope_gradient[active] = current
    self._update_slope_patch_geometry(active, current)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = self._ids(env_ids)
    self._ensure_slope_state()
    self._env.treadmill_slope_gradient[env_ids] = 0.0
    self._env.treadmill_slope_target_gradient[env_ids] = 0.0
    self._hide_ice_patch(env_ids)
    self._hide_slope_patch(env_ids)
    automatic_ids = env_ids[~self._manual_mode[env_ids]]
    manual_ids = env_ids[self._manual_mode[env_ids]]
    self._set_break(automatic_ids)
    if len(manual_ids) > 0:
      self._clear_all_event_effects(manual_ids)
      self._phase[manual_ids] = self.BREAK
      self._time_remaining[manual_ids] = 0.0
      for event_index, event_name in enumerate(self.MANUAL_EVENT_NAMES):
        enabled_ids = manual_ids[self._manual_events[manual_ids, event_index]]
        self._apply_manual_event(enabled_ids, event_name)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_asset_cfg: SceneEntityCfg,
    terrain_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    normal_friction_range: tuple[float, float],
    ice_friction_range: tuple[float, float],
    wind_force_ranges: dict[str, tuple[float, float]],
    slope_gradient_range: tuple[float, float],
    active_slope_magnitude_range: tuple[float, float],
    slope_piece_count: int,
    slope_outer_fraction_range: tuple[float, float],
    slope_inner_fraction_range: tuple[float, float],
    event_patch_ahead_range_m: tuple[float, float],
    rough_surface_height_range_m: tuple[float, float],
    rough_forward_range_m: tuple[float, float],
    rough_lateral_range_m: tuple[float, float],
    rock_height_range_m: tuple[float, float],
    rock_count: int,
    rough_rows: int,
    rough_cols: int,
    event_duration_range_s: tuple[float, float],
    break_duration_range_s: tuple[float, float],
    max_delta_per_step: float,
    rough_clear_delta_per_step_m: float,
  ) -> None:
    del (
      env,
      asset_cfg,
      friction_asset_cfg,
      terrain_cfg,
      robot_cfg,
      normal_friction_range,
      ice_friction_range,
      wind_force_ranges,
      slope_piece_count,
      slope_outer_fraction_range,
      slope_inner_fraction_range,
      event_patch_ahead_range_m,
      rough_surface_height_range_m,
      rough_forward_range_m,
      rough_lateral_range_m,
      rock_height_range_m,
      rock_count,
      rough_rows,
      rough_cols,
      event_duration_range_s,
      break_duration_range_s,
      max_delta_per_step,
      rough_clear_delta_per_step_m,
    )
    maximum_slope = max(abs(slope_gradient_range[0]), abs(slope_gradient_range[1]))
    if active_slope_magnitude_range[1] > maximum_slope:
      raise ValueError("Active slope magnitude exceeds slope_gradient_range.")
    env_ids = self._ids(env_ids)
    env_ids = env_ids[~self._manual_mode[env_ids]]
    if len(env_ids) == 0:
      return
    self._time_remaining[env_ids] -= self._env.step_dt

    expired = env_ids[self._time_remaining[env_ids] <= 0.0]
    # Snapshot before mutating phases. Otherwise a newly activated wind/ice
    # phase would also look like an expired active phase in this same call and
    # be cleared immediately.
    expired_phase = self._phase[expired].clone()
    break_expired = expired[expired_phase == self.BREAK]
    self._activate_random_event(break_expired)

    finished = expired[
      (expired_phase == self.WIND)
      | (expired_phase == self.ICE)
      | (expired_phase == self.SLOPE)
      | (expired_phase == self.BUMPS)
    ]
    self._set_break(finished)

    self._advance_slope(env_ids)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    super().debug_vis(visualizer)
