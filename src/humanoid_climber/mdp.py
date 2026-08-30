"""Project-specific MDP terms for Hum Climber."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_FORCE_AXES = ("x", "y", "z")


class apply_wind_force:
  """Apply wind and visualize its world-frame direction at the robot torso."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._wind_labels = {}
    self._velocity_labels = {}

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
          if key not in self._wind_labels:
            container = server.scene.add_3d_gui_container(
              name=f"/wind/status/{env_idx}/{body_id}",
              position=label_pos,
            )
            with container:
              label = server.gui.add_html(html)
            self._wind_labels[key] = (container, label)
          else:
            container, label = self._wind_labels[key]
            container.position = label_pos
            label.content = html

      self._draw_velocity_labels(visualizer, env_idx)

  def _draw_velocity_labels(
    self, visualizer: DebugVisualizer, env_idx: int
  ) -> None:
    """Place values below the command and actual planar-velocity arrows."""
    server = getattr(visualizer, "server", None)
    if server is None:
      return

    command_term = self._env.command_manager.get_term("twist")
    command = command_term.command[env_idx, :2]
    actual = self._asset.data.root_link_lin_vel_b[env_idx, :2]
    base_pos = self._asset.data.root_link_pos_w[env_idx]
    base_mat = matrix_from_quat(
      self._asset.data.root_link_quat_w[env_idx].unsqueeze(0)
    )[0]
    scale = command_term.cfg.viz.scale
    z_offset = command_term.cfg.viz.z_offset
    scene_offset = getattr(visualizer, "_scene_offset", 0.0)

    values = (
      ("command", command, "#5353b8", "CMD"),
      ("actual", actual, "#00a7e8", "ACTUAL"),
    )
    for row, (name, velocity, accent, title) in enumerate(values):
      local_tip = torch.cat((velocity, velocity.new_tensor([z_offset])))
      position = (
        base_pos + base_mat @ (local_tip * scale)
      ).detach().cpu().numpy()
      position[2] -= 0.12 + row * 0.13
      position += scene_offset
      speed = velocity.norm().item()
      text = f"{title}  {speed:.2f} m/s"
      html = (
        '<div style="font-weight:900;color:#000;background:rgba(255,255,255,.9);'
        f'border-left:6px solid {accent};padding:3px 7px;border-radius:4px;'
        f'white-space:nowrap">{text}</div>'
      )
      key = (env_idx, name)
      if key not in self._velocity_labels:
        container = server.scene.add_3d_gui_container(
          name=f"/velocity/status/{env_idx}/{name}",
          position=position,
        )
        with container:
          label = server.gui.add_html(html)
        self._velocity_labels[key] = (container, label)
      else:
        container, label = self._velocity_labels[key]
        container.position = position
        label.content = html

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Participate in manager lifecycle so debug visualization is discovered."""
    del env_ids