"""Local inference adapters for the deterministic specialist bank."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CHECKPOINTS = {
  "flat": PROJECT_ROOT / "ckpt" / "g1_velocity_model_final.pt",
  "ice_incline": PROJECT_ROOT / "ckpt" / "recovered" / "model_34400.pt",
  "wind": PROJECT_ROOT / "ckpt" / "wind-specialist" / "training" / "g1_flat_wind"
  / "base-finetune" / "2026-08-30_04-45-48_flat-wind-base-finetune"
  / "model_34998.pt",
  # The rough specialist intentionally starts as an explicit copy of flat.
  "rough": PROJECT_ROOT / "ckpt" / "g1_velocity_model_final.pt",
}
RECOVERY_CHECKPOINT = (
  PROJECT_ROOT / "ckpt" / "recovery-specialist" / "training" / "g1_recovery"
  / "supine-native-tracking" / "2026-08-30_04-45-47_supine-native-tracking"
  / "model_19999.pt"
)
RECOVERY_MOTION = (
  PROJECT_ROOT / "private_assets" / "recovery" / "g1_humanup_getup_50hz.npz"
)


def _actor_tensor(observations: Any) -> torch.Tensor:
  if isinstance(observations, Mapping):
    observations = observations["actor"]
  if not isinstance(observations, torch.Tensor):
    raise TypeError("Policy observations must contain an actor tensor")
  return observations


class CheckpointActor:
  """Small deterministic actor loader independent of MjLab's single checkpoint UI."""

  def __init__(self, path: Path, *, device: torch.device | str) -> None:
    if not path.is_file():
      raise FileNotFoundError(f"Specialist checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["actor_state_dict"]
    self.path = path
    self.device = torch.device(device)
    self.mean = state["obs_normalizer._mean"].to(self.device)
    self.std = state["obs_normalizer._std"].to(self.device)
    self.layers = tuple(
      (
        state[f"mlp.{index}.weight"].to(self.device),
        state[f"mlp.{index}.bias"].to(self.device),
      )
      for index in (0, 2, 4, 6)
    )

  @property
  def observation_size(self) -> int:
    return int(self.mean.shape[-1])

  @property
  def action_size(self) -> int:
    return int(self.layers[-1][0].shape[0])

  def tensor_action(self, observations: torch.Tensor) -> torch.Tensor:
    if observations.shape[-1] != self.observation_size:
      raise ValueError(
        f"{self.path.name} expects {self.observation_size} observations, "
        f"received {observations.shape[-1]}"
      )
    value = (observations.to(self.device) - self.mean) / self.std.clamp_min(1.0e-6)
    value = value.clamp(-5.0, 5.0)
    for layer_index, (weight, bias) in enumerate(self.layers):
      value = F.linear(value, weight, bias)
      if layer_index + 1 < len(self.layers):
        value = F.elu(value)
    return value

  def __call__(self, observations: Any) -> torch.Tensor:
    return self.tensor_action(_actor_tensor(observations))


class SpecialistPolicyBank:
  """Preloaded compatible locomotion actors with an honest rough-policy alias."""

  def __init__(self, *, device: torch.device | str) -> None:
    actors: dict[Path, CheckpointActor] = {}
    self._policies: dict[str, CheckpointActor] = {}
    for key, path in POLICY_CHECKPOINTS.items():
      resolved = path.resolve()
      actor = actors.get(resolved)
      if actor is None:
        actor = CheckpointActor(resolved, device=device)
        if actor.observation_size != 99:
          raise ValueError(f"Locomotion policy {key} does not use 99 observations")
        actors[resolved] = actor
      self._policies[key] = actor

  def action(self, policy_key: str, observations: Any) -> torch.Tensor:
    try:
      policy = self._policies[policy_key]
    except KeyError as exc:
      raise KeyError(f"Unknown locomotion specialist: {policy_key}") from exc
    return policy(observations)


class RecoveryPolicyAdapter:
  """Build the tracking actor's 160-value observation inside the velocity task."""

  def __init__(self, env: Any, *, device: torch.device | str) -> None:
    self._env = getattr(env, "unwrapped", env)
    self._actor = CheckpointActor(RECOVERY_CHECKPOINT, device=device)
    if self._actor.observation_size != 160:
      raise ValueError("Recovery policy must use the native 160-observation interface")
    if self._actor.action_size != 29:
      raise ValueError("Recovery policy must use the native 29-action G1 interface")
    with np.load(RECOVERY_MOTION) as motion:
      motion_fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
      self._joint_pos = torch.as_tensor(
        motion["joint_pos"], dtype=torch.float32, device=device
      )
      self._joint_vel = torch.as_tensor(
        motion["joint_vel"], dtype=torch.float32, device=device
      )
      self._body_pos = torch.as_tensor(
        motion["body_pos_w"], dtype=torch.float32, device=device
      )
      self._body_quat = torch.as_tensor(
        motion["body_quat_w"], dtype=torch.float32, device=device
      )
      self._body_lin_vel = torch.as_tensor(
        motion["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      self._body_ang_vel = torch.as_tensor(
        motion["body_ang_vel_w"], dtype=torch.float32, device=device
      )
    if motion_fps != 50:
      raise ValueError(f"Recovery motion must run at 50 Hz, received {motion_fps} Hz")
    if self._joint_pos.ndim != 2 or self._joint_pos.shape[1] != 29:
      raise ValueError("Recovery motion must contain 29 G1 joint positions per frame")
    if self._joint_vel.shape != self._joint_pos.shape:
      raise ValueError("Recovery motion joint position/velocity shapes must match")
    robot = self._env.scene["robot"]
    self._anchor_index = robot.body_names.index("torso_link")
    self._frame = 0
    self._xy_offset: torch.Tensor | None = None
    self._active = False

  @property
  def active(self) -> bool:
    return self._active

  @property
  def finished(self) -> bool:
    return self._frame >= len(self._joint_pos) - 1

  def start(self, env_idx: int, walking_observations: Any) -> None:
    """Recreate the standalone frame-zero state before recovery inference."""
    actor_obs = _actor_tensor(walking_observations)
    if actor_obs.ndim != 2 or actor_obs.shape[1] != 99:
      raise ValueError("Integrated recovery requires the 99-value walking observation")
    if not 0 <= env_idx < actor_obs.shape[0]:
      raise IndexError(f"Recovery environment index out of range: {env_idx}")
    robot = self._env.scene["robot"]
    # The standalone play task uses MotionCommandCfg(sampling_mode="start") and
    # therefore always begins at frame zero. Joint-only nearest-frame matching
    # could skip the initial roll/rise phase after a fall, placing the policy
    # outside the state distribution in which it was demonstrated to recover.
    self._frame = 0
    current_anchor = robot.data.body_link_pos_w[env_idx, self._anchor_index]
    reference_anchor = self._body_pos[self._frame, self._anchor_index]
    env_origin = self._env.scene.env_origins[env_idx]
    joint_rmse = torch.mean(
      (robot.data.joint_pos[env_idx] - self._joint_pos[self._frame]).square()
    ).sqrt()
    current_quat = robot.data.body_link_quat_w[env_idx, self._anchor_index]
    reference_quat = self._body_quat[self._frame, self._anchor_index]
    quat_dot = torch.dot(current_quat, reference_quat).abs().clamp(0.0, 1.0)
    orientation_error_deg = math.degrees(2.0 * math.acos(float(quat_dot.item())))
    height_error_m = float(
      (current_anchor[2] - env_origin[2] - reference_anchor[2]).item()
    )
    print(
      "[RECOVERY START] "
      f"reference_frame=0 joint_rmse={float(joint_rmse.item()):.3f} rad "
      f"torso_height_error={height_error_m:+.3f} m "
      f"torso_orientation_error={orientation_error_deg:.1f} deg "
      "canonicalized=true"
    )

    # Standalone playback does not ask this policy to recover from an arbitrary
    # terminal walking pose: MotionCommand writes the frame-zero reference root,
    # joints, and velocities into MuJoCo first. Reproduce that contract here,
    # retaining only the fallen robot's world XY location.
    root_reference = self._body_pos[self._frame, 0]
    current_root = robot.data.root_link_pos_w[env_idx]
    self._xy_offset = current_root[:2] - env_origin[:2] - root_reference[:2]
    root_pos = root_reference.clone() + env_origin
    root_pos[:2] += self._xy_offset
    root_state = torch.cat(
      (
        root_pos,
        self._body_quat[self._frame, 0],
        self._body_lin_vel[self._frame, 0],
        self._body_ang_vel[self._frame, 0],
      )
    )[None, :]
    env_ids = torch.tensor(
      [env_idx], device=robot.data.joint_pos.device, dtype=torch.long
    )
    robot.write_joint_state_to_sim(
      self._joint_pos[self._frame][None, :],
      self._joint_vel[self._frame][None, :],
      env_ids=env_ids,
    )
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.reset(env_ids=env_ids)
    self._env.sim.forward()
    self._active = True

  def reset(self) -> None:
    self._frame = 0
    self._xy_offset = None
    self._active = False

  def action(self, env_idx: int, walking_observations: Any) -> torch.Tensor:
    if not self._active or self._xy_offset is None:
      raise RuntimeError("Recovery policy must be started before inference")
    actor_obs = _actor_tensor(walking_observations)
    robot = self._env.scene["robot"]
    frame = min(self._frame, len(self._joint_pos) - 1)
    reference_pos = self._body_pos[frame, self._anchor_index].clone()
    reference_pos[:2] += self._xy_offset
    reference_pos = reference_pos[None, :] + self._env.scene.env_origins[env_idx, None]
    reference_quat = self._body_quat[frame, self._anchor_index][None, :]
    robot_pos = robot.data.body_link_pos_w[env_idx, self._anchor_index][None, :]
    robot_quat = robot.data.body_link_quat_w[env_idx, self._anchor_index][None, :]
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
      robot_pos, robot_quat, reference_pos, reference_quat
    )
    anchor_ori_b = matrix_from_quat(anchor_quat_b)[..., :2].reshape(1, 6)
    recovery_obs = torch.cat(
      (
        self._joint_pos[frame][None, :],
        self._joint_vel[frame][None, :],
        anchor_pos_b,
        anchor_ori_b,
        actor_obs[env_idx : env_idx + 1, 0:3],
        actor_obs[env_idx : env_idx + 1, 3:6],
        actor_obs[env_idx : env_idx + 1, 9:38],
        actor_obs[env_idx : env_idx + 1, 38:67],
        actor_obs[env_idx : env_idx + 1, 67:96],
      ),
      dim=1,
    )
    action = self._actor.tensor_action(recovery_obs)
    self._frame = min(self._frame + 1, len(self._joint_pos) - 1)
    return action
