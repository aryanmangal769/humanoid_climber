"""Convert the published HumanUP joint trajectory to MjLab's G1 motion format."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

HUMANUP_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)
INITIAL_ROOT_QUAT_WXYZ = np.array(
  [-0.86587, -0.0098234, 0.49986, 0.017525], dtype=np.float64
)
STANDING_ROOT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=Path, help="NPZ with joint_pos and head_height")
  parser.add_argument("output", type=Path)
  parser.add_argument("--duration", type=float, default=8.0)
  parser.add_argument("--fps", type=int, default=50)
  return parser.parse_args()


def _interpolate(values: np.ndarray, frame_count: int) -> np.ndarray:
  source_phase = np.linspace(0.0, 1.0, values.shape[0])
  target_phase = np.linspace(0.0, 1.0, frame_count)
  return np.stack(
    [np.interp(target_phase, source_phase, values[:, i]) for i in range(values.shape[1])],
    axis=1,
  )


def _slerp(start: np.ndarray, end: np.ndarray, phase: np.ndarray) -> np.ndarray:
  start = start / np.linalg.norm(start)
  end = end / np.linalg.norm(end)
  dot = float(np.dot(start, end))
  if dot < 0.0:
    end = -end
    dot = -dot
  dot = np.clip(dot, -1.0, 1.0)
  if dot > 0.9995:
    result = start[None] + phase[:, None] * (end - start)[None]
    return result / np.linalg.norm(result, axis=1, keepdims=True)
  theta = np.arccos(dot)
  return (
    np.sin((1.0 - phase) * theta)[:, None] * start[None]
    + np.sin(phase * theta)[:, None] * end[None]
  ) / np.sin(theta)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
  result = quat.copy()
  result[..., 1:] *= -1.0
  return result


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
  rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
  return np.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    axis=-1,
  )


def _angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
  result = np.zeros((len(quats), quats.shape[1], 3), dtype=np.float32)
  for frame in range(len(quats)):
    before = max(frame - 1, 0)
    after = min(frame + 1, len(quats) - 1)
    span = max((after - before) * dt, dt)
    delta = _quat_multiply(quats[after], _quat_conjugate(quats[before]))
    flip = delta[..., :1] < 0.0
    delta = np.where(flip, -delta, delta)
    vector_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., 0], 1.0e-8, None))
    axis = delta[..., 1:] / np.clip(vector_norm[..., None], 1.0e-8, None)
    result[frame] = axis * (angle / span)[..., None]
  return result


def build_motion(input_path: Path, output_path: Path, duration: float, fps: int) -> None:
  source = np.load(input_path)
  source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
  source_head_height = np.asarray(source["head_height"], dtype=np.float64).reshape(-1)
  if source_joint_pos.ndim != 2 or source_joint_pos.shape[1] != len(HUMANUP_JOINT_NAMES):
    raise ValueError(f"Expected HumanUP joint_pos shape (frames, 23), got {source_joint_pos.shape}")
  if source_head_height.shape[0] != source_joint_pos.shape[0]:
    raise ValueError("head_height and joint_pos must have equal frame counts")

  frame_count = round(duration * fps) + 1
  dt = 1.0 / fps
  humanup_joint_pos = _interpolate(source_joint_pos, frame_count)
  head_height = _interpolate(source_head_height[:, None], frame_count)[:, 0]
  rise_phase = (head_height - head_height.min()) / max(np.ptp(head_height), 1.0e-8)
  rise_phase = np.maximum.accumulate(rise_phase)
  rise_phase = rise_phase * rise_phase * (3.0 - 2.0 * rise_phase)
  root_quat = _slerp(INITIAL_ROOT_QUAT_WXYZ, STANDING_ROOT_QUAT_WXYZ, rise_phase)

  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  joint_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
    for idx in range(1, model.njnt)
  )
  if len(joint_names) != 29:
    raise ValueError(f"Expected 29 MjLab G1 joints, got {len(joint_names)}")
  joint_indexes = {name: idx for idx, name in enumerate(joint_names)}

  joint_pos = np.zeros((frame_count, len(joint_names)), dtype=np.float64)
  for source_idx, name in enumerate(HUMANUP_JOINT_NAMES):
    joint_pos[:, joint_indexes[name]] = humanup_joint_pos[:, source_idx]

  root_z = 0.056526 + rise_phase * (0.783675 - 0.056526)
  qpos = np.zeros((frame_count, model.nq), dtype=np.float64)
  qpos[:, 2] = root_z
  qpos[:, 3:7] = root_quat
  qpos[:, 7:] = joint_pos

  joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
  body_pos_w = np.empty((frame_count, model.nbody - 1, 3), dtype=np.float32)
  body_quat_w = np.empty((frame_count, model.nbody - 1, 4), dtype=np.float32)
  for frame in range(frame_count):
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    body_pos_w[frame] = data.xpos[1:]
    body_quat_w[frame] = data.xquat[1:]
  body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
  body_ang_vel_w = _angular_velocity(body_quat_w, dt)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_path,
    fps=np.array([fps], dtype=np.int32),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    source_head_height=head_height.astype(np.float32),
  )
  print(f"Wrote {frame_count} frames ({duration:.2f}s at {fps}Hz) to {output_path}")


def main() -> None:
  args = parse_args()
  build_motion(args.input, args.output, args.duration, args.fps)


if __name__ == "__main__":
  main()
