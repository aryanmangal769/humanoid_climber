"""Small dependency-light loader for the exported Unitree G1 velocity policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "vendor/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
)


class G1VelocityPolicy:
    """Run an exported 98->29 ELU MLP without requiring ONNX Runtime."""

    def __init__(self, path: str | Path = DEFAULT_CHECKPOINT):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"G1 policy checkpoint is missing: {self.path}")
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise RuntimeError("The dashboard policy loader needs the 'onnx' package") from exc

        onnx_model = onnx.load(str(self.path))
        graph = onnx_model.graph
        tensors = {item.name: numpy_helper.to_array(item).astype(np.float32) for item in graph.initializer}
        self.mean = tensors["obs_normalizer._mean"].reshape(-1)
        std_tensor = tensors.get("obs_normalizer._std")
        if std_tensor is None:
            std_tensor = tensors.get("onnx::Div_24")
        if std_tensor is None:
            raise ValueError(f"Policy {self.path} has no observation-normalizer standard deviation")
        self.std = std_tensor.reshape(-1)
        self.weights = [tensors[f"mlp.{index}.weight"] for index in (0, 2, 4, 6)]
        self.biases = [tensors[f"mlp.{index}.bias"] for index in (0, 2, 4, 6)]
        self.metadata = {item.key: item.value for item in onnx_model.metadata_props}
        self.default_joint_pos = self._csv("default_joint_pos", 29)
        self.action_scale = self._csv("action_scale", 29)
        self.stiffness = self._csv("joint_stiffness", 29)
        self.damping = self._csv("joint_damping", 29)
        self.joint_names = tuple(self.metadata.get("joint_names", "").split(","))
        if self.mean.size not in {98, 99} or self.std.size != self.mean.size or self.weights[-1].shape[0] != 29:
            raise ValueError(f"Unexpected G1 policy schema in {self.path}")
        self.last_action = np.zeros(29, dtype=np.float32)
        self.inference_count = 0

    def _csv(self, key: str, size: int) -> np.ndarray:
        values = np.fromstring(self.metadata.get(key, ""), sep=",", dtype=np.float32)
        if values.size != size:
            raise ValueError(f"Policy metadata {key!r} has {values.size} values, expected {size}")
        return values

    def observation(
        self,
        data: Any,
        model: Any,
        *,
        command: tuple[float, float, float] = (0.5, 0.0, 0.0),
        period: float = 0.6,
    ) -> np.ndarray:
        """Build the exact actor ordering recorded in the exported metadata."""
        root = 1  # G1 pelvis is the first non-world body in this MJCF.
        rotation = np.asarray(data.xmat[root], dtype=np.float32).reshape(3, 3)
        gravity_b = rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        command_array = np.asarray(command, dtype=np.float32)
        phase = np.zeros(2, dtype=np.float32)
        if np.linalg.norm(command_array) >= 0.1:
            angle = 2.0 * np.pi * ((float(data.time) % period) / period)
            phase[:] = (np.sin(angle), np.cos(angle))
        joint_pos = np.asarray(data.qpos[7:], dtype=np.float32) - self.default_joint_pos
        joint_vel = np.asarray(data.qvel[6:], dtype=np.float32)
        import mujoco

        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if sensor_id < 0:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
        if sensor_id >= 0:
            start = model.sensor_adr[sensor_id]
            base_ang_vel = np.asarray(data.sensordata[start : start + 3], dtype=np.float32)
        else:
            base_ang_vel = rotation.T @ np.asarray(data.qvel[3:6], dtype=np.float32)
        if self.mean.size == 98:
            return np.concatenate(
                (base_ang_vel, gravity_b, command_array, phase, joint_pos, joint_vel, self.last_action)
            ).astype(np.float32)

        # MjLab 1.6 velocity actor ordering (99 values): body-frame linear and
        # angular velocity, projected gravity, joint position/velocity, prior
        # action, then the three-axis twist command. This layout has no gait
        # phase term and must not be confused with the bundled 98-value actor.
        base_lin_vel = rotation.T @ np.asarray(data.qvel[:3], dtype=np.float32)
        return np.concatenate(
            (base_lin_vel, base_ang_vel, gravity_b, joint_pos, joint_vel, self.last_action, command_array)
        ).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = (np.asarray(observation, dtype=np.float32).reshape(1, -1) - self.mean) / self.std
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight.T + bias
            if index < len(self.weights) - 1:
                # ``np.where`` evaluates both branches. Calling expm1 on the
                # unused, large positive activations therefore emitted an
                # overflow warning even though ONNX ELU leaves them unchanged.
                negative = value <= 0.0
                value[negative] = np.expm1(value[negative])
        action = value.reshape(-1).astype(np.float32)
        self.last_action = action.copy()
        self.inference_count += 1
        return action

    def target_positions(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized policy output into the G1 position targets."""
        return self.default_joint_pos + self.action_scale * np.asarray(action, dtype=np.float32)

    def configure_mujoco_actuators(self, model: Any) -> None:
        """Apply the PD gains exported with the checkpoint to MuJoCo.

        The Menagerie scene supplies position actuators in the correct order,
        but its generic gains differ from the gains used to train this policy.
        MuJoCo's position actuator force is ``kp*(target-q) - kv*qvel``.
        """
        import mujoco

        actuator_joints = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            actuator_joints.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            )
        if tuple(actuator_joints) != self.joint_names:
            raise ValueError(
                "Policy joint order does not match the MuJoCo actuator order: "
                f"policy={self.joint_names}, mujoco={tuple(actuator_joints)}"
            )
        if model.nu != self.stiffness.size or model.nu != self.damping.size:
            raise ValueError(f"Policy has {self.stiffness.size} gains for {model.nu} actuators")
        model.actuator_gainprm[:, 0] = self.stiffness
        model.actuator_biasprm[:, 1] = -self.stiffness
        model.actuator_biasprm[:, 2] = -self.damping

    def status(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.path),
            "checkpoint_exists": self.path.is_file(),
            "input_size": int(self.mean.size),
            "action_size": int(self.weights[-1].shape[0]),
            "joint_count": len(self.joint_names),
            "inference_count": self.inference_count,
            "run_path": self.metadata.get("run_path", "unknown"),
        }
