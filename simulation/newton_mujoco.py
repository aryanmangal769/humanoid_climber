"""Newton-owned G1 dynamics mirrored into native MuJoCo for rendering.

Newton imports the canonical G1 MJCF and advances it with ``SolverMuJoCo``.
The native MuJoCo model is a lossless pose mirror used by the existing renderer.
The queued body-wrench API is the coupling seam for future MPM impulses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from .snow import SnowLayer

MENAGERIE_G1_SCENE = (
    Path(__file__).resolve().parents[1]
    / "vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/scene_mjx.xml"
)


class NewtonMuJoCoBridge:
    """Advance a Newton model through MuJoCo and mirror it for native rendering."""

    def __init__(
        self,
        scene: str | Path = MENAGERIE_G1_SCENE,
        *,
        dt: float | None = None,
        device: str | None = None,
    ) -> None:
        self.scene = Path(scene).resolve()
        if not self.scene.is_file():
            raise FileNotFoundError(f"G1 scene is missing: {self.scene}")

        self.native_model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.native_data = mujoco.MjData(self.native_model)
        self.dt = float(dt if dt is not None else self.native_model.opt.timestep)
        self.device = wp.get_device(device)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(str(self.scene), up_axis="Z", parse_visuals=False)
        self.model = builder.finalize(device=self.device)
        self.solver = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.device.is_cpu,
            use_mujoco_contacts=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._queued_wrenches = np.zeros((self.model.body_count, 6), dtype=np.float32)
        self._body_ids = {
            label.rsplit("/", 1)[-1]: index for index, label in enumerate(self.model.body_label)
        }
        self.sim_time = 0.0
        self.snow = SnowLayer("snow")
        self.reset()

    @staticmethod
    def _mujoco_to_newton_q(qpos: np.ndarray) -> np.ndarray:
        """Convert MuJoCo's free-joint wxyz quaternion to Newton's xyzw."""
        result = np.asarray(qpos, dtype=np.float32).copy()
        result[3:7] = qpos[[4, 5, 6, 3]]
        return result

    @staticmethod
    def _newton_to_mujoco_q(joint_q: np.ndarray) -> np.ndarray:
        """Convert Newton's free-joint xyzw quaternion to MuJoCo's wxyz."""
        result = np.asarray(joint_q, dtype=np.float64).copy()
        result[3:7] = joint_q[[6, 3, 4, 5]]
        return result

    def reset(self, keyframe: str = "home") -> None:
        """Reset both representations from a named native MuJoCo keyframe."""
        key_id = mujoco.mj_name2id(self.native_model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"MuJoCo keyframe does not exist: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.native_model, self.native_data, key_id)

        joint_q = self._mujoco_to_newton_q(self.native_data.qpos)
        joint_qd = np.asarray(self.native_data.qvel, dtype=np.float32)
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        self.state_1.joint_q.assign(joint_q)
        self.state_1.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        # Newton 1.5 renamed the public target buffers and may expose position
        # targets either in coordinate layout (including the free-joint
        # quaternion) or in the legacy DoF layout. Preserve both layouts so
        # this bridge remains valid across imported Menagerie models.
        target_q = np.zeros_like(self.control.joint_target_q.numpy())
        if target_q.shape == joint_q.shape:
            target_q[:] = joint_q
        elif target_q.shape == (self.model.joint_dof_count,):
            # The imported Menagerie position actuators occupy the scalar DoFs
            # after the six free-base velocity coordinates.
            target_q[6:] = np.asarray(self.native_data.qpos[7:], dtype=np.float32)
        else:
            raise RuntimeError(
                "Unexpected Newton joint_target_q layout: "
                f"{target_q.shape}; coordinates={joint_q.shape}, "
                f"dofs={self.model.joint_dof_count}"
            )
        self.control.joint_target_q.assign(target_q)
        self.control.joint_target_qd.zero_()
        self.control.joint_f.zero_()
        self._queued_wrenches.fill(0.0)
        self.sim_time = 0.0
        self._sync_native_from_newton()

    def queue_body_wrench(
        self,
        body: str,
        force: Sequence[float],
        torque: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Queue one world-frame force/torque for the next Newton step.

        MPM coupling converts a collected collider impulse to force with
        ``force = impulse / dt`` and queues it here. Multiple calls accumulate.
        """
        if body not in self._body_ids:
            raise KeyError(f"Unknown body {body!r}; available names: {sorted(self._body_ids)}")
        force_array = np.asarray(force, dtype=np.float32)
        torque_array = np.asarray(torque, dtype=np.float32)
        if force_array.shape != (3,) or torque_array.shape != (3,):
            raise ValueError("force and torque must each contain exactly three values")
        index = self._body_ids[body]
        self._queued_wrenches[index, :3] += force_array
        self._queued_wrenches[index, 3:] += torque_array

    def queue_collider_impulse(
        self,
        body: str,
        impulse: Sequence[float],
        position: Sequence[float] | None = None,
    ) -> None:
        """Convert an MPM-style collider impulse into the next-step body wrench."""
        impulse_array = np.asarray(impulse, dtype=np.float32)
        if impulse_array.shape != (3,):
            raise ValueError("impulse must contain exactly three values")
        force = impulse_array / self.dt
        torque = np.zeros(3, dtype=np.float32)
        if position is not None:
            point = np.asarray(position, dtype=np.float32)
            if point.shape != (3,):
                raise ValueError("position must contain exactly three values")
            native_body = mujoco.mj_name2id(
                self.native_model, mujoco.mjtObj.mjOBJ_BODY, body
            )
            if native_body < 0:
                raise KeyError(f"Body {body!r} is missing from the native MuJoCo mirror")
            torque = np.cross(point - self.native_data.xipos[native_body], force)
        self.queue_body_wrench(body, force=force, torque=torque)

    def step(self, steps: int = 1) -> None:
        """Advance Newton/MuJoCo and update the native render mirror."""
        if steps < 1:
            raise ValueError("steps must be at least one")
        for _ in range(steps):
            self.state_0.clear_forces()
            self.state_0.body_f.assign(self._queued_wrenches)
            self._queued_wrenches.fill(0.0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.dt
        self._sync_native_from_newton()

    def _sync_native_from_newton(self) -> None:
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        self.native_data.qpos[:] = self._newton_to_mujoco_q(joint_q)
        self.native_data.qvel[:] = joint_qd
        self.native_data.time = self.sim_time
        mujoco.mj_forward(self.native_model, self.native_data)

    def status(self) -> dict[str, Any]:
        """Return serializable integration and capability metadata."""
        joint_q = self.state_0.joint_q.numpy()
        return {
            "newton_version": str(getattr(newton, "__version__", "unknown")),
            "newton_device": str(self.device),
            "cuda_available": bool(wp.is_cuda_available()),
            "rigid_solver": type(self.solver).__name__,
            "rigid_backend": "MuJoCo CPU" if self.device.is_cpu else "MuJoCo Warp",
            "mpm_solver_available": hasattr(newton.solvers, "SolverImplicitMPM"),
            "mpm_runtime_ready": not self.device.is_cpu,
            "scene": str(self.scene),
            "bodies": int(self.model.body_count),
            "joints": int(self.model.joint_count),
            "dofs": int(self.model.joint_dof_count),
            "collision_shapes": int(self.model.shape_count),
            "sim_time": self.sim_time,
            "base_position": [float(value) for value in joint_q[:3]],
            "native_pose_mirror_max_error": float(
                np.max(np.abs(self.native_data.qpos - self._newton_to_mujoco_q(joint_q)))
            ),
            "coupling_boundary": "collider impulses converted to world-frame body wrenches",
            "snow_material": self.snow.manifest(),
            "snow_mpm_parameters": self.snow.mpm_parameters(),
        }
