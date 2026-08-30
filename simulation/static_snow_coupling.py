"""Experimental direct two-way coupling for a static multilayer snow snapshot.

This is the migration target for the dashboard's current compatibility path.
Unlike ``NewtonSnowPatch``, it does not create kinematic proxy feet, predict a
sinkage target, or mirror snow deformation into a rigid support heightfield.
The real Newton-imported G1 collision geometry is registered directly as the
Implicit MPM collider and collected impulses are queued back into the rigid
Newton/MuJoCo bridge.

The first version intentionally uses a flat rigid substrate. Everest terrain is
wired in after this coupling contract is verified in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import newton
import warp as wp
from newton.solvers import SolverImplicitMPM

from .newton_mujoco import NewtonMuJoCoBridge
from .snow import SnowColumn


@dataclass(frozen=True)
class CouplingStats:
    sim_time: float
    particle_count: int
    max_surface_drop_m: float
    coupled_bodies: int
    total_impulse_ns: float


class StaticSnowTwoWayCoupler:
    """Direct G1 <-> MPM coupling for a fixed initial snow column."""

    def __init__(
        self,
        bridge: NewtonMuJoCoBridge,
        column: SnowColumn,
        *,
        center_xy: tuple[float, float] = (0.0, 0.0),
        size_xy: tuple[float, float] = (1.8, 1.2),
        base_z: float = 0.0,
        voxel_size: float = 0.05,
        dt: float = 0.01,
    ) -> None:
        self.bridge = bridge
        self.column = column
        self.center_xy = tuple(float(value) for value in center_xy)
        self.size_xy = tuple(float(value) for value in size_xy)
        self.base_z = float(base_z)
        self.voxel_size = float(voxel_size)
        self.dt = float(dt)
        self.device = bridge.device
        self.sim_time = 0.0
        self.last_total_impulse_ns = 0.0
        self.last_coupled_bodies = 0

        self._build_snow_model()
        self._build_solver()
        self._initial_surface = self._surface_heights()

    def _build_snow_model(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverImplicitMPM.register_custom_attributes(builder)

        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        masses: list[float] = []
        radii: list[float] = []
        material_ids: list[int] = []
        attributes: dict[str, list[float]] = {
            key: []
            for key in (
                "mpm:young_modulus",
                "mpm:poisson_ratio",
                "mpm:damping",
                "mpm:hardening",
                "mpm:friction",
                "mpm:yield_pressure",
                "mpm:tensile_yield_ratio",
                "mpm:yield_stress",
                "mpm:dilatancy",
            )
        }

        nx = max(4, math.ceil(self.size_xy[0] / self.voxel_size))
        ny = max(4, math.ceil(self.size_xy[1] / self.voxel_size))
        dx = self.size_xy[0] / nx
        dy = self.size_xy[1] / ny
        self.surface_resolution = (nx, ny)
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]

        accumulated = 0.0
        for layer_id, layer in reversed(list(enumerate(self.column.layers))):
            nz = max(1, math.ceil(layer.thickness_m / self.voxel_size))
            dz = layer.thickness_m / nz
            layer_bottom = self.base_z + accumulated
            particle_mass = (
                layer.density_kg_m3
                * self.size_xy[0]
                * self.size_xy[1]
                * layer.thickness_m
                / (nx * ny * nz)
            )
            radius = 0.5 * float(np.cbrt(dx * dy * dz))
            material = layer.newton_attributes()
            for ix in range(nx):
                x = x0 + (ix + 0.5) * dx
                for iy in range(ny):
                    y = y0 + (iy + 0.5) * dy
                    for iz in range(nz):
                        z = layer_bottom + (iz + 0.5) * dz
                        positions.append([x, y, z])
                        velocities.append([0.0, 0.0, 0.0])
                        masses.append(particle_mass)
                        radii.append(radius)
                        material_ids.append(layer_id)
                        for name, values in attributes.items():
                            values.append(material[name])
            accumulated += layer.thickness_m

        builder.add_particles(
            pos=positions,
            vel=velocities,
            mass=masses,
            radius=radii,
            custom_attributes=attributes,
        )
        self.model = builder.finalize(device=self.device)
        self.material_ids = np.asarray(material_ids, dtype=np.uint8)
        self.initial_positions = np.asarray(positions, dtype=np.float32)
        self.total_mass_kg = float(sum(masses))

    def _build_solver(self) -> None:
        config = SolverImplicitMPM.Config()
        config.voxel_size = self.voxel_size
        config.grid_type = "sparse" if self.device.is_cuda else "dense"
        # Keep Newton's APIC default. It preserves angular momentum and is much
        # less dissipative than the compatibility path's forced PIC transfer.
        config.tolerance = 1.0e-6
        config.max_iterations = 50 if self.device.is_cuda else 24
        config.critical_fraction = 0.0
        config.strain_basis = "P0"
        self.solver = SolverImplicitMPM(self.model, config)
        self.solver.setup_collider(model=self.bridge.model)
        self.collider_body_index = self.solver.collider_body_index.numpy()

        self.state = self.model.state()
        self.state.body_q = wp.empty_like(self.bridge.state_0.body_q)
        self.state.body_qd = wp.empty_like(self.bridge.state_0.body_qd)
        self.state.body_f = wp.zeros_like(self.bridge.state_0.body_f)
        self._sync_colliders()

    def lift_robot_to_surface(self, clearance_m: float = 0.015) -> None:
        """Place the imported G1 above the static snow surface for sandbox tests."""
        joint_q = self.bridge.state_0.joint_q.numpy()
        joint_q[2] += self.column.depth + float(clearance_m)
        self.bridge.state_0.joint_q.assign(joint_q)
        self.bridge.state_1.joint_q.assign(joint_q)
        newton.eval_fk(
            self.bridge.model,
            self.bridge.state_0.joint_q,
            self.bridge.state_0.joint_qd,
            self.bridge.state_0,
        )
        newton.eval_fk(
            self.bridge.model,
            self.bridge.state_1.joint_q,
            self.bridge.state_1.joint_qd,
            self.bridge.state_1,
        )
        self.bridge._sync_native_from_newton()
        self._sync_colliders()

    def _sync_colliders(self) -> None:
        self.state.body_q.assign(self.bridge.state_0.body_q)
        self.state.body_qd.assign(self.bridge.state_0.body_qd)
        self.state.body_f.zero_()

    def _queue_reaction_impulses(self) -> tuple[int, float]:
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse_np = impulses.numpy()
        position_np = positions.numpy()
        collider_np = collider_ids.numpy()
        body_impulses: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for index, collider_id in enumerate(collider_np):
            if collider_id < 0 or collider_id >= len(self.collider_body_index):
                continue
            body_id = int(self.collider_body_index[collider_id])
            if body_id < 0:
                continue
            body_impulses.setdefault(body_id, []).append((impulse_np[index], position_np[index]))

        coupled = 0
        total_norm = 0.0
        labels = self.bridge.model.body_label
        for body_id, samples in body_impulses.items():
            impulse = np.sum([sample[0] for sample in samples], axis=0)
            magnitudes = np.asarray([np.linalg.norm(sample[0]) for sample in samples])
            if float(magnitudes.sum()) > 1.0e-12:
                point = np.average(np.asarray([sample[1] for sample in samples]), axis=0, weights=magnitudes)
            else:
                point = np.mean(np.asarray([sample[1] for sample in samples]), axis=0)
            label = str(labels[body_id]).rsplit("/", 1)[-1]
            if label not in self.bridge._body_ids:
                continue
            self.bridge.queue_collider_impulse(label, impulse, point)
            coupled += 1
            total_norm += float(np.linalg.norm(impulse))
        return coupled, total_norm

    def step(self, rigid_steps: int = 1) -> CouplingStats:
        """Advance G1 and the static pack with direct collider impulse exchange."""
        if rigid_steps < 1:
            raise ValueError("rigid_steps must be at least one")
        self.bridge.step(rigid_steps)
        self._sync_colliders()
        self.solver.step(self.state, self.state, contacts=None, control=None, dt=self.dt)
        self.solver.project_outside(self.state, self.state, self.dt)
        coupled, total_impulse = self._queue_reaction_impulses()
        self.sim_time += self.dt
        self.last_coupled_bodies = coupled
        self.last_total_impulse_ns = total_impulse
        current = self._surface_heights()
        max_drop = float(np.max(np.maximum(0.0, self._initial_surface - current), initial=0.0))
        return CouplingStats(
            sim_time=self.sim_time,
            particle_count=int(self.model.particle_count),
            max_surface_drop_m=max_drop,
            coupled_bodies=coupled,
            total_impulse_ns=total_impulse,
        )

    def _surface_heights(self) -> np.ndarray:
        nx, ny = self.surface_resolution
        positions = self.state.particle_q.numpy()
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]
        ix = np.clip(((positions[:, 0] - x0) / self.size_xy[0] * nx).astype(int), 0, nx - 1)
        iy = np.clip(((positions[:, 1] - y0) / self.size_xy[1] * ny).astype(int), 0, ny - 1)
        flat = iy * nx + ix
        heights = np.full(nx * ny, self.base_z, dtype=np.float32)
        np.maximum.at(heights, flat, positions[:, 2])
        return heights

    def status(self) -> dict[str, Any]:
        return {
            "mode": "experimental_direct_static_two_way",
            "device": str(self.device),
            "particle_count": int(self.model.particle_count),
            "particle_mass_kg": self.total_mass_kg,
            "voxel_size_m": self.voxel_size,
            "step_dt": self.dt,
            "transfer_scheme": "apic",
            "proxy_feet": False,
            "predicted_sinkage": False,
            "rigid_snow_heightfield": False,
            "substrate": "flat compatibility floor (Everest terrain next)",
            "coupling": "real G1 colliders -> Newton MPM -> collected impulses -> SolverMuJoCo",
        }
