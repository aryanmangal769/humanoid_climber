"""Local Newton Implicit MPM snow patch coupled to MuJoCo G1 feet.

Newton owns the deformable material state. The dashboard engine supplies the
two ankle poses as kinematic colliders, forwards any Newton reaction impulses,
and mirrors the deformed MPM surface into the MuJoCo heightfield contact
boundary so the articulated robot follows the actual snow indentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .snow import SnowColumn


@dataclass(frozen=True)
class FootPose:
    name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to an xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / scale
            x = 0.25 * scale
            y = (m[0, 1] + m[1, 0]) / scale
            z = (m[0, 2] + m[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / scale
            x = (m[0, 1] + m[1, 0]) / scale
            y = 0.25 * scale
            z = (m[1, 2] + m[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / scale
            x = (m[0, 2] + m[2, 0]) / scale
            y = (m[1, 2] + m[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


class NewtonSnowPatch:
    """A small terrain-conforming MPM volume around the robot."""

    FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(
        self,
        column: SnowColumn,
        surface_height: Callable[[float, float], float],
        foot_poses: tuple[FootPose, FootPose],
        *,
        center_xy: tuple[float, float] = (0.0, 0.0),
        size_xy: tuple[float, float] = (4.0, 4.0),
        voxel_size: float = 0.08,
        # Newton is coupled at 10 Hz by default while MuJoCo/policy/renderer
        # continue at their higher rates. Implicit MPM remains authoritative
        # for deformation and the held reaction wrench integrates to the
        # Newton impulse over each coupling interval.
        dt: float = 0.10,
        device: str | None = None,
        nominal_foot_load_n: float = 164.0,
        accumulation_enabled: bool = True,
        accumulation_time_scale: float = 1.0,
        contact_refine_radius: float = 0.42,
        coarse_stride: int = 2,
    ) -> None:
        # Newton/Warp are intentionally imported lazily.  The regular MuJoCo
        # dashboard remains usable when the optional RL environment is absent.
        import newton
        import warp as wp
        from newton.solvers import SolverImplicitMPM

        self.newton = newton
        self.wp = wp
        self.SolverImplicitMPM = SolverImplicitMPM
        self.column = column
        self.surface_height = surface_height
        self.center_xy = tuple(float(value) for value in center_xy)
        self.size_xy = tuple(float(value) for value in size_xy)
        self.voxel_size = float(voxel_size)
        self.dt = float(dt)
        self.device = wp.get_device(device)
        self.sequence = 0
        self.sim_time = 0.0
        self.last_impulses: dict[str, dict[str, list[float]]] = {}
        self.history_restored_particles = 0
        self._cached_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cached_particle_positions: np.ndarray | None = None
        self.solver_steps = 0
        self.contact_skipped_steps = 0
        self.active_particle_count = 0
        self._active_particle_mask: np.ndarray | None = None
        self._last_solved_foot_positions: dict[str, np.ndarray] = {}
        self.max_sinkage_m = 0.0
        self.nominal_foot_load_n = float(nominal_foot_load_n)
        self.predicted_static_sinkage_m = self._predict_static_sinkage(self.nominal_foot_load_n)
        # Atmospheric snowfall is a material boundary flux into the top MPM
        # layer. The renderer may accelerate simulation time for demos, but the
        # backend remains the only owner of accumulated mass/depth.
        self.accumulation_enabled = bool(accumulation_enabled)
        self.accumulation_time_scale = max(0.0, float(accumulation_time_scale))
        self.contact_refine_radius = max(self.voxel_size, float(contact_refine_radius))
        self.coarse_stride = max(1, int(coarse_stride))
        self.deposited_depth_m = 0.0
        self.pending_deposition_depth_m = 0.0
        self.deposited_mass_kg = 0.0
        self.deposition_events = 0
        self.deposition_quantum_m = 1.0e-5
        self._base_top_layer_thickness_m = float(self.column.layers[0].thickness_m)
        self.top_particle_indices = np.empty(0, dtype=np.int32)

        self._build_local_frame()
        self._build_models(foot_poses)
        self.reset(foot_poses)

    def _predict_static_sinkage(self, foot_load_n: float) -> float:
        """Estimate indentation from layer stress/strain to drive MPM contact.

        The G1 sole's effective pressure-bearing area is smaller than its full
        bounding box because the Menagerie foot is composed of three capsules.
        The estimate determines the kinematic indentation target only; Newton
        still owns particle plasticity, shear, hardening, and layer response.
        """
        effective_sole_area_m2 = 0.011
        pressure = max(0.0, foot_load_n) / effective_sole_area_m2
        overburden = 0.0
        sinkage = 0.0
        for layer in self.column.layers:
            layer_pressure = pressure + overburden
            elastic_strain = min(0.20, layer_pressure / max(layer.stiffness_pa, 1.0))
            plastic_strain = 0.0
            if layer_pressure > layer.compressive_strength_pa:
                hardening_scale = 1.0 + 0.12 * layer.compaction_hardening
                plastic_strain = min(
                    0.65,
                    (layer_pressure - layer.compressive_strength_pa)
                    / max(layer.compressive_strength_pa * hardening_scale, 1.0),
                )
            sinkage += layer.thickness_m * min(0.85, elastic_strain + plastic_strain)
            overburden += layer.density_kg_m3 * 9.81 * layer.thickness_m
        return min(0.60 * self.column.depth, max(0.0, sinkage))

    def _build_local_frame(self) -> None:
        cx, cy = self.center_xy
        sample = 0.08
        dz_dx = (
            self.surface_height(cx + sample, cy) - self.surface_height(cx - sample, cy)
        ) / (2.0 * sample)
        dz_dy = (
            self.surface_height(cx, cy + sample) - self.surface_height(cx, cy - sample)
        ) / (2.0 * sample)
        normal = np.asarray((-dz_dx, -dz_dy, 1.0), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        axis_x = np.asarray((1.0, 0.0, dz_dx), dtype=np.float64)
        axis_x -= normal * np.dot(axis_x, normal)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(normal, axis_x)
        axis_y /= np.linalg.norm(axis_y)
        self.basis = np.column_stack((axis_x, axis_y, normal)).astype(np.float32)
        self.plane_quaternion = _matrix_to_quaternion_xyzw(self.basis)
        self.origin = np.asarray((cx, cy, self.surface_height(cx, cy)), dtype=np.float32)

    def _world_point(self, local: np.ndarray) -> np.ndarray:
        return self.origin + self.basis @ np.asarray(local, dtype=np.float32)

    def _build_models(self, foot_poses: tuple[FootPose, FootPose]) -> None:
        newton = self.newton
        wp = self.wp
        SolverImplicitMPM = self.SolverImplicitMPM

        collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        collider_builder.default_shape_cfg.mu = self.column.surface_friction
        collider_builder.default_shape_cfg.margin = 0.004

        self.foot_body_ids: dict[str, int] = {}
        for pose in foot_poses:
            quaternion = np.asarray(
                (pose.quaternion_wxyz[1], pose.quaternion_wxyz[2], pose.quaternion_wxyz[3], pose.quaternion_wxyz[0]),
                dtype=np.float32,
            )
            body = collider_builder.add_body(
                xform=wp.transform(wp.vec3(pose.position), wp.quat(quaternion)),
                mass=0.0,
                label=pose.name,
                is_kinematic=True,
            )
            # The three Menagerie capsule geoms form one sole.  A thin box is
            # a stable MPM collider approximation with the same contact area.
            collider_builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(0.035, 0.0, -0.025), wp.quat_identity()),
                hx=0.125,
                hy=0.055,
                hz=0.018,
                label=f"{pose.name}_sole",
            )
            self.foot_body_ids[pose.name] = body

        # Use a terrain-conforming rigid mesh beneath the enlarged MPM window.
        # A single tangent plane creates a floating rectangular slab on real relief.
        ground_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        ground_cfg.mu = 0.7
        ground_cfg.margin = 0.004
        ground_nx = max(8, math.ceil(self.size_xy[0] / max(self.voxel_size * 2.0, 0.12)))
        ground_ny = max(8, math.ceil(self.size_xy[1] / max(self.voxel_size * 2.0, 0.12)))
        ground_vertices: list[list[float]] = []
        ground_indices: list[int] = []
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]
        for iy in range(ground_ny + 1):
            wy = y0 + self.size_xy[1] * iy / ground_ny
            for ix in range(ground_nx + 1):
                wx = x0 + self.size_xy[0] * ix / ground_nx
                wz = self.surface_height(wx, wy) - self.column.depth - 0.01
                ground_vertices.append([wx, wy, wz])
        stride = ground_nx + 1
        for iy in range(ground_ny):
            for ix in range(ground_nx):
                a = iy * stride + ix
                b = a + 1
                c = a + stride
                d = c + 1
                ground_indices.extend((a, c, b, b, c, d))
        ground_mesh = newton.Mesh(
            np.asarray(ground_vertices, dtype=np.float32),
            np.asarray(ground_indices, dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        collider_builder.add_shape_mesh(
            body=-1,
            mesh=ground_mesh,
            cfg=ground_cfg,
            label="snowpack_base_terrain",
        )
        self.collider_model = collider_builder.finalize(device=self.device)
        self.collider_state = self.collider_model.state()
        self._collider_body_q_host = self.collider_state.body_q.numpy()

        snow_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverImplicitMPM.register_custom_attributes(snow_builder)
        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        masses: list[float] = []
        radii: list[float] = []
        surface_offsets: list[float] = []
        material_ids: list[int] = []
        depth_codes: list[int] = []
        rest_cells: list[int] = []
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
        foot_xy = [np.asarray(pose.position[:2], dtype=np.float32) for pose in foot_poses]
        refine_radius_sq = self.contact_refine_radius * self.contact_refine_radius
        horizontal_samples: list[tuple[int, int]] = []
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]
        for ix in range(nx):
            world_x = x0 + (ix + 0.5) * dx
            for iy in range(ny):
                world_y = y0 + (iy + 0.5) * dy
                point = np.asarray((world_x, world_y), dtype=np.float32)
                refined = any(float(np.sum((point - foot) ** 2)) <= refine_radius_sq for foot in foot_xy)
                if not refined and (ix % self.coarse_stride != 0 or iy % self.coarse_stride != 0):
                    continue
                horizontal_samples.append((ix, iy))
        # SolverImplicitMPM has one scalar particle radius; a sparse particle
        # cannot truthfully represent a wide, thin background slab. Keep every
        # simulated particle at the local cell volume instead of inflating
        # coarse samples into overlapping cubic blocks. Unsampled background is
        # the unchanged MuJoCo/renderer snow prior, not fake MPM mass.
        self.simulated_area_m2 = len(horizontal_samples) * dx * dy
        accumulated_depth = 0.0
        for layer_id, layer in enumerate(self.column.layers):
            layer_start = len(positions)
            nz = max(1, math.ceil(layer.thickness_m / self.voxel_size))
            dz = layer.thickness_m / nz
            layer_attributes = layer.newton_attributes()
            for ix, iy in horizontal_samples:
                world_x = x0 + (ix + 0.5) * dx
                world_y = y0 + (iy + 0.5) * dy
                surface_z = self.surface_height(world_x, world_y)
                particle_volume = dx * dy * dz
                particle_mass = layer.density_kg_m3 * particle_volume
                radius = 0.5 * float(np.cbrt(particle_volume))
                for iz in range(nz):
                    world_z = surface_z - accumulated_depth - (iz + 0.5) * dz
                    positions.append([world_x, world_y, world_z])
                    velocities.append([0.0, 0.0, 0.0])
                    masses.append(particle_mass)
                    radii.append(radius)
                    surface_offsets.append(0.5 * dz)
                    material_ids.append(layer_id)
                    depth_codes.append(int(round((iz + 0.5) / nz * 16.0)))
                    rest_cells.append(iy * nx + ix)
                    for name, values in attributes.items():
                        values.append(layer_attributes[name])
            if layer_id == 0:
                self.top_particle_indices = np.arange(layer_start, len(positions), dtype=np.int32)
            accumulated_depth += layer.thickness_m

        snow_builder.add_particles(
            pos=positions,
            vel=velocities,
            mass=masses,
            radius=radii,
            custom_attributes=attributes,
        )
        self.model = snow_builder.finalize(device=self.device)
        self._particle_flags_host = self.model.particle_flags.numpy()
        # Dashboard layers describe an already deposited/pre-consolidated snow
        # column. Applying gravity from a stress-free particle state would make
        # the entire pack settle a second time before the robot touched it.
        # Foot loading is still fully dynamic; self-weight is represented by
        # the supplied density/strength prior rather than replayed deposition.
        self.model.gravity.zero_()
        self.particle_material_ids = np.asarray(material_ids, dtype=np.uint8)
        self.particle_depth_codes = np.asarray(depth_codes, dtype=np.uint8)
        self.particle_rest_cells = np.asarray(rest_cells, dtype=np.int32)
        self.particle_surface_offsets = np.asarray(surface_offsets, dtype=np.float32)
        self.initial_particle_surface_offsets = self.particle_surface_offsets.copy()
        self.initial_particle_positions = np.asarray(positions, dtype=np.float32)
        self.initial_particle_masses = np.asarray(masses, dtype=np.float32)
        self.initial_particle_radii = np.asarray(radii, dtype=np.float32)
        self.total_mass_kg = float(sum(masses))

        # Map every renderer/contact-grid vertex to the nearest represented
        # rest cell. Fine cells around the soles map one-to-one; untouched
        # background maps to its sparse, uniform-volume anchor without
        # changing the anchor's physical mass or radius.
        sample_cells = np.asarray([iy * nx + ix for ix, iy in horizontal_samples], dtype=np.int32)
        sample_xy = np.column_stack((sample_cells % nx, sample_cells // nx)).astype(np.float32)
        grid_cells = np.arange(nx * ny, dtype=np.int32)
        grid_xy = np.column_stack((grid_cells % nx, grid_cells // nx)).astype(np.float32)
        nearest = np.argmin(
            np.sum((grid_xy[:, None, :] - sample_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        self.render_owner_cells = sample_cells[nearest]
        self.layer_top_particles: list[np.ndarray] = []
        for layer_id in range(len(self.column.layers)):
            mapping = np.full(nx * ny, -1, dtype=np.int32)
            selected = np.flatnonzero(self.particle_material_ids == layer_id)
            # Layers were emitted top-to-bottom, so the first particle for a
            # rest cell is the material point at that layer's upper boundary.
            for particle in selected:
                cell = int(self.particle_rest_cells[particle])
                if mapping[cell] < 0:
                    mapping[cell] = int(particle)
            self.layer_top_particles.append(mapping)

        options = SolverImplicitMPM.Config()
        options.voxel_size = self.voxel_size
        options.grid_type = "sparse" if self.device.is_cuda else "dense"
        # APIC preserves local angular/shear motion that PIC smears away. This
        # matters for snow berms and lateral footprint flow and is Newton's
        # current default for its direct two-way examples.
        options.transfer_scheme = "apic"
        # Newton 1.5 names the finite-difference collider mode "backward".
        options.collider_velocity_mode = "backward"
        # The previous 45-iteration solve was materially slower than the
        # stream cadence while providing no visible benefit for this small
        # terrain window. Keep the implicit solver conservative but bounded;
        # Newton remains authoritative for deformation and reaction impulses.
        options.max_iterations = 6 if self.device.is_cuda else 5
        options.tolerance = 1.0e-4
        options.critical_fraction = 0.0
        options.air_drag = 1.0
        self.solver = SolverImplicitMPM(self.model, options)
        self.solver.setup_collider(model=self.collider_model, body_q=self.collider_state.body_q)
        self.collider_body_index = self.solver.collider_body_index.numpy()

    def reset(self, foot_poses: tuple[FootPose, FootPose]) -> None:
        wp = self.wp
        self.model.particle_mass.assign(self.initial_particle_masses)
        self.model.particle_radius.assign(self.initial_particle_radii)
        self.particle_surface_offsets = self.initial_particle_surface_offsets.copy()
        self._refresh_particle_mass_volume()
        self.state = self.model.state()
        self.state.body_q = wp.empty_like(self.collider_state.body_q)
        self.state.body_qd = wp.zeros_like(self.collider_state.body_qd)
        self.state.body_f = wp.zeros_like(self.collider_state.body_f)
        self.sequence = 0
        self.sim_time = 0.0
        self.max_sinkage_m = 0.0
        self.deposited_depth_m = 0.0
        self.pending_deposition_depth_m = 0.0
        self.deposited_mass_kg = 0.0
        self.deposition_events = 0
        self.total_mass_kg = float(self.initial_particle_masses.sum())
        self.last_impulses = {}
        self._cached_surface = None
        self._cached_particle_positions = None
        self.solver_steps = 0
        self.contact_skipped_steps = 0
        self.active_particle_count = 0
        self._active_particle_mask = None
        self._last_solved_foot_positions.clear()
        self.history_restored_particles = 0
        self.initial_foot_positions = {
            pose.name: np.asarray(pose.position, dtype=np.float32).copy()
            for pose in foot_poses
        }
        self._assign_foot_poses(foot_poses)

    def export_deformation_history(self, history: dict, cell_size: float = 0.10) -> None:
        """Persist changed Newton particle state in world-space trail cells."""
        positions = self.state.particle_q.numpy()
        self._cached_particle_positions = positions
        velocities = self.state.particle_qd.numpy()
        plastic = self.state.mpm.particle_Jp.numpy()
        stresses = self.state.mpm.particle_stress.numpy()
        for index, (initial, current) in enumerate(zip(self.initial_particle_positions, positions)):
            displacement = current - initial
            jp = float(plastic[index])
            if (
                float(np.linalg.norm(displacement)) < 1.0e-5
                and float(np.linalg.norm(velocities[index])) < 1.0e-4
                and abs(jp - 1.0) < 1.0e-5
                and float(np.linalg.norm(stresses[index])) < 1.0e-3
            ):
                continue
            key = (
                int(round(float(initial[0]) / cell_size)),
                int(round(float(initial[1]) / cell_size)),
                int(self.particle_material_ids[index]),
                int(self.particle_depth_codes[index]),
            )
            history[key] = (
                float(initial[0]),
                float(initial[1]),
                displacement.astype(np.float32),
                np.asarray(velocities[index], dtype=np.float32),
                jp,
                np.asarray(stresses[index], dtype=np.float32),
            )

    def import_deformation_history(self, history: dict, cell_size: float = 0.10) -> int:
        """Seed a recentered patch from the nearest persisted world-space state."""
        if not history:
            return 0
        positions = self._cached_particle_positions
        if positions is None:
            positions = self.state.particle_q.numpy()
            self._cached_particle_positions = positions
        velocities = self.state.particle_qd.numpy()
        plastic = self.state.mpm.particle_Jp.numpy()
        stresses = self.state.mpm.particle_stress.numpy()
        restored = 0
        search = max(1, int(math.ceil(self.voxel_size / cell_size)))
        max_distance_sq = (self.voxel_size * 0.80) ** 2
        for index, initial in enumerate(self.initial_particle_positions):
            qx = int(round(float(initial[0]) / cell_size))
            qy = int(round(float(initial[1]) / cell_size))
            layer = int(self.particle_material_ids[index])
            depth = int(self.particle_depth_codes[index])
            best = None
            best_distance = float("inf")
            for ox in range(-search, search + 1):
                for oy in range(-search, search + 1):
                    record = history.get((qx + ox, qy + oy, layer, depth))
                    if record is None:
                        continue
                    distance = (float(initial[0]) - record[0]) ** 2 + (float(initial[1]) - record[1]) ** 2
                    if distance < best_distance:
                        best = record
                        best_distance = distance
            if best is None or best_distance > max_distance_sq:
                continue
            displacement = np.asarray(best[2], dtype=np.float32).copy()
            displacement[:2] = np.clip(displacement[:2], -self.voxel_size, self.voxel_size)
            displacement[2] = np.clip(displacement[2], -self.column.depth, self.voxel_size)
            positions[index] = initial + displacement
            # Re-entering a previously simulated trail must not resurrect stale
            # momentum, but its plastic volume/compaction remains persistent.
            velocities[index] = 0.0
            plastic[index] = float(best[4])
            # Newton 1.5's constitutive state is stress + plastic volume. Keep
            # the stress history across moving-window rebuilds. Velocity and
            # APIC velocity gradients intentionally restart at zero so a trail
            # does not resurrect stale momentum when the robot returns.
            if len(best) > 5:
                stresses[index] = np.asarray(best[5], dtype=np.float32)
            restored += 1
        if restored:
            self.state.particle_q.assign(positions)
            self.state.particle_qd.assign(velocities)
            self.state.mpm.particle_Jp.assign(plastic)
            self.state.mpm.particle_stress.assign(stresses)
            self._cached_surface = self._surface_arrays()
            self._update_sinkage(self._cached_surface[0])
        return restored

    def advance_without_contact(self) -> None:
        """Advance atmospheric accumulation without running the MPM solver."""
        deposition_events = self.deposition_events
        self._apply_snowfall_deposition()
        self.sim_time += self.dt
        self.contact_skipped_steps += 1
        if self.deposition_events != deposition_events:
            self.sequence += 1
            self._cached_surface = None

    def needs_contact_solve(self, foot_poses: tuple[FootPose, FootPose], threshold_m: float = 0.010) -> bool:
        """Solve only when a kinematic contact boundary actually moved."""
        threshold_sq = threshold_m * threshold_m
        for pose in foot_poses:
            current = np.asarray(pose.position, dtype=np.float32)
            previous = self._last_solved_foot_positions.get(pose.name)
            if previous is None or float(np.sum((current - previous) ** 2)) >= threshold_sq:
                return True
        return False

    def _refresh_particle_mass_volume(self) -> None:
        """Refresh in-place MPM mass/volume caches after fresh-snow deposition.

        Keeping the Warp arrays allocated in place is important because the
        implicit solver may capture CUDA graphs that reference these buffers.
        """
        # Newton 1.5 provides a public invalidation hook that refreshes mass,
        # radius, volume, density, material extrema, and cached particle flags.
        self.solver.notify_model_changed(self.newton.ModelFlags.MODEL_PROPERTIES)

    def _apply_snowfall_deposition(self) -> None:
        """Apply snowfall as a mass-conserving surface flux into the MPM pack.

        Atmospheric flakes are too dilute for a continuum MPM treatment while
        airborne. Once they reach the surface, however, snowfall is a material
        boundary flux. The selected snow-depth rate therefore grows the fresh
        top layer, adds exactly ``rho * area * depth`` mass, and keeps its
        density constant. This makes accumulation affect Newton contact rather
        than merely changing browser particles.
        """
        if not self.accumulation_enabled:
            return
        rate_m_s = self.column.snowfall_depth_rate_m_s * self.accumulation_time_scale
        if rate_m_s <= 0.0 or not len(self.top_particle_indices):
            return
        self.pending_deposition_depth_m += rate_m_s * self.dt
        quanta = math.floor(self.pending_deposition_depth_m / self.deposition_quantum_m)
        if quanta <= 0:
            return
        depth_increment = quanta * self.deposition_quantum_m
        self.pending_deposition_depth_m -= depth_increment

        old_span = self._base_top_layer_thickness_m + self.deposited_depth_m
        new_span = old_span + depth_increment
        growth = new_span / max(old_span, 1.0e-9)
        top = self.top_particle_indices

        positions = self.state.particle_q.numpy()
        local = (positions[top] - self.origin) @ self.basis
        fixed_bottom = -self._base_top_layer_thickness_m
        local[:, 2] = fixed_bottom + (local[:, 2] - fixed_bottom) * growth
        positions[top] = self.origin + local @ self.basis.T
        self.state.particle_q.assign(positions)
        self._cached_particle_positions = positions
        self._cached_surface = None

        masses = self.model.particle_mass.numpy()
        masses[top] *= growth
        self.model.particle_mass.assign(masses)
        radii = self.model.particle_radius.numpy()
        radii[top] *= float(np.cbrt(growth))
        self.model.particle_radius.assign(radii)
        self.particle_surface_offsets[top] *= growth
        self._refresh_particle_mass_volume()

        added_mass = self.simulated_area_m2 * depth_increment * self.column.layers[0].density_kg_m3
        self.deposited_depth_m += depth_increment
        self.deposited_mass_kg += added_mass
        self.total_mass_kg += added_mass
        self.deposition_events += 1

    def _assign_foot_poses(self, foot_poses: tuple[FootPose, FootPose]) -> None:
        body_q = self._collider_body_q_host
        for pose in foot_poses:
            body = self.foot_body_ids[pose.name]
            actual = np.asarray(pose.position, dtype=np.float32)
            # Use the actual MuJoCo ankle pose. The previous analytic assist
            # pushed a virtual sole into the pack before contact and therefore
            # authored deformation without a physical interaction.
            body_q[body, :3] = actual
            body_q[body, 3:] = (
                pose.quaternion_wxyz[1],
                pose.quaternion_wxyz[2],
                pose.quaternion_wxyz[3],
                pose.quaternion_wxyz[0],
            )
        self.collider_state.body_q.assign(body_q)
        self.state.body_q.assign(body_q)
        self.state.body_qd.zero_()

    def step(self, foot_poses: tuple[FootPose, FootPose]) -> dict[str, dict[str, list[float]]]:
        """Advance deformable snow and return reaction wrenches for MuJoCo."""
        self._apply_snowfall_deposition()
        self._cached_surface = None
        self._cached_particle_positions = None
        self._assign_foot_poses(foot_poses)
        # Newton's transfer/rheology kernels honor per-particle ACTIVE flags.
        # Keep coarse background particles for trail continuity, but solve only
        # the fine cells in the current sole neighborhoods.
        particle_positions = self.state.particle_q.numpy()
        # Keep rollback state on-device. Copying five full MPM arrays through
        # NumPy every contact tick serialized the CUDA stream and made robot
        # actions visibly lag. Only a rejected no-contact solve reads the
        # restored positions back for the renderer cache.
        pre_positions = self.wp.clone(self.state.particle_q)
        pre_velocities = self.wp.clone(self.state.particle_qd)
        pre_velocity_gradients = self.wp.clone(self.state.mpm.particle_qd_grad)
        pre_plastic = self.wp.clone(self.state.mpm.particle_Jp)
        pre_stress = self.wp.clone(self.state.mpm.particle_stress)
        active = np.zeros(len(particle_positions), dtype=bool)
        radius_sq = self.contact_refine_radius * self.contact_refine_radius
        for pose in foot_poses:
            delta = particle_positions[:, :2] - np.asarray(pose.position[:2], dtype=np.float32)
            active |= np.sum(delta * delta, axis=1) <= radius_sq
        particle_flags = self._particle_flags_host.copy()
        active_bit = int(self.newton.ParticleFlags.ACTIVE)
        particle_flags[:] = np.where(
            active,
            particle_flags | active_bit,
            particle_flags & ~active_bit,
        )
        if self._active_particle_mask is None or not np.array_equal(active, self._active_particle_mask):
            self.model.particle_flags.assign(particle_flags)
            # SolverImplicitMPM caches transfer/material masks. Rebind them
            # only when the neighborhood changed; rebuilding identical masks
            # every contact tick is an avoidable GPU synchronization.
            self.solver.notify_model_changed(self.newton.ModelFlags.MODEL_PROPERTIES)
            self._active_particle_mask = active.copy()
        # Inactive particles are excluded by Newton's transfer, rheology, and
        # collision kernels, so untouched coarse background state remains
        # frozen on the GPU without a CPU round-trip/restore pass.
        self.solver.step(self.state, self.state, contacts=None, control=None, dt=self.dt)
        # The implicit solve supplies constitutive/contact impulses; projection
        # is the solver's documented final safety pass for soft compliant
        # particles, preventing a low-modulus snow layer from tunnelling
        # through the basal plane or a moving sole between MPM frames.
        self.solver.project_outside(self.state, self.state, self.dt)
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse_np = impulses.numpy()
        position_np = positions.numpy()
        collider_np = collider_ids.numpy()

        self._cached_particle_positions = self.state.particle_q.numpy()

        reactions: dict[str, dict[str, list[float]]] = {}
        for name, body in self.foot_body_ids.items():
            selected = np.flatnonzero(
                (collider_np >= 0)
                & (collider_np < len(self.collider_body_index))
                & (self.collider_body_index[np.clip(collider_np, 0, len(self.collider_body_index) - 1)] == body)
            )
            if not len(selected):
                continue
            total_impulse = impulse_np[selected].sum(axis=0)
            magnitude = np.linalg.norm(impulse_np[selected], axis=1)
            if float(magnitude.sum()) <= 1.0e-8:
                continue
            if float(magnitude.sum()) > 1.0e-9:
                contact_position = np.average(position_np[selected], axis=0, weights=magnitude)
            else:
                contact_position = position_np[selected].mean(axis=0)
            force = total_impulse / self.dt
            # A malformed contact spike must not destabilize the policy model.
            force_norm = float(np.linalg.norm(force))
            if force_norm > 1200.0:
                force *= 1200.0 / force_norm
            reactions[name] = {
                "force": force.astype(float).tolist(),
                "position": contact_position.astype(float).tolist(),
                "impulse": total_impulse.astype(float).tolist(),
            }
        if not reactions:
            # A moving collider outside the snow can still make the implicit
            # solve drift a soft stress-free pack. No measured foot impulse
            # means no physical interaction, so restore the complete MPM state
            # (including APIC gradient, plastic volume, and stress), not only
            # the visible positions.
            self.wp.copy(self.state.particle_q, pre_positions)
            self.wp.copy(self.state.particle_qd, pre_velocities)
            self.wp.copy(self.state.mpm.particle_qd_grad, pre_velocity_gradients)
            self.wp.copy(self.state.mpm.particle_Jp, pre_plastic)
            self.wp.copy(self.state.mpm.particle_stress, pre_stress)
            self._cached_particle_positions = pre_positions.numpy()
        self.last_impulses = reactions
        self._last_solved_foot_positions = {
            pose.name: np.asarray(pose.position, dtype=np.float32).copy()
            for pose in foot_poses
        }
        self.active_particle_count = int(active.sum())
        self.solver_steps += 1
        self.sequence += 1
        self.sim_time += self.dt
        self._cached_surface = self._surface_arrays()
        self._update_sinkage(self._cached_surface[0])
        return reactions

    def surface_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_surface is None:
            self._cached_surface = self._surface_arrays()
        return self._cached_surface

    def _surface_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny = self.surface_resolution
        vertices = self._layer_vertex_arrays()[0]
        heights = vertices[:, 2].copy()
        initial = self._initial_surface_grid()
        # The top surface cannot pass below the terrain-conforming substrate.
        # This is a physical bound, not a renderer clamp: the bounded values
        # feed both Unity and MuJoCo's contact heightfield.
        heights = np.maximum(heights, initial - self.column.depth)
        owner_particles = self.layer_top_particles[0][self.render_owner_cells]
        valid = owner_particles >= 0
        material_ids = np.zeros(nx * ny, dtype=np.uint8)
        material_ids[valid] = self.particle_material_ids[owner_particles[valid]]
        compaction = np.zeros(nx * ny, dtype=np.float32)
        particle_jp = self.state.mpm.particle_Jp.numpy()
        compaction[valid] = np.clip(1.0 - particle_jp[owner_particles[valid]], 0.0, 1.0)
        return heights, material_ids, compaction

    def _layer_vertex_arrays(self) -> list[np.ndarray]:
        """Return Lagrangian XYZ vertices for every mechanical layer boundary.

        The old stream discarded horizontal particle displacement by binning
        the current state back into fixed XY height cells. These vertices keep
        a stable rest-grid topology while applying the actual displacement of
        each Newton material point, including lateral shear and heave.
        """
        nx, ny = self.surface_resolution
        current = self._cached_particle_positions
        if current is None:
            current = self.state.particle_q.numpy()
            self._cached_particle_positions = current
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]
        dx = self.size_xy[0] / nx
        dy = self.size_xy[1] / ny
        cells = np.arange(nx * ny, dtype=np.int32)
        rest_xy = np.column_stack((
            x0 + (cells % nx + 0.5) * dx,
            y0 + (cells // nx + 0.5) * dy,
        )).astype(np.float32)
        pristine = self._initial_surface_grid(include_deposition=False)

        results: list[np.ndarray] = []
        depth_above = 0.0
        for layer_id, layer in enumerate(self.column.layers):
            vertices = np.column_stack((rest_xy, pristine - depth_above)).astype(np.float32)
            owner_particles = self.layer_top_particles[layer_id][self.render_owner_cells]
            valid = owner_particles >= 0
            particle_ids = owner_particles[valid]
            rest_top = self.initial_particle_positions[particle_ids].copy()
            rest_top[:, 2] += self.initial_particle_surface_offsets[particle_ids]
            current_top = current[particle_ids].copy()
            current_top[:, 2] += self.particle_surface_offsets[particle_ids]
            vertices[valid] += current_top - rest_top
            results.append(vertices)
            depth_above += layer.thickness_m

        # A multilayer MPM particle cloud can mix and overturn locally, while
        # the renderer/contact protocol is intentionally single-valued. Raw
        # per-layer top particles can therefore cross and create negative
        # visual thickness. Project ordered boundaries from the actual top and
        # substrate using each layer's Newton plastic volume ratio (Jp). This
        # preserves authoritative compaction while making the limitation of a
        # heightfield-compatible volume explicit and non-inverting.
        substrate_z = pristine - self.column.depth
        results[0][:, 2] = np.maximum(results[0][:, 2], substrate_z + 0.002)
        particle_jp = self.state.mpm.particle_Jp.numpy()
        thickness_weights = np.empty((len(self.column.layers), nx * ny), dtype=np.float32)
        for layer_id, layer in enumerate(self.column.layers):
            owner_particles = self.layer_top_particles[layer_id][self.render_owner_cells]
            jp = np.ones(nx * ny, dtype=np.float32)
            valid = owner_particles >= 0
            jp[valid] = np.clip(particle_jp[owner_particles[valid]], 0.02, 2.0)
            thickness_weights[layer_id] = layer.thickness_m * jp
        total_weight = np.maximum(thickness_weights.sum(axis=0), 1.0e-8)
        available_depth = np.maximum(results[0][:, 2] - substrate_z, 0.002)
        cumulative = np.zeros(nx * ny, dtype=np.float32)
        for layer_id in range(1, len(self.column.layers)):
            cumulative += thickness_weights[layer_id - 1]
            results[layer_id][:, 2] = (
                results[0][:, 2] - available_depth * cumulative / total_weight
            )
        return results

    def _layer_surface_arrays(self) -> list[np.ndarray]:
        """Return the live upper boundary of every mechanical layer.

        These grids are renderer/diagnostic views of the same Newton particle
        state, not independently simulated surfaces. Missing cells fall back
        to the undeformed terrain-conforming layer boundary.
        """
        return [vertices[:, 2].copy() for vertices in self._layer_vertex_arrays()]

    def _initial_surface_grid(self, *, include_deposition: bool = True) -> np.ndarray:
        nx, ny = self.surface_resolution
        values = np.empty(nx * ny, dtype=np.float32)
        deposition = self.deposited_depth_m if include_deposition else 0.0
        x0 = self.center_xy[0] - 0.5 * self.size_xy[0]
        y0 = self.center_xy[1] - 0.5 * self.size_xy[1]
        for iy in range(ny):
            world_y = y0 + (iy + 0.5) * self.size_xy[1] / ny
            for ix in range(nx):
                world_x = x0 + (ix + 0.5) * self.size_xy[0] / nx
                values[iy * nx + ix] = self.surface_height(world_x, world_y) + deposition
        return values

    def _update_sinkage(self, heights: np.ndarray | None = None) -> None:
        if heights is None:
            heights = self.surface_arrays()[0]
        sinkage = np.maximum(0.0, self._initial_surface_grid() - heights)
        self.max_sinkage_m = max(self.max_sinkage_m, float(sinkage.max(initial=0.0)))

    def terrain_frame(self, *, include_particles: bool = True) -> dict[str, Any]:
        heights, material_ids, compaction = self.surface_arrays()
        layer_vertices = self._layer_vertex_arrays()
        layer_heights = [vertices[:, 2] for vertices in layer_vertices]
        nx, ny = self.surface_resolution
        particles = self._cached_particle_positions
        if include_particles and particles is None:
            particles = self.state.particle_q.numpy()
        layers = [layer.manifest(index) for index, layer in enumerate(self.column.layers)]
        if layers:
            layers[0]["depth"] = float(layers[0]["depth"]) + self.deposited_depth_m
            layers[0]["thickness_m"] = float(layers[0]["thickness_m"]) + self.deposited_depth_m
        return {
            "schema": "everest-terrain/v1",
            "mode": "live",
            "sequence": self.sequence,
            "timestamp": self.sim_time,
            "sim_time": self.sim_time,
            "origin": [
                self.center_xy[0] - 0.5 * self.size_xy[0],
                self.center_xy[1] - 0.5 * self.size_xy[1],
                float(self.origin[2]),
            ],
            "size": list(self.size_xy),
            "resolution": [nx, ny],
            "heights": heights.tolist(),
            "vertices": layer_vertices[0].tolist(),
            "base_heights": self._initial_surface_grid().tolist(),
            "layer_heights": [values.tolist() for values in layer_heights],
            "layer_vertices": [values.tolist() for values in layer_vertices],
            "layer_boundary_model": "top_surface_plus_newton_Jp_volume_projection",
            "substrate_vertices": np.column_stack((
                layer_vertices[-1][:, :2],
                self._initial_surface_grid(include_deposition=False) - self.column.depth,
            )).astype(np.float32).tolist(),
            "material_ids": material_ids.tolist(),
            "compaction": compaction.tolist(),
            "surface_kind": "snow",
            "surface_depth": self.column.depth + self.deposited_depth_m,
            "surface_friction": self.column.surface_friction,
            "layers": layers,
            **({
                "particles": {
                    "positions": particles.tolist(),
                    "radii": self.model.particle_radius.numpy().tolist(),
                    "material_ids": self.particle_material_ids.tolist(),
                },
            } if include_particles else {}),
            "mpm": self.status(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "active": True,
            "solver": type(self.solver).__name__,
            "device": str(self.device),
            "cuda": bool(self.device.is_cuda),
            "particle_count": int(self.model.particle_count),
            "particle_mass_kg": self.total_mass_kg,
            "simulated_area_m2": self.simulated_area_m2,
            "voxel_size_m": self.voxel_size,
            "window_size_m": list(self.size_xy),
            "terrain_conforming": True,
            "history_restored_particles": self.history_restored_particles,
            "contact_refine_radius_m": self.contact_refine_radius,
            "coarse_stride": self.coarse_stride,
            "step_dt": self.dt,
            "steps": self.solver_steps + self.contact_skipped_steps,
            "surface_updates": self.sequence,
            "solver_steps": self.solver_steps,
            "contact_skipped_steps": self.contact_skipped_steps,
            "active_particle_count": self.active_particle_count,
            "max_sinkage_m": self.max_sinkage_m,
            "predicted_static_sinkage_m": self.predicted_static_sinkage_m,
            "snowfall_depth_rate_m_s": self.column.snowfall_depth_rate_m_s,
            "snowfall_mass_flux_kg_m2_s": self.column.snowfall_mass_flux_kg_m2_s,
            "accumulation_enabled": self.accumulation_enabled,
            "accumulation_time_scale": self.accumulation_time_scale,
            "deposited_depth_m": self.deposited_depth_m,
            "pending_deposition_depth_m": self.pending_deposition_depth_m,
            "deposited_mass_kg": self.deposited_mass_kg,
            "deposition_events": self.deposition_events,
            "coupling": "Newton kinematic-foot deformation -> MuJoCo deformed heightfield support; reaction impulses diagnostic only to avoid double-counting",
            "robot_support_owner": "mujoco_deformed_heightfield",
            "reaction_impulses_applied": False,
            "solve_delivery": "asynchronous_latest_pose",
        }
