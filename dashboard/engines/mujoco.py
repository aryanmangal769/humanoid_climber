"""MuJoCo telemetry adapter for Unitree RL MjLab's canonical G1 scene."""

from __future__ import annotations

import copy
import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any
import xml.etree.ElementTree as ET

# The backend is headless; native MuJoCo subset previews need an offscreen GL
# platform selected before mujoco is imported. Operators may still override it.
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

from ..g1_model import G1_XML
from ..policy import DEFAULT_CHECKPOINT, G1VelocityPolicy
from simulation.newton_snow import FootPose, NewtonSnowPatch
from simulation.policy_supervisor import PolicySupervisor, predict_imbalance
from simulation.snow import SURFACES, SnowLayer


ROOT = Path(__file__).resolve().parents[2]
G1_SCENE = G1_XML
G1_ASSETS = G1_SCENE.parent / "assets"
PHYSICS_SCENE = (
    ROOT / "vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/scene_mjx.xml"
)
PHYSICS_SOURCE = "google-deepmind/mujoco_menagerie/unitree_g1"
PHYSICS_SOURCE_URL = "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1"
PHYSICS_SOURCE_REVISION = "1b86ece576591213e2b666ebf59508454200ca97"
RENDER_SOURCE = "unitreerobotics/unitree_rl_mjlab"
RENDER_SOURCE_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
UNITY_TERRAIN_MANIFEST = ROOT / "maps/everest_local_terrain.json"
TERRAIN_MANIFEST = (
    UNITY_TERRAIN_MANIFEST if UNITY_TERRAIN_MANIFEST.is_file()
    else ROOT / "maps/everest_terrain.json"
)


def _resample_crop(
    values: np.ndarray,
    *,
    source_rows: int,
    source_columns: int,
    output_rows: int,
    output_columns: int,
    crop: tuple[float, float, float, float],
) -> np.ndarray:
    """Bilinearly crop/resample a south-to-north height grid."""
    u0, u1, v0, v1 = crop
    source = np.asarray(values, dtype=np.float64).reshape(source_rows, source_columns)
    xs = np.linspace(u0 * (source_columns - 1), u1 * (source_columns - 1), output_columns)
    ys = np.linspace(v0 * (source_rows - 1), v1 * (source_rows - 1), output_rows)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(source_columns - 1, x0 + 1)
    y1 = np.minimum(source_rows - 1, y0 + 1)
    tx = xs - x0
    ty = ys - y0
    a = source[np.ix_(y0, x0)]
    b = source[np.ix_(y0, x1)]
    c = source[np.ix_(y1, x0)]
    d = source[np.ix_(y1, x1)]
    return (
        a * (1.0 - ty[:, None]) * (1.0 - tx[None, :])
        + b * (1.0 - ty[:, None]) * tx[None, :]
        + c * ty[:, None] * (1.0 - tx[None, :])
        + d * ty[:, None] * tx[None, :]
    )


def _load_everest_model() -> tuple[mujoco.MjModel, dict[str, Any]]:
    """Compile the G1 scene with the generated Everest tile as a collider."""
    if not TERRAIN_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Everest terrain manifest is missing: {TERRAIN_MANIFEST}. "
            "Run maps/build_everest_visual.py first."
        )
    terrain = json.loads(TERRAIN_MANIFEST.read_text())
    width = int(terrain["grid_width"])
    depth = int(terrain["grid_height"])
    heights = np.asarray(terrain["heights"], dtype=np.float32)
    if heights.size != width * depth:
        raise ValueError("Everest terrain height count does not match its grid dimensions")
    world_width = float(terrain["world_width_m"])
    world_depth = float(terrain["world_depth_m"])
    center = np.asarray(terrain["terrain_center"], dtype=np.float32)
    if center.shape != (3,) or world_width <= 0 or world_depth <= 0:
        raise ValueError("Everest terrain manifest has invalid placement or dimensions")
    elevation_min = float(heights.min())
    elevation_range = float(heights.max() - elevation_min)
    if elevation_range <= 0:
        raise ValueError("Everest terrain has no vertical relief")

    height_grid = heights.reshape(depth, width)
    # MuJoCo hfield rows increase from the negative-y edge. New renderer
    # products are already south-to-north; retain the old preview fallback's
    # historical north-to-south conversion for compatibility.
    if terrain.get("row_order") != "south_to_north":
        height_grid = height_grid[::-1]
    normalized = ((height_grid - elevation_min) / elevation_range).ravel()
    heightfield_png = BytesIO()
    Image.fromarray(
        np.round(normalized.reshape(depth, width) * 65535).astype(np.uint16),
        mode="I;16",
    ).save(heightfield_png, format="PNG")
    asset_name = "everest_heightfield.png"
    spec = mujoco.MjSpec.from_file(
        str(PHYSICS_SCENE), assets={asset_name: heightfield_png.getvalue()}
    )
    hfield_base = 0.1
    spec.add_hfield(
        name="everest",
        file=asset_name,
        size=(world_width / 2, world_depth / 2, elevation_range, hfield_base),
    )
    floor = spec.geom("floor")
    if floor is None:
        raise ValueError("G1 scene has no floor geom")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "everest"
    # The fourth hfield size is solid depth below the minimum surface; it does
    # not shift the sampled elevation surface itself.
    floor.pos = [
        float(center[0]),
        float(center[1]),
        float(center[2]) + elevation_min,
    ]
    floor.size = [0.0, 0.0, 0.0]
    model = spec.compile()
    model.hfield_data[:] = normalized
    return model, terrain


class MuJoCoEngine:
    """Publish complete body-pose snapshots behind an engine-neutral API."""

    def __init__(self, telemetry_hz: float = 30.0, checkpoint: str | Path = DEFAULT_CHECKPOINT):
        if not PHYSICS_SCENE.is_file():
            raise FileNotFoundError(
                f"Menagerie G1 physics scene is missing: {PHYSICS_SCENE}. "
                "Run scripts/setup-playground.sh."
            )
        self.model, self.everest_terrain = _load_everest_model()
        self._terrain_source = copy.deepcopy(self.everest_terrain)
        self._terrain_edit = {
            "crop": [0.0, 1.0, 0.0, 1.0],
            "scale_xy": 1.0,
            "scale_z": 1.0,
        }
        self.data = mujoco.MjData(self.model)
        self.snow = SnowLayer("snow")
        self.snow.apply_to_mujoco(self.model)
        self._base_hfield_data = np.asarray(self.model.hfield_data, dtype=np.float64).copy()
        self._snow_patch: NewtonSnowPatch | None = None
        self._snow_deformation_history: dict[tuple[int, int, int, int], tuple] = {}
        self._snow_history_frames: list[dict[str, Any]] = []
        self._snow_history_revision = 0
        self._snow_history_limit = 256
        self._terrain_snapshot_cache: dict[str, Any] | None = None
        self._terrain_snapshot_mpm_sequence = -1
        self._snow_mpm_error: str | None = None
        self._mpm_reactions: dict[str, dict[str, list[float]]] = {}
        self._next_mpm_time = 0.0
        self._foot_pair_margins = {
            pair_id: float(self.model.pair_margin[pair_id])
            for pair_id in self._foot_floor_pairs()
        }
        self._weather: dict[str, Any] | None = None
        self._weather_friction_scale = 1.0
        self._surface_friction_override: float | None = None
        self._wind_force_n = 0.0
        self._wind_direction_deg = 0.0
        # The whole DEM remains a MuJoCo heightfield. Only this moving radius
        # gets expensive Newton MPM multilayer material.
        # Keep the default live window large enough to cover both feet and the
        # next step, but small enough for interactive implicit-MPM streaming.
        # Operators can still expand/refine it for offline fidelity checks.
        self._physics_radius_m = 1.25
        self._mpm_min_voxel_size_m = 0.10
        self._physics_detail_cells = 24
        self._mpm_coupling_hz = 3.0
        self._mpm_contact_refine_radius_m = 0.55
        self._mpm_coarse_stride = 4
        self._patch_recenter_fraction = 0.70
        self._snow_accumulation_enabled = True
        self._weather_time_scale = 1.0

        # Explicit non-physical transport mode for environment/debug testing.
        # Normal mode always keeps MuJoCo/Newton authoritative.
        self._cheat_mode = False
        self._manual_force_mode = False
        self._manual_nudge_force_n = 20.0
        self._manual_turn_torque_nm = 8.0
        self._cheat_speed_m_s = 1.6
        self._cheat_yaw_rate_rad_s = 1.4
        self._cheat_root_clearance_m = 0.0
        self._cheat_yaw_rad = 0.0
        self._cheat_joint_qpos: np.ndarray | None = None
        self._hold_joint_qpos: np.ndarray | None = None
        self._stand_lock_xy = np.zeros(2, dtype=np.float64)
        self._stand_lock_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._stand_lock_enabled = True
        self._stand_lock_min_z = 0.0
        self._safety_pose_active = False
        self._safety_pose_started_at = 0.0
        self._safety_pose_start_qpos: np.ndarray | None = None
        self._safety_pose_hold_qpos: np.ndarray | None = None
        self._safety_pose_attack_seconds = 0.12
        self._safety_pose_settle_seconds = 0.45
        # The bundled checkpoint was trained on flat terrain. Start stationary
        # on the Everest slope; walking commands remain available over the API.
        self._command = (0.0, 0.0, 0.0)
        self._policy: G1VelocityPolicy | None = None
        self._policy_error: str | None = None
        try:
            self._policy = G1VelocityPolicy(checkpoint)
            self._policy.configure_mujoco_actuators(self.model)
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            self._policy_error = f"{type(exc).__name__}: {exc}"
        # The MJX scene uses five iterations for accelerator throughput.
        # Native MuJoCo needs ten to resolve impacts after a slip on ice
        # without triggering its numerical-instability auto-reset.
        self.model.opt.iterations = max(10, self.model.opt.iterations)
        self._policy_period = 0.02
        self._next_policy_time = 0.0
        self._next_supervisor_time = 0.0
        self._supervisor_cooldown_until = 0.0
        self._policy_supervisor = PolicySupervisor()
        self._policy_selection_key = "auto"
        self._last_auto_route_signature: tuple[str, str] | None = None
        self._policy_registry = self._build_policy_registry()
        if self._policy is not None:
            self._policy_supervisor.activate_policy(
                "flat", "Flat-ground walker", str(self._policy.path), sim_time=0.0
            )
        self._subset_preview_enabled = False
        self._subset_preview_sequence = 0
        self._subset_renderer = None
        self._subset_preview_error: str | None = None
        self.period = 1.0 / telemetry_hz
        self._body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            for body_id in range(1, self.model.nbody)
        ]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = True
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._reset_frames_remaining = 3
        self._telemetry_error: str | None = None
        self._simulation_fault: str | None = None
        self._started_at = time.time()
        self._snapshot: dict[str, Any] = {}
        self._scene = self._build_scene_manifest()
        with self._lock:
            self._reset_to_home()
            self._publish_snapshot()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mujoco-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._subset_renderer is not None:
            try:
                self._subset_renderer.close()
            except Exception:
                pass
            self._subset_renderer = None

    def reset(self) -> None:
        with self._lock:
            self._simulation_fault = None
            self._reset_to_home()
            self._publish_snapshot()

    def _reset_to_home(self) -> None:
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home < 0:
            raise RuntimeError("Menagerie G1 scene is missing its home keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        self.data.ctrl[:] = 0.0
        self._next_policy_time = 0.0
        if self._policy is not None:
            self._policy.last_action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        self._align_home_to_local_terrain()
        self._cheat_root_clearance_m = float(
            self.data.qpos[2]
            - self._terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
        )
        self._cheat_yaw_rad = self._yaw_from_quaternion_wxyz(self.data.qpos[3:7])
        self._cheat_joint_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
        self._hold_joint_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
        self._stand_lock_xy = np.asarray(self.data.qpos[:2], dtype=np.float64).copy()
        self._stand_lock_quat = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
        self._stand_lock_enabled = True
        self._stand_lock_min_z = float(self.data.qpos[2] - 0.08)
        self.model.hfield_data[:] = self._base_hfield_data
        self._mpm_reactions = {}
        self._next_mpm_time = 0.0
        self._snow_deformation_history.clear()
        self._snow_history_frames.clear()
        self._snow_history_revision += 1
        self._policy_supervisor.failure_candidate_frames = 0
        self._policy_supervisor.failure_latched = False
        self._policy_supervisor.stage = "monitoring"
        self._policy_supervisor.request_id = None
        self._policy_supervisor.request_manifest = None
        self._policy_supervisor.requested_at = None
        self._safety_pose_active = False
        self._safety_pose_started_at = 0.0
        self._safety_pose_start_qpos = None
        self._safety_pose_hold_qpos = None
        self._supervisor_cooldown_until = 0.0
        self._policy_supervisor.risk = self._policy_supervisor.risk.__class__(0.0, 0.0, 0.0, 2, False)
        self._policy_supervisor.route = self._policy_supervisor.route.__class__(
            "flat / nominal terrain", "flat", "Flat-ground walker", 0.94, False,
            "Nominal conditions fit the stock policy envelope."
        )
        self._last_auto_route_signature = None
        if self._snow_patch is not None:
            self._snow_patch.reset(self._foot_poses())
        # Keep the marker alive across several publications so a slower poll
        # cannot silently miss a reset event.
        self._reset_frames_remaining = 3

    def _align_home_to_local_terrain(self) -> None:
        """Align the canonical flat-floor keyframe to the local DEM tangent.

        The origin is sloped enough that the upstream flat-floor pose starts
        with only one sole touching. The controller then tips/slides before
        snow settlement can be distinguished from a fall. Preserve the home
        articulation and heading, rotate the floating base onto the local
        tangent, and lower it only until both MuJoCo soles report contact.
        """
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        sample = 0.08
        dz_dx = (
            self._terrain_height(x + sample, y) - self._terrain_height(x - sample, y)
        ) / (2.0 * sample)
        dz_dy = (
            self._terrain_height(x, y + sample) - self._terrain_height(x, y - sample)
        ) / (2.0 * sample)
        normal = np.asarray((-dz_dx, -dz_dy, 1.0), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        yaw = self._yaw_from_quaternion_wxyz(self.data.qpos[3:7])
        forward = np.asarray(
            (math.cos(yaw), math.sin(yaw), dz_dx * math.cos(yaw) + dz_dy * math.sin(yaw)),
            dtype=np.float64,
        )
        forward -= normal * np.dot(forward, normal)
        forward /= np.linalg.norm(forward)
        left = np.cross(normal, forward)
        left /= np.linalg.norm(left)
        rotation = np.column_stack((forward, left, normal))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        self.data.qpos[3:7] = quaternion
        mujoco.mj_forward(self.model, self.data)

        # Descend at sub-millimetre resolution. The upper bound is far below
        # a snow layer thickness and only compensates the sole/DEM mismatch
        # introduced by rotating the authored flat-floor pose.
        for _ in range(24):
            if all(item["contact"] for item in self._foot_contact_telemetry().values()):
                break
            self.data.qpos[2] -= 0.0005
            mujoco.mj_forward(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

    def _publish_snapshot(self) -> None:
        """Replace the latest frame atomically; readers never see partial poses."""
        self._sequence += 1
        self._snapshot = {
            "schema": "everest-viewer/v1",
            "sequence": self._sequence,
            "timestamp": time.time(),
            "sim_time": float(self.data.time),
            "step_dt": float(self.model.opt.timestep),
            "reset_frame": self._reset_frames_remaining > 0,
            "engine": "newton+mujoco" if self._snow_patch is not None else "mujoco",
            "source": PHYSICS_SOURCE,
            "source_url": PHYSICS_SOURCE_URL,
            "source_revision": PHYSICS_SOURCE_REVISION,
            "render_source": RENDER_SOURCE,
            "render_source_revision": RENDER_SOURCE_REVISION,
            "body_names": self._body_names,
            "body_pos_w": self.data.xpos[1:].tolist(),
            "body_quat_w": self.data.xquat[1:].tolist(),
            "base_linear_velocity": [float(value) for value in self.data.qvel[:3]],
            "base_angular_velocity": [float(value) for value in self.data.qvel[3:6]],
            **self._joint_telemetry(),
            "command": {
                "forward": float(self._command[0]),
                "lateral": float(self._command[1]),
                "yaw": float(self._command[2]),
            },
            "feet": self._foot_contact_telemetry(),
            "paused": self._paused,
            "cheat_mode": self._cheat_mode,
            "surface": self.snow.surface,
            "snow": self._snow_manifest(),
            "weather": self._weather,
            "weather_parameters": self._weather_parameters(),
            "policy": self._policy_status(),
            "simulation_fault": self._simulation_fault,
        }
        self._reset_frames_remaining = max(0, self._reset_frames_remaining - 1)

    def frame(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def scene_manifest(self) -> dict[str, Any]:
        manifest = copy.deepcopy(self._scene)
        manifest["terrain"] = self._snow_manifest()
        manifest["terrain_tile"] = {
            "url": "/everest-terrain.json",
            "schema": self.everest_terrain["schema"],
            "grid_width": self.everest_terrain["grid_width"],
            "grid_height": self.everest_terrain["grid_height"],
            "world_width_m": self.everest_terrain["world_width_m"],
            "world_depth_m": self.everest_terrain["world_depth_m"],
            "vertical_relief_m": self.everest_terrain.get("vertical_relief_m"),
            "collision": "mujoco_hfield",
        }
        manifest["weather"] = self._weather
        manifest["weather_parameters"] = self._weather_parameters()
        manifest["terrain_edit"] = copy.deepcopy(self._terrain_edit)
        manifest["terrain_stream"] = {
            "schema": "everest-terrain/v1",
            "url": "/api/terrain/frame",
            "rate_hz": 15,
        }
        return manifest

    def terrain_tile(self) -> dict[str, Any]:
        """Return the currently edited physical terrain artifact."""
        with self._lock:
            return copy.deepcopy(self.everest_terrain)

    def terrain_frame(self, *, include_particles: bool = True) -> dict[str, Any]:
        """Return the active local material surface without bloating pose frames."""
        with self._lock:
            if self._snow_patch is not None:
                if self._terrain_snapshot_cache is not None and self._terrain_snapshot_mpm_sequence == self._snow_patch.sequence:
                    return copy.deepcopy(self._terrain_snapshot_cache)
                frame = copy.deepcopy(self._snow_patch.terrain_frame(include_particles=include_particles))
                # Renderer stream sequencing must remain monotonic even when a
                # moving-radius Newton patch is rebuilt/recentered.
                frame["sequence"] = self._sequence
                frame["timestamp"] = float(self.data.time)
                frame["sim_time"] = float(self.data.time)
                self._terrain_snapshot_cache = frame
                self._terrain_snapshot_mpm_sequence = self._snow_patch.sequence
                return copy.deepcopy(frame)
            if self.snow.surface in {"ice", "rock"}:
                return self._rigid_active_surface_frame(self.snow.surface)
            return {
                "schema": "everest-terrain/v1",
                "mode": "inactive",
                "sequence": self._sequence,
                "mpm": {"active": False, "error": self._snow_mpm_error},
            }

    def snow_history(self) -> dict[str, Any]:
        """Return renderer snapshots of the persistent traveled snow path."""
        with self._lock:
            return {
                "schema": "everest-snow-history/v1",
                "sequence": self._snow_history_revision,
                "patches": copy.deepcopy(self._snow_history_frames),
            }

    def _rigid_active_surface_frame(self, surface_kind: str) -> dict[str, Any]:
        """Sample a backend-authoritative rigid active patch for ice rendering."""
        radius = self._physics_radius_m
        size = 2.0 * radius
        center_x = float(self.data.qpos[0])
        center_y = float(self.data.qpos[1])
        step = max(0.08, self._effective_mpm_voxel_size())
        n = max(12, min(128, int(math.ceil(size / step))))
        x0 = center_x - radius
        y0 = center_y - radius
        heights = np.empty(n * n, dtype=np.float32)
        for iy in range(n):
            wy = y0 + (iy + 0.5) * size / n
            for ix in range(n):
                wx = x0 + (ix + 0.5) * size / n
                heights[iy * n + ix] = self._terrain_height(wx, wy)

        material = SURFACES[surface_kind]
        friction = (
            self._surface_friction_override
            if self._surface_friction_override is not None
            else material.friction
        ) * self._weather_friction_scale
        return {
            "schema": "everest-terrain/v1",
            "mode": "live",
            "sequence": self._sequence,
            "timestamp": float(self.data.time),
            "sim_time": float(self.data.time),
            "origin": [x0, y0, self._terrain_height(center_x, center_y)],
            "size": [size, size],
            "resolution": [n, n],
            "heights": heights.tolist(),
            "material_ids": [0] * (n * n),
            "compaction": [0.0] * (n * n),
            "surface_kind": surface_kind,
            "surface_depth": 0.0,
            "surface_friction": friction,
            "layers": [{
                "id": 0,
                "type": "ICE" if surface_kind == "ice" else "ROCK",
                "name": "Glacier ice" if surface_kind == "ice" else "Everest rock",
                "label": "Glacier ice" if surface_kind == "ice" else "Everest rock",
                "color": list(material.color),
                "depth": 0.0,
                "thickness_m": 0.0,
                "density_kg_m3": material.density,
                "stiffness_pa": material.young_modulus,
                "compressive_strength_pa": material.yield_pressure,
                "shear_strength_pa": material.cohesion,
                "compaction_hardening": 0.0,
                "bond_strength_below_pa": 0.0,
            }],
            "mpm": {
                "active": False,
                "rigid": True,
                "solver": "MuJoCo hfield",
                "device": "cpu",
                "particle_count": 0,
                "steps": self._sequence,
                "window_size_m": [size, size],
                "terrain_conforming": True,
            },
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            age = time.time() - self._started_at
            return {
                "schema": "everest-state/v1",
                "engine": "newton+mujoco" if self._snow_patch is not None else "mujoco",
                "model": "Unitree G1",
                "source": PHYSICS_SOURCE,
                "source_url": PHYSICS_SOURCE_URL,
                "source_revision": PHYSICS_SOURCE_REVISION,
                "render_source": RENDER_SOURCE,
                "render_source_revision": RENDER_SOURCE_REVISION,
                "scene": str(PHYSICS_SCENE.relative_to(ROOT)),
                "render_assets": str(G1_SCENE.relative_to(ROOT)),
                "bodies": int(self.model.nbody),
                "joints": int(self.model.njnt),
                "actuators": int(self.model.nu),
                "paused": self._paused,
                "frames": self._sequence,
                "telemetry_hz": round(self._sequence / age, 1) if age > 0 else 0.0,
                "sim_time": round(float(self.data.time), 6),
                "telemetry_error": self._telemetry_error,
                "simulation_fault": self._simulation_fault,
                "surface": self.snow.surface,
                "snow": self._snow_manifest(),
                "terrain_collision": (
                    "newton_mpm_with_mujoco_hfield_feedback"
                    if self._snow_patch is not None else "everest_hfield"
                ),
                "weather": self._weather,
                "weather_parameters": self._weather_parameters(),
                "surface_friction_override": self._surface_friction_override,
                "simulation_settings": {
                    "physics_radius_m": self._physics_radius_m,
                    "mpm_min_voxel_size_m": self._mpm_min_voxel_size_m,
                    "mpm_voxel_size_m": self._effective_mpm_voxel_size(),
                    "physics_detail_cells": self._physics_detail_cells,
                    "mpm_coupling_hz": self._mpm_coupling_hz,
                    "mpm_contact_refine_radius_m": self._mpm_contact_refine_radius_m,
                    "mpm_coarse_stride": self._mpm_coarse_stride,
                    "patch_recenter_fraction": self._patch_recenter_fraction,
                    "snow_accumulation_enabled": self._snow_accumulation_enabled,
                    "weather_time_scale": self._weather_time_scale,
                    "cheat_mode": self._cheat_mode,
                    "manual_force_mode": self._manual_force_mode,
                    "manual_nudge_force_n": self._manual_nudge_force_n,
                    "manual_turn_torque_nm": self._manual_turn_torque_nm,
                    "stand_lock_active": (
                        self._stand_lock_enabled
                        and not self._cheat_mode
                        and not self._manual_force_mode
                        and not any(abs(item) > 1.0e-6 for item in self._command)
                    ),
                    "stand_lock_max_settlement_m": 0.08,
                    "safety_pose": self._safety_pose_status(),
                    "failure_detector_cooldown_s": max(
                        0.0, self._supervisor_cooldown_until - float(self.data.time)
                    ),
                    "cheat_speed_m_s": self._cheat_speed_m_s,
                    "cheat_yaw_rate_rad_s": self._cheat_yaw_rate_rad_s,
                },
                "terrain_edit": copy.deepcopy(self._terrain_edit),
                "snow_history": {
                    "particle_records": len(self._snow_deformation_history),
                    "patches": len(self._snow_history_frames),
                    "revision": self._snow_history_revision,
                },
                "policy": self._policy_status(),
            }

    def weather_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "everest-weather-state/v1",
                "weather": copy.deepcopy(self._weather),
                "parameters": self._weather_parameters(),
            }

    def control(self, action: str, value: Any = None) -> None:
        # Movement and pause are latency-sensitive edge controls. A Newton MPM
        # solve can hold the state lock for hundreds of milliseconds, so do
        # not queue these behind it. Tuple/bool assignment is atomic under the
        # GIL; the simulation thread observes the new values at its next
        # MuJoCo substep. Snapshot publication remains on the main state lock.
        if action == "pause":
            if self._safety_pose_active:
                raise ValueError("active safety posture keeps physics live")
            self._paused = bool(value)
            return
        if action == "command" and isinstance(value, (list, tuple)) and len(value) == 3:
            command = tuple(float(item) for item in value)
            if (
                self._policy_supervisor.stage == "waiting_checkpoint"
                and any(abs(item) > 1.0e-9 for item in command)
            ):
                raise ValueError("safety posture is waiting for a compatible checkpoint or reset")
            if (
                not self._cheat_mode
                and not self._weather_parameters()["movement_allowed"]
                and any(abs(item) > 1e-9 for item in command)
            ):
                raise ValueError("Movement is disabled by the active weather risk gate")
            self._command = command
            if any(abs(item) > 1.0e-6 for item in command):
                self._stand_lock_enabled = False
                if self._policy_supervisor.stage == "policy_active":
                    self._supervisor_cooldown_until = float(self.data.time) + 1.0
            return

        with self._lock:
            if action == "reset":
                self._simulation_fault = None
                self._reset_to_home()
                self._publish_snapshot()
            elif action == "play":
                # Start the velocity-policy demo atomically. Keeping command
                # and pause in one request avoids browser frame polling racing
                # the two state changes and leaving the engine paused.
                if self._policy_supervisor.stage == "waiting_checkpoint":
                    self._command = (0.0, 0.0, 0.0)
                    self._paused = False
                    self._publish_snapshot()
                    return
                if not self._weather_parameters()["movement_allowed"]:
                    raise ValueError("Movement is disabled by the active weather risk gate")
                self._command = (-0.1, 0.0, 0.0)
                self._supervisor_cooldown_until = float(self.data.time) + 1.0
                self._manual_force_mode = False
                self._paused = False
                self._publish_snapshot()
            elif action == "surface" and isinstance(value, str):
                if value not in SURFACES:
                    raise ValueError(f"Unknown surface {value!r}; choose {sorted(SURFACES)}")
                column = self.snow.column
                self._surface_friction_override = None
                self.snow = SnowLayer(value, self.snow.depth)
                if column is not None:
                    self.snow.column = column
                self.snow.apply_to_mujoco(self.model)
                self._apply_surface_friction()
                if value == "snow" and column is not None:
                    self._rebuild_snow_patch()
                else:
                    self._deactivate_snow_patch()
                self._scene["terrain"] = self._snow_manifest()
                self._publish_snapshot()
            elif action == "surface_friction":
                if self.snow.surface == "snow":
                    raise ValueError("snow friction is part of snow_parameters")
                friction = float(value)
                if not 0.01 <= friction <= 1.5:
                    raise ValueError("surface friction must be between 0.01 and 1.5")
                self._surface_friction_override = friction
                self._apply_surface_friction()
                self._publish_snapshot()
            elif action == "snow_parameters":
                self._surface_friction_override = None
                self.snow.configure_column(value)
                self.snow.apply_to_mujoco(self.model)
                self._apply_surface_friction()
                self._rebuild_snow_patch()
                self._scene["terrain"] = self._snow_manifest()
                self._publish_snapshot()
            elif action == "snow_forcing" and isinstance(value, dict):
                if self.snow.column is None:
                    raise ValueError("snow_forcing requires an active multilayer snow column")
                merged = self.snow.column.manifest()
                for key in ("snowfall_mm_h", "wind_speed_m_s", "wind_direction_deg", "temperature_c", "slope_deg"):
                    if key in value:
                        merged[key] = value[key]
                updated = type(self.snow.column).from_payload(merged)
                self.snow.column = updated
                self.snow.depth = updated.depth
                if self._snow_patch is not None:
                    self._snow_patch.column = updated
                self._scene["terrain"] = self._snow_manifest()
                self._publish_snapshot()
            elif action == "simulation_settings" and isinstance(value, dict):
                radius = float(value.get("physics_radius_m", self._physics_radius_m))
                min_voxel = float(value.get(
                    "mpm_min_voxel_size_m",
                    value.get("mpm_voxel_size_m", self._mpm_min_voxel_size_m),
                ))
                detail_cells = int(value.get("physics_detail_cells", self._physics_detail_cells))
                coupling_hz = float(value.get("mpm_coupling_hz", self._mpm_coupling_hz))
                refine_radius = float(value.get("mpm_contact_refine_radius_m", self._mpm_contact_refine_radius_m))
                coarse_stride = int(value.get("mpm_coarse_stride", self._mpm_coarse_stride))
                recenter = float(value.get("patch_recenter_fraction", self._patch_recenter_fraction))
                accumulation_enabled = bool(value.get("snow_accumulation_enabled", self._snow_accumulation_enabled))
                weather_time_scale = float(value.get("weather_time_scale", self._weather_time_scale))
                cheat_speed = float(value.get("cheat_speed_m_s", self._cheat_speed_m_s))
                cheat_yaw = float(value.get("cheat_yaw_rate_rad_s", self._cheat_yaw_rate_rad_s))
                if not 0.75 <= radius <= 6.0:
                    raise ValueError("physics_radius_m must be between 0.75 and 6.0")
                if not 0.05 <= min_voxel <= 0.25:
                    raise ValueError("mpm_min_voxel_size_m must be between 0.05 and 0.25")
                if not 24 <= detail_cells <= 96:
                    raise ValueError("physics_detail_cells must be between 24 and 96")
                if not 2.0 <= coupling_hz <= 30.0:
                    raise ValueError("mpm_coupling_hz must be between 2 and 30")
                if not 0.30 <= refine_radius <= 1.25:
                    raise ValueError("mpm_contact_refine_radius_m must be between 0.30 and 1.25")
                if not 1 <= coarse_stride <= 4:
                    raise ValueError("mpm_coarse_stride must be between 1 and 4")
                if not 0.25 <= recenter <= 0.75:
                    raise ValueError("patch_recenter_fraction must be between 0.25 and 0.75")
                if not 0.0 <= weather_time_scale <= 600.0:
                    raise ValueError("weather_time_scale must be between 0 and 600")
                if not 0.1 <= cheat_speed <= 5.0:
                    raise ValueError("cheat_speed_m_s must be between 0.1 and 5.0")
                if not 0.1 <= cheat_yaw <= 4.0:
                    raise ValueError("cheat_yaw_rate_rad_s must be between 0.1 and 4.0")
                rebuild = (
                    abs(radius - self._physics_radius_m) > 1.0e-6
                    or abs(min_voxel - self._mpm_min_voxel_size_m) > 1.0e-6
                    or detail_cells != self._physics_detail_cells
                    or abs(coupling_hz - self._mpm_coupling_hz) > 1.0e-6
                    or abs(refine_radius - self._mpm_contact_refine_radius_m) > 1.0e-6
                    or coarse_stride != self._mpm_coarse_stride
                )
                self._physics_radius_m = radius
                self._mpm_min_voxel_size_m = min_voxel
                self._physics_detail_cells = detail_cells
                self._mpm_coupling_hz = coupling_hz
                self._mpm_contact_refine_radius_m = refine_radius
                self._mpm_coarse_stride = coarse_stride
                self._patch_recenter_fraction = recenter
                self._snow_accumulation_enabled = accumulation_enabled
                self._weather_time_scale = weather_time_scale
                self._cheat_speed_m_s = cheat_speed
                self._cheat_yaw_rate_rad_s = cheat_yaw
                if rebuild and self.snow.column is not None and self.snow.surface == "snow":
                    self._rebuild_snow_patch()
                elif self._snow_patch is not None:
                    self._snow_patch.accumulation_enabled = self._snow_accumulation_enabled
                    self._snow_patch.accumulation_time_scale = self._weather_time_scale
                self._publish_snapshot()
            elif action == "cheat_mode":
                enabled = bool(value)
                if enabled and self._policy_supervisor.stage == "waiting_checkpoint":
                    raise ValueError("safety posture is waiting for a compatible checkpoint or reset")
                if enabled != self._cheat_mode:
                    if enabled:
                        self._stand_lock_enabled = False
                    self._cheat_mode = enabled
                    if enabled:
                        self._manual_force_mode = False
                    self._command = (0.0, 0.0, 0.0)
                    self._next_policy_time = float(self.data.time)
                    self._cheat_root_clearance_m = float(
                        self.data.qpos[2]
                        - self._terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
                    )
                    self._cheat_yaw_rad = self._yaw_from_quaternion_wxyz(self.data.qpos[3:7])
                    self._cheat_joint_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
                    self.data.qvel[:] = 0.0
                    self.data.qacc[:] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                self._publish_snapshot()
            elif action == "manual_force_mode":
                if bool(value) and self._policy_supervisor.stage == "waiting_checkpoint":
                    raise ValueError("safety posture is waiting for a compatible checkpoint or reset")
                self._manual_force_mode = bool(value)
                if self._manual_force_mode:
                    self._stand_lock_enabled = False
                self._command = (0.0, 0.0, 0.0)
                self._publish_snapshot()
            elif action == "policy_select":
                if self._policy_supervisor.stage == "waiting_checkpoint":
                    raise ValueError("safety posture is waiting for a compatible checkpoint or reset")
                key = str(value or "").strip()
                if key == "auto":
                    self._policy_selection_key = key
                    checkpoint = next(item["checkpoint"] for item in self._policy_registry if item["key"] == "flat")
                    policy = G1VelocityPolicy(checkpoint)
                    policy.configure_mujoco_actuators(self.model)
                    self._policy = policy
                    self._policy_supervisor.activate_policy(
                        "flat", "Flat-ground walker", checkpoint, sim_time=float(self.data.time)
                    )
                    self._last_auto_route_signature = None
                    self._policy_supervisor.log(
                        "SELECTOR",
                        "Deterministic selector enabled; execution remains tied to loaded compatible checkpoints.",
                        sim_time=float(self.data.time),
                    )
                    self._log_auto_route_execution()
                elif key == "flat":
                    self._policy_selection_key = key
                    checkpoint = next(item["checkpoint"] for item in self._policy_registry if item["key"] == "flat")
                    policy = G1VelocityPolicy(checkpoint)
                    policy.configure_mujoco_actuators(self.model)
                    self._policy = policy
                    self._policy_supervisor.activate_policy(
                        "flat", "Flat-ground walker", checkpoint, sim_time=float(self.data.time)
                    )
                    self._supervisor_cooldown_until = float(self.data.time) + 1.0
                elif key in {"ice_incline", "wind", "rough"}:
                    self._policy_selection_key = key
                    self._return_demo_pretrained(key)
                elif key == self._policy_supervisor.active_policy_key and self._policy_supervisor.demo_pretrained:
                    pass
                else:
                    raise ValueError("policy unavailable; return a demo-pretrained checkpoint or provide a compatible ONNX file")
                self._publish_snapshot()
            elif action == "retrain_request":
                self._enter_safe_wait_and_request_training(
                    "Operator requested retraining for the current Newton region."
                )
                self._publish_snapshot()
            elif action == "demo_failure":
                self._policy_supervisor.failure_latched = True
                self._policy_supervisor.stage = "failure_detected"
                self._policy_supervisor.log(
                    "FAILURE DETECTED",
                    "Demo failure injected through the operator UI.",
                    sim_time=float(self.data.time),
                )
                self._enter_safe_wait_and_request_training("Demo failure confirmed; moved to safe wait.")
                self._publish_snapshot()
            elif action == "demo_return_pretrained":
                key = str(value or self._policy_supervisor.route.requested_key)
                if key not in {"ice_incline", "wind", "rough"}:
                    key = "ice_incline"
                self._policy_selection_key = key
                self._return_demo_pretrained(key)
                self._publish_snapshot()
            elif action == "checkpoint_return" and isinstance(value, dict):
                path = Path(str(value.get("path") or "")).expanduser().resolve()
                key = str(value.get("key") or self._policy_supervisor.route.requested_key)
                label = str(value.get("label") or f"Returned {key} policy")
                policy = G1VelocityPolicy(path)
                policy.configure_mujoco_actuators(self.model)
                self._policy = policy
                self._safety_pose_active = False
                self._safety_pose_start_qpos = None
                self._safety_pose_hold_qpos = None
                self._hold_joint_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
                self._policy_selection_key = key
                self._policy_supervisor.activate_policy(
                    key, label, str(path), sim_time=float(self.data.time), demo_pretrained=False
                )
                self._supervisor_cooldown_until = float(self.data.time) + 1.0
                self._paused = False
                self._stand_lock_enabled = False
                self._publish_snapshot()
            elif action == "subset_preview":
                self._subset_preview_enabled = bool(value)
                self._publish_snapshot()
            elif action == "terrain_edit" and isinstance(value, dict):
                self._apply_terrain_edit(value)
                if self.snow.column is not None and self.snow.surface == "snow":
                    self._rebuild_snow_patch()
                self._publish_snapshot()
            elif action == "weather" and isinstance(value, dict):
                self._apply_weather(value)
                self._publish_snapshot()

    def _build_scene_manifest(self) -> dict[str, Any]:
        xml_root = ET.parse(G1_SCENE).getroot()
        mesh_files = {
            mesh.get("name"): mesh.get("file")
            for mesh in xml_root.findall("./asset/mesh")
            if mesh.get("name") and mesh.get("file")
        }
        visuals: list[dict[str, Any]] = []

        def numbers(element: ET.Element, attribute: str, fallback: tuple[float, ...]) -> list[float]:
            raw = element.get(attribute)
            return [float(value) for value in raw.split()] if raw else list(fallback)

        # The STL files retain the robot link coordinates authored upstream.
        # Read local geom transforms from the MJCF rather than compiled
        # ``geom_pos`` values, which include MuJoCo's internal mesh centering.
        def visit(body: ET.Element) -> None:
            body_name = body.get("name")
            for geom in body.findall("geom"):
                mesh_name = geom.get("mesh")
                filename = mesh_files.get(mesh_name)
                if not body_name or not filename:
                    continue
                if geom.get("contype") != "0" or geom.get("group") != "1":
                    continue
                visuals.append({
                    "id": f"geom-{len(visuals)}",
                    "body": body_name,
                    "mesh": mesh_name,
                    "url": f"/assets/unitree_g1/{filename}",
                    "position": numbers(geom, "pos", (0.0, 0.0, 0.0)),
                    "quaternion": numbers(geom, "quat", (1.0, 0.0, 0.0, 0.0)),
                    "scale": numbers(geom, "scale", (1.0, 1.0, 1.0)),
                    "rgba": numbers(geom, "rgba", (0.7, 0.7, 0.7, 1.0)),
                })
            for child in body.findall("body"):
                visit(child)

        for body in xml_root.findall("./worldbody/body"):
            visit(body)
        return {
            "schema": "everest-scene/v1",
            "model": "Unitree G1",
            "up_axis": "z",
            "quaternion_order": "wxyz",
            "source": RENDER_SOURCE,
            "source_revision": RENDER_SOURCE_REVISION,
            "physics_source": PHYSICS_SOURCE,
            "physics_source_url": PHYSICS_SOURCE_URL,
            "physics_source_revision": PHYSICS_SOURCE_REVISION,
            "visuals": visuals,
            "terrain": self._snow_manifest(),
        }

    def _apply_terrain_edit(self, payload: dict[str, Any]) -> None:
        """Apply a physical crop/scale edit to the shared Everest heightfield.

        The MuJoCo collider and browser terrain artifact are rebuilt from the
        same resampled source grid, preserving the project's one-unit-one-metre
        contract. The edit intentionally keeps the heightfield resolution
        fixed; crop changes sampling density rather than reallocating MuJoCo.
        """
        crop_raw = payload.get("crop", self._terrain_edit["crop"])
        if not isinstance(crop_raw, (list, tuple)) or len(crop_raw) != 4:
            raise ValueError("terrain crop must be [u0, u1, v0, v1]")
        crop = tuple(float(value) for value in crop_raw)
        u0, u1, v0, v1 = crop
        if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
            raise ValueError("terrain crop bounds must be ordered within 0..1")
        if u1 - u0 < 0.08 or v1 - v0 < 0.08:
            raise ValueError("terrain crop must retain at least 8% of each axis")
        scale_xy = float(payload.get("scale_xy", self._terrain_edit["scale_xy"]))
        scale_z = float(payload.get("scale_z", self._terrain_edit["scale_z"]))
        if not 0.25 <= scale_xy <= 4.0:
            raise ValueError("terrain scale_xy must be between 0.25 and 4")
        if not 0.1 <= scale_z <= 5.0:
            raise ValueError("terrain scale_z must be between 0.1 and 5")

        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hfield = int(self.model.geom_dataid[floor])
        rows = int(self.model.hfield_nrow[hfield])
        columns = int(self.model.hfield_ncol[hfield])
        source_rows = int(self._terrain_source["grid_height"])
        source_columns = int(self._terrain_source["grid_width"])
        source_heights = np.asarray(self._terrain_source["heights"], dtype=np.float64)
        # Source rows are north-to-south; convert once to MuJoCo/browser world
        # orientation where rows grow from south to north.
        source_world = source_heights.reshape(source_rows, source_columns)[::-1]
        cropped = _resample_crop(
            source_world,
            source_rows=source_rows,
            source_columns=source_columns,
            output_rows=rows,
            output_columns=columns,
            crop=crop,
        )
        scaled = cropped * scale_z
        elevation_min = float(scaled.min())
        elevation_range = float(scaled.max() - elevation_min)
        if elevation_range <= 1.0e-6:
            raise ValueError("edited terrain has no usable vertical relief")
        normalized = (scaled - elevation_min) / elevation_range

        source_width = float(self._terrain_source["world_width_m"])
        source_depth = float(self._terrain_source["world_depth_m"])
        source_center = np.asarray(self._terrain_source["terrain_center"], dtype=np.float64)
        crop_center_u = 0.5 * (u0 + u1)
        crop_center_v = 0.5 * (v0 + v1)
        center_x = float(source_center[0] + (crop_center_u - 0.5) * source_width)
        center_y = float(source_center[1] + (crop_center_v - 0.5) * source_depth)
        world_width = source_width * (u1 - u0) * scale_xy
        world_depth = source_depth * (v1 - v0) * scale_xy

        self.model.hfield_size[hfield, 0] = world_width / 2.0
        self.model.hfield_size[hfield, 1] = world_depth / 2.0
        self.model.hfield_size[hfield, 2] = elevation_range
        self.model.geom_pos[floor, :3] = (
            center_x,
            center_y,
            float(source_center[2]) + elevation_min,
        )
        self.model.hfield_data[:] = normalized.ravel()
        self._base_hfield_data = normalized.ravel().astype(np.float64, copy=True)

        edited = copy.deepcopy(self._terrain_source)
        edited["grid_width"] = columns
        edited["grid_height"] = rows
        edited["world_width_m"] = world_width
        edited["world_depth_m"] = world_depth
        edited["vertical_relief_m"] = elevation_range
        edited["terrain_center"] = [center_x, center_y, float(source_center[2])]
        # Browser terrain tiles are north-to-south, so flip back before export.
        edited["heights"] = scaled[::-1].ravel().astype(float).tolist()
        edited["route"] = []
        edited["editor"] = {
            "crop": list(crop),
            "scale_xy": scale_xy,
            "scale_z": scale_z,
            "source": "maps/everest_terrain.json",
        }
        self.everest_terrain = edited
        self._terrain_edit = {
            "crop": list(crop),
            "scale_xy": scale_xy,
            "scale_z": scale_z,
        }
        mujoco.mj_forward(self.model, self.data)

    def _snow_manifest(self) -> dict[str, Any]:
        manifest = self.snow.manifest()
        if self._snow_patch is not None:
            manifest.update({
                "mpm_ready": True,
                "mpm_active": True,
                "physics_mode": "newton_implicit_mpm_hfield_coupled",
                "telemetry_url": "/api/terrain/frame",
                "mpm": self._snow_patch.status(),
            })
        else:
            manifest.update({
                "mpm_active": False,
                "mpm": {"active": False, "error": self._snow_mpm_error},
            })
        return manifest

    def _policy_status(self) -> dict[str, Any]:
        if self._policy is None:
            base = {"enabled": False, "error": self._policy_error}
        else:
            base = {"enabled": True, "command": self._command, **self._policy.status()}
        base.update({
            "selected_policy_key": self._policy_selection_key,
            "registry": copy.deepcopy(self._policy_registry),
            "supervisor": self._policy_supervisor.manifest(),
            "subset_preview_enabled": self._subset_preview_enabled,
            "subset_preview_error": self._subset_preview_error,
        })
        return base

    def _build_policy_registry(self) -> list[dict[str, Any]]:
        checkpoint = str(self._policy.path) if self._policy is not None else str(DEFAULT_CHECKPOINT)
        return [
            {"key": "auto", "label": "Deterministic selector", "status": "selector", "checkpoint": None},
            {"key": "flat", "label": "Flat-ground walker", "status": "available", "checkpoint": checkpoint},
            {"key": "ice_incline", "label": "Low-friction incline", "status": "demo_pretrained", "checkpoint": checkpoint, "surrogate": True},
            {"key": "wind", "label": "Wind walker", "status": "demo_pretrained", "checkpoint": checkpoint, "surrogate": True},
            {"key": "rough", "label": "Rough-terrain walker", "status": "demo_pretrained", "checkpoint": checkpoint, "surrogate": True},
            {"key": "recovery", "label": "Supine recovery", "status": "incompatible_unavailable", "checkpoint": None},
        ]

    def _supervisor_context(self) -> dict[str, Any]:
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        sample = 0.08
        dz_dx = (self._terrain_height(x + sample, y) - self._terrain_height(x - sample, y)) / (2.0 * sample)
        dz_dy = (self._terrain_height(x, y + sample) - self._terrain_height(x, y - sample)) / (2.0 * sample)
        return {
            "slope_gradient": math.hypot(dz_dx, dz_dy),
            "friction": float(self._weather_parameters()["effective_friction"]),
            "roughness_m": float(self._snow_patch.max_sinkage_m if self._snow_patch is not None else 0.0),
            "wind_force_n": float(self._wind_force_n),
            "surface": self.snow.surface,
        }

    def _measure_failure_risk(self):
        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        rotation = np.asarray(self.data.xmat[pelvis], dtype=np.float64).reshape(3, 3)
        up_body = rotation.T @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        angular_body = rotation.T @ np.asarray(self.data.qvel[3:6], dtype=np.float64)
        contacts = self._foot_contact_telemetry()
        feet = sum(bool(item["contact"]) for item in contacts.values())
        return predict_imbalance(tuple(up_body), tuple(angular_body), feet)

    def _capture_retrain_subset(self) -> tuple[str, str]:
        request_id = f"snow-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        directory = ROOT / "runs" / "retrain_requests" / request_id
        directory.mkdir(parents=True, exist_ok=False)
        terrain = (
            copy.deepcopy(self._snow_patch.terrain_frame(include_particles=False))
            if self._snow_patch is not None
            else {
                "mode": "rigid",
                "surface_kind": self.snow.surface,
                "origin": [
                    float(self.data.qpos[0]),
                    float(self.data.qpos[1]),
                    self._terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1])),
                ],
                "size": [2.0 * self._physics_radius_m, 2.0 * self._physics_radius_m],
            }
        )
        manifest = {
            "schema": "everest-rl-subset/v1",
            "request_id": request_id,
            "created_at": time.time(),
            "sim_time": float(self.data.time),
            "source": "current Newton moving window",
            "policy_request": self._policy_supervisor.route.requested_key,
            "failure_risk": self._policy_supervisor.manifest()["detector"],
            "environment": {
                "surface": self.snow.surface,
                "weather": copy.deepcopy(self._weather),
                "context": self._supervisor_context(),
                "terrain": terrain,
                "robot_qpos": self.data.qpos.tolist(),
                "robot_qvel": self.data.qvel.tolist(),
                "command": list(self._command),
                "feet": self._foot_contact_telemetry(),
            },
            "training": {
                "status": "requested_not_launched",
                "reason": "No external trainer endpoint is configured.",
                "expected_return": "compatible 98-observation / 29-action ONNX checkpoint",
            },
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return request_id, str(path)

    def _enter_safe_wait_and_request_training(self, reason: str) -> None:
        if self._policy_supervisor.stage == "waiting_checkpoint":
            return
        request_id, manifest = self._capture_retrain_subset()
        self._command = (0.0, 0.0, 0.0)
        self._manual_force_mode = False
        self._cheat_mode = False
        self._stand_lock_enabled = False
        self._safety_pose_active = True
        self._safety_pose_started_at = float(self.data.time)
        self._safety_pose_start_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
        self._safety_pose_hold_qpos = None
        self._paused = False
        self._policy_supervisor.log(
            "SAFETY POSTURE",
            f"{reason} Physics remains live; commanding a low four-point protective stance.",
            sim_time=float(self.data.time),
        )
        self._policy_supervisor.request_training(request_id, manifest, sim_time=float(self.data.time))
        self._subset_preview_enabled = True

    def _return_demo_pretrained(self, key: str) -> None:
        spec = next((item for item in self._policy_registry if item["key"] == key), None)
        if spec is None or spec.get("status") != "demo_pretrained":
            raise ValueError("selected policy has no demo-pretrained return")
        policy = G1VelocityPolicy(spec["checkpoint"])
        policy.configure_mujoco_actuators(self.model)
        self._policy = policy
        self._safety_pose_active = False
        self._safety_pose_start_qpos = None
        self._safety_pose_hold_qpos = None
        self._hold_joint_qpos = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
        self._policy_supervisor.activate_policy(
            key,
            str(spec["label"]),
            str(spec["checkpoint"]),
            sim_time=float(self.data.time),
            demo_pretrained=True,
        )
        self._paused = False
        self._stand_lock_enabled = False
        self._supervisor_cooldown_until = float(self.data.time) + 1.0

    def _four_point_safety_target(self, *, aggressive: bool) -> np.ndarray:
        """Port humanoid_climber's deterministic hands-and-feet safety pose."""
        if self._policy is None:
            target = np.asarray(self.data.qpos[7:], dtype=np.float64).copy()
            names = [
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    int(self.model.actuator_trnid[index, 0]),
                ) or ""
                for index in range(self.model.nu)
            ]
        else:
            target = np.asarray(self._policy.default_joint_pos, dtype=np.float64).copy()
            names = list(self._policy.joint_names)
        for index, name in enumerate(names):
            if "hip_pitch" in name:
                target[index] = -0.75 if aggressive else -0.65
            elif "knee" in name:
                target[index] = 1.30 if aggressive else 1.15
            elif "ankle_pitch" in name:
                target[index] = -0.55 if aggressive else -0.50
            elif name == "left_hip_roll_joint":
                target[index] = 0.15
            elif name == "right_hip_roll_joint":
                target[index] = -0.15
            elif "hip_yaw" in name or "ankle_roll" in name:
                target[index] = 0.0
            elif name == "waist_pitch_joint":
                target[index] = 0.48 if aggressive else 0.45
            elif "waist_roll" in name or "waist_yaw" in name:
                target[index] = 0.0
            elif "shoulder_pitch" in name:
                # In the Menagerie G1 coordinate frame, a modest negative
                # shoulder pitch swings the forearms forward/down. The old
                # -1.55 value folded the arms backward and left the robot
                # sprawled on its back instead of bracing on its hands.
                target[index] = -0.45
            elif name == "left_shoulder_roll_joint":
                target[index] = 0.25
            elif name == "right_shoulder_roll_joint":
                target[index] = -0.25
            elif "shoulder_yaw" in name:
                target[index] = 0.0
            elif "elbow" in name:
                target[index] = 1.50
            elif "wrist_pitch" in name:
                target[index] = -0.45
            elif "wrist_roll" in name or "wrist_yaw" in name:
                target[index] = 0.0
        return np.clip(
            target,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def _safety_pose_control(self) -> np.ndarray:
        start = (
            self._safety_pose_start_qpos
            if self._safety_pose_start_qpos is not None
            else np.asarray(self.data.qpos[7:], dtype=np.float64)
        )
        elapsed = max(0.0, float(self.data.time) - self._safety_pose_started_at)
        if elapsed >= 1.0:
            # Once contact has arrested the incident, stop chasing an idealized
            # kinematic target that may be infeasible on the local slope. Hold
            # the achieved joint configuration and let live contacts carry it.
            if self._safety_pose_hold_qpos is None:
                self._safety_pose_hold_qpos = np.asarray(
                    self.data.qpos[7:], dtype=np.float64
                ).copy()
            return np.clip(
                self._safety_pose_hold_qpos,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        aggressive = self._four_point_safety_target(aggressive=True)
        sustained = self._four_point_safety_target(aggressive=False)
        if elapsed < self._safety_pose_attack_seconds:
            alpha = elapsed / max(self._safety_pose_attack_seconds, 1.0e-6)
            return (1.0 - alpha) * start + alpha * aggressive
        alpha = min(
            1.0,
            (elapsed - self._safety_pose_attack_seconds)
            / max(self._safety_pose_settle_seconds - self._safety_pose_attack_seconds, 1.0e-6),
        )
        return (1.0 - alpha) * aggressive + alpha * sustained

    def _ground_support_bodies(self) -> list[str]:
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        bodies: set[str] = set()
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geoms = (int(contact.geom1), int(contact.geom2))
            if floor not in geoms:
                continue
            other = geoms[1] if geoms[0] == floor else geoms[0]
            body = int(self.model.geom_bodyid[other])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if name:
                bodies.add(name)
        return sorted(bodies)

    def _safety_pose_status(self) -> dict[str, Any]:
        elapsed = max(0.0, float(self.data.time) - self._safety_pose_started_at)
        return {
            "active": self._safety_pose_active,
            "kind": "four_point_protective_stance",
            "physics_live": self._safety_pose_active and not self._paused,
            "elapsed_s": elapsed if self._safety_pose_active else 0.0,
            "transition_progress": (
                min(1.0, elapsed / max(self._safety_pose_settle_seconds, 1.0e-6))
                if self._safety_pose_active else 0.0
            ),
            "phase": (
                "stabilizing" if self._safety_pose_active and elapsed < 1.0
                else "contact_hold" if self._safety_pose_active else "inactive"
            ),
            "support_bodies": self._ground_support_bodies() if self._safety_pose_active else [],
        }

    def _log_auto_route_execution(self) -> None:
        if self._policy_selection_key != "auto":
            return
        route = self._policy_supervisor.route
        signature = (route.terrain_type, route.requested_key)
        if signature == self._last_auto_route_signature:
            return
        self._last_auto_route_signature = signature
        if route.requested_key == "flat":
            message = "Selector chose nominal terrain; executing the loaded Flat-ground walker checkpoint."
        else:
            message = (
                f"Selector requested {route.requested_label}, but no compatible specialist checkpoint is loaded; "
                "executing the Flat-ground walker fallback."
            )
        self._policy_supervisor.log("EXECUTION", message, sim_time=float(self.data.time))

    def subset_preview(self) -> dict[str, Any] | None:
        if not self._subset_preview_enabled:
            return None
        with self._lock:
            try:
                if self._subset_renderer is None:
                    self._subset_renderer = mujoco.Renderer(self.model, height=240, width=320)
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = (float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2]) * 0.45)
                camera.distance = 2.8
                camera.azimuth = 135.0
                camera.elevation = -28.0
                self._subset_renderer.update_scene(self.data, camera=camera)
                pixels = self._subset_renderer.render()
                output = BytesIO()
                Image.fromarray(pixels).save(output, format="JPEG", quality=72)
                self._subset_preview_sequence += 1
                self._subset_preview_error = None
                return {
                    "schema": "everest-mujoco-subset-preview/v1",
                    "sequence": self._subset_preview_sequence,
                    "request_id": self._policy_supervisor.request_id,
                    "encoding": "jpeg/base64",
                    "width": 320,
                    "height": 240,
                    "image": base64.b64encode(output.getvalue()).decode("ascii"),
                    "caption": "Raw MuJoCo view of the current Newton-window RL subset",
                }
            except Exception as exc:
                self._subset_preview_error = f"{type(exc).__name__}: {exc}"
                return None

    def _foot_floor_pairs(self) -> list[int]:
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        result: list[int] = []
        for pair_id in range(self.model.npair):
            geom_ids = (int(self.model.pair_geom1[pair_id]), int(self.model.pair_geom2[pair_id]))
            if floor not in geom_ids:
                continue
            other = geom_ids[1] if geom_ids[0] == floor else geom_ids[0]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if "foot" in name:
                result.append(pair_id)
        return result

    def _foot_poses(self) -> tuple[FootPose, FootPose]:
        poses: list[FootPose] = []
        for name in NewtonSnowPatch.FOOT_NAMES:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"Menagerie G1 is missing MPM foot body {name!r}")
            poses.append(FootPose(
                name=name,
                position=tuple(float(value) for value in self.data.xpos[body]),
                quaternion_wxyz=tuple(float(value) for value in self.data.xquat[body]),
            ))
        return (poses[0], poses[1])

    def _joint_telemetry(self) -> dict[str, Any]:
        """Renderer-facing actuator/joint state in the policy's canonical order."""
        if self._policy is not None:
            names = list(self._policy.joint_names)
        else:
            names = []
            for actuator_id in range(self.model.nu):
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                names.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    or f"joint_{joint_id}"
                )
        return {
            "joint_names": names,
            "joint_positions": [float(value) for value in self.data.qpos[7 : 7 + len(names)]],
            "joint_velocities": [float(value) for value in self.data.qvel[6 : 6 + len(names)]],
            "joint_torques": [float(value) for value in self.data.actuator_force[: len(names)]],
        }

    def _foot_contact_telemetry(self) -> dict[str, Any]:
        """Aggregate MuJoCo contacts per foot for renderer/debug overlays."""
        result: dict[str, Any] = {}
        for side, body_name in zip(("left", "right"), NewtonSnowPatch.FOOT_NAMES):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_ids = {
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
            }
            normal_force = 0.0
            tangent = np.zeros(2, dtype=np.float64)
            penetration = 0.0
            contact_position = np.asarray(self.data.xpos[body_id], dtype=np.float64)
            in_contact = False
            for contact_id in range(self.data.ncon):
                contact = self.data.contact[contact_id]
                if int(contact.geom1) not in geom_ids and int(contact.geom2) not in geom_ids:
                    continue
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                normal_force += abs(float(force[0]))
                tangent += force[1:3]
                penetration = max(penetration, max(0.0, -float(contact.dist)))
                contact_position = np.asarray(contact.pos, dtype=np.float64)
                in_contact = True
            body_linear_velocity = np.asarray(self.data.cvel[body_id, 3:6], dtype=np.float64)
            result[side] = {
                "position": [float(value) for value in contact_position],
                "normal_force_n": normal_force,
                "tangential_force_n": [float(value) for value in tangent],
                "penetration_m": penetration,
                "slip_speed_m_s": float(np.linalg.norm(body_linear_velocity[:2])),
                "contact": in_contact,
            }
        return result

    def _terrain_height(self, x: float, y: float) -> float:
        """Bilinearly sample the immutable Everest contact heightfield."""
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hfield = int(self.model.geom_dataid[floor])
        rows = int(self.model.hfield_nrow[hfield])
        columns = int(self.model.hfield_ncol[hfield])
        hx, hy, hz = (float(value) for value in self.model.hfield_size[hfield, :3])
        floor_position = self.model.geom_pos[floor]
        column = np.clip((x - floor_position[0] + hx) / (2.0 * hx) * (columns - 1), 0, columns - 1)
        row = np.clip((y - floor_position[1] + hy) / (2.0 * hy) * (rows - 1), 0, rows - 1)
        x0, y0 = int(np.floor(column)), int(np.floor(row))
        x1, y1 = min(columns - 1, x0 + 1), min(rows - 1, y0 + 1)
        tx, ty = float(column - x0), float(row - y0)
        grid = self._base_hfield_data.reshape(rows, columns)
        normalized = (
            grid[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + grid[y0, x1] * tx * (1.0 - ty)
            + grid[y1, x0] * (1.0 - tx) * ty
            + grid[y1, x1] * tx * ty
        )
        return float(floor_position[2] + normalized * hz)

    def _set_deformable_foot_contacts(self, active: bool) -> None:
        for pair_id, default_margin in self._foot_pair_margins.items():
            self.model.pair_margin[pair_id] = (
                -min(self.snow.depth, 1.0) if active else default_margin
            )

    def _deactivate_snow_patch(self) -> None:
        self._snow_patch = None
        self._mpm_reactions = {}
        self.model.hfield_data[:] = self._base_hfield_data
        self._set_deformable_foot_contacts(False)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
        w, x, y, z = (float(value) for value in quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _maybe_recenter_snow_patch(self) -> None:
        if self._snow_patch is None or self.snow.surface != "snow":
            return
        dx = float(self.data.qpos[0]) - self._snow_patch.center_xy[0]
        dy = float(self.data.qpos[1]) - self._snow_patch.center_xy[1]
        threshold = max(0.35, self._physics_radius_m * self._patch_recenter_fraction)
        if dx * dx + dy * dy > threshold * threshold:
            self._archive_snow_patch()
            self._rebuild_snow_patch()

    def _archive_snow_patch(self) -> None:
        """Store the outgoing Newton window before constructing the next one."""
        if self._snow_patch is None:
            return
        self._snow_patch.export_deformation_history(self._snow_deformation_history)
        while len(self._snow_deformation_history) > 250_000:
            self._snow_deformation_history.pop(next(iter(self._snow_deformation_history)))
        frame = copy.deepcopy(self._snow_patch.terrain_frame(include_particles=False))
        frame.pop("particles", None)
        frame.pop("layer_heights", None)
        sinkage = np.asarray(frame["base_heights"], dtype=np.float32) - np.asarray(
            frame["heights"], dtype=np.float32
        )
        compaction = np.asarray(frame["compaction"], dtype=np.float32)
        if float(np.max(np.abs(sinkage), initial=0.0)) < 1.0e-4 and float(compaction.max(initial=0.0)) < 1.0e-4:
            return
        frame["history_revision"] = self._snow_history_revision + 1
        self._snow_history_frames.append(frame)
        self._snow_history_frames = self._snow_history_frames[-self._snow_history_limit:]
        self._snow_history_revision += 1

    def _effective_mpm_voxel_size(self) -> float:
        """Adaptive voxel size that keeps radius changes within a cell budget."""
        diameter = 2.0 * self._physics_radius_m
        adaptive = diameter / max(1, self._physics_detail_cells)
        return max(self._mpm_min_voxel_size_m, adaptive)

    def _advance_cheat_to(self, target: float) -> None:
        """Translate the floating base directly while still driving local material."""
        dt = float(self.model.opt.timestep)
        while self.data.time + dt * 0.5 < target:
            forward_norm, lateral_norm, yaw_norm = self._command
            self._cheat_yaw_rad += float(yaw_norm) * self._cheat_yaw_rate_rad_s * dt
            c = math.cos(self._cheat_yaw_rad)
            s = math.sin(self._cheat_yaw_rad)
            forward = np.asarray((c, s), dtype=np.float64)
            left = np.asarray((-s, c), dtype=np.float64)
            delta = (
                forward * float(forward_norm)
                + left * float(lateral_norm)
            ) * self._cheat_speed_m_s * dt

            self.data.qpos[0] += delta[0]
            self.data.qpos[1] += delta[1]
            self.data.qpos[2] = (
                self._terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
                + self._cheat_root_clearance_m
            )
            half = 0.5 * self._cheat_yaw_rad
            self.data.qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
            if (
                self._cheat_joint_qpos is not None
                and len(self._cheat_joint_qpos) == len(self.data.qpos[7:])
            ):
                self.data.qpos[7:] = self._cheat_joint_qpos
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            self._maybe_recenter_snow_patch()
            self._step_mpm_if_due()

            self.data.time += dt
            mujoco.mj_forward(self.model, self.data)

    def _step_mpm_if_due(self) -> None:
        patch = self._snow_patch
        if patch is None or self.data.time + 1.0e-9 < self._next_mpm_time:
            return
        if self._feet_have_contact():
            foot_poses = self._foot_poses()
            if patch.needs_contact_solve(foot_poses):
                self._mpm_reactions = patch.step(foot_poses)
                self._apply_mpm_surface_to_hfield()
            else:
                patch.advance_without_contact()
        else:
            self._mpm_reactions = {}
            patch.advance_without_contact()
        self._next_mpm_time += patch.dt

    def _rebuild_snow_patch(self) -> None:
        if self.snow.column is None or self.snow.surface != "snow":
            self._deactivate_snow_patch()
            return
        center = (float(self.data.qpos[0]), float(self.data.qpos[1]))
        try:
            patch = NewtonSnowPatch(
                self.snow.column,
                self._terrain_height,
                self._foot_poses(),
                center_xy=center,
                size_xy=(2.0 * self._physics_radius_m, 2.0 * self._physics_radius_m),
                voxel_size=self._effective_mpm_voxel_size(),
                dt=1.0 / self._mpm_coupling_hz,
                contact_refine_radius=self._mpm_contact_refine_radius_m,
                coarse_stride=self._mpm_coarse_stride,
                nominal_foot_load_n=float(self.model.body_mass.sum() * 9.81 / 2.0),
                accumulation_enabled=self._snow_accumulation_enabled,
                accumulation_time_scale=self._weather_time_scale,
            )
        except Exception as exc:
            self._snow_mpm_error = f"{type(exc).__name__}: {exc}"
            self._deactivate_snow_patch()
            raise ValueError(f"Newton MPM initialization failed: {self._snow_mpm_error}") from exc
        self._snow_patch = patch
        patch.history_restored_particles = patch.import_deformation_history(
            self._snow_deformation_history
        )
        self._snow_mpm_error = None
        self._terrain_snapshot_cache = None
        self._terrain_snapshot_mpm_sequence = -1
        self._mpm_reactions = {}
        self._next_mpm_time = float(self.data.time)
        self.model.hfield_data[:] = self._base_hfield_data
        # Keep MuJoCo's sole contact as the stable support boundary, but move
        # that boundary to Newton's deformed surface every MPM step. Newton
        # still supplies any resolved foot impulses; MuJoCo no longer pins the
        # robot to the undeformed Everest elevation.
        self._set_deformable_foot_contacts(False)
        mujoco.mj_forward(self.model, self.data)

    def _apply_mpm_surface_to_hfield(self) -> None:
        if self._snow_patch is None:
            return
        nx, ny = self._snow_patch.surface_resolution
        current = self._snow_patch.surface_arrays()[0].astype(np.float64).reshape(ny, nx)
        initial = self._snow_patch._initial_surface_grid().reshape(ny, nx)
        sink = np.maximum(0.0, initial - current)
        deposited_vertical = (
            self._snow_patch.deposited_depth_m * float(self._snow_patch.basis[2, 2])
        )
        origin_x = self._snow_patch.center_xy[0] - 0.5 * self._snow_patch.size_xy[0]
        origin_y = self._snow_patch.center_xy[1] - 0.5 * self._snow_patch.size_xy[1]
        size_x, size_y = self._snow_patch.size_xy

        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hfield = int(self.model.geom_dataid[floor])
        rows = int(self.model.hfield_nrow[hfield])
        columns = int(self.model.hfield_ncol[hfield])
        hx, hy, hz = (float(value) for value in self.model.hfield_size[hfield, :3])
        floor_position = self.model.geom_pos[floor]
        xs = floor_position[0] - hx + np.arange(columns) * (2.0 * hx / (columns - 1))
        ys = floor_position[1] - hy + np.arange(rows) * (2.0 * hy / (rows - 1))
        selected_columns = np.flatnonzero((xs >= origin_x) & (xs <= origin_x + size_x))
        selected_rows = np.flatnonzero((ys >= origin_y) & (ys <= origin_y + size_y))
        target = self._base_hfield_data.reshape(rows, columns).copy()
        for row in selected_rows:
            v = np.clip((ys[row] - origin_y) / size_y * (ny - 1), 0, ny - 1)
            y0, y1 = int(np.floor(v)), min(ny - 1, int(np.floor(v)) + 1)
            ty = float(v - y0)
            for column in selected_columns:
                u = np.clip((xs[column] - origin_x) / size_x * (nx - 1), 0, nx - 1)
                x0, x1 = int(np.floor(u)), min(nx - 1, int(np.floor(u)) + 1)
                tx = float(u - x0)
                sinkage = (
                    sink[y0, x0] * (1.0 - tx) * (1.0 - ty)
                    + sink[y0, x1] * tx * (1.0 - ty)
                    + sink[y1, x0] * (1.0 - tx) * ty
                    + sink[y1, x1] * tx * ty
                )
                target[row, column] = np.clip(
                    self._base_hfield_data.reshape(rows, columns)[row, column]
                    + (deposited_vertical - sinkage) / hz,
                    0.0,
                    1.0,
                )
        self.model.hfield_data[:] = target.ravel()

    def _feet_have_contact(self) -> bool:
        """Return true only when MuJoCo reports an actual foot-floor contact."""
        foot_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in NewtonSnowPatch.FOOT_NAMES
        }
        foot_geoms = {
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in foot_bodies
        }
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if floor in (int(contact.geom1), int(contact.geom2)) and (
                int(contact.geom1) in foot_geoms or int(contact.geom2) in foot_geoms
            ):
                return True
        return False

    def _apply_mpm_reaction_forces(self) -> None:
        """Do not add Newton reactions on top of heightfield support.

        The compatibility architecture mirrors the Newton deformation into
        MuJoCo's floor heightfield, which already supplies the complete robot
        support force. Applying collected kinematic-collider impulses as well
        double-counted the same contact. Keep the impulses as diagnostics until
        the direct Newton rigid/MPM coupler replaces heightfield support.
        """
        # Retain ``_mpm_reactions`` for telemetry/calibration, but deliberately
        # leave ``xfrc_applied`` unchanged in this support mode.

    def _weather_parameters(self) -> dict[str, Any]:
        return {
            "wind_force_n": self._wind_force_n,
            "wind_direction_deg": self._wind_direction_deg,
            "friction_scale": self._weather_friction_scale,
            "effective_friction": (self._surface_friction_override if self._surface_friction_override is not None else self.snow.material.friction) * self._weather_friction_scale,
            "visibility_scale": (
                self._weather.get("simulation", {}).get("visibility_scale", 1.0)
                if self._weather else 1.0
            ),
            "movement_allowed": (
                self._weather.get("simulation", {}).get("movement_allowed", True)
                if self._weather else True
            ),
        }

    def _apply_weather(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != "everest-weather/v1":
            raise ValueError("Weather payload must use schema everest-weather/v1")
        conditions = payload.get("conditions") or {}
        simulation = payload.get("simulation") or {}
        try:
            wind_scale = float(simulation.get("wind_force_scale", 0.0))
            friction_scale = float(simulation.get("terrain_friction_scale", 1.0))
            visibility_scale = float(simulation.get("visibility_scale", 1.0))
            direction_deg = float(conditions.get("wind_direction_deg") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Weather simulation parameters must be numeric") from exc
        if (
            not 0.0 <= wind_scale <= 1.0
            or not 0.0 < friction_scale <= 1.0
            or not 0.0 <= visibility_scale <= 1.0
        ):
            raise ValueError("Weather simulation scales are outside their allowed ranges")
        if not isinstance(simulation.get("movement_allowed", True), bool):
            raise ValueError("Weather movement_allowed parameter must be boolean")
        self._weather = copy.deepcopy(payload)
        self._weather_friction_scale = friction_scale
        self._wind_force_n = 120.0 * wind_scale
        self._wind_direction_deg = direction_deg % 360.0
        if not bool(simulation.get("movement_allowed", True)):
            self._command = (0.0, 0.0, 0.0)
        self._apply_surface_friction()

    def _apply_surface_friction(self) -> None:
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0:
            raise ValueError("G1 scene has no floor geom")
        base_friction = self._surface_friction_override if self._surface_friction_override is not None else self.snow.material.friction
        friction = base_friction * self._weather_friction_scale
        self.model.geom_friction[floor, 0] = friction
        for pair_id in range(self.model.npair):
            if floor in (self.model.pair_geom1[pair_id], self.model.pair_geom2[pair_id]):
                self.model.pair_friction[pair_id, :2] = friction

    def _advance_to(self, target: float) -> None:
        """Advance deterministically and detect MuJoCo's silent auto-reset."""
        if self._cheat_mode:
            self._advance_cheat_to(target)
            return
        while self.data.time + self.model.opt.timestep * 0.5 < target:
            moving_command = any(abs(item) > 1.0e-6 for item in self._command)
            if self._safety_pose_active:
                self.data.ctrl[:] = self._safety_pose_control()
            elif (
                self._policy is not None
                and moving_command
                and not self._manual_force_mode
                and self.data.time + 1.0e-9 >= self._next_policy_time
            ):
                observation = self._policy.observation(
                    self.data,
                    self.model,
                    command=(0.0, 0.0, 0.0) if self._manual_force_mode else self._command,
                )
                action = np.clip(self._policy(observation), -1.0, 1.0)
                target_pos = self._policy.target_positions(action)
                self.data.ctrl[:] = np.clip(
                    target_pos,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
                self._next_policy_time += self._policy_period
            elif not moving_command and not self._manual_force_mode and self._hold_joint_qpos is not None:
                # A zero velocity command is a stand/hold request, not an
                # instruction to let the flat-terrain policy free-run. The
                # bundled checkpoint has no slope-aware foothold controller;
                # allowing it to produce autonomous motion at startup makes
                # the robot slide/fall and look like it is sinking through
                # the snow. Explicit nonzero commands still use the policy.
                self.data.ctrl[:] = np.clip(
                    self._hold_joint_qpos,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
            previous_time = float(self.data.time)
            previous_qpos = self.data.qpos.copy()
            previous_qvel = self.data.qvel.copy()
            self._maybe_recenter_snow_patch()
            if self._snow_patch is not None:
                try:
                    self._step_mpm_if_due()
                except Exception as exc:
                    self._snow_mpm_error = f"{type(exc).__name__}: {exc}"
                    self._deactivate_snow_patch()
                    raise FloatingPointError(f"Newton MPM failed: {self._snow_mpm_error}") from exc
            self.data.xfrc_applied[:] = 0.0
            self._apply_mpm_reaction_forces()
            if self._wind_force_n > 0.0:
                pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                # Open-Meteo uses the meteorological direction the wind comes
                # from, clockwise from north. MuJoCo uses +X east, +Y north.
                direction = self._wind_direction_deg * np.pi / 180.0
                self.data.xfrc_applied[pelvis, 0] = -self._wind_force_n * np.sin(direction)
                self.data.xfrc_applied[pelvis, 1] = -self._wind_force_n * np.cos(direction)
            if self._safety_pose_active:
                pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                # Dampen planar/rotational momentum while the joint controller
                # moves into the protective pose. Position and orientation
                # remain physically unconstrained so gravity and contacts stay live.
                self.data.xfrc_applied[pelvis, :2] -= np.asarray(self.data.qvel[:2]) * 90.0
                self.data.xfrc_applied[pelvis, 3:6] -= np.asarray(self.data.qvel[3:6]) * 18.0
            if self._manual_force_mode:
                pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                yaw = self._yaw_from_quaternion_wxyz(self.data.xquat[pelvis])
                forward = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
                self.data.xfrc_applied[pelvis, :2] += forward * float(self._command[0]) * self._manual_nudge_force_n
                self.data.xfrc_applied[pelvis, 5] += float(self._command[2]) * self._manual_turn_torque_nm
                # Physical drag keeps a held key a controlled nudge instead of
                # integrating into an ever-growing slide down the slope.
                self.data.xfrc_applied[pelvis, :2] -= np.asarray(self.data.qvel[:2]) * 42.0
                self.data.xfrc_applied[pelvis, 5] -= float(self.data.qvel[5]) * 7.0
            mujoco.mj_step(self.model, self.data)
            if (
                self.data.time + 1.0e-9 >= self._next_supervisor_time
                and self.data.time >= self._supervisor_cooldown_until
            ):
                risk = self._measure_failure_risk()
                confirmed = self._policy_supervisor.observe(
                    risk,
                    self._supervisor_context(),
                    sim_time=float(self.data.time),
                )
                self._log_auto_route_execution()
                self._next_supervisor_time = float(self.data.time) + self._policy_period
                if confirmed:
                    self._enter_safe_wait_and_request_training(
                        "Deterministic IMU/contact detector confirmed imminent failure."
                    )
                    break
            if self._stand_lock_enabled and not moving_command and not self._manual_force_mode:
                # Settlement-inspection constraint: keep the unsupported
                # flat-ground controller from translating or toppling the
                # floating base while leaving vertical motion and all foot/
                # snow contacts dynamic. This is reported in state as
                # stand_lock_active and disengages for every explicit control
                # or locomotion command.
                self.data.qpos[:2] = self._stand_lock_xy
                self.data.qpos[3:7] = self._stand_lock_quat
                self.data.qpos[2] = max(float(self.data.qpos[2]), self._stand_lock_min_z)
                self.data.qvel[:2] = 0.0
                self.data.qvel[3:6] = 0.0
                mujoco.mj_forward(self.model, self.data)
            finite = all(
                np.isfinite(values).all()
                for values in (self.data.qpos, self.data.qvel, self.data.qacc)
            )
            if not finite or self.data.time <= previous_time:
                # Preserve the last valid renderer-visible pose. Unity should
                # see a paused fault and decide when to Reset; the backend must
                # not silently teleport the robot home after instability.
                self.data.qpos[:] = previous_qpos
                self.data.qvel[:] = previous_qvel
                self.data.qacc[:] = 0.0
                self.data.time = previous_time
                mujoco.mj_forward(self.model, self.data)
                raise FloatingPointError(
                    "MuJoCo produced a non-finite state or reset after numerical instability"
                )

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                with self._lock:
                    if not self._paused:
                        target = float(self.data.time) + self.period
                        try:
                            self._advance_to(target)
                        except FloatingPointError as exc:
                            self._simulation_fault = str(exc)
                            self._paused = True
                    self._publish_snapshot()
                self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))
        except Exception as exc:
            with self._lock:
                self._telemetry_error = f"{type(exc).__name__}: {exc}"
