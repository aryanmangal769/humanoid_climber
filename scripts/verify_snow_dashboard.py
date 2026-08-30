"""Verify dashboard snow contacts, policy inference, and Newton MPM material wiring."""

from __future__ import annotations

import json
import math

import newton
import mujoco
import numpy as np
import onnx
import warp as wp
from newton.solvers import SolverImplicitMPM
from onnx.reference import ReferenceEvaluator

from dashboard.engines.mujoco import MuJoCoEngine
from simulation.snow import SnowLayer


LIVE_SNOW_PARAMETERS = {
    "surface_friction": 0.35,
    "layers": [
        {
            "type": "POWDER", "label": "Fresh snow", "color": [0.94, 0.97, 1.0],
            "thickness_m": 0.03, "density_kg_m3": 120.0, "stiffness_pa": 60000.0,
            "compressive_strength_pa": 4000.0, "shear_strength_pa": 1200.0,
            "compaction_hardening": 4.0, "bond_strength_below_pa": 800.0,
        },
        {
            "type": "CRUST", "label": "Wind crust", "color": [0.72, 0.82, 0.9],
            "thickness_m": 0.06, "density_kg_m3": 380.0, "stiffness_pa": 600000.0,
            "compressive_strength_pa": 70000.0, "shear_strength_pa": 25000.0,
            "compaction_hardening": 18.0, "bond_strength_below_pa": 6000.0,
        },
        {
            "type": "POWDER", "label": "Weak layer", "color": [0.82, 0.89, 0.95],
            "thickness_m": 0.25, "density_kg_m3": 210.0, "stiffness_pa": 140000.0,
            "compressive_strength_pa": 12000.0, "shear_strength_pa": 3500.0,
            "compaction_hardening": 7.0, "bond_strength_below_pa": 1200.0,
        },
        {
            "type": "DENSE_SNOW", "label": "Dense old snow", "color": [0.57, 0.69, 0.78],
            "thickness_m": 0.4, "density_kg_m3": 520.0, "stiffness_pa": 1500000.0,
            "compressive_strength_pa": 180000.0, "shear_strength_pa": 75000.0,
            "compaction_hardening": 28.0, "bond_strength_below_pa": 12000.0,
        },
    ],
}


def rollout(surface: str, command: tuple[float, float, float], seconds: float) -> dict[str, object]:
    """Run through the engine's deterministic stepping path, faster than wall time."""
    engine = MuJoCoEngine()
    engine.control("surface", surface)
    engine.control("command", command)
    start = np.asarray(engine.data.qpos[:3]).copy()
    min_height = float(engine.data.qpos[2])
    max_acceleration = 0.0
    while engine.data.time + engine.model.opt.timestep * 0.5 < seconds:
        engine._advance_to(min(seconds, float(engine.data.time) + engine.period))
        if not all(
            np.isfinite(values).all()
            for values in (engine.data.qpos, engine.data.qvel, engine.data.qacc)
        ):
            raise RuntimeError(f"{surface} rollout produced non-finite state")
        min_height = min(min_height, float(engine.data.qpos[2]))
        max_acceleration = max(max_acceleration, float(np.max(np.abs(engine.data.qacc))))
    end = np.asarray(engine.data.qpos[:3]).copy()
    floor = engine.model.geom("floor").id
    return {
        "sim_time": float(engine.data.time),
        "start_position": start.tolist(),
        "end_position": end.tolist(),
        "displacement_xy": float(np.linalg.norm(end[:2] - start[:2])),
        "min_pelvis_height": min_height,
        "end_contacts": int(engine.data.ncon),
        "end_floor_contacts": sum(
            floor in (engine.data.contact[index].geom1, engine.data.contact[index].geom2)
            for index in range(engine.data.ncon)
        ),
        "max_abs_qacc": max_acceleration,
        "policy_inferences": int(engine._policy.inference_count),
        "fell": bool(min_height < 0.6),
    }


def main() -> None:
    engine = MuJoCoEngine()
    floor = engine.model.geom("floor").id
    if engine.model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_HFIELD:
        raise RuntimeError("Dashboard floor is not the Everest heightfield collider")
    terrain = engine.everest_terrain
    hfield_size = engine.model.hfield_size[0]
    expected_size = (
        float(terrain["world_width_m"]),
        float(terrain["world_depth_m"]),
        float(terrain["vertical_relief_m"]),
    )
    actual_size = (float(hfield_size[0] * 2), float(hfield_size[1] * 2), float(hfield_size[2]))
    if not np.allclose(actual_size, expected_size, atol=1e-6):
        raise RuntimeError(f"Visual/collision terrain scale mismatch: {actual_size} != {expected_size}")
    policy = engine._policy
    if policy is None:
        raise RuntimeError(engine.state()["policy"])
    observation = policy.observation(engine.data, engine.model, command=(0.3, 0.0, 0.0))
    action = policy(observation)
    if observation.shape != (98,) or action.shape != (29,):
        raise RuntimeError(f"Unexpected policy shapes: {observation.shape}, {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError("Policy produced non-finite actions")
    reference_action = ReferenceEvaluator(onnx.load(policy.path)).run(
        None, {"obs": observation.reshape(1, -1)}
    )[0].reshape(-1)
    inference_error = float(np.max(np.abs(action - reference_action)))
    if inference_error > 1.0e-5:
        raise RuntimeError(f"Dependency-light policy inference differs from ONNX: {inference_error}")

    friction = {}
    pair_friction = {}
    floor = engine.model.geom("floor").id
    for surface in ("snow", "ice"):
        engine.control("surface", surface)
        friction[surface] = float(engine.model.geom_friction[floor, 0])
        pair_friction[surface] = sorted({
            float(engine.model.pair_friction[pair_id, 0])
            for pair_id in range(engine.model.npair)
            if floor in (engine.model.pair_geom1[pair_id], engine.model.pair_geom2[pair_id])
        })
    if not math.isclose(friction["snow"], 0.35) or not math.isclose(friction["ice"], 0.08):
        raise RuntimeError(f"Surface friction was not applied: {friction}")
    if pair_friction != {"snow": [0.35], "ice": [0.08]}:
        raise RuntimeError(f"Contact-pair friction was not applied: {pair_friction}")

    actuator_joints = [
        mujoco.mj_id2name(
            engine.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(engine.model.actuator_trnid[index, 0]),
        )
        for index in range(engine.model.nu)
    ]
    if tuple(actuator_joints) != policy.joint_names:
        raise RuntimeError("Policy and actuator joint orders differ")
    if not np.allclose(engine.model.actuator_gainprm[:, 0], policy.stiffness):
        raise RuntimeError("Checkpoint stiffness was not applied to MuJoCo")
    if not np.allclose(-engine.model.actuator_biasprm[:, 2], policy.damping):
        raise RuntimeError("Checkpoint damping was not applied to MuJoCo")
    engine.control("play")
    if engine._paused or engine.state()["policy"]["command"] != (-0.1, 0.0, 0.0):
        raise RuntimeError("Play did not atomically start the velocity-policy demo")
    engine.control("pause", True)

    rollouts = {
        "snow_stand": rollout("snow", (0.0, 0.0, 0.0), 3.0),
        "ice_stand": rollout("ice", (0.0, 0.0, 0.0), 3.0),
    }
    for name, result in rollouts.items():
        if result["fell"] or result["end_floor_contacts"] <= 0:
            raise RuntimeError(f"G1 did not remain supported by the heightfield in {name}: {result}")

    # Exercise the actual dashboard control path, not just the particle helper.
    live_engine = MuJoCoEngine()
    live_engine.control("snow_parameters", LIVE_SNOW_PARAMETERS)
    live_patch = live_engine._snow_patch
    if live_patch is None:
        raise RuntimeError(f"Dashboard did not activate Newton MPM: {live_engine.state()['snow']}")
    top = live_patch.particle_material_ids == 0
    top_density = live_patch.model.particle_mass.numpy()[top] / (
        8.0 * live_patch.model.particle_radius.numpy()[top] ** 3
    )
    expected_fields = {
        "young_modulus": 60000.0,
        "yield_pressure": 4000.0,
        "yield_stress": 1200.0,
        "hardening": 4.0,
    }
    actual_fields = {
        name: float(getattr(live_patch.model.mpm, name).numpy()[top][0])
        for name in expected_fields
    }
    if actual_fields != expected_fields or not np.allclose(top_density, 120.0, rtol=1.0e-5):
        raise RuntimeError(
            f"Dashboard sliders did not reach Newton material fields: {actual_fields}, density={top_density}"
        )
    initial_pelvis_height = float(live_engine.data.qpos[2])
    live_engine._advance_to(0.8)
    robot_sinkage = initial_pelvis_height - float(live_engine.data.qpos[2])
    terrain_frame = live_engine.terrain_frame()
    if not 0.01 < robot_sinkage < 0.25:
        raise RuntimeError(f"G1 did not settle plausibly into Newton snow: {robot_sinkage} m")
    if live_patch.max_sinkage_m <= robot_sinkage or terrain_frame.get("mode") != "live":
        raise RuntimeError("Newton deformation was not published to the terrain stream")
    if not np.isfinite(live_engine.data.qpos).all() or live_engine.data.ncon <= 0:
        raise RuntimeError("G1 lost stable MuJoCo support after Newton snow deformation")
    live_coupling = {
        "robot_sinkage_m": robot_sinkage,
        "snow_sinkage_m": live_patch.max_sinkage_m,
        "predicted_static_sinkage_m": live_patch.predicted_static_sinkage_m,
        "particle_count": live_patch.model.particle_count,
        "material_fields": actual_fields,
        "top_density_kg_m3": float(top_density.mean()),
        "terrain_resolution": terrain_frame["resolution"],
        "contacts": int(live_engine.data.ncon),
    }

    snow = SnowLayer("snow")
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    snow.emit_newton_particle_grid(
        builder,
        bounds_lo=(-0.2, -0.2, 0.0),
        bounds_hi=(0.2, 0.2, 0.08),
        voxel_size=0.08,
        particles_per_cell=1,
    )
    particle_count = int(builder.particle_count)
    particle_mass = float(sum(builder.particle_mass))
    expected_particle_mass = 0.4 * 0.4 * 0.08 * snow.material.density
    device = wp.get_device()
    model = builder.finalize(device=device)
    snow.configure_newton_particles(model)
    if particle_count <= 0:
        raise RuntimeError("Newton snow pack emitted no particles")
    if not math.isclose(particle_mass, expected_particle_mass, rel_tol=1.0e-6):
        raise RuntimeError(
            f"Newton particle mass does not preserve requested density: {particle_mass} != {expected_particle_mass}"
        )
    mpm_options = SolverImplicitMPM.Config()
    mpm_options.voxel_size = 0.08
    mpm_options.grid_type = "sparse" if device.is_cuda else "dense"
    mpm_options.max_iterations = 8
    mpm_solver = SolverImplicitMPM(model, mpm_options)
    mpm_state_0 = model.state()
    mpm_state_1 = model.state()
    mpm_solver.step(mpm_state_0, mpm_state_1, None, None, 0.005)
    if not np.isfinite(mpm_state_1.particle_q.numpy()).all():
        raise RuntimeError("Newton MPM CPU/CUDA smoke step produced non-finite particles")

    print(json.dumps({
        "policy": policy.status(),
        "action_range": [float(action.min()), float(action.max())],
        "onnx_reference_max_error": inference_error,
        "surface_friction": friction,
        "contact_pair_friction": pair_friction,
        "terrain_collider": {
            "type": "mujoco_hfield",
            "resolution": [int(engine.model.hfield_ncol[0]), int(engine.model.hfield_nrow[0])],
            "size_m": actual_size,
        },
        "rollouts": rollouts,
        "live_newton_coupling": live_coupling,
        "snow": snow.manifest(),
        "newton_device": str(device),
        "newton_mpm_particle_count": particle_count,
        "newton_mpm_particle_mass_kg": particle_mass,
        "newton_mpm_step_ready": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
