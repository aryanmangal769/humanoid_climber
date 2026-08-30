"""Focused Newton 1.5 snow-state and 3D deformation regression."""

from __future__ import annotations

import json

import numpy as np

from simulation.newton_snow import FootPose, NewtonSnowPatch
from simulation.snow import SnowColumn, SnowColumnLayer


def _pose(name: str, x: float, z: float) -> FootPose:
    return FootPose(
        name=name,
        position=(x, 0.0, z),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def main() -> None:
    column = SnowColumn(
        surface_friction=0.32,
        layers=(
            SnowColumnLayer(
                type="POWDER",
                label="Powder",
                color=(0.94, 0.97, 1.0),
                thickness_m=0.10,
                density_kg_m3=140.0,
                stiffness_pa=60_000.0,
                compressive_strength_pa=4_000.0,
                shear_strength_pa=1_500.0,
                compaction_hardening=6.0,
                bond_strength_below_pa=800.0,
            ),
            SnowColumnLayer(
                type="DENSE_SNOW",
                label="Dense snow",
                color=(0.72, 0.82, 0.90),
                thickness_m=0.20,
                density_kg_m3=360.0,
                stiffness_pa=600_000.0,
                compressive_strength_pa=70_000.0,
                shear_strength_pa=25_000.0,
                compaction_hardening=18.0,
                bond_strength_below_pa=6_000.0,
            ),
        ),
    )
    airborne = (
        _pose(NewtonSnowPatch.FOOT_NAMES[0], -0.15, 0.35),
        _pose(NewtonSnowPatch.FOOT_NAMES[1], 0.15, 0.35),
    )
    patch = NewtonSnowPatch(
        column,
        lambda _x, _y: 0.0,
        airborne,
        size_xy=(1.2, 1.0),
        voxel_size=0.08,
        dt=0.05,
        contact_refine_radius=0.32,
        coarse_stride=3,
        accumulation_enabled=False,
    )

    q_before = patch.state.particle_q.numpy().copy()
    qd_before = patch.state.particle_qd.numpy().copy()
    grad_before = patch.state.mpm.particle_qd_grad.numpy().copy()
    jp_before = patch.state.mpm.particle_Jp.numpy().copy()
    stress_before = patch.state.mpm.particle_stress.numpy().copy()
    no_contact = patch.step(airborne)
    if no_contact:
        raise RuntimeError(f"Airborne feet produced Newton reactions: {no_contact}")
    state_errors = {
        "position": float(np.max(np.abs(patch.state.particle_q.numpy() - q_before))),
        "velocity": float(np.max(np.abs(patch.state.particle_qd.numpy() - qd_before))),
        "velocity_gradient": float(np.max(np.abs(patch.state.mpm.particle_qd_grad.numpy() - grad_before))),
        "plastic_volume": float(np.max(np.abs(patch.state.mpm.particle_Jp.numpy() - jp_before))),
        "stress": float(np.max(np.abs(patch.state.mpm.particle_stress.numpy() - stress_before))),
    }
    if max(state_errors.values()) > 1.0e-7:
        raise RuntimeError(f"No-contact solve changed snow state: {state_errors}")

    frame_before = patch.terrain_frame(include_particles=False)
    contact = (
        _pose(NewtonSnowPatch.FOOT_NAMES[0], -0.15, 0.018),
        _pose(NewtonSnowPatch.FOOT_NAMES[1], 0.15, 0.018),
    )
    reactions = patch.step(contact)
    if not reactions:
        raise RuntimeError("Penetrating soles produced no Newton reaction impulse")
    frame_after = patch.terrain_frame(include_particles=False)
    vertices_before = np.asarray(frame_before["vertices"], dtype=np.float32)
    vertices_after = np.asarray(frame_after["vertices"], dtype=np.float32)
    displacement = vertices_after - vertices_before
    max_lateral = float(np.linalg.norm(displacement[:, :2], axis=1).max(initial=0.0))
    max_vertical = float(np.abs(displacement[:, 2]).max(initial=0.0))
    if max_lateral <= 1.0e-5 or max_vertical <= 1.0e-5:
        raise RuntimeError(
            f"3D stream discarded Newton displacement: lateral={max_lateral}, vertical={max_vertical}"
        )
    if not np.allclose(vertices_after[:, 2], frame_after["heights"], atol=1.0e-6):
        raise RuntimeError("Streamed XYZ vertices and MuJoCo support heights diverged")
    if len(frame_after["layer_vertices"]) != len(column.layers):
        raise RuntimeError("Not every mechanical layer has a live XYZ boundary")
    if len(frame_after["substrate_vertices"]) != len(frame_after["heights"]):
        raise RuntimeError("Snow volume substrate topology does not match the active surface")
    boundaries = [
        np.asarray(values, dtype=np.float32)
        for values in frame_after["layer_vertices"]
    ] + [np.asarray(frame_after["substrate_vertices"], dtype=np.float32)]
    minimum_layer_thickness = min(
        float(np.min(boundaries[layer][:, 2] - boundaries[layer + 1][:, 2]))
        for layer in range(len(column.layers))
    )
    if minimum_layer_thickness < -1.0e-6:
        raise RuntimeError(f"Projected multilayer volume inverted: {minimum_layer_thickness} m")

    expected_mass = patch.simulated_area_m2 * sum(
        layer.density_kg_m3 * layer.thickness_m for layer in column.layers
    )
    if not np.isclose(patch.total_mass_kg, expected_mass, rtol=1.0e-5):
        raise RuntimeError(f"Active-domain mass mismatch: {patch.total_mass_kg} != {expected_mass}")
    per_layer_radius_spread = {}
    radii = patch.model.particle_radius.numpy()
    for layer_id in range(len(column.layers)):
        selected = radii[patch.particle_material_ids == layer_id]
        spread = float(selected.max(initial=0.0) - selected.min(initial=np.inf))
        per_layer_radius_spread[str(layer_id)] = spread
        if spread > 1.0e-7:
            raise RuntimeError(f"Layer {layer_id} still contains inflated coarse particles: {spread}")

    history: dict = {}
    patch.export_deformation_history(history)
    restored_patch = NewtonSnowPatch(
        column,
        lambda _x, _y: 0.0,
        airborne,
        size_xy=(1.2, 1.0),
        voxel_size=0.08,
        dt=0.05,
        contact_refine_radius=0.32,
        coarse_stride=3,
        accumulation_enabled=False,
    )
    restored_count = restored_patch.import_deformation_history(history)
    if restored_count <= 0:
        raise RuntimeError("Moving-window rebuild restored no deformed particles")
    if float(np.linalg.norm(restored_patch.state.mpm.particle_stress.numpy())) <= 1.0e-6:
        raise RuntimeError("Moving-window rebuild discarded Newton constitutive stress")

    print(json.dumps({
        "device": str(patch.device),
        "particles": int(patch.model.particle_count),
        "simulated_area_m2": patch.simulated_area_m2,
        "mass_kg": patch.total_mass_kg,
        "no_contact_state_error": state_errors,
        "contact_bodies": sorted(reactions),
        "max_lateral_deformation_m": max_lateral,
        "max_vertical_deformation_m": max_vertical,
        "layer_radius_spread_m": per_layer_radius_spread,
        "history_records": len(history),
        "history_restored_particles": restored_count,
        "minimum_layer_thickness_m": minimum_layer_thickness,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
