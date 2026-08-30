"""Smoke-test the experimental direct static G1 <-> Newton MPM coupling."""

from __future__ import annotations

import json

from simulation.newton_mujoco import NewtonMuJoCoBridge
from simulation.snow import SnowColumn, SnowColumnLayer
from simulation.static_snow_coupling import StaticSnowTwoWayCoupler


def main() -> None:
    column = SnowColumn(
        surface_friction=0.28,
        layers=(
            SnowColumnLayer(
                type="POWDER",
                label="Soft static powder",
                color=(0.94, 0.97, 1.0),
                thickness_m=0.20,
                density_kg_m3=100.0,
                stiffness_pa=30_000.0,
                compressive_strength_pa=2_000.0,
                shear_strength_pa=500.0,
                compaction_hardening=2.0,
                bond_strength_below_pa=200.0,
            ),
        ),
    )
    bridge = NewtonMuJoCoBridge(dt=0.002)
    coupler = StaticSnowTwoWayCoupler(
        bridge,
        column,
        size_xy=(1.4, 1.0),
        voxel_size=0.05,
        dt=0.01,
    )
    coupler.lift_robot_to_surface(clearance_m=0.01)
    samples = []
    for step in range(30):
        stats = coupler.step(rigid_steps=5)
        if step in {0, 4, 9, 19, 29}:
            samples.append({
                "step": step + 1,
                "surface_drop_m": stats.max_surface_drop_m,
                "coupled_bodies": stats.coupled_bodies,
                "total_impulse_ns": stats.total_impulse_ns,
                "base_z_m": bridge.status()["base_position"][2],
            })
    print(json.dumps({"coupler": coupler.status(), "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
