#!/usr/bin/env python3
"""Stress the active four-point posture across representative fall directions."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from dashboard.engines.mujoco import MuJoCoEngine


def quaternion_wxyz(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def main() -> None:
    cases = (
        ("upright", 0.0, 0.0, 0.0, 0.0),
        ("left_roll", 0.35, 0.0, 0.60, 0.0),
        ("right_roll", -0.35, 0.0, -0.60, 0.0),
        ("forward_pitch", 0.0, 0.35, 0.0, 0.60),
        ("back_pitch", 0.0, -0.35, 0.0, -0.60),
    )
    results = []
    for name, roll, pitch, roll_rate, pitch_rate in cases:
        engine = MuJoCoEngine()
        try:
            engine.data.qpos[3:7] = quaternion_wxyz(roll, pitch)
            engine.data.qvel[3:6] = (roll_rate, pitch_rate, 0.0)
            mujoco.mj_forward(engine.model, engine.data)
            target = engine._four_point_safety_target(aggressive=False)
            initial_error = float(np.linalg.norm(engine.data.qpos[7:] - target))
            engine.control("demo_failure")
            engine._advance_to(float(engine.data.time) + 1.0)
            state = engine.state()
            final_error = float(np.linalg.norm(engine.data.qpos[7:] - target))
            safety = state["simulation_settings"]["safety_pose"]
            assert np.isfinite(engine.data.qpos).all(), name
            assert not state["paused"], name
            assert safety["active"] and safety["physics_live"], name
            assert safety["transition_progress"] >= 1.0, name
            assert final_error < initial_error * 0.50, (name, initial_error, final_error)
            assert safety["support_bodies"], name
            engine._advance_to(float(engine.data.time) + 3.0)
            long_state = engine.state()
            long_safety = long_state["simulation_settings"]["safety_pose"]
            assert np.isfinite(engine.data.qpos).all(), f"{name} long hold"
            assert not long_state["paused"], f"{name} long hold"
            assert long_safety["phase"] == "contact_hold", name
            assert long_safety["support_bodies"], f"{name} long hold"
            results.append((name, final_error, long_safety["support_bodies"]))
        finally:
            engine.stop()
    for name, error, supports in results:
        print(f"PASS {name}: target_error={error:.3f}, support={','.join(supports)}")


if __name__ == "__main__":
    main()
