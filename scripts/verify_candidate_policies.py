#!/usr/bin/env python3
"""Verify external MjLab 1.6 velocity candidates in the Everest main engine."""

from __future__ import annotations

import numpy as np

from dashboard.engines.mujoco import MuJoCoEngine


def main() -> None:
    cases = (
        ("flat_mjlab_1_6", None),
        ("ice_incline", 0.12),
        ("wind", None),
    )
    results = []
    for key, friction in cases:
        engine = MuJoCoEngine(enable_newton=False, telemetry_hz=60.0)
        try:
            registry = {item["key"]: item for item in engine.state()["policy"]["registry"]}
            candidate = registry[key]
            assert candidate["status"] == "candidate_available", candidate
            assert candidate["input_size"] == 99
            assert candidate["action_size"] == 29
            engine.control("policy_select", key)
            if friction is not None:
                engine.control("surface", "ice")
                engine.control("surface_friction", friction)
            if key == "wind":
                engine.control("weather", {
                    "schema": "everest-weather/v1",
                    "conditions": {"wind_direction_deg": 0.0},
                    "simulation": {
                        "wind_force_scale": 0.25,
                        "terrain_friction_scale": 1.0,
                        "visibility_scale": 1.0,
                        "movement_allowed": True,
                    },
                })
            engine.control("command", [0.05, 0.0, 0.0])
            engine.control("pause", False)
            for _ in range(45):
                engine._advance_to(float(engine.data.time) + engine.period)
            state = engine.state()
            assert state["simulation_fault"] is None
            assert np.isfinite(engine.data.qpos).all()
            assert state["policy"]["inference_count"] > 0
            results.append((key, state["policy"]["inference_count"]))
        finally:
            engine.stop()

    recovery = registry["recovery"]
    assert recovery["status"] == "incompatible_160_observation"
    assert recovery["input_size"] == 160
    print("candidate-policies: PASS")
    for key, inferences in results:
        print(f"  {key}: 99 -> 29, {inferences} main-engine inferences")
    print("  recovery: correctly held incompatible at 160 -> 29")


if __name__ == "__main__":
    main()
