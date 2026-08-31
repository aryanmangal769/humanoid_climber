#!/usr/bin/env python3
"""Verify the reactive weather-driven autonomous demo contract."""

from __future__ import annotations

import numpy as np

from dashboard.engines.mujoco import MuJoCoEngine


def main() -> None:
    engine = MuJoCoEngine(
        telemetry_hz=60.0,
        enable_newton=False,
        demo="autonomous-showcase",
    )
    stages: list[str] = []
    failure_xy: np.ndarray | None = None
    try:
        assert engine.state()["demo"]["stage"] == "journey"
        assert engine.state()["demo"]["wind_force_n"] == 0.0
        safe = dict(zip(
            engine._policy.joint_names,
            engine._four_point_safety_target(aggressive=False),
        ))
        assert safe["left_shoulder_pitch_joint"] == -1.55
        assert safe["left_shoulder_roll_joint"] == 0.60
        assert safe["right_shoulder_roll_joint"] == -0.60
        assert safe["left_elbow_joint"] == 0.65
        assert engine._safety_pose_attack_seconds == 0.08
        assert engine._safety_pose_settle_seconds == 0.40

        # Match Unity's weather path: a rapid 108 N physical crosswind change.
        engine.control("weather", {
            "schema": "everest-weather/v1",
            "conditions": {"wind_direction_deg": 90.0},
            "simulation": {
                "wind_force_scale": 0.90,
                "terrain_friction_scale": 1.0,
                "visibility_scale": 1.0,
                "movement_allowed": True,
            },
        })

        for _ in range(2500):
            engine._advance_to(float(engine.data.time) + engine.period)
            state = engine.state()
            stage = state["demo"]["stage"]
            if not stages or stages[-1] != stage:
                stages.append(stage)
            if stage == "safety_hold" and failure_xy is None:
                failure_xy = np.asarray(engine._demo_failure_xy).copy()
                assert state["demo"]["training_view_active"]
                assert not state["paused"]
            if stage == "training_attempt_2":
                engine.control("demo_skip_phase")
                break

        assert stages == [
            "journey", "safety_hold", "training_attempt_1", "training_attempt_2",
        ], stages
        assert failure_xy is not None
        state = engine.state()
        assert state["demo"]["stage"] == "journey_adapted"
        assert state["demo"]["recovered_once"]
        assert not state["demo"]["training_view_active"]
        assert np.linalg.norm(engine.data.qpos[:2] - failure_xy) < 1.0e-6
        assert engine._policy_supervisor.active_policy_key == "wind"

        engine.control("demo_stop", True)
        stopped_time = float(engine.data.time)
        engine._advance_to(stopped_time + 0.20)
        assert engine.state()["demo"]["operator_stopped"]
        assert engine.state()["policy"]["command"] == (0.0, 0.0, 0.0)
        assert float(engine.data.time) > stopped_time  # physics stays live
        engine.control("demo_stop", False)
        assert not engine.state()["demo"]["operator_stopped"]
    finally:
        engine.stop()

    print("autonomous-demo: PASS")
    print("stages:", " -> ".join(stages), "-> journey_adapted")
    print("reactive wind failure + same-place resume + live STOP: PASS")


if __name__ == "__main__":
    main()
