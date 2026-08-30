"""Smoke-test Newton's MuJoCo solver and the native G1 pose mirror."""

from __future__ import annotations

import json
import math

from simulation import NewtonMuJoCoBridge


def main() -> None:
    bridge = NewtonMuJoCoBridge()
    initial_x = bridge.status()["base_position"][0]

    # Exercise the exact seam that MPM collider impulses will use.
    bridge.queue_collider_impulse("pelvis", impulse=(100.0 * bridge.dt, 0.0, 0.0))
    bridge.step(steps=2)
    status = bridge.status()
    status["wrench_smoke_delta_x"] = status["base_position"][0] - initial_x

    values = status["base_position"]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Newton produced a non-finite G1 base pose: {values}")
    if status["native_pose_mirror_max_error"] > 1.0e-7:
        raise RuntimeError("Native MuJoCo mirror diverged from Newton state")
    if status["wrench_smoke_delta_x"] <= 0.0:
        raise RuntimeError("Queued Newton body wrench did not move the G1 pelvis")

    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
