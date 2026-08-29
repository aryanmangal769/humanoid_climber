"""Verify that MuJoCo Playground can construct and reset the native Unitree G1 task."""

from __future__ import annotations

import json

import jax
from mujoco_playground import locomotion

from .g1_model import validate_g1


def main() -> None:
    env = locomotion.load("G1JoystickFlatTerrain")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    result = validate_g1()
    result.update(
        observation_shapes={name: list(value.shape) for name, value in state.obs.items()},
        reward_shape=list(state.reward.shape),
        done_shape=list(state.done.shape),
        backend=jax.default_backend(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
