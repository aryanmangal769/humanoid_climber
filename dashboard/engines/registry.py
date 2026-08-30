"""Engine discovery metadata; adapters share ``ViewerEngine`` from protocol.py."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def catalog() -> list[dict[str, object]]:
    """Return capabilities without importing optional physics runtimes."""
    return [
        {
            "id": "mujoco",
            "label": "MuJoCo",
            "available": True,
            "active": True,
            "capabilities": ["pose-telemetry", "scene-manifest", "state", "control"],
            "source": "mujoco_menagerie/unitree_g1 + unitree_rl_mjlab policy",
        },
        {
            "id": "newton",
            "label": "Newton + MuJoCo",
            "available": (ROOT / "vendor" / "newton" / "newton").is_dir(),
            "active": False,
            "capabilities": ["pose-telemetry", "coupled-step", "body-wrench-boundary"],
            "source": "newton-physics/newton / SolverMuJoCo",
        },
    ]
