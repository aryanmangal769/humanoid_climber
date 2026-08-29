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
            "capabilities": ["frames", "state", "control"],
            "source": "unitree_rl_mjlab",
        },
        {
            "id": "newton",
            "label": "Newton",
            "available": (ROOT / "vendor" / "newton" / "newton").is_dir(),
            "active": False,
            "capabilities": ["planned:frames", "planned:state", "planned:control"],
            "source": "newton-physics/newton",
        },
    ]
