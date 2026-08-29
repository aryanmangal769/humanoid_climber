"""Canonical Unitree RL MjLab G1 model metadata."""

from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G1_XML = ROOT / "vendor" / "unitree_rl_mjlab" / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml"


@lru_cache(maxsize=1)
def validate_g1() -> dict[str, object]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    return {
        "xml": str(G1_XML),
        "mujoco": mujoco.__version__,
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "environment": "Unitree-G1-Flat",
        "source": "unitreerobotics/unitree_rl_mjlab",
        "source_revision": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
    }
