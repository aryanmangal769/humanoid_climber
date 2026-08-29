"""Small, explicit G1 model helpers used by the dashboard checks."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = ROOT / "vendor" / "mujoco_playground"
G1_XML = (
    PLAYGROUND
    / "mujoco_playground"
    / "_src"
    / "locomotion"
    / "g1"
    / "xmls"
    / "scene_mjx_feetonly_flat_terrain.xml"
)


def validate_g1() -> dict:
    """Load the native Playground G1 environment and return useful metadata."""
    if not G1_XML.is_file():
        raise FileNotFoundError(f"G1 XML is missing: {G1_XML}")
    try:
        import mujoco
    except ImportError as exc:  # dashboard can run without the sim installed
        return {"xml": str(G1_XML), "mujoco": None, "error": str(exc)}

    try:
        from mujoco_playground import locomotion
        env = locomotion.load("G1JoystickFlatTerrain")
        model = env.mj_model
    except Exception as exc:
        return {"xml": str(G1_XML), "mujoco": mujoco.__version__, "error": str(exc)}
    return {
        "xml": str(G1_XML),
        "mujoco": mujoco.__version__,
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "environment": "G1JoystickFlatTerrain",
    }
