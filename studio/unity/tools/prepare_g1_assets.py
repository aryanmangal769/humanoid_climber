"""Convert canonical Unitree G1 STL visuals into Unity-ready OBJ resources.

The backend and STL assets are right-handed/Z-up. Unity is left-handed/Y-up,
and Unity's OBJ importer itself reflects the OBJ X coordinate. We therefore
bake the proper rotation ``(x, y, z) -> (-x, z, y)`` into the OBJ. Unity's
implicit X reflection then produces the intended runtime mesh coordinates
``(x, z, y)``, exactly matching ``EverestCoordinates``.

The baked transform has positive determinant, so triangle winding remains
unchanged. Unity handles the winding change associated with its own OBJ
handedness conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vendor/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/assets"
DEST = ROOT / "studio/unity/EverestSim/Assets/Resources/G1"


def convert(path: Path) -> Path:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a triangle mesh")

    vertices = np.asarray(loaded.vertices, dtype=np.float64).copy()
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 0] *= -1.0
    faces = np.asarray(loaded.faces, dtype=np.int64).copy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    target = DEST / f"{path.stem}.obj"
    target.write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=True))
    return target


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    DEST.mkdir(parents=True, exist_ok=True)

    converted = []
    for source in sorted(SOURCE.glob("*.STL")):
        converted.append(convert(source))

    # Keep the runtime resource folder deterministic when upstream changes.
    expected = {path.name for path in converted}
    for old in DEST.glob("*.obj"):
        if old.name not in expected:
            old.unlink()

    print(f"Converted {len(converted)} Unitree G1 meshes -> {DEST}")


if __name__ == "__main__":
    main()
