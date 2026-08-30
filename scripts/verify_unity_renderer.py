"""Validate the Unity renderer handoff against a live Everest backend."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import trimesh
import websockets


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = ROOT / "studio/unity/EverestSim"
RESOURCES = UNITY_ROOT / "Assets/Resources/G1"
G1_SOURCE = ROOT / "vendor/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/assets"
PORT = int(os.environ.get("EVEREST_BACKEND_PORT", "18765"))
URL = os.environ.get("EVEREST_BACKEND_URL", f"ws://127.0.0.1:{PORT}")


async def receive_initial() -> dict[str, dict]:
    wanted = {"scene", "terrain", "macro_terrain", "frame", "snow", "state", "environment"}
    found: dict[str, dict] = {}
    async with websockets.connect(URL, max_size=16 * 1024 * 1024, compression=None) as ws:
        while wanted - found.keys():
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
            kind = message.get("type")
            if kind in wanted:
                found[kind] = message["data"]
    return found


def validate(messages: dict[str, dict]) -> None:
    scene = messages["scene"]
    terrain = messages["terrain"]
    macro = messages["macro_terrain"]
    frame = messages["frame"]
    snow = messages["snow"]
    state = messages["state"]

    assert scene["schema"] == "everest-scene/v1"
    assert scene["up_axis"] == "z"
    assert scene["handedness"] == "right"
    assert scene["quaternion_order"] == "wxyz"
    assert len(scene["body_names"]) == len(frame["body_names"])
    assert len(frame["body_names"]) == len(frame["body_pos_w"]) == len(frame["body_quat_w"])

    missing = []
    for visual in scene["visuals"]:
        stem = Path(visual["asset"]).stem
        if not (RESOURCES / f"{stem}.obj").is_file():
            missing.append(stem)
    assert not missing, f"Unity G1 resources missing: {missing}"

    # Unity's OBJ importer reflects X. The generated resource must therefore
    # contain (-x, z, y), or Unity renders asymmetric links and lettering as a
    # mirror image relative to the authoritative body transforms.
    source = trimesh.load(G1_SOURCE / "logo_link.STL", force="mesh", process=False)
    converted = trimesh.load(RESOURCES / "logo_link.obj", force="mesh", process=False)
    source_bounds = source.bounds
    converted_bounds = converted.bounds
    expected_min = [-source_bounds[1, 0], source_bounds[0, 2], source_bounds[0, 1]]
    expected_max = [-source_bounds[0, 0], source_bounds[1, 2], source_bounds[1, 1]]
    assert np.allclose(converted_bounds[0], expected_min, atol=1e-6), (
        "G1 OBJ minimum bounds do not compensate for Unity's X reflection: "
        f"{converted_bounds[0]} != {expected_min}"
    )
    assert np.allclose(converted_bounds[1], expected_max, atol=1e-6), (
        "G1 OBJ maximum bounds do not compensate for Unity's X reflection: "
        f"{converted_bounds[1]} != {expected_max}"
    )

    assert terrain["grid_width"] * terrain["grid_height"] == len(terrain["heights"])
    assert terrain["grid_width"] == 257 and terrain["grid_height"] == 257
    assert macro["grid_width"] * macro["grid_height"] == len(macro["heights"])

    nx, ny = snow["resolution"]
    assert nx * ny == len(snow["heights"])
    assert nx * ny == len(snow["vertices"])
    assert all(len(vertex) == 3 for vertex in snow["vertices"])
    assert nx * ny == len(snow["base_heights"])
    assert nx * ny == len(snow["compaction"])
    assert nx * ny == len(snow["material_ids"])
    assert 1 <= len(snow["layers"]) <= 6
    assert len(snow["layer_heights"]) == len(snow["layers"])
    assert all(nx * ny == len(values) for values in snow["layer_heights"])
    assert len(snow["layer_vertices"]) == len(snow["layers"])
    assert all(nx * ny == len(values) for values in snow["layer_vertices"])
    assert nx * ny == len(snow["substrate_vertices"])
    assert state["simulation_fault"] is None

    scripts = UNITY_ROOT / "Assets/Scripts"
    shaders = UNITY_ROOT / "Assets/Shaders"
    required_scripts = {
        "EverestRuntime.cs",
        "EverestBackendClient.cs",
        "EverestCoordinates.cs",
        "EverestRobotRenderer.cs",
        "EverestTerrainRenderer.cs",
        "EverestSnowRenderer.cs",
        "EverestEnvironmentRenderer.cs",
        "EverestCameraController.cs",
        "EverestHud.cs",
    }
    absent = sorted(name for name in required_scripts if not (scripts / name).is_file())
    assert not absent, f"Unity scripts missing: {absent}"
    required_shaders = {
        "EverestSnow.shader": 'Shader "Everest/Snow"',
        "EverestTerrain.shader": 'Shader "Everest/Terrain"',
        "EverestWireTerrain.shader": 'Shader "Everest/WireTerrain"',
        "EverestVolumetricClouds.shader": 'Shader "Everest/VolumetricClouds"',
        "EverestSnowfall.shader": 'Shader "Everest/Snowfall"',
        "EverestSnowLayers.shader": 'Shader "Everest/SnowLayers"',
    }
    for filename, declaration in required_shaders.items():
        path = shaders / filename
        assert path.is_file(), f"Unity shader missing: {filename}"
        assert declaration in path.read_text(), f"Shader declaration mismatch: {filename}"

    print(
        "Unity renderer handoff OK: "
        f"{len(scene['body_names'])} bodies, {len(scene['visuals'])} visuals, "
        f"terrain {terrain['grid_width']}x{terrain['grid_height']}, "
        f"macro {macro['grid_width']}x{macro['grid_height']}, "
        f"snow {nx}x{ny}/{len(snow['layers'])} layers, "
        f"Newton {state['newton']['device']}"
    )


async def main() -> None:
    validate(await receive_initial())


if __name__ == "__main__":
    asyncio.run(main())
