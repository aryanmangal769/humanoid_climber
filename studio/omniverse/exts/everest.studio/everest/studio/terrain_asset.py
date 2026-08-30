from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _resample_crop(
    heights: np.ndarray,
    crop: Sequence[float],
) -> np.ndarray:
    """Crop a height grid using normalized [x0, x1, y0, y1] bounds."""
    x0, x1, y0, y1 = (float(value) for value in crop)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("crop must satisfy 0 <= min < max <= 1 on both axes")
    rows, columns = heights.shape
    col0 = int(round(x0 * (columns - 1)))
    col1 = int(round(x1 * (columns - 1))) + 1
    row0 = int(round(y0 * (rows - 1)))
    row1 = int(round(y1 * (rows - 1))) + 1
    cropped = heights[row0:row1, col0:col1]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        raise ValueError("terrain crop is too small")
    return cropped


def export_terrain_usd(
    source_json: str | Path,
    output_usd: str | Path,
    *,
    crop: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
    scale_xy: float = 1.0,
    scale_z: float = 1.0,
    snow_depth_m: float = 0.08,
) -> Path:
    """Export the Everest height artifact as a real-metre USD triangle mesh."""
    source_json = Path(source_json)
    output_usd = Path(output_usd)
    payload = json.loads(source_json.read_text())
    width = int(payload["grid_width"])
    depth = int(payload["grid_height"])
    source = np.asarray(payload["heights"], dtype=np.float64).reshape(depth, width)
    heights = _resample_crop(source, crop)
    rows, columns = heights.shape

    x0, x1, y0, y1 = (float(value) for value in crop)
    world_width = float(payload["world_width_m"]) * (x1 - x0) * float(scale_xy)
    world_depth = float(payload["world_depth_m"]) * (y1 - y0) * float(scale_xy)
    center = np.asarray(payload.get("terrain_center", [0.0, 0.0, 0.0]), dtype=np.float64)
    source_relief = float(heights.max() - heights.min())
    z0 = float(heights.min())
    # Preserve the source minimum elevation so scale=1 is exactly the same
    # metre-space artifact used by the simulation. Vertical exaggeration grows
    # relief upward from that fixed base instead of silently translating it.
    scaled_heights = z0 + (heights - z0) * float(scale_z)

    xs = np.linspace(-world_width / 2.0, world_width / 2.0, columns)
    ys = np.linspace(-world_depth / 2.0, world_depth / 2.0, rows)
    points = [
        Gf.Vec3f(float(x), float(y), float(center[2] + scaled_heights[row, col]))
        for row, y in enumerate(ys)
        for col, x in enumerate(xs)
    ]

    face_counts: list[int] = []
    face_indices: list[int] = []
    for row in range(rows - 1):
        for col in range(columns - 1):
            a = row * columns + col
            b = a + 1
            c = a + columns
            d = c + 1
            face_counts.extend((3, 3))
            face_indices.extend((a, b, c, b, d, c))

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/EverestTerrain")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/EverestTerrain/Mesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    material = UsdShade.Material.Define(stage, "/EverestTerrain/Looks/RockSnow")
    shader = UsdShade.Shader.Define(stage, "/EverestTerrain/Looks/RockSnow/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.42, 0.46, 0.50))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    snow_points = [Gf.Vec3f(point[0], point[1], point[2] + float(snow_depth_m)) for point in points]
    snow_mesh = UsdGeom.Mesh.Define(stage, "/EverestTerrain/SnowSurface")
    snow_mesh.CreatePointsAttr(snow_points)
    snow_mesh.CreateFaceVertexCountsAttr(face_counts)
    snow_mesh.CreateFaceVertexIndicesAttr(face_indices)
    snow_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    snow_mesh.CreateDoubleSidedAttr(False)
    snow_material = UsdShade.Material.Define(stage, "/EverestTerrain/Looks/Snow")
    snow_shader = UsdShade.Shader.Define(stage, "/EverestTerrain/Looks/Snow/PreviewSurface")
    snow_shader.CreateIdAttr("UsdPreviewSurface")
    snow_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.95, 0.98))
    snow_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.76)
    snow_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    snow_material.CreateSurfaceOutput().ConnectToSource(snow_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(snow_mesh.GetPrim()).Bind(snow_material)
    snow_mesh.GetPrim().CreateAttribute("everest:visualSnowDepthM", Sdf.ValueTypeNames.Float).Set(float(snow_depth_m))

    prim = root.GetPrim()
    prim.CreateAttribute("everest:source", Sdf.ValueTypeNames.String).Set(str(source_json))
    prim.CreateAttribute("everest:crop", Sdf.ValueTypeNames.Float4).Set(tuple(float(v) for v in crop))
    prim.CreateAttribute("everest:scaleXY", Sdf.ValueTypeNames.Float).Set(float(scale_xy))
    prim.CreateAttribute("everest:scaleZ", Sdf.ValueTypeNames.Float).Set(float(scale_z))
    prim.CreateAttribute("everest:sourceReliefM", Sdf.ValueTypeNames.Float).Set(source_relief)
    stage.GetRootLayer().Save()
    return output_usd
