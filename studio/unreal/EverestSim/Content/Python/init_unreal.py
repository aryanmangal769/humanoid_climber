"""One-time editor bootstrap for G1 meshes and simple PBR materials."""
from __future__ import annotations
import pathlib
import unreal

PROJECT = pathlib.Path(unreal.Paths.project_dir()).resolve()
ROOT = PROJECT.parents[2]
G1_ASSETS = PROJECT / "SourceData/G1"
DEST = "/Game/Robots/G1/Meshes"
MAT_DEST = "/Game/Everest/Materials"


def import_g1() -> None:
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    tasks = []
    for path in sorted(G1_ASSETS.glob("*.obj")):
        asset_path = f"{DEST}/{path.stem}"
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            continue
        task = unreal.AssetImportTask()
        task.filename = str(path)
        task.destination_path = DEST
        task.destination_name = path.stem
        task.automated = True
        task.replace_existing = False
        task.save = True
        tasks.append(task)
    if tasks:
        tools.import_asset_tasks(tasks)
        unreal.log(f"Everest: imported {len(tasks)} Unitree G1 meshes")


def material(name: str, base, roughness: float, metallic: float = 0.0, vertex_color: bool = False):
    path = f"{MAT_DEST}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, MAT_DEST, unreal.Material, unreal.MaterialFactoryNew())
    if vertex_color:
        color = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionVertexColor, -400, -80)
        unreal.MaterialEditingLibrary.connect_material_property(color, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        color = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -400, -80)
        color.set_editor_property("constant", unreal.LinearColor(*base, 1.0))
        unreal.MaterialEditingLibrary.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    r = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionConstant, -400, 80)
    r.set_editor_property("r", roughness)
    unreal.MaterialEditingLibrary.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
    m = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionConstant, -400, 160)
    m.set_editor_property("r", metallic)
    unreal.MaterialEditingLibrary.connect_material_property(m, "", unreal.MaterialProperty.MP_METALLIC)
    unreal.MaterialEditingLibrary.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def main() -> None:
    import_g1()
    material("M_EverestTerrain", (0.2, 0.24, 0.28), 0.82, vertex_color=True)
    material("M_NewtonSnow", (0.92, 0.96, 1.0), 0.62, vertex_color=True)
    material("M_G1Robot", (0.11, 0.13, 0.15), 0.28, metallic=0.72)
    level_path = "/Game/Everest/Maps/Everest"
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not unreal.EditorAssetLibrary.does_asset_exist(level_path):
        levels.new_level(level_path, False)
    else:
        levels.load_level(level_path)
    unreal.log("Everest Unreal bootstrap ready: MuJoCo robot / Newton snow / DEM renderer")

try:
    main()
except Exception as exc:
    unreal.log_error(f"Everest Unreal bootstrap failed: {exc}")
