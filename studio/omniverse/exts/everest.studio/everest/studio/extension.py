from __future__ import annotations

from pathlib import Path

import omni.ext
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdLux

from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
from isaacsim.physics.newton import get_active_physics_engine

from .terrain_asset import export_terrain_usd


class EverestStudioExtension(omni.ext.IExt):
    """Everest-specific tooling layered onto Kit's native Base Editor UI."""

    def on_startup(self, ext_id: str) -> None:
        manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_path = Path(manager.get_extension_path(ext_id)).resolve()
        self._repo = self._ext_path.parents[3]
        self._assets = self._repo / "studio/omniverse/assets"
        self._terrain_json = self._repo / "maps/everest_terrain.json"
        self._terrain_usd = self._assets / "everest_terrain.usda"
        self._g1_mjcf = self._repo / "vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/g1_mjx.xml"
        self._g1_usd = self._find_existing_g1_usd()
        self._scene_usd = self._assets / "everest_scene.usda"
        self._assets.mkdir(parents=True, exist_ok=True)

        self._models = {
            "crop_x0": ui.SimpleFloatModel(0.0),
            "crop_x1": ui.SimpleFloatModel(1.0),
            "crop_y0": ui.SimpleFloatModel(0.0),
            "crop_y1": ui.SimpleFloatModel(1.0),
            "scale_xy": ui.SimpleFloatModel(1.0),
            "scale_z": ui.SimpleFloatModel(1.0),
            "depth": ui.SimpleFloatModel(0.40),
            "density": ui.SimpleFloatModel(120.0),
            "stiffness": ui.SimpleFloatModel(60000.0),
            "compression": ui.SimpleFloatModel(4000.0),
            "shear": ui.SimpleFloatModel(1200.0),
            "hardening": ui.SimpleFloatModel(4.0),
            "friction": ui.SimpleFloatModel(0.35),
        }
        self._status_text = "Ready"
        self._status_label = None
        self._window = ui.Window("Everest", width=360, height=720)
        self._window.frame.set_build_fn(self._build_ui)

        if not self._terrain_usd.exists():
            self._export_terrain()
        # Open a terrain-first stage immediately. Robot import is intentionally
        # deferred because the first MJCF -> USD conversion can be expensive.
        self._build_scene(import_robot=True)

    def on_shutdown(self) -> None:
        if getattr(self, "_window", None):
            self._window.destroy()
            self._window = None

    def _build_ui(self) -> None:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8, height=0):
                ui.Label("EVEREST STUDIO", style={"font_size": 18})
                ui.Label("Native USD · RTX · Newton", style={"color": 0xFF8C8C8C})
                ui.Separator()
                ui.Button("Build / Open Scene", height=30, clicked_fn=self._build_scene)
                self._status_label = ui.Label(self._status_text, word_wrap=True)

                self._section("Terrain")
                self._float_row("Crop X min", "crop_x0", 0.0, 0.99)
                self._float_row("Crop X max", "crop_x1", 0.01, 1.0)
                self._float_row("Crop Y min", "crop_y0", 0.0, 0.99)
                self._float_row("Crop Y max", "crop_y1", 0.01, 1.0)
                self._float_row("Physical XY scale", "scale_xy", 0.05, 20.0)
                self._float_row("Physical Z scale", "scale_z", 0.05, 20.0)
                ui.Button("Rebuild Terrain USD", height=28, clicked_fn=self._rebuild_terrain)

                self._section("Static snow snapshot")
                ui.Label("No accumulation. These values describe the initial pack.", word_wrap=True)
                self._float_row("Depth (m)", "depth", 0.01, 2.5)
                self._float_row("Density (kg/m³)", "density", 30.0, 950.0)
                self._float_row("Young modulus (Pa)", "stiffness", 1000.0, 1.0e8)
                self._float_row("Compression yield (Pa)", "compression", 100.0, 1.0e7)
                self._float_row("Shear yield (Pa)", "shear", 50.0, 5.0e6)
                self._float_row("Hardening", "hardening", 0.0, 100.0)
                self._float_row("Foot friction", "friction", 0.01, 1.5)
                ui.Button("Apply Snow Parameters", height=28, clicked_fn=self._apply_snow_metadata)

                self._section("Simulation")
                ui.Label(f"Physics backend: {get_active_physics_engine()}")
                ui.Label("Newton two-way MPM integration replaces the legacy heightfield shim.", word_wrap=True)

    def _section(self, title: str) -> None:
        ui.Spacer(height=5)
        ui.Separator()
        ui.Label(title, style={"font_size": 15})

    def _set_status(self, value: str) -> None:
        self._status_text = value
        if self._status_label is not None:
            self._status_label.text = value

    def _float_row(self, label: str, key: str, minimum: float, maximum: float) -> None:
        with ui.HStack(height=24, spacing=8):
            ui.Label(label, width=180)
            ui.FloatDrag(self._models[key], min=minimum, max=maximum, step=0.01)

    def _crop(self) -> tuple[float, float, float, float]:
        return (
            self._models["crop_x0"].as_float,
            self._models["crop_x1"].as_float,
            self._models["crop_y0"].as_float,
            self._models["crop_y1"].as_float,
        )

    def _export_terrain(self) -> None:
        export_terrain_usd(
            self._terrain_json,
            self._terrain_usd,
            crop=self._crop(),
            scale_xy=self._models["scale_xy"].as_float,
            scale_z=self._models["scale_z"].as_float,
            snow_depth_m=self._models["depth"].as_float,
        )

    def _rebuild_terrain(self) -> None:
        try:
            self._export_terrain()
            self._set_status(f"Terrain rebuilt: {self._terrain_usd.name}")
            self._build_scene(import_robot=False)
        except Exception as exc:
            self._set_status(f"Terrain error: {exc}")

    def _find_existing_g1_usd(self) -> Path | None:
        candidates = sorted(self._assets.glob("*/g1_mjx.usda"))
        if candidates:
            return candidates[0]
        candidates = sorted(self._assets.glob("**/*.usda"))
        for candidate in candidates:
            if candidate.name != "everest_terrain.usda" and candidate.name != "everest_scene.usda":
                return candidate
        return None

    def _import_g1(self) -> Path:
        if self._g1_usd is not None and self._g1_usd.exists():
            return self._g1_usd
        config = MJCFImporterConfig(
            mjcf_path=str(self._g1_mjcf),
            usd_path=str(self._assets),
            import_scene=False,
            merge_mesh=True,
            collision_from_visuals=False,
            fix_base=False,
            run_multi_physics_conversion=False,
        )
        self._g1_usd = Path(MJCFImporter(config).import_mjcf())
        return self._g1_usd

    def _build_scene(self, import_robot: bool = True) -> None:
        try:
            self._export_terrain()
            g1_usd = self._import_g1() if import_robot else self._find_existing_g1_usd()
            stage = Usd.Stage.CreateNew(str(self._scene_usd))
            stage.SetMetadata("metersPerUnit", 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            stage.DefinePrim("/World/Terrain", "Xform").GetReferences().AddReference(str(self._terrain_usd))
            if g1_usd is not None and g1_usd.exists():
                stage.DefinePrim("/World/G1", "Xform").GetReferences().AddReference(str(g1_usd))
            snow = stage.DefinePrim("/World/Snow", "Xform")
            self._author_snow_metadata(snow)

            dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
            dome.CreateIntensityAttr(450.0)
            key = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
            key.CreateIntensityAttr(3000.0)
            key.CreateAngleAttr(0.8)
            stage.GetRootLayer().Save()
            omni.usd.get_context().open_stage(str(self._scene_usd))
            self._set_status(f"Opened {self._scene_usd.name} · {get_active_physics_engine()}")
        except Exception as exc:
            self._set_status(f"Scene error: {exc}")

    def _author_snow_metadata(self, prim) -> None:
        values = {
            "everest:snow:depthM": self._models["depth"].as_float,
            "everest:snow:densityKgM3": self._models["density"].as_float,
            "everest:snow:youngModulusPa": self._models["stiffness"].as_float,
            "everest:snow:compressiveYieldPa": self._models["compression"].as_float,
            "everest:snow:shearYieldPa": self._models["shear"].as_float,
            "everest:snow:hardening": self._models["hardening"].as_float,
            "everest:snow:footFriction": self._models["friction"].as_float,
        }
        for name, value in values.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=True).Set(float(value))
        prim.CreateAttribute("everest:snow:mode", Sdf.ValueTypeNames.String, custom=True).Set("static_snapshot")

    def _apply_snow_metadata(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            self._set_status("Open the Everest scene first")
            return
        prim = stage.GetPrimAtPath("/World/Snow")
        if not prim:
            prim = stage.DefinePrim("/World/Snow", "Xform")
        self._author_snow_metadata(prim)
        stage.GetRootLayer().Save()
        self._set_status("Static snow parameters saved to USD")
