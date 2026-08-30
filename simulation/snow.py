"""Snow/ice mechanics shared by the MuJoCo and Newton integrations.

The dashboard validates a multilayer column here, applies exposed-surface
friction to MuJoCo, and maps the internal mechanics to Newton Implicit MPM.
The local MPM patch runs on CUDA when available and on a reduced dense CPU grid
otherwise; the immutable surface presets remain the rigid fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco


@dataclass(frozen=True)
class SnowMaterial:
    name: str
    friction: float
    density: float
    young_modulus: float
    poisson_ratio: float
    yield_pressure: float
    cohesion: float
    color: tuple[float, float, float]


SURFACES = {
    "snow": SnowMaterial("snow", 0.35, 250.0, 2.0e5, 0.30, 2.0e4, 1500.0, (0.82, 0.88, 0.94)),
    "ice": SnowMaterial("ice", 0.08, 917.0, 8.0e6, 0.33, 2.0e5, 0.0, (0.52, 0.72, 0.88)),
    "rock": SnowMaterial("rock", 0.82, 2700.0, 5.0e7, 0.25, 2.0e6, 5.0e5, (0.30, 0.29, 0.27)),
    "nominal": SnowMaterial("nominal", 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, (0.10, 0.12, 0.15)),
}


@dataclass(frozen=True)
class SnowColumnLayer:
    """One mechanically active layer, ordered from the surface downward."""

    type: str
    label: str
    color: tuple[float, float, float]
    thickness_m: float
    density_kg_m3: float
    stiffness_pa: float
    compressive_strength_pa: float
    shear_strength_pa: float
    compaction_hardening: float
    bond_strength_below_pa: float

    def newton_attributes(self) -> dict[str, float]:
        """Map dashboard mechanics to Newton Implicit MPM attributes."""
        # Newton has no explicit layer-interface cohesion field.  The ratio of
        # interface bond to compressive strength is therefore used as the
        # tensile yield ratio, while the measured shear strength maps directly
        # to deviatoric yield stress.  Internal friction is deliberately kept
        # separate from G1/sole Coulomb friction.
        tensile_ratio = min(
            0.5,
            max(0.01, self.bond_strength_below_pa / max(self.compressive_strength_pa, 1.0)),
        )
        internal_friction = min(
            0.65,
            max(0.05, 0.08 + 0.35 * self.shear_strength_pa / max(self.compressive_strength_pa, 1.0)),
        )
        # Newton's fully coupled dilatancy solve is useful for offline
        # calibration but currently halves the interactive GPU stream rate.
        # Keep the attribute explicit and disabled in the realtime preset;
        # APIC still preserves lateral material flow. A future fidelity preset
        # can opt into calibrated per-layer values deliberately.
        dilatancy = 0.0
        return {
            "mpm:young_modulus": self.stiffness_pa,
            "mpm:poisson_ratio": 0.30,
            "mpm:damping": 0.02,
            "mpm:hardening": self.compaction_hardening,
            "mpm:friction": internal_friction,
            "mpm:yield_pressure": self.compressive_strength_pa,
            "mpm:tensile_yield_ratio": tensile_ratio,
            "mpm:yield_stress": self.shear_strength_pa,
            # Explicit realtime value; see the performance note above.
            "mpm:dilatancy": dilatancy,
        }

    def manifest(self, index: int) -> dict[str, Any]:
        return {
            "id": index,
            "type": self.type,
            "name": self.label,
            "label": self.label,
            "color": list(self.color),
            "depth": self.thickness_m,
            "thickness_m": self.thickness_m,
            "density_kg_m3": self.density_kg_m3,
            "stiffness_pa": self.stiffness_pa,
            "compressive_strength_pa": self.compressive_strength_pa,
            "shear_strength_pa": self.shear_strength_pa,
            "compaction_hardening": self.compaction_hardening,
            "bond_strength_below_pa": self.bond_strength_below_pa,
            "newton": self.newton_attributes(),
        }


@dataclass(frozen=True)
class SnowColumn:
    """Validated dashboard state consumed by MuJoCo and Newton MPM."""

    surface_friction: float
    layers: tuple[SnowColumnLayer, ...]
    snowfall_mm_h: float = 0.0
    wind_speed_m_s: float = 0.0
    wind_direction_deg: float = 0.0
    temperature_c: float = -10.0
    slope_deg: float = 0.0

    @property
    def depth(self) -> float:
        return sum(layer.thickness_m for layer in self.layers)

    @property
    def snowfall_depth_rate_m_s(self) -> float:
        """Fresh-snow depth flux in metres per simulated second."""
        return self.snowfall_mm_h * 1.0e-3 / 3600.0

    @property
    def snowfall_mass_flux_kg_m2_s(self) -> float:
        """Mass flux using the measured/selected density of the fresh top layer."""
        return self.snowfall_depth_rate_m_s * self.layers[0].density_kg_m3

    @classmethod
    def from_payload(cls, payload: Any) -> "SnowColumn":
        if not isinstance(payload, dict):
            raise ValueError("Snow parameters must be a JSON object")
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, list) or not 1 <= len(raw_layers) <= 6:
            raise ValueError("Snow parameters require 1-6 mechanical layers")

        friction = _finite_in_range(payload.get("surface_friction"), "surface_friction", 0.01, 1.5)
        layers: list[SnowColumnLayer] = []
        for index, raw in enumerate(raw_layers):
            if not isinstance(raw, dict):
                raise ValueError(f"Snow layer {index + 1} must be an object")
            color_raw = raw.get("color", (0.9, 0.95, 1.0))
            if not isinstance(color_raw, (list, tuple)) or len(color_raw) != 3:
                raise ValueError(f"Snow layer {index + 1} color must have three channels")
            color = tuple(
                _finite_in_range(value, f"layers[{index}].color", 0.0, 1.0)
                for value in color_raw
            )
            layers.append(SnowColumnLayer(
                type=str(raw.get("type", "SNOW"))[:32],
                label=str(raw.get("label", f"Layer {index + 1}"))[:64],
                color=color,
                thickness_m=_finite_in_range(raw.get("thickness_m"), f"layers[{index}].thickness_m", 0.005, 1.0),
                density_kg_m3=_finite_in_range(raw.get("density_kg_m3"), f"layers[{index}].density_kg_m3", 30.0, 950.0),
                stiffness_pa=_finite_in_range(raw.get("stiffness_pa"), f"layers[{index}].stiffness_pa", 1.0e3, 1.0e8),
                compressive_strength_pa=_finite_in_range(raw.get("compressive_strength_pa"), f"layers[{index}].compressive_strength_pa", 100.0, 1.0e7),
                shear_strength_pa=_finite_in_range(raw.get("shear_strength_pa"), f"layers[{index}].shear_strength_pa", 50.0, 5.0e6),
                compaction_hardening=_finite_in_range(raw.get("compaction_hardening"), f"layers[{index}].compaction_hardening", 0.0, 100.0),
                bond_strength_below_pa=_finite_in_range(raw.get("bond_strength_below_pa"), f"layers[{index}].bond_strength_below_pa", 0.0, 5.0e6),
            ))
        if sum(layer.thickness_m for layer in layers) > 2.5:
            raise ValueError("Total active snow depth cannot exceed 2.5 m")
        snowfall_mm_h = _finite_in_range(payload.get("snowfall_mm_h", 0.0), "snowfall_mm_h", 0.0, 200.0)
        wind_speed_m_s = _finite_in_range(payload.get("wind_speed_m_s", 0.0), "wind_speed_m_s", 0.0, 100.0)
        wind_direction_deg = _finite_in_range(payload.get("wind_direction_deg", 0.0), "wind_direction_deg", 0.0, 360.0) % 360.0
        temperature_c = _finite_in_range(payload.get("temperature_c", -10.0), "temperature_c", -80.0, 20.0)
        slope_deg = _finite_in_range(payload.get("slope_deg", 0.0), "slope_deg", 0.0, 60.0)
        return cls(
            surface_friction=friction,
            layers=tuple(layers),
            snowfall_mm_h=snowfall_mm_h,
            wind_speed_m_s=wind_speed_m_s,
            wind_direction_deg=wind_direction_deg,
            temperature_c=temperature_c,
            slope_deg=slope_deg,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "surface_friction": self.surface_friction,
            "depth": self.depth,
            "snowfall_mm_h": self.snowfall_mm_h,
            "snowfall_depth_rate_m_s": self.snowfall_depth_rate_m_s,
            "snowfall_mass_flux_kg_m2_s": self.snowfall_mass_flux_kg_m2_s,
            "wind_speed_m_s": self.wind_speed_m_s,
            "wind_direction_deg": self.wind_direction_deg,
            "temperature_c": self.temperature_c,
            "slope_deg": self.slope_deg,
            "layers": [layer.manifest(index) for index, layer in enumerate(self.layers)],
        }


def _finite_in_range(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


class SnowLayer:
    """Surface parameters, MuJoCo contact application, and render manifest."""

    def __init__(self, surface: str = "snow", depth: float = 0.12):
        if surface not in SURFACES:
            raise ValueError(f"Unknown surface {surface!r}; choose {sorted(SURFACES)}")
        if depth < 0.0:
            raise ValueError("snow depth must be non-negative")
        self.surface = surface
        self.depth = float(depth)
        self.column: SnowColumn | None = None

    @property
    def material(self) -> SnowMaterial:
        if self.column is None or self.surface != "snow":
            return SURFACES[self.surface]
        top = self.column.layers[0]
        return SnowMaterial(
            "snow",
            self.column.surface_friction,
            top.density_kg_m3,
            top.stiffness_pa,
            0.30,
            top.compressive_strength_pa,
            top.bond_strength_below_pa,
            top.color,
        )

    def configure_column(self, payload: Any) -> SnowColumn:
        """Validate and install live multilayer mechanics from the dashboard."""
        column = SnowColumn.from_payload(payload)
        self.column = column
        self.surface = "snow"
        self.depth = column.depth
        return column

    def apply_to_mujoco(self, model: mujoco.MjModel) -> None:
        """Set the floor's Coulomb friction for the selected surface."""
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0:
            raise ValueError("G1 scene has no floor geom")
        model.geom_friction[floor, 0] = self.material.friction
        # MuJoCo's torsional/rolling terms remain conservative and small.
        model.geom_friction[floor, 1:] = (0.005, 0.0001)
        # Menagerie's MJX scene uses explicit contact pairs, whose friction
        # overrides geom mixing. Update every pair that references the floor.
        for pair_id in range(model.npair):
            if floor in (model.pair_geom1[pair_id], model.pair_geom2[pair_id]):
                model.pair_friction[pair_id, :2] = self.material.friction
                model.pair_friction[pair_id, 2:] = (0.005, 0.0001, 0.0001)

    def manifest(self) -> dict[str, Any]:
        """Describe deterministic snow/ice geometry for the Three.js layer."""
        result = {
            "surface": self.surface,
            "depth": self.depth,
            "friction": self.material.friction,
            "density": self.material.density,
            "young_modulus": self.material.young_modulus,
            "poisson_ratio": self.material.poisson_ratio,
            "yield_pressure": self.material.yield_pressure,
            "cohesion": self.material.cohesion,
            "color": self.material.color,
            "mpm_material": self.surface == "snow" and self.column is not None,
            "mpm_ready": self.surface == "snow" and self.column is not None,
            "newton_compatible": True,
            "physics_mode": "newton_mpm_pending" if self.surface == "snow" and self.column is not None else "mujoco_rigid_friction",
            "calibration": "baseline; site-specific snow density and strength must be identified",
        }
        if self.column is not None:
            result.update(self.column.manifest())
        return result

    def mpm_parameters(self) -> dict[str, float]:
        """Return names matching Newton SolverImplicitMPM particle attributes."""
        material = self.material
        result = {
            "density": material.density,
            "young_modulus": material.young_modulus,
            "poisson_ratio": material.poisson_ratio,
            "friction": (
                self.column.layers[0].newton_attributes()["mpm:friction"]
                if self.column else 0.10
            ),
            "yield_pressure": material.yield_pressure,
            "yield_stress": self.column.layers[0].shear_strength_pa if self.column else material.cohesion,
            "hardening": self.column.layers[0].compaction_hardening if self.column else 10.0,
            "tensile_yield_ratio": (
                self.column.layers[0].newton_attributes()["mpm:tensile_yield_ratio"]
                if self.column else 0.05
            ),
            "dilatancy": (
                self.column.layers[0].newton_attributes()["mpm:dilatancy"]
                if self.column else 0.0
            ),
        }
        return result

    def configure_newton_particles(self, model: Any) -> None:
        """Apply this layer's material values to a Newton MPM model.

        Call after ``SolverImplicitMPM.register_custom_attributes(builder)`` and
        ``builder.finalize()``. Newton owns the actual constitutive update; this
        method only supplies the calibrated material parameters.
        """
        for name, value in self.mpm_parameters().items():
            field = getattr(getattr(model, "mpm", None), name, None)
            if field is not None:
                field.fill_(value)

    def emit_newton_particle_grid(
        self,
        builder: Any,
        *,
        bounds_lo: tuple[float, float, float] = (-3.0, -3.0, 0.0),
        bounds_hi: tuple[float, float, float] = (3.0, 3.0, 0.12),
        voxel_size: float = 0.08,
        particles_per_cell: int = 2,
    ) -> None:
        """Add a uniform snow volume to a Newton ``ModelBuilder``.

        This follows Newton's documented ``add_particle_grid`` convention and
        deliberately leaves solver construction to the caller so the same
        rigid G1 model can be registered as the MPM collider.
        """
        import numpy as np
        import warp as wp

        if self.surface == "nominal":
            return
        lo = np.asarray(bounds_lo, dtype=np.float32)
        hi = np.asarray(bounds_hi, dtype=np.float32)
        resolution = np.maximum(1, np.ceil(particles_per_cell * (hi - lo) / voxel_size).astype(int))
        cell = (hi - lo) / resolution
        radius = float(np.max(cell) * 0.5)
        particle_count = int(np.prod(resolution))
        mass = float(np.prod(hi - lo) * self.material.density / particle_count)
        builder.add_particle_grid(
            pos=wp.vec3(lo + 0.5 * cell),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(resolution[0]),
            dim_y=int(resolution[1]),
            dim_z=int(resolution[2]),
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=mass,
            jitter=0.15 * radius,
            radius_mean=radius,
        )
