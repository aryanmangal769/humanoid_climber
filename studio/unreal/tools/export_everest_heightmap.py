#!/usr/bin/env python3
"""Export the repo DEM as a UE Landscape-compatible 16-bit R16 heightmap."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "maps/output_hh.tif"
OUT = ROOT / "studio/unreal/EverestSim/SourceData"
W, H = 1009, 1513  # 16x24 Landscape components at 63 quads + 1
WEST, EAST = 86.65, 86.95
SOUTH, NORTH = 27.63, 28.05

im = Image.open(SRC).resize((W, H), Image.Resampling.BILINEAR)
elev = np.asarray(im, dtype=np.float64)
lo, hi = float(np.nanmin(elev)), float(np.nanmax(elev))
norm = np.clip((elev - lo) / max(1e-9, hi - lo), 0, 1)
u16 = np.rint(norm * 65535.0).astype("<u2")
OUT.mkdir(parents=True, exist_ok=True)
raw = OUT / "everest_macro.r16"
u16.tofile(raw)
(raw.with_suffix(".json")).write_text(json.dumps({"width": W, "height": H, "bbp": 16}, indent=2))
center_lat = (NORTH + SOUTH) / 2
width_m = (EAST - WEST) * 111_320 * math.cos(math.radians(center_lat))
depth_m = (NORTH - SOUTH) * 111_320
z_scale = (hi - lo) * 100.0 / 512.0
meta = {
    "source": str(SRC.relative_to(ROOT)),
    "bounds_lon_lat": [WEST, SOUTH, EAST, NORTH],
    "width_m": width_m, "depth_m": depth_m,
    "min_elevation_m": lo, "max_elevation_m": hi,
    "mid_elevation_m": (lo + hi) / 2,
    "unreal_landscape": {
        "resolution": [W, H],
        "x_scale_cm": width_m * 100.0 / (W - 1),
        "y_scale_cm": depth_m * 100.0 / (H - 1),
        "z_scale": z_scale,
        "actor_z_cm": (lo + hi) * 50.0
    }
}
(OUT / "everest_macro.metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {raw} {W}x{H}; elevation {lo:.1f}-{hi:.1f}m")
