"""Build true-scale Unity/backend terrain products from the 2 m Everest DEM."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage


MAPS = Path(__file__).resolve().parent
DEM_2M = MAPS / "HMA2_DCG_SMB_KHU_ANNUAL_DEM_20151102T0517Z_V01.0.tif"
ROUTE = MAPS / "export.geojson"
LOCAL_OUTPUT = MAPS / "everest_local_terrain.json"
MACRO_OUTPUT = MAPS / "everest_macro_terrain.json"

LOCAL_SIZE_M = 1_200.0
LOCAL_GRID = 257
MACRO_WIDTH = 321
SOURCE_EPSG = 32645  # WGS84 / UTM zone 45N, encoded in the HMA2 GeoTIFF.


def _lonlat_to_utm45(lon: float, lat: float) -> tuple[float, float]:
    """Use the system PROJ install without adding a Python geospatial dependency."""
    process = subprocess.run(
        [
            "/usr/bin/cs2cs",
            "+proj=longlat",
            "+datum=WGS84",
            "+to",
            "+proj=utm",
            "+zone=45",
            "+datum=WGS84",
        ],
        input=f"{lon} {lat}\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    parts = process.stdout.split()
    if len(parts) < 2:
        raise RuntimeError(f"cs2cs returned no UTM coordinate: {process.stdout!r}")
    return float(parts[0]), float(parts[1])


def _georef(image: Image.Image) -> tuple[float, float, float, float]:
    pixel_scale = image.tag_v2.get(33550)
    tiepoint = image.tag_v2.get(33922)
    if not pixel_scale or not tiepoint:
        raise ValueError(f"2 m DEM has no GeoTIFF pixel scale/tiepoint metadata: {DEM_2M}")
    dx, dy = float(pixel_scale[0]), float(pixel_scale[1])
    origin_e, origin_n = float(tiepoint[3]), float(tiepoint[4])
    if abs(dx - 2.0) > 1e-6 or abs(dy - 2.0) > 1e-6:
        raise ValueError(f"Expected 2 m DEM, got {dx} x {dy} m pixels")
    return origin_e, origin_n, dx, dy


def _bilinear_window(
    image: Image.Image,
    *,
    eastings: np.ndarray,
    northings: np.ndarray,
    origin_e: float,
    origin_n: float,
    dx: float,
    dy: float,
    nodata: float | None,
) -> np.ndarray:
    """Sample a small projected target grid without loading the whole 2 m DEM."""
    px = (eastings - origin_e) / dx
    py = (origin_n - northings) / dy
    x_min = max(0, int(math.floor(float(px.min()))) - 1)
    x_max = min(image.width - 1, int(math.ceil(float(px.max()))) + 1)
    y_min = max(0, int(math.floor(float(py.min()))) - 1)
    y_max = min(image.height - 1, int(math.ceil(float(py.max()))) + 1)
    if px.min() < 0 or px.max() > image.width - 1 or py.min() < 0 or py.max() > image.height - 1:
        raise ValueError("Requested local terrain window extends outside the 2 m DEM")

    crop = np.asarray(image.crop((x_min, y_min, x_max + 1, y_max + 1)), dtype=np.float64)
    valid = np.isfinite(crop)
    if nodata is not None:
        valid &= ~np.isclose(crop, nodata)
    if not valid.any():
        raise ValueError("Local 2 m DEM crop contains only nodata")
    if not valid.all():
        indices = ndimage.distance_transform_edt(
            ~valid,
            return_distances=False,
            return_indices=True,
        )
        crop = crop[tuple(indices)]
    xx, yy = np.meshgrid(px - x_min, py - y_min)
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    x1 = np.minimum(crop.shape[1] - 1, x0 + 1)
    y1 = np.minimum(crop.shape[0] - 1, y0 + 1)
    tx = xx - x0
    ty = yy - y0
    return (
        crop[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + crop[y0, x1] * tx * (1.0 - ty)
        + crop[y1, x0] * (1.0 - tx) * ty
        + crop[y1, x1] * tx * ty
    )


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")))


def main() -> None:
    if not DEM_2M.is_file():
        raise FileNotFoundError(f"2 m Everest DEM missing: {DEM_2M}")
    route = json.loads(ROUTE.read_text())["features"][0]
    anchor_lon, anchor_lat = route["geometry"]["coordinates"][0]
    anchor_e, anchor_n = _lonlat_to_utm45(float(anchor_lon), float(anchor_lat))

    image = Image.open(DEM_2M)
    origin_e, origin_n, dx, dy = _georef(image)
    nodata_raw = image.tag_v2.get(42113)
    nodata = float(nodata_raw) if nodata_raw is not None else None

    # Output rows deliberately run south -> north so row index and +Y agree
    # in MuJoCo, Newton, and the Unity contract.
    axis = np.linspace(-LOCAL_SIZE_M / 2.0, LOCAL_SIZE_M / 2.0, LOCAL_GRID)
    eastings = anchor_e + axis
    northings = anchor_n + axis
    local_abs = _bilinear_window(
        image,
        eastings=eastings,
        northings=northings,
        origin_e=origin_e,
        origin_n=origin_n,
        dx=dx,
        dy=dy,
        nodata=nodata,
    )
    anchor_elevation = float(local_abs[LOCAL_GRID // 2, LOCAL_GRID // 2])
    local = local_abs - anchor_elevation
    _write(LOCAL_OUTPUT, {
        "schema": "everest-terrain/v1",
        "name": "Everest local physical terrain (HMA2 2 m DEM)",
        "product": "local_physical",
        "source_dem": DEM_2M.name,
        "source_crs": f"EPSG:{SOURCE_EPSG}",
        "source_resolution_m": 2.0,
        "grid_width": LOCAL_GRID,
        "grid_height": LOCAL_GRID,
        "world_width_m": LOCAL_SIZE_M,
        "world_depth_m": LOCAL_SIZE_M,
        "sample_spacing_m": LOCAL_SIZE_M / (LOCAL_GRID - 1),
        "terrain_center": [0.0, 0.0, 0.0],
        "row_order": "south_to_north",
        "anchor_lon_lat": [anchor_lon, anchor_lat],
        "anchor_utm_m": [anchor_e, anchor_n],
        "anchor_elevation_m": anchor_elevation,
        "min_height_m": float(local.min()),
        "max_height_m": float(local.max()),
        "vertical_relief_m": float(np.ptp(local)),
        "heights": np.round(local, 4).ravel().tolist(),
    })

    # The macro visual terrain is also sourced from the 2 m DEM. It is merely
    # downsampled for transport/rendering; no coarse DEM is mixed into it.
    macro_height = max(2, round(image.height * MACRO_WIDTH / image.width))
    source = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(source)
    if nodata is not None:
        valid &= ~np.isclose(source, nodata)
    weighted = np.where(valid, source, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    weighted_small = np.asarray(
        Image.fromarray(weighted, mode="F").resize(
            (MACRO_WIDTH, macro_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    weights_small = np.asarray(
        Image.fromarray(weights, mode="F").resize(
            (MACRO_WIDTH, macro_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    macro_abs = np.zeros_like(weighted_small)
    valid_small = weights_small > 1.0e-4
    macro_abs[valid_small] = weighted_small[valid_small] / weights_small[valid_small]
    if not valid_small.all():
        if not valid_small.any():
            raise ValueError("2 m DEM macro product contains only nodata")
        indices = ndimage.distance_transform_edt(
            ~valid_small,
            return_distances=False,
            return_indices=True,
        )
        macro_abs = macro_abs[tuple(indices)]
    macro_abs = macro_abs[::-1]
    macro = macro_abs - anchor_elevation
    world_width_m = (image.width - 1) * dx
    world_depth_m = (image.height - 1) * dy
    center_e = origin_e + 0.5 * world_width_m
    center_n = origin_n - 0.5 * world_depth_m
    _write(MACRO_OUTPUT, {
        "schema": "everest-terrain/v1",
        "name": "Everest macro visual terrain (HMA2 2 m DEM)",
        "product": "macro_visual",
        "source_dem": DEM_2M.name,
        "source_crs": f"EPSG:{SOURCE_EPSG}",
        "source_resolution_m": 2.0,
        "grid_width": MACRO_WIDTH,
        "grid_height": macro_height,
        "world_width_m": world_width_m,
        "world_depth_m": world_depth_m,
        "terrain_center": [center_e - anchor_e, center_n - anchor_n, 0.0],
        "row_order": "south_to_north",
        "anchor_lon_lat": [anchor_lon, anchor_lat],
        "anchor_utm_m": [anchor_e, anchor_n],
        "anchor_elevation_m": anchor_elevation,
        "min_height_m": float(macro.min()),
        "max_height_m": float(macro.max()),
        "vertical_relief_m": float(np.ptp(macro)),
        "heights": np.round(macro, 2).ravel().tolist(),
    })
    print(
        f"Wrote {LOCAL_OUTPUT} ({LOCAL_GRID}x{LOCAL_GRID}, {LOCAL_SIZE_M:.0f} m square) "
        f"from native {dx:.0f} m DEM"
    )
    print(
        f"Wrote {MACRO_OUTPUT} ({MACRO_WIDTH}x{macro_height}, "
        f"{world_width_m/1000:.2f}x{world_depth_m/1000:.2f} km) from same 2 m DEM"
    )


if __name__ == "__main__":
    main()
