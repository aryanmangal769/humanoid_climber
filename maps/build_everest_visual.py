from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

MAPS = Path(__file__).resolve().parent
DEM = MAPS / "output_hh.tif"
ROUTE = MAPS / "export.geojson"
OUTPUT = MAPS / "everest_terrain.json"

WEST, EAST = 86.65, 86.95
SOUTH, NORTH = 27.63, 28.05
WORLD_DEPTH_M = 12.0
GRID_WIDTH = 180


def sample(grid: np.ndarray, lon: float, lat: float) -> float:
    """Bilinearly sample a north-up elevation grid."""
    rows, cols = grid.shape
    x = np.clip((lon - WEST) / (EAST - WEST) * (cols - 1), 0, cols - 1)
    y = np.clip((NORTH - lat) / (NORTH - SOUTH) * (rows - 1), 0, rows - 1)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    fx, fy = x - x0, y - y0
    return float(
        grid[y0, x0] * (1 - fx) * (1 - fy)
        + grid[y0, x1] * fx * (1 - fy)
        + grid[y1, x0] * (1 - fx) * fy
        + grid[y1, x1] * fx * fy
    )


def main() -> None:
    image = Image.open(DEM)
    rows = round(image.height * GRID_WIDTH / image.width)
    small = image.resize((GRID_WIDTH, rows), Image.Resampling.BILINEAR)
    elevations = np.asarray(small, dtype=np.float32)

    feature = json.loads(ROUTE.read_text())["features"][0]
    coordinates = feature["geometry"]["coordinates"]
    anchor_lon, anchor_lat = coordinates[0]

    center_lat = (NORTH + SOUTH) / 2
    width_m = (EAST - WEST) * 111_320 * math.cos(math.radians(center_lat))
    depth_m = (NORTH - SOUTH) * 111_320
    world_scale = WORLD_DEPTH_M / depth_m
    world_width_m = width_m * world_scale

    route_start_elevation = sample(elevations, anchor_lon, anchor_lat)
    elevation_min = float(elevations.min())
    elevation_max = float(elevations.max())
    # Use one scale for all three axes so the collision slopes match Everest's
    # source aspect ratio instead of applying visual vertical exaggeration.
    vertical_scale = world_scale

    # Terrain vertex heights are relative to the route start, placing the G1
    # at visual ground level at the start of the South Col line.
    heights = ((elevations - route_start_elevation) * vertical_scale).ravel()
    route = []
    for lon, lat in coordinates:
        route.append([
            (lon - anchor_lon) * 111_320 * math.cos(math.radians(center_lat)) * world_scale,
            (lat - anchor_lat) * 111_320 * world_scale,
            (sample(elevations, lon, lat) - route_start_elevation) * vertical_scale + 0.025,
        ])

    OUTPUT.write_text(json.dumps({
        "schema": "everest-terrain/v1",
        "name": feature["properties"].get("name", "Everest route"),
        "grid_width": GRID_WIDTH,
        "grid_height": rows,
        "world_width_m": world_width_m,
        "world_depth_m": WORLD_DEPTH_M,
        "world_scale": world_scale,
        "vertical_relief_m": (elevation_max - elevation_min) * world_scale,
        "terrain_center": [
            ((WEST + EAST) / 2 - anchor_lon) * 111_320
            * math.cos(math.radians(center_lat)) * world_scale,
            ((SOUTH + NORTH) / 2 - anchor_lat) * 111_320
            * world_scale,
            0.0,
        ],
        "min_elevation_m": elevation_min,
        "max_elevation_m": elevation_max,
        "route_start_elevation_m": route_start_elevation,
        "heights": [round(float(value), 6) for value in heights],
        "route": route,
    }, separators=(",", ":")))

    print(f"Wrote {OUTPUT} ({GRID_WIDTH}×{rows} terrain, {len(route)} route points)")


if __name__ == "__main__":
    main()
