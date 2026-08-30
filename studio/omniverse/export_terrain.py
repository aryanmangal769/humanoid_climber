#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "studio/omniverse/exts/everest.studio/everest/studio/terrain_asset.py"
SPEC = importlib.util.spec_from_file_location("everest_terrain_asset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load terrain exporter: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
export_terrain_usd = MODULE.export_terrain_usd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Everest terrain JSON to USD")
    parser.add_argument("--source", default=str(ROOT / "maps/everest_terrain.json"))
    parser.add_argument("--output", default=str(ROOT / "studio/omniverse/assets/everest_terrain.usda"))
    parser.add_argument("--crop", nargs=4, type=float, default=(0.0, 1.0, 0.0, 1.0))
    parser.add_argument("--scale-xy", type=float, default=1.0)
    parser.add_argument("--scale-z", type=float, default=1.0)
    parser.add_argument("--snow-depth", type=float, default=0.08)
    args = parser.parse_args()
    result = export_terrain_usd(
        args.source,
        args.output,
        crop=args.crop,
        scale_xy=args.scale_xy,
        scale_z=args.scale_z,
        snow_depth_m=args.snow_depth,
    )
    print(result)


if __name__ == "__main__":
    main()
