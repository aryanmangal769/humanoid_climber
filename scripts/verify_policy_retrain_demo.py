#!/usr/bin/env python3
"""Verify the Everest failure/retrain/checkpoint demo lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
import copy

from dashboard.engines.mujoco import MuJoCoEngine
from simulation.unity_bridge import DEFAULT_SNOW


def main() -> None:
    engine = MuJoCoEngine()
    try:
        # Match the real Unity bridge startup path so this acceptance test
        # verifies capture of an actual live Newton/MPM region, not the
        # engine's pre-configuration rigid fallback.
        engine.control("snow_parameters", copy.deepcopy(DEFAULT_SNOW))
        initial = engine.state()["policy"]
        assert initial["selected_policy_key"] == "auto"
        assert initial["supervisor"]["active_policy_key"] == "flat"
        keys = {item["key"] for item in initial["registry"]}
        assert {"auto", "flat", "ice_incline", "wind", "rough", "recovery"} <= keys

        engine.control("demo_failure")
        waiting = engine.state()
        supervisor = waiting["policy"]["supervisor"]
        assert waiting["paused"]
        assert supervisor["stage"] == "waiting_checkpoint"
        manifest_path = Path(supervisor["request_manifest"])
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == "everest-rl-subset/v1"
        terrain = manifest["environment"]["terrain"]
        assert terrain["mode"] == "live"
        assert terrain["vertices"]
        assert terrain["layer_vertices"]
        assert terrain["mpm"]["solver"]
        assert manifest["training"]["status"] == "requested_not_launched"

        preview = engine.subset_preview()
        assert preview and preview["encoding"] == "jpeg/base64"
        assert len(preview["image"]) > 1000

        engine.control("demo_return_pretrained", "ice_incline")
        policy = engine.state()["policy"]
        assert policy["selected_policy_key"] == "ice_incline"
        active = policy["supervisor"]
        assert active["stage"] == "policy_active"
        assert active["active_policy_key"] == "ice_incline"
        assert active["demo_pretrained"]
        print(json.dumps({
            "request_id": supervisor["request_id"],
            "manifest": str(manifest_path),
            "preview_bytes_base64": len(preview["image"]),
            "active_policy": active["active_policy_label"],
            "demo_pretrained": active["demo_pretrained"],
        }, indent=2))
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
