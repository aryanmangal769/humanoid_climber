#!/usr/bin/env python3
"""Measure robot and snow-column settlement immediately after a reset."""

from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np
import websockets


URL = os.environ.get("EVEREST_BACKEND_URL", "ws://127.0.0.1:18766")
SIM_SECONDS = float(os.environ.get("EVEREST_SINKAGE_SECONDS", "3.0"))


async def send_control(ws, action, value) -> None:
    await ws.send(json.dumps({"type": "control", "action": action, "value": value}))
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
        if message.get("type") == "control_ack":
            data = message.get("data") or {}
            if not data.get("ok"):
                raise RuntimeError(data)
            return


def summarize(frame: dict, snow: dict, state: dict) -> dict:
    body_names = frame["body_names"]
    pelvis = np.asarray(frame["body_pos_w"][body_names.index("pelvis")], dtype=float)
    vertices = np.asarray(snow["vertices"], dtype=float)
    pristine = np.asarray(snow["base_heights"], dtype=float)
    heights = np.asarray(snow["heights"], dtype=float)
    substrate = np.asarray(snow["substrate_vertices"], dtype=float)[:, 2]
    sinkage = pristine - heights
    feet = {}
    for side in ("left", "right"):
        foot = frame["feet"][side]
        position = np.asarray(foot["position"], dtype=float)
        cell = int(np.argmin(np.sum((vertices[:, :2] - position[:2]) ** 2, axis=1)))
        feet[side] = {
            "contact": bool(foot["contact"]),
            "normal_force_n": float(foot["normal_force_n"]),
            "foot_z_m": float(position[2]),
            "surface_z_m": float(heights[cell]),
            "substrate_z_m": float(substrate[cell]),
            "local_sinkage_m": float(sinkage[cell]),
            "remaining_column_m": float(heights[cell] - substrate[cell]),
        }
    return {
        "sim_time": float(state["sim_time"]),
        "pelvis": pelvis.tolist(),
        "max_sinkage_m": float(sinkage.max(initial=0.0)),
        "p90_sinkage_m": float(np.percentile(sinkage, 90)),
        "mean_sinkage_m": float(np.mean(np.maximum(sinkage, 0.0))),
        "max_compaction": float(np.max(snow["compaction"], initial=0.0)),
        "newton_steps": int(state["newton"]["steps"]),
        "newton_solver_steps": int(state["newton"]["solver_steps"]),
        "stand_lock_active": bool(
            state.get("simulation_settings", {}).get("stand_lock_active", False)
        ),
        "feet": feet,
    }


async def main() -> None:
    latest: dict[str, dict] = {}
    samples: list[dict] = []
    async with websockets.connect(URL, max_size=16 * 1024 * 1024, compression=None) as ws:
        await send_control(ws, "pause", True)
        await send_control(ws, "reset", None)
        await send_control(ws, "command", [0.0, 0.0, 0.0])
        await send_control(ws, "pause", False)
        start_sim_time: float | None = None
        next_sample = 0.0
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
            kind = message.get("type")
            if kind in {"frame", "snow", "state"}:
                latest[kind] = message["data"]
            if not {"frame", "snow", "state"}.issubset(latest):
                continue
            sim_time = float(latest["state"]["sim_time"])
            if start_sim_time is None:
                start_sim_time = sim_time
                next_sample = sim_time
            if sim_time + 1.0e-6 >= next_sample:
                samples.append(summarize(latest["frame"], latest["snow"], latest["state"]))
                next_sample += 0.25
            if sim_time >= start_sim_time + SIM_SECONDS:
                break
        await ws.send(json.dumps({"type": "control", "action": "pause", "value": True}))
    if not samples:
        raise RuntimeError("startup sinkage probe produced no samples")
    first, final = samples[0], samples[-1]
    displacement_xy = float(np.linalg.norm(
        np.asarray(final["pelvis"][:2]) - np.asarray(first["pelvis"][:2])
    ))
    pelvis_settlement = float(first["pelvis"][2] - final["pelvis"][2])
    remaining_column = min(
        float(item["remaining_column_m"])
        for item in final["feet"].values()
    )
    result = {
        "sim_seconds": final["sim_time"],
        "stand_lock_active": final["stand_lock_active"],
        "pelvis_displacement_xy_m": displacement_xy,
        "pelvis_settlement_m": pelvis_settlement,
        "max_sinkage_m": final["max_sinkage_m"],
        "p90_sinkage_m": final["p90_sinkage_m"],
        "minimum_remaining_column_m": remaining_column,
        "max_compaction": final["max_compaction"],
        "samples": samples if os.environ.get("EVEREST_SINKAGE_VERBOSE") == "1" else None,
    }
    assert result["stand_lock_active"], result
    assert displacement_xy < 1.0e-4, result
    assert pelvis_settlement < 0.08, result
    assert result["max_sinkage_m"] < 0.10, result
    assert result["p90_sinkage_m"] < 0.01, result
    assert remaining_column > 0.25, result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
