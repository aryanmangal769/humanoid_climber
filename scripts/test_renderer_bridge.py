"""End-to-end contract smoke test for the Unity WebSocket bridge."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import websockets


ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("EVEREST_BACKEND_TEST_PORT", "18765"))
URL = f"ws://127.0.0.1:{PORT}"


def _port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


async def _recv_until(ws, wanted: set[str], timeout: float = 25.0) -> dict[str, dict]:
    found: dict[str, dict] = {}
    deadline = time.monotonic() + timeout
    while wanted - found.keys():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"missing bridge messages: {sorted(wanted - found.keys())}")
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
        if message.get("type") in wanted:
            found[message["type"]] = message["data"]
    return found


async def run_test() -> None:
    process = None
    if not _port_open():
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib{':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''}"
        process = subprocess.Popen(
            [str(ROOT / "scripts/start-simulation-backend.sh"), "--port", str(PORT)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(120):
            if _port_open():
                break
            if process.poll() is not None:
                raise RuntimeError(process.stdout.read())
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError(f"simulation backend did not open port {PORT}")

    try:
        async with websockets.connect(URL, max_size=16 * 1024 * 1024, compression=None) as ws:
            initial = await _recv_until(ws, {"scene", "terrain", "macro_terrain", "frame", "snow", "state"})
            assert initial["scene"]["schema"] == "everest-scene/v1"
            assert initial["terrain"]["grid_width"] == 257
            assert initial["terrain"]["world_width_m"] >= 1199
            assert initial["macro_terrain"]["source_resolution_m"] == 2.0
            assert initial["macro_terrain"]["world_width_m"] > 10_000
            assert len(initial["frame"]["body_names"]) == len(initial["frame"]["body_pos_w"])
            assert initial["snow"]["compaction"]
            assert len(initial["snow"]["vertices"]) == len(initial["snow"]["heights"])
            assert len(initial["snow"]["layer_vertices"]) == len(initial["snow"]["layers"])
            before_pos = np.asarray(initial["frame"]["body_pos_w"][0], dtype=float)
            before_vertices = np.asarray(initial["snow"]["vertices"], dtype=float)
            before_steps = int(initial["state"]["newton"]["steps"])
            before_sim_time = float(initial["state"]["sim_time"])

            await ws.send(json.dumps({"type": "control", "action": "command", "value": [0.15, 0.0, 0.0]}))
            await ws.send(json.dumps({"type": "control", "action": "pause", "value": False}))
            rollout_started = time.monotonic()
            latest: dict[str, dict] = {}
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                # The first Newton/Warp contact step may JIT additional GPU
                # kernels after the socket is already live. Allow that one-off
                # compile without treating it as a streaming deadlock.
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
                kind = message.get("type")
                if kind in {"frame", "snow", "state"}:
                    latest[kind] = message["data"]
                state = latest.get("state")
                if state and state.get("simulation_fault"):
                    raise AssertionError(state["simulation_fault"])
                if (
                    state
                    and float(state["sim_time"]) >= before_sim_time + 0.8
                    and int(state["newton"]["steps"]) > before_steps
                    and "frame" in latest
                    and "snow" in latest
                ):
                    break
            else:
                raise TimeoutError("bridge did not advance one simulated second")
            later = latest
            rollout_wall_seconds = time.monotonic() - rollout_started
            rollout_sim_seconds = float(later["state"]["sim_time"]) - before_sim_time
            after_pos = np.asarray(later["frame"]["body_pos_w"][0], dtype=float)
            after_steps = int(later["state"]["newton"]["steps"])
            snow = later["snow"]
            after_vertices = np.asarray(snow["vertices"], dtype=float)
            vertex_delta = after_vertices - before_vertices
            sinkage = np.asarray(snow["base_heights"], dtype=float) - np.asarray(
                snow["heights"], dtype=float
            )
            moved = float(np.linalg.norm(after_pos - before_pos))
            assert moved > 1.0e-3, f"G1 did not move enough: {moved} m"
            assert after_steps > before_steps, f"Newton did not step: {before_steps} -> {after_steps}"
            assert float(np.percentile(sinkage, 90)) < 0.05, (
                f"untouched Newton snow collapsed: p90 sinkage={np.percentile(sinkage, 90):.4f} m"
            )
            assert float(sinkage.max(initial=0.0)) <= float(snow["surface_depth_m"]) + 1.0e-5, (
                "Newton surface passed below its terrain-conforming substrate"
            )
            assert np.allclose(after_vertices[:, 2], snow["heights"], atol=1.0e-6), (
                "3D renderer vertices and MuJoCo support heights diverged"
            )
            assert float(np.linalg.norm(vertex_delta[:, :2], axis=1).max(initial=0.0)) > 1.0e-4, (
                "stream discarded Newton lateral/shear deformation"
            )
            assert float(np.abs(vertex_delta[:, 2]).max(initial=0.0)) > 1.0e-4, (
                "stream discarded Newton vertical deformation"
            )
            assert later["state"]["simulation_fault"] is None, later["state"]["simulation_fault"]
            await ws.send(json.dumps({"type": "control", "action": "pause", "value": True}))
            print(
                f"renderer bridge OK: G1 moved {moved:.4f} m; "
                f"Newton steps {before_steps}->{after_steps}; "
                f"max vertex lateral={np.linalg.norm(vertex_delta[:, :2], axis=1).max(initial=0.0):.5f} m, "
                f"vertical={np.abs(vertex_delta[:, 2]).max(initial=0.0):.5f} m; "
                f"stream={rollout_sim_seconds / max(rollout_wall_seconds, 1.0e-6):.2f}x realtime"
            )
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            output = process.stdout.read() if process.stdout is not None else ""
            if output and os.environ.get("EVEREST_TEST_BACKEND_LOG") == "1":
                print(output)


if __name__ == "__main__":
    asyncio.run(run_test())
