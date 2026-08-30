#!/usr/bin/env python3
"""Verify Unity operator controls against a disposable backend instance."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import websockets

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("EVEREST_SIM_CONTROL_TEST_PORT", "18767"))
URL = f"ws://127.0.0.1:{PORT}"


def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


async def recv_until(ws, wanted: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        if msg.get("type") == wanted:
            return msg["data"]
    raise TimeoutError(wanted)


async def control(ws, action, value=None, *, expected_ok=True):
    await ws.send(json.dumps({"type": "control", "action": action, "value": value}))
    ack = await recv_until(ws, "control_ack")
    assert bool(ack.get("ok")) is expected_ok, ack
    return ack


async def main() -> None:
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
    try:
        for _ in range(120):
            if port_open():
                break
            if process.poll() is not None:
                raise RuntimeError(process.stdout.read())
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("backend did not start")

        async with websockets.connect(URL, max_size=16 * 1024 * 1024, compression=None) as ws:
            state = await recv_until(ws, "state")
            assert state.get("data_mode") == "sim", state
            assert state.get("surface") == "snow", state

            initial_time = float(state["sim_time"])
            await control(ws, "play")
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                state = await recv_until(ws, "state")
                if float(state["sim_time"]) > initial_time + 0.05:
                    break
            else:
                raise AssertionError("RUN/play did not start the policy simulation")
            assert state.get("simulation_settings", {}).get("stand_lock_active") is False, state
            await control(ws, "pause", True)

            await control(ws, "simulation_settings", {
                "physics_radius_m": 1.5,
                "mpm_voxel_size_m": 0.10,
                "patch_recenter_fraction": 0.5,
                "cheat_speed_m_s": 1.8,
                "cheat_yaw_rate_rad_s": 1.2,
            })
            state = None
            snow_frame = None
            while state is None or snow_frame is None:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
                if message.get("type") == "state":
                    candidate = message["data"]
                    settings = candidate.get("simulation_settings", {})
                    if abs(float(settings.get("physics_radius_m", 0.0)) - 1.5) < 1e-6:
                        state = candidate
                elif message.get("type") == "snow":
                    snow_frame = message["data"]
            assert abs(float(snow_frame["size"][0]) - 3.0) < 1e-5, snow_frame["size"]

            ack = await control(ws, "mode", "live", expected_ok=False)
            assert ack.get("code") == "live_not_configured", ack
            state = await recv_until(ws, "state")
            assert state.get("data_mode") == "sim", state
            await control(ws, "surface", "ice")
            await control(ws, "surface_friction", 0.12)
            while True:
                state = await recv_until(ws, "state")
                if state.get("surface") == "ice":
                    break
            assert abs(float(state.get("surface_friction_override")) - 0.12) < 1e-6, state
            ice_frame = await recv_until(ws, "snow")
            assert ice_frame.get("surface_kind") == "ice", ice_frame.get("surface_kind")

            await control(ws, "surface", "rock")
            while True:
                state = await recv_until(ws, "state")
                if state.get("surface") == "rock":
                    break
            rock_frame = await recv_until(ws, "snow")
            assert rock_frame.get("surface_kind") == "rock", rock_frame.get("surface_kind")

            await control(ws, "surface", "snow")
            while True:
                state = await recv_until(ws, "state")
                if state.get("surface") == "snow" and state.get("newton", {}).get("active"):
                    break

            # Exercise the operator-facing failure -> active safety posture -> checkpoint
            # lifecycle over the same WebSocket Unity uses.  The bridge starts
            # with DEFAULT_SNOW, so the captured subset must be a live Newton
            # multilayer window rather than the rigid fallback.
            await control(ws, "demo_failure")
            waiting = None
            subset = None
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and (waiting is None or subset is None):
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
                if message.get("type") == "state":
                    candidate = message["data"]
                    supervisor = candidate.get("policy", {}).get("supervisor", {})
                    if supervisor.get("stage") == "waiting_checkpoint":
                        waiting = candidate
                elif message.get("type") == "subset_view":
                    subset = message["data"]
            assert waiting is not None, "demo failure did not enter safety posture"
            assert waiting.get("paused") is False
            assert waiting["policy"]["selected_policy_key"] == "auto"
            safety = waiting.get("simulation_settings", {}).get("safety_pose", {})
            assert safety.get("active") is True, safety
            assert safety.get("physics_live") is True, safety
            safety_start_time = float(waiting["sim_time"])
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                candidate = await recv_until(ws, "state", timeout=3.0)
                safety = candidate.get("simulation_settings", {}).get("safety_pose", {})
                if (
                    float(candidate["sim_time"]) >= safety_start_time + 0.5
                    and float(safety.get("transition_progress", 0.0)) >= 1.0
                ):
                    waiting = candidate
                    break
            else:
                raise AssertionError("safety posture did not keep physics advancing")
            supervisor = waiting["policy"]["supervisor"]
            manifest_path = Path(supervisor["request_manifest"])
            manifest = json.loads(manifest_path.read_text())
            terrain = manifest["environment"]["terrain"]
            assert terrain.get("mode") == "live", terrain
            assert terrain.get("vertices"), terrain
            assert terrain.get("layer_vertices"), terrain
            assert terrain.get("mpm", {}).get("solver"), terrain
            assert subset and subset.get("encoding") == "jpeg/base64"

            await control(ws, "play")
            held_pause = await control(ws, "pause", True, expected_ok=False)
            assert "physics live" in held_pause.get("message", "").lower(), held_pause
            held_manual = await control(ws, "manual_force_mode", True, expected_ok=False)
            assert "safe" in held_manual.get("message", "").lower(), held_manual
            held_command = await control(ws, "command", [1.0, 0.0, 0.0], expected_ok=False)
            assert "safe" in held_command.get("message", "").lower(), held_command

            bad_return = await control(ws, "checkpoint_return", {
                "key": "ice_incline",
                "label": "Invalid smoke-test checkpoint",
                "path": "/tmp/everest-missing-policy.onnx",
            }, expected_ok=False)
            assert "checkpoint" in bad_return.get("message", "").lower() or "onnx" in bad_return.get("message", "").lower(), bad_return

            await control(ws, "demo_return_pretrained", "ice_incline")
            active = None
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                candidate = await recv_until(ws, "state", timeout=3.0)
                supervisor = candidate.get("policy", {}).get("supervisor", {})
                if supervisor.get("stage") == "policy_active":
                    active = candidate
                    break
            assert active is not None, "demo checkpoint did not activate"
            assert active["policy"]["selected_policy_key"] == "ice_incline"
            assert active["policy"]["supervisor"]["active_policy_key"] == "ice_incline"
            assert active["policy"]["supervisor"]["demo_pretrained"] is True

            await control(ws, "weather", {
                "temperature_c": -24.0,
                "wind_speed_m_s": 22.0,
                "wind_direction_deg": 310.0,
                "snowfall_mm_h": 12.0,
                "visibility_scale": 0.55,
                "cloud_density": 0.61,
                "cloud_coverage": 0.72,
                "cloud_radius_m": 180.0,
                "cloud_altitude_m": 55.0,
                "cloud_thickness_m": 40.0,
                "cloud_speed": 0.5,
                "cloud_quality": 0.7,
                "movement_allowed": True,
            })
            env_state = await recv_until(ws, "environment")
            assert abs(float(env_state["wind_speed_m_s"]) - 22.0) < 1e-6
            assert abs(float(env_state["cloud_radius_m"]) - 180.0) < 1e-6

            await control(ws, "manual_force_mode", True)
            while True:
                state = await recv_until(ws, "state")
                if state.get("manual_force_mode"):
                    break
            await control(ws, "manual_force_mode", False)

            before = await recv_until(ws, "frame")
            before_x = float(before["body_pos_w"][0][0])
            await control(ws, "cheat_mode", True)
            await control(ws, "command", [1.0, 0.0, 0.0])
            await control(ws, "pause", False)
            deadline = time.monotonic() + 5.0
            moved = False
            while time.monotonic() < deadline:
                frame = await recv_until(ws, "frame")
                if abs(float(frame["body_pos_w"][0][0]) - before_x) > 0.03:
                    moved = True
                    break
            assert moved, "cheat mode did not translate the floating base"
            await control(ws, "command", [0.0, 0.0, 0.0])
            await control(ws, "cheat_mode", False)
            await control(ws, "pause", True)

            print("Unity sim controls OK: radius/LOD, snow history, force control, snow/ice/rock, clouds, and cheat transport")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    asyncio.run(main())
