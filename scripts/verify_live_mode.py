#!/usr/bin/env python3
"""Verify configured LIVE switching, provenance, staleness, and lockout."""

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
PORT = int(os.environ.get("EVEREST_LIVE_TEST_PORT", "18768"))
URL = f"ws://127.0.0.1:{PORT}"
REPLAY = ROOT / "tests/fixtures/live_replay.json"
PENDING: dict[str, list[dict]] = {}


def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


async def receive(ws, wanted: str, predicate=lambda _: True, timeout: float = 10.0):
    queued = PENDING.get(wanted, [])
    for index, data in enumerate(queued):
        if predicate(data):
            queued.pop(index)
            return data
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        kind = message.get("type")
        data = message.get("data", {})
        if kind == wanted:
            if predicate(data):
                return data
        else:
            PENDING.setdefault(str(kind), []).append(data)
    raise TimeoutError(wanted)


async def control(ws, action, value, *, ok=True):
    await ws.send(json.dumps({"type": "control", "action": action, "value": value}))
    ack = await receive(ws, "control_ack")
    assert bool(ack.get("ok")) is ok, ack
    return ack


async def main() -> None:
    if port_open():
        raise RuntimeError(f"test port {PORT} is already occupied")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib{':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''}"
    process = subprocess.Popen(
        [
            str(ROOT / "scripts/start-simulation-backend.sh"),
            "--port", str(PORT),
            "--live-adapter", "replay",
            "--live-replay", str(REPLAY),
            "--live-stale-ms", "1500",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(160):
            if port_open():
                break
            if process.poll() is not None:
                raise RuntimeError(process.stdout.read())
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("LIVE test backend did not start")

        async with websockets.connect(URL, max_size=16 * 1024 * 1024, compression=None) as ws:
            initial = await receive(ws, "state")
            assert initial["data_mode"] == "sim"
            sim_epoch = initial["source"]["epoch"]

            await control(ws, "mode", "live")
            state = await receive(ws, "state", lambda item: item.get("data_mode") == "live")
            assert state["control_authority"] == "read_only"
            assert state["source"]["kind"] == "replay"
            assert state["source"]["epoch"] > sim_epoch
            assert state["source"]["status"] == "connected", state["source"]

            frame = await receive(ws, "frame", lambda item: item.get("data_mode") == "live")
            assert frame["engine"] == "replay"
            assert frame["body_pos_w"][0] == [1.0, 2.0, 0.82]
            assert frame["source_epoch"] == state["source"]["epoch"]
            environment = await receive(ws, "environment", lambda item: item.get("data_mode") == "live")
            assert abs(float(environment["wind_speed_m_s"]) - 20.0) < 1e-6
            assert environment["provenance"] == "south-col-weather-station"
            sensors = await receive(ws, "sensors")
            assert sensors["foot_pressure"]["left_n"] == 280.0

            ack = await control(ws, "reset", None, ok=False)
            assert ack["code"] == "live_read_only"

            stale = await receive(
                ws,
                "state",
                lambda item: item.get("data_mode") == "live" and item.get("source", {}).get("status") == "stale",
                timeout=5.0,
            )
            assert stale["source"]["channels"]["robot"]["status"] == "stale"

            await control(ws, "mode", "sim")
            sim_state = await receive(
                ws,
                "state",
                lambda item: item.get("data_mode") == "sim" and item.get("source", {}).get("epoch", 0) > frame["source_epoch"],
            )
            sim_frame = await receive(
                ws,
                "frame",
                lambda item: item.get("data_mode") == "sim" and item.get("source_epoch", 0) > frame["source_epoch"],
            )
            assert sim_state["control_authority"] == "simulation"
            assert sim_frame["engine"] == "newton+mujoco"
            assert sim_frame["source_epoch"] > frame["source_epoch"]
            print("LIVE mode OK: replay provenance, all channels, stale state, read-only lockout, and SIM restore")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    asyncio.run(main())
