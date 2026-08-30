#!/usr/bin/env python3
"""Realtime bridge: MuJoCo owns G1, Newton owns snow, Unreal renders both."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/newton"))

import websockets
from websockets.asyncio.server import ServerConnection

import dashboard.engines.mujoco as mujoco_engine

LOCAL_TERRAIN = ROOT / "studio/unreal/config/everest_robot_terrain.json"
MACRO_TERRAIN = ROOT / "studio/unreal/config/everest_macro_terrain.json"
mujoco_engine.TERRAIN_MANIFEST = LOCAL_TERRAIN
MuJoCoEngine = mujoco_engine.MuJoCoEngine

DEFAULT_SNOW = ROOT / "studio/unreal/config/default_snow.json"


class EverestUnrealBridge:
    def __init__(self, host: str, port: int, telemetry_hz: float) -> None:
        self.host = host
        self.port = port
        self.engine = MuJoCoEngine(telemetry_hz=telemetry_hz)
        self.clients: set[ServerConnection] = set()
        self.stop = asyncio.Event()
        self.last_sequence = -1
        self.last_snow_sequence = -1

    def configure_default_snow(self, path: Path = DEFAULT_SNOW) -> None:
        payload = json.loads(path.read_text())
        self.engine.control("snow_parameters", payload)

    def warm_up_physics(self) -> None:
        # Compile Warp/Newton kernels before Unreal starts receiving frames.
        # Reset immediately afterward so warmup never becomes simulation state.
        patch = getattr(self.engine, "_snow_patch", None)
        if patch is not None:
            patch.step(self.engine._foot_poses())
            self.engine.reset()

    async def _send(self, ws: ServerConnection, message_type: str, data: dict[str, Any]) -> None:
        await ws.send(json.dumps({"type": message_type, "data": data}, separators=(",", ":")))

    async def handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        try:
            await self._send(ws, "scene", self.engine.scene_manifest())
            await self._send(ws, "terrain", self.engine.terrain_tile())
            await self._send(ws, "macro_terrain", json.loads(MACRO_TERRAIN.read_text()))
            await self._send(ws, "state", self.engine.state())
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("type") != "control":
                        continue
                    action = str(msg.get("action", ""))
                    value = msg.get("value")
                    if action == "command":
                        if not isinstance(value, list) or len(value) != 3:
                            raise ValueError("command must be [forward, lateral, yaw_rate]")
                        command = [float(v) for v in value]
                        if any(abs(v) > 1e-6 for v in command):
                            self.engine.control("pause", False)
                        self.engine.control("command", command)
                    elif action == "pause":
                        self.engine.control("pause", bool(value))
                    elif action == "reset":
                        self.engine.control("reset")
                    elif action in {"snow_parameters", "weather", "terrain_edit", "surface"}:
                        self.engine.control(action, value)
                    await self._send(ws, "state", self.engine.state())
                except Exception as exc:
                    await self._send(ws, "error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.clients.discard(ws)

    async def broadcast(self, message_type: str, data: dict[str, Any]) -> None:
        if not self.clients:
            return
        payload = json.dumps({"type": message_type, "data": data}, separators=(",", ":"))
        stale: list[ServerConnection] = []
        for ws in tuple(self.clients):
            try:
                await ws.send(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)

    async def pump(self) -> None:
        while not self.stop.is_set():
            frame = self.engine.frame()
            sequence = int(frame.get("sequence", -1))
            if sequence != self.last_sequence:
                self.last_sequence = sequence
                await self.broadcast("frame", frame)
                if sequence - self.last_snow_sequence >= 2:
                    self.last_snow_sequence = sequence
                    await self.broadcast("snow", self.engine.terrain_frame())
            await asyncio.sleep(1.0 / 120.0)

    async def run(self) -> None:
        self.configure_default_snow()
        self.warm_up_physics()
        self.engine.start()
        try:
            async with websockets.serve(
                self.handler,
                self.host,
                self.port,
                max_size=None,
                compression="deflate",
                ping_interval=10,
                ping_timeout=10,
            ):
                state = self.engine.state()
                print(
                    f"Everest Unreal bridge ws://{self.host}:{self.port} | "
                    f"engine={state['engine']} terrain={state['terrain_collision']} "
                    f"snow_mpm={state['snow'].get('mpm', {}).get('active', False)}",
                    flush=True,
                )
                await self.pump()
        finally:
            self.engine.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--telemetry-hz", type=float, default=30.0)
    p.add_argument("--probe", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    bridge = EverestUnrealBridge(args.host, args.port, args.telemetry_hz)
    if args.probe:
        bridge.configure_default_snow()
        state = bridge.engine.state()
        print(json.dumps({
            "engine": state["engine"],
            "terrain_collision": state["terrain_collision"],
            "snow": state["snow"],
            "bodies": state["bodies"],
            "terrain_grid": [bridge.engine.terrain_tile()["grid_width"], bridge.engine.terrain_tile()["grid_height"]],
        }, indent=2))
        bridge.engine.stop()
        return
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bridge.stop.set)
        except NotImplementedError:
            pass
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
