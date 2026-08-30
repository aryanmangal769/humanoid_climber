"""Renderer-neutral WebSocket bridge for Unity.

Unity receives authoritative MuJoCo/Newton state and sends high-level controls.
It never imports either physics engine.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import time
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from dashboard.engines.mujoco import MuJoCoEngine, ROOT
from simulation.data_sources import LiveDataSource
from simulation.live_telemetry import (
    CompositeTelemetryAdapter,
    JsonFileTelemetryAdapter,
    OpenMeteoWeatherAdapter,
    ReplayTelemetryAdapter,
    UdpTelemetryAdapter,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MACRO_TERRAIN = ROOT / "maps/everest_macro_terrain.json"

DEFAULT_SNOW = {
    "surface_friction": 0.36,
    "snowfall_mm_h": 1.5,
    "wind_speed_m_s": 9.0,
    "wind_direction_deg": 250.0,
    "temperature_c": -18.0,
    "slope_deg": 18.0,
    "layers": [
        {
            "type": "POWDER",
            "label": "Fresh powder",
            "color": [0.94, 0.97, 1.0],
            "thickness_m": 0.08,
            "density_kg_m3": 120.0,
            "stiffness_pa": 35_000.0,
            "compressive_strength_pa": 3_500.0,
            "shear_strength_pa": 1_800.0,
            "compaction_hardening": 12.0,
            "bond_strength_below_pa": 2_500.0,
        },
        {
            "type": "DENSE_SNOW",
            "label": "Settled snow",
            "color": [0.80, 0.88, 0.95],
            "thickness_m": 0.32,
            "density_kg_m3": 320.0,
            "stiffness_pa": 500_000.0,
            "compressive_strength_pa": 55_000.0,
            "shear_strength_pa": 20_000.0,
            "compaction_hardening": 18.0,
            "bond_strength_below_pa": 7_500.0,
        },
    ],
}

DEFAULT_ENVIRONMENT = {
    "data_mode": "sim",
    "temperature_c": -18.0,
    "wind_speed_m_s": 9.0,
    "wind_direction_deg": 250.0,
    "snowfall_mm_h": 1.5,
    "visibility_scale": 0.88,
    "cloud_density": 0.42,
    "cloud_coverage": 0.58,
    "cloud_radius_m": 170.0,
    "cloud_altitude_m": 38.0,
    "cloud_thickness_m": 46.0,
    "cloud_speed": 0.40,
    "cloud_quality": 0.55,
    "movement_allowed": True,
}


def _message(kind: str, data: Any) -> str:
    return json.dumps({"type": kind, "data": data}, separators=(",", ":"))


class ControlRejected(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class UnityRendererBridge:
    def __init__(
        self,
        *,
        particles: bool = False,
        live_source: LiveDataSource | None = None,
    ) -> None:
        if not MACRO_TERRAIN.is_file():
            raise FileNotFoundError(
                f"Macro terrain missing: {MACRO_TERRAIN}. Run maps/build_unity_terrain.py."
            )
        self.engine = MuJoCoEngine(telemetry_hz=60.0)
        self.engine.control("snow_parameters", copy.deepcopy(DEFAULT_SNOW))
        self.data_mode = "sim"
        self.environment = copy.deepcopy(DEFAULT_ENVIRONMENT)
        self.live_source = live_source
        self.source_epoch = 1
        self._live_generation = live_source.generation if live_source is not None else 0
        self.include_particles = particles
        self.macro_terrain = json.loads(MACRO_TERRAIN.read_text())
        self.engine.start()
        if self.live_source is not None:
            layout = self.engine.frame()
            self.live_source.set_expected_layout(
                list(layout.get("body_names", [])),
                list(layout.get("joint_names", [])),
            )

    def close(self) -> None:
        if self.live_source is not None:
            self.live_source.close()
        self.engine.stop()

    def _refresh_live_epoch(self) -> None:
        if self.data_mode != "live" or self.live_source is None:
            return
        generation = self.live_source.generation
        if generation != self._live_generation:
            self._live_generation = generation
            self.source_epoch += 1

    def scene(self) -> dict[str, Any]:
        source = self.engine.scene_manifest()
        visuals = []
        for visual in source.get("visuals", []):
            item = copy.deepcopy(visual)
            url = item.pop("url", "")
            item["asset"] = Path(url).name if url else f"{item.get('mesh', 'mesh')}.obj"
            visuals.append(item)
        return {
            "schema": "everest-scene/v1",
            "model": "Unitree G1",
            "units": "metres",
            "up_axis": "z",
            "handedness": "right",
            "quaternion_order": "wxyz",
            "body_names": self.engine.frame().get("body_names", []),
            "visuals": visuals,
        }

    def local_terrain(self) -> dict[str, Any]:
        result = self.engine.terrain_tile()
        result["units"] = "metres"
        result["up_axis"] = "z"
        result["handedness"] = "right"
        return result

    def macro(self) -> dict[str, Any]:
        result = copy.deepcopy(self.macro_terrain)
        result["units"] = "metres"
        result["up_axis"] = "z"
        result["handedness"] = "right"
        return result

    def frame(self) -> dict[str, Any] | None:
        self._refresh_live_epoch()
        if self.data_mode == "live":
            if self.live_source is None:
                return None
            live = self.live_source.frame()
            if live is None:
                return None
            live["data_mode"] = "live"
            live["source_epoch"] = self.source_epoch
            return live
        frame = self.engine.frame()
        return {
            "schema": "everest-viewer/v1",
            "sequence": int(frame["sequence"]),
            "timestamp": float(frame["timestamp"]),
            "sim_time": float(frame["sim_time"]),
            "engine": "newton+mujoco",
            "body_names": frame["body_names"],
            "body_pos_w": frame["body_pos_w"],
            "body_quat_w": frame["body_quat_w"],
            "base_linear_velocity": frame["base_linear_velocity"],
            "base_angular_velocity": frame["base_angular_velocity"],
            "joint_names": frame["joint_names"],
            "joint_positions": frame["joint_positions"],
            "joint_velocities": frame["joint_velocities"],
            "joint_torques": frame["joint_torques"],
            "command": frame["command"],
            "feet": frame["feet"],
            "paused": bool(frame["paused"]),
            "data_mode": "sim",
            "source_epoch": self.source_epoch,
        }

    def snow(self) -> dict[str, Any] | None:
        self._refresh_live_epoch()
        if self.data_mode == "live":
            if self.live_source is None:
                return None
            result = self.live_source.snow()
            if result is not None:
                result["source_epoch"] = self.source_epoch
                result["data_mode"] = "live"
            return result
        terrain = self.engine.terrain_frame(include_particles=self.include_particles)
        if terrain.get("mode") != "live":
            return None
        result = {
            "schema": "everest-snow-surface/v1",
            "sequence": int(terrain["sequence"]),
            "timestamp": time.time(),
            "sim_time": float(terrain["sim_time"]),
            "origin": terrain["origin"],
            "size": terrain["size"],
            "resolution": terrain["resolution"],
            "heights": terrain["heights"],
            "vertices": terrain.get("vertices"),
            "base_heights": terrain.get("base_heights", terrain["heights"]),
            "layer_heights": terrain.get("layer_heights", []),
            "layer_vertices": terrain.get("layer_vertices", []),
            "substrate_vertices": terrain.get("substrate_vertices"),
            "layer_boundary_model": terrain.get("layer_boundary_model"),
            "compaction": terrain["compaction"],
            "material_ids": terrain["material_ids"],
            "surface_kind": terrain.get("surface_kind", "snow"),
            "surface_depth_m": float(terrain["surface_depth"]),
            "surface_friction": float(terrain["surface_friction"]),
            "layers": terrain["layers"],
            "mpm": terrain["mpm"],
        }
        if self.include_particles:
            result["particles"] = terrain.get("particles")
        result["source_epoch"] = self.source_epoch
        result["data_mode"] = "sim"
        return result

    def terrain(self) -> dict[str, Any] | None:
        if self.data_mode == "sim":
            result = self.local_terrain()
            result["source_epoch"] = self.source_epoch
            result["data_mode"] = "sim"
            result.setdefault("sequence", 0)
            return result
        if self.live_source is None:
            return None
        result = self.live_source.terrain()
        if result is not None:
            result["source_epoch"] = self.source_epoch
            result["data_mode"] = "live"
        return result

    def environment_payload(self) -> dict[str, Any]:
        if self.data_mode == "live" and self.live_source is not None:
            return self.live_source.environment()
        return copy.deepcopy(self.environment)

    def sensors(self) -> dict[str, Any] | None:
        if self.data_mode != "live" or self.live_source is None:
            return None
        result = self.live_source.sensors()
        if result is not None:
            result["source_epoch"] = self.source_epoch
            result["data_mode"] = "live"
        return result

    def snow_history(self) -> dict[str, Any]:
        return self.engine.snow_history()

    def state(self) -> dict[str, Any]:
        self._refresh_live_epoch()
        raw = self.engine.state()
        snow = raw.get("snow") or {}
        mpm = snow.get("mpm") or {}
        policy = raw.get("policy") or {}
        if self.data_mode == "live" and self.live_source is not None:
            source = self.live_source.health()
            source["epoch"] = self.source_epoch
        else:
            source = {
                "kind": "newton+mujoco",
                "name": "local-simulator",
                "status": "connected" if raw.get("telemetry_error") is None else "disconnected",
                "epoch": self.source_epoch,
                "sample_time": time.time(),
                "age_ms": 0.0,
                "stale_after_ms": None,
                "last_error": raw.get("telemetry_error"),
                "channels": {
                    "robot": {"status": "connected", "age_ms": 0.0, "provenance": "MuJoCo"},
                    "weather": {"status": "connected", "age_ms": 0.0, "provenance": "operator/sim"},
                    "terrain": {"status": "connected", "age_ms": 0.0, "provenance": "static DEM"},
                    "snow": {"status": "connected", "age_ms": 0.0, "provenance": "Newton predicted"},
                    "sensors": {"status": "unavailable", "age_ms": None, "provenance": None},
                },
            }
        result = {
            "schema": "everest-state/v1",
            "sequence": int(raw.get("frames", 0)),
            "timestamp": time.time(),
            "sim_time": float(raw.get("sim_time", 0.0)),
            "engine": "newton+mujoco" if self.data_mode == "sim" else source["kind"],
            "data_mode": self.data_mode,
            "source": source,
            "control_authority": "simulation" if self.data_mode == "sim" else "read_only",
            "surface": raw.get("surface", "snow"),
            "surface_friction_override": raw.get("surface_friction_override"),
            "simulation_settings": copy.deepcopy(raw.get("simulation_settings") or {}),
            "cheat_mode": bool((raw.get("simulation_settings") or {}).get("cheat_mode", False)),
            "manual_force_mode": bool((raw.get("simulation_settings") or {}).get("manual_force_mode", False)),
            "mujoco": {"active": raw.get("telemetry_error") is None},
            "newton": {
                "active": bool(raw.get("snow", {}).get("mpm_active")),
                "solver": mpm.get("solver", "SolverImplicitMPM"),
                "device": mpm.get("device"),
                "particle_count": int(mpm.get("particle_count", 0)),
                "steps": int(mpm.get("steps", 0)),
                "solver_steps": int(mpm.get("solver_steps", 0)),
                "contact_skipped_steps": int(mpm.get("contact_skipped_steps", 0)),
                "active_particle_count": int(mpm.get("active_particle_count", 0)),
                "voxel_size_m": mpm.get("voxel_size_m"),
                "window_size_m": copy.deepcopy(mpm.get("window_size_m")),
                "terrain_conforming": bool(mpm.get("terrain_conforming", False)),
                "history_restored_particles": int(mpm.get("history_restored_particles", 0)),
                "contact_refine_radius_m": mpm.get("contact_refine_radius_m"),
                "coarse_stride": int(mpm.get("coarse_stride", 1)),
                "accumulation_enabled": bool(mpm.get("accumulation_enabled", False)),
                "accumulation_time_scale": float(mpm.get("accumulation_time_scale", 1.0)),
                "deposited_depth_m": float(mpm.get("deposited_depth_m", 0.0)),
                "deposited_mass_kg": float(mpm.get("deposited_mass_kg", 0.0)),
            },
            "policy": {"active": bool(policy.get("enabled")), "inference_count": policy.get("inference_count", 0)},
            "paused": bool(raw.get("paused", True)),
            "simulation_fault": raw.get("simulation_fault") or raw.get("telemetry_error"),
            "snow_history": copy.deepcopy(raw.get("snow_history") or {}),
        }
        if self.data_mode == "live":
            # These fields describe the parked local simulator and must not be
            # mistaken for live physics/source health.
            result["paused"] = False
            result["simulation_fault"] = source.get("last_error")
            result["local_simulator"] = {
                "paused": bool(raw.get("paused", True)),
                "sim_time": float(raw.get("sim_time", 0.0)),
                "publishing": False,
            }
        return result

    def _normalize_weather(self, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("schema") == "everest-weather/v1":
            conditions = value.get("conditions") or {}
            simulation = value.get("simulation") or {}
            self.environment.update({
                "temperature_c": conditions.get("temperature_c", self.environment["temperature_c"]),
                "wind_speed_m_s": float(conditions.get("wind_speed_kmh", 0.0)) / 3.6,
                "wind_direction_deg": conditions.get("wind_direction_deg", self.environment["wind_direction_deg"]),
                "visibility_scale": simulation.get("visibility_scale", self.environment["visibility_scale"]),
                "movement_allowed": simulation.get("movement_allowed", self.environment["movement_allowed"]),
            })
            return value
        self.environment.update({key: value[key] for key in self.environment if key in value})
        wind = float(self.environment["wind_speed_m_s"])
        return {
            "schema": "everest-weather/v1",
            "conditions": {
                "temperature_c": float(self.environment["temperature_c"]),
                "wind_speed_kmh": wind * 3.6,
                "wind_direction_deg": float(self.environment["wind_direction_deg"]),
            },
            "simulation": {
                "wind_force_scale": min(1.0, max(0.0, wind / 45.0)),
                "terrain_friction_scale": 1.0,
                "visibility_scale": min(1.0, max(0.0, float(self.environment["visibility_scale"]))),
                "movement_allowed": bool(self.environment["movement_allowed"]),
            },
        }

    def control(self, action: str, value: Any) -> None:
        if action == "mode":
            mode = str(value or "").strip().lower()
            if mode not in {"sim", "live"}:
                raise ValueError("mode must be sim or live")
            if mode == self.data_mode:
                return
            if mode == "live" and self.live_source is None:
                raise ControlRejected(
                    "LIVE is not configured; start the backend with a live adapter",
                    code="live_not_configured",
                )
            if mode == "live":
                # Stop the local simulator's held command before changing the
                # publisher. Selecting LIVE never forwards a hardware command.
                self.engine.control("command", [0.0, 0.0, 0.0])
                self.engine.control("manual_force_mode", False)
                self.engine.control("cheat_mode", False)
            self.data_mode = mode
            self.environment["data_mode"] = mode
            self.source_epoch += 1
            return
        if self.data_mode == "live":
            raise ControlRejected(
                f"{action} is read-only in LIVE mode",
                code="live_read_only",
            )
        if action == "surface":
            self.engine.control("surface", value)
            return
        if action == "surface_friction":
            self.engine.control("surface_friction", value)
            return
        if action == "simulation_settings":
            if not isinstance(value, dict):
                raise ValueError("simulation_settings control requires an object")
            self.engine.control("simulation_settings", value)
            return
        if action == "cheat_mode":
            self.engine.control("cheat_mode", bool(value))
            return
        if action == "manual_force_mode":
            self.engine.control("manual_force_mode", bool(value))
            return
        if action == "weather":
            if not isinstance(value, dict):
                raise ValueError("weather control requires an object")
            self.engine.control("weather", self._normalize_weather(value))
            if self.data_mode == "sim" and self.engine.snow.surface == "snow" and self.engine.snow.column is not None:
                self.engine.control("snow_forcing", {
                    "temperature_c": float(self.environment["temperature_c"]),
                    "wind_speed_m_s": float(self.environment["wind_speed_m_s"]),
                    "wind_direction_deg": float(self.environment["wind_direction_deg"]),
                    "snowfall_mm_h": float(self.environment["snowfall_mm_h"]),
                })
            return
        if action in {"command", "pause", "reset", "snow_parameters"}:
            self.engine.control(action, value)
            if action == "snow_parameters" and isinstance(value, dict):
                for key in ("temperature_c", "wind_speed_m_s", "wind_direction_deg", "snowfall_mm_h"):
                    if key in value:
                        self.environment[key] = value[key]
            return
        raise ValueError(f"Unsupported control action: {action}")


async def _serve_client(socket: ServerConnection, bridge: UnityRendererBridge) -> None:
    await socket.send(_message("scene", bridge.scene()))
    await socket.send(_message("terrain", bridge.local_terrain()))
    await socket.send(_message("macro_terrain", bridge.macro()))
    await socket.send(_message("environment", bridge.environment_payload()))
    await socket.send(_message("state", bridge.state()))

    async def receive_controls() -> None:
        async for raw in socket:
            try:
                payload = json.loads(raw)
                if payload.get("type") != "control":
                    raise ValueError("client messages must use type=control")
                bridge.control(str(payload.get("action")), payload.get("value"))
                await socket.send(_message("control_ack", {"action": payload.get("action"), "ok": True}))
                if payload.get("action") == "mode":
                    await socket.send(_message("state", bridge.state()))
                    await socket.send(_message("environment", bridge.environment_payload()))
            except Exception as exc:
                error = {"ok": False, "message": str(exc)}
                if isinstance(exc, ControlRejected):
                    error["code"] = exc.code
                await socket.send(_message("control_ack", error))

    async def publish() -> None:
        next_frame = next_snow = next_state = 0.0
        last_frame: tuple[int, int] | None = None
        last_snow: tuple[int, int] | None = None
        last_terrain: tuple[int, int] | None = None
        last_sensors: tuple[int, float] | None = None
        last_history = -1
        last_fault: str | None = None
        while True:
            now = asyncio.get_running_loop().time()
            if now >= next_frame:
                frame = bridge.frame()
                frame_key = None if frame is None else (
                    int(frame.get("source_epoch", 0)), int(frame.get("sequence", 0))
                )
                if frame is not None and frame_key != last_frame:
                    await socket.send(_message("frame", frame))
                    last_frame = frame_key
                next_frame = now + 1.0 / 60.0
            if now >= next_snow:
                snow = bridge.snow()
                snow_key = None if snow is None else (
                    int(snow.get("source_epoch", 0)), int(snow.get("sequence", 0))
                )
                if snow is not None and snow_key != last_snow:
                    await socket.send(_message("snow", snow))
                    last_snow = snow_key
                terrain = bridge.terrain()
                terrain_key = None if terrain is None else (
                    int(terrain.get("source_epoch", 0)), int(terrain.get("sequence", 0))
                )
                if terrain is not None and terrain_key != last_terrain:
                    await socket.send(_message("terrain", terrain))
                    last_terrain = terrain_key
                sensors = bridge.sensors()
                sensors_key = None if sensors is None else (
                    int(sensors.get("source_epoch", 0)), float(sensors.get("sample_time", 0.0))
                )
                if sensors is not None and sensors_key != last_sensors:
                    await socket.send(_message("sensors", sensors))
                    last_sensors = sensors_key
                if bridge.data_mode == "sim":
                    history = bridge.snow_history()
                    if history["sequence"] != last_history:
                        await socket.send(_message("snow_history", history))
                        last_history = history["sequence"]
                next_snow = now + 1.0 / 15.0
            if now >= next_state:
                state = bridge.state()
                await socket.send(_message("state", state))
                await socket.send(_message("environment", bridge.environment_payload()))
                fault = state.get("simulation_fault")
                if fault and fault != last_fault:
                    await socket.send(_message("fault", {
                        "source": "live_telemetry" if bridge.data_mode == "live" else "physics",
                        "message": fault,
                        "sim_time": state["sim_time"],
                    }))
                last_fault = fault
                next_state = now + 0.5
            await asyncio.sleep(0.003)

    receiver = asyncio.create_task(receive_controls())
    publisher = asyncio.create_task(publish())
    done, pending = await asyncio.wait((receiver, publisher), return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        task.result()


def _probe(bridge: UnityRendererBridge, host: str, port: int) -> None:
    state = bridge.state()
    local = bridge.local_terrain()
    snow = bridge.snow()
    print("MuJoCo: OK" if state["mujoco"]["active"] else "MuJoCo: FAILED")
    print(f"G1: OK ({len(bridge.frame()['body_names'])} bodies)")
    print("policy: OK" if state["policy"]["active"] else "policy: FAILED")
    print()
    print("Newton: OK" if state["newton"]["active"] else "Newton: FAILED")
    print(f"solver: {state['newton']['solver']}")
    print(f"device: {state['newton']['device']}")
    try:
        import warp as wp
        gpu = wp.get_device(state["newton"]["device"]).name if state["newton"]["device"] else "unavailable"
    except Exception:
        gpu = "unavailable"
    print(f"GPU: {gpu}")
    print()
    print(f"terrain: OK ({local['grid_width']}x{local['grid_height']}, {local['world_width_m']:.0f}m x {local['world_depth_m']:.0f}m)")
    print(f"snow layers: {len(snow['layers']) if snow else 0}")
    print(f"MPM particles: {state['newton']['particle_count']}")
    print()
    print(f"WebSocket: ws://{host}:{port}")


async def _run(
    host: str,
    port: int,
    particles: bool,
    live_source: LiveDataSource | None,
) -> None:
    bridge = UnityRendererBridge(particles=particles, live_source=live_source)
    try:
        async with websockets.serve(
            lambda socket: _serve_client(socket, bridge),
            host,
            port,
            max_size=16 * 1024 * 1024,
            compression=None,
            ping_interval=20,
        ):
            print(f"Everest simulation backend listening on ws://{host}:{port}", flush=True)
            if live_source is None:
                print("LIVE source: disabled", flush=True)
            else:
                print(f"LIVE source: {live_source.kind} ({live_source.name})", flush=True)
            await asyncio.Future()
    finally:
        bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--particles", action="store_true", help="include raw MPM particles in snow messages")
    parser.add_argument(
        "--live-adapter",
        choices=("disabled", "replay", "json-file", "udp"),
        default="disabled",
        help="read-only robot/sensor telemetry transport",
    )
    parser.add_argument("--live-endpoint", help="path for json-file or bind host:port for udp")
    parser.add_argument("--live-replay", help="everest-live-replay/v1 JSON or JSONL fixture")
    parser.add_argument("--live-replay-speed", type=float, default=1.0)
    parser.add_argument("--live-stale-ms", type=float, default=250.0)
    parser.add_argument(
        "--live-weather",
        choices=("disabled", "open-meteo"),
        default="disabled",
        help="optional independent live weather channel",
    )
    parser.add_argument("--live-weather-location", default="south-col")
    parser.add_argument("--live-weather-interval", type=float, default=300.0)
    args = parser.parse_args()
    adapters = []
    if args.live_adapter == "replay":
        if not args.live_replay:
            parser.error("--live-adapter replay requires --live-replay")
        adapters.append(ReplayTelemetryAdapter(args.live_replay, speed=args.live_replay_speed))
    elif args.live_adapter == "json-file":
        if not args.live_endpoint:
            parser.error("--live-adapter json-file requires --live-endpoint")
        adapters.append(JsonFileTelemetryAdapter(args.live_endpoint))
    elif args.live_adapter == "udp":
        if not args.live_endpoint:
            parser.error("--live-adapter udp requires --live-endpoint host:port")
        adapters.append(UdpTelemetryAdapter(args.live_endpoint))
    if args.live_weather == "open-meteo":
        adapters.append(OpenMeteoWeatherAdapter(
            args.live_weather_location,
            interval_s=args.live_weather_interval,
        ))
    live_source = None
    if adapters:
        adapter = adapters[0] if len(adapters) == 1 else CompositeTelemetryAdapter(adapters)
        live_source = LiveDataSource(
            adapter,
            default_environment=DEFAULT_ENVIRONMENT,
            stale_after_ms=args.live_stale_ms,
        )
    if args.probe:
        bridge = UnityRendererBridge(particles=args.particles, live_source=live_source)
        try:
            _probe(bridge, args.host, args.port)
        finally:
            bridge.close()
        return
    asyncio.run(_run(args.host, args.port, args.particles, live_source))


if __name__ == "__main__":
    main()
