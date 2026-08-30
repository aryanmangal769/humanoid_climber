"""Renderer data sources for truthful SIM/LIVE source switching."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
import math
import time
from typing import Any, Protocol

from simulation.live_telemetry import LiveTelemetryAdapter


LIVE_CHANNELS = ("robot", "weather", "terrain", "snow", "sensors")


class RendererDataSource(Protocol):
    def frame(self) -> dict[str, Any] | None: ...
    def environment(self) -> dict[str, Any]: ...
    def snow(self) -> dict[str, Any] | None: ...
    def terrain(self) -> dict[str, Any] | None: ...
    def health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class LiveSampleError(ValueError):
    pass


@dataclass
class _Channel:
    data: dict[str, Any]
    sample_time: float
    receipt_time: float
    provenance: str


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise LiveSampleError("boolean is not a valid sample timestamp")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    else:
        raise LiveSampleError("sample_time must be Unix seconds or ISO-8601")
    if not math.isfinite(result):
        raise LiveSampleError("sample_time must be finite")
    return result


def _unwrap_channel(value: Any, fallback_time: float) -> tuple[dict[str, Any], float, str | None]:
    if not isinstance(value, dict):
        raise LiveSampleError("live channel payload must be an object")
    if isinstance(value.get("data"), dict):
        data = copy.deepcopy(value["data"])
        sample_time = _timestamp(value.get("sample_time", fallback_time))
        provenance = value.get("provenance")
        return data, sample_time, str(provenance) if provenance is not None else None
    return copy.deepcopy(value), fallback_time, None


def _validate_robot(frame: dict[str, Any]) -> None:
    required = ("body_names", "body_pos_w", "body_quat_w")
    missing = [key for key in required if not isinstance(frame.get(key), list)]
    if missing:
        raise LiveSampleError(f"robot channel missing arrays: {', '.join(missing)}")
    body_count = len(frame["body_names"])
    if len(frame["body_pos_w"]) != body_count or len(frame["body_quat_w"]) != body_count:
        raise LiveSampleError("robot body name/position/quaternion lengths differ")
    for position in frame["body_pos_w"]:
        if not isinstance(position, list) or len(position) != 3:
            raise LiveSampleError("robot positions must be xyz triples")
    for quaternion in frame["body_quat_w"]:
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            raise LiveSampleError("robot quaternions must be WXYZ quadruples")
    joint_names = frame.get("joint_names", [])
    if not isinstance(joint_names, list):
        raise LiveSampleError("robot joint_names must be an array")
    for key in ("joint_positions", "joint_velocities", "joint_torques"):
        values = frame.get(key, [])
        if not isinstance(values, list):
            raise LiveSampleError(f"robot {key} must be an array")
        if values and len(values) != len(joint_names):
            raise LiveSampleError(f"robot joint_names/{key} lengths differ")


def _validate_grid(data: dict[str, Any], *, channel: str) -> None:
    if channel == "terrain":
        width = int(data.get("grid_width", 0))
        height = int(data.get("grid_height", 0))
    else:
        resolution = data.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise LiveSampleError("snow resolution must be [width, height]")
        width, height = int(resolution[0]), int(resolution[1])
    heights = data.get("heights")
    if width < 2 or height < 2 or not isinstance(heights, list) or len(heights) != width * height:
        raise LiveSampleError(f"{channel} grid dimensions do not match heights")


def weather_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw environment or everest-weather/v1 payload for Unity."""
    if payload.get("schema") != "everest-weather/v1":
        return copy.deepcopy(payload)
    conditions = payload.get("conditions") or {}
    simulation = payload.get("simulation") or {}
    snow_prior = simulation.get("snow_prior") or {}
    wind_kmh = float(conditions.get("wind_speed_kmh") or 0.0)
    cloud_percent = float(conditions.get("cloud_cover_percent") or 0.0)
    return {
        "temperature_c": float(conditions.get("temperature_c") or 0.0),
        "wind_speed_m_s": wind_kmh / 3.6,
        "wind_gust_m_s": float(conditions.get("wind_gust_kmh") or wind_kmh) / 3.6,
        "wind_direction_deg": float(conditions.get("wind_direction_deg") or 0.0),
        "snowfall_mm_h": float(snow_prior.get("snowfall_rate_mm_h") or 0.0),
        "visibility_scale": float(simulation.get("visibility_scale", 1.0)),
        "cloud_density": min(1.0, max(0.0, cloud_percent / 100.0)),
        "cloud_coverage": min(1.0, max(0.0, cloud_percent / 100.0)),
        "movement_allowed": bool(simulation.get("movement_allowed", True)),
        "simulation_parameters": copy.deepcopy(simulation),
        "weather_payload": copy.deepcopy(payload),
    }


class LiveDataSource:
    """Normalize, retain, and report freshness for an external telemetry source."""

    def __init__(
        self,
        adapter: LiveTelemetryAdapter,
        *,
        default_environment: dict[str, Any],
        stale_after_ms: float = 250.0,
        future_tolerance_ms: float = 1000.0,
        reorder_tolerance_ms: float = 5.0,
    ) -> None:
        if stale_after_ms <= 0.0:
            raise ValueError("live stale threshold must be positive")
        self.adapter = adapter
        self.kind = adapter.kind
        self.name = adapter.name
        self.stale_after_ms = float(stale_after_ms)
        self.future_tolerance_ms = float(future_tolerance_ms)
        self.reorder_tolerance_ms = float(reorder_tolerance_ms)
        self._default_environment = copy.deepcopy(default_environment)
        self._channels: dict[str, _Channel] = {}
        self._last_error: str | None = None
        self._rejected_samples = 0
        self._generation = 0
        self._adapter_generation = int(adapter.health().get("generation", 0))
        self._robot_status = "disconnected"
        self._expected_body_names: set[str] = set()
        self._expected_joint_names: set[str] = set()

    def set_expected_layout(self, body_names: list[str], joint_names: list[str]) -> None:
        self._expected_body_names = {str(name) for name in body_names}
        self._expected_joint_names = {str(name) for name in joint_names}

    @property
    def generation(self) -> int:
        self._poll()
        return self._generation

    def _poll(self) -> None:
        try:
            samples = self.adapter.poll()
        except Exception as exc:
            self._last_error = str(exc)
            return
        adapter_health = self.adapter.health()
        adapter_generation = int(adapter_health.get("generation", 0))
        if adapter_generation != self._adapter_generation:
            self._adapter_generation = adapter_generation
            self._generation += 1
        if adapter_health.get("last_error"):
            self._last_error = str(adapter_health["last_error"])
        for sample in samples:
            try:
                self._accept(sample)
                self._last_error = None
            except (LiveSampleError, TypeError, ValueError) as exc:
                self._last_error = str(exc)
                self._rejected_samples += 1

        robot_status = self._channel_health("robot")["status"]
        if robot_status == "connected" and self._robot_status in {"stale", "disconnected", "unavailable"}:
            self._generation += 1
        self._robot_status = robot_status

    def _accept(self, sample: dict[str, Any]) -> None:
        if not isinstance(sample, dict):
            raise LiveSampleError("LIVE adapter emitted a non-object sample")
        now = time.time()
        fallback_time = _timestamp(sample.get("sample_time", now))
        provenance_map = sample.get("provenance") if isinstance(sample.get("provenance"), dict) else {}
        staged: list[tuple[str, _Channel]] = []
        for channel_name in LIVE_CHANNELS:
            if channel_name not in sample:
                continue
            data, sample_time, nested_provenance = _unwrap_channel(sample[channel_name], fallback_time)
            if sample_time > now + self.future_tolerance_ms / 1000.0:
                raise LiveSampleError(f"future {channel_name} sample rejected")
            previous = self._channels.get(channel_name)
            if previous and sample_time < previous.sample_time - self.reorder_tolerance_ms / 1000.0:
                raise LiveSampleError(f"non-monotonic {channel_name} sample rejected")
            if channel_name == "robot":
                _validate_robot(data)
            elif channel_name in {"terrain", "snow"}:
                _validate_grid(data, channel=channel_name)
            provenance = nested_provenance or provenance_map.get(channel_name) or self.name
            staged.append((channel_name, _Channel(data, sample_time, now, str(provenance))))
        if not staged:
            raise LiveSampleError("LIVE sample contains no supported channels")
        for channel_name, channel in staged:
            self._channels[channel_name] = channel

    def _channel_health(self, name: str) -> dict[str, Any]:
        channel = self._channels.get(name)
        if channel is None:
            return {
                "status": "unavailable",
                "sample_time": None,
                "receipt_time": None,
                "age_ms": None,
                "provenance": None,
            }
        age_ms = max(0.0, (time.time() - channel.sample_time) * 1000.0)
        result = {
            "status": "connected" if age_ms <= self.stale_after_ms else "stale",
            "sample_time": channel.sample_time,
            "receipt_time": channel.receipt_time,
            "age_ms": age_ms,
            "provenance": channel.provenance,
        }
        if name == "robot":
            bodies = {str(item) for item in channel.data.get("body_names", [])}
            joints = {str(item) for item in channel.data.get("joint_names", [])}
            result["missing_bodies"] = sorted(self._expected_body_names - bodies)
            result["missing_joints"] = sorted(self._expected_joint_names - joints)
        return result

    def health(self) -> dict[str, Any]:
        self._poll()
        channels = {name: self._channel_health(name) for name in LIVE_CHANNELS}
        robot = channels["robot"]
        adapter_health = self.adapter.health()
        if robot["status"] == "unavailable":
            status = "disconnected"
        else:
            status = robot["status"]
        if adapter_health.get("status") == "disconnected" and status == "connected":
            status = "disconnected"
        return {
            "kind": self.kind,
            "name": self.name,
            "status": status,
            "sample_time": robot["sample_time"],
            "age_ms": robot["age_ms"],
            "stale_after_ms": self.stale_after_ms,
            "last_error": self._last_error or adapter_health.get("last_error"),
            "rejected_samples": self._rejected_samples,
            "channels": channels,
        }

    def frame(self) -> dict[str, Any] | None:
        self._poll()
        channel = self._channels.get("robot")
        if channel is None:
            return None
        frame = copy.deepcopy(channel.data)
        frame.setdefault("schema", "everest-viewer/v1")
        frame.setdefault("timestamp", channel.sample_time)
        frame.setdefault("sim_time", 0.0)
        frame.setdefault("sequence", 0)
        frame.setdefault("engine", self.kind)
        frame.setdefault("base_linear_velocity", [0.0, 0.0, 0.0])
        frame.setdefault("base_angular_velocity", [0.0, 0.0, 0.0])
        frame.setdefault("joint_names", [])
        frame.setdefault("joint_positions", [])
        frame.setdefault("joint_velocities", [])
        frame.setdefault("joint_torques", [])
        frame.setdefault("command", [0.0, 0.0, 0.0])
        frame.setdefault("feet", {})
        frame["paused"] = False
        frame["provenance"] = channel.provenance
        return frame

    def environment(self) -> dict[str, Any]:
        self._poll()
        result = copy.deepcopy(self._default_environment)
        result["data_mode"] = "live"
        channel = self._channels.get("weather")
        health = self._channel_health("weather")
        if channel is not None:
            result.update(weather_environment(channel.data))
        result["channel_status"] = health["status"]
        result["sample_time"] = health["sample_time"]
        result["age_ms"] = health["age_ms"]
        result["provenance"] = health["provenance"] or "static-render-context"
        return result

    def snow(self) -> dict[str, Any] | None:
        self._poll()
        channel = self._channels.get("snow")
        if channel is None:
            return None
        result = copy.deepcopy(channel.data)
        result.setdefault("schema", "everest-snow-surface/v1")
        result.setdefault("timestamp", channel.sample_time)
        result.setdefault("sequence", 0)
        result["provenance"] = channel.provenance
        result["measurement_status"] = self._channel_health("snow")["status"]
        return result

    def terrain(self) -> dict[str, Any] | None:
        self._poll()
        channel = self._channels.get("terrain")
        if channel is None:
            return None
        result = copy.deepcopy(channel.data)
        result.setdefault("schema", "everest-terrain/v1")
        result.setdefault("timestamp", channel.sample_time)
        result.setdefault("sequence", 0)
        result["provenance"] = channel.provenance
        result["measurement_status"] = self._channel_health("terrain")["status"]
        return result

    def sensors(self) -> dict[str, Any] | None:
        self._poll()
        channel = self._channels.get("sensors")
        if channel is None:
            return None
        result = copy.deepcopy(channel.data)
        result["provenance"] = channel.provenance
        result["sample_time"] = channel.sample_time
        return result

    def close(self) -> None:
        self.adapter.close()
