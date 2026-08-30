"""Transport adapters for read-only LIVE telemetry.

Adapters emit renderer-neutral ``everest-live-sample/v1`` dictionaries.  They
do not know about MuJoCo, Newton, Unity, or robot command transport.  A sensor
gateway can therefore publish the same JSON over a watched file or local UDP
without gaining any control authority over the physical robot.
"""

from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Protocol


LIVE_SAMPLE_SCHEMA = "everest-live-sample/v1"


class LiveTelemetryAdapter(Protocol):
    """Read-only source of already normalized live telemetry samples."""

    kind: str
    name: str

    def poll(self) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _load_samples(path: Path) -> tuple[list[dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        samples = [json.loads(line) for line in text.splitlines() if line.strip()]
        return samples, False

    if isinstance(payload, list):
        return payload, False
    if not isinstance(payload, dict):
        raise ValueError("live telemetry must be a JSON object, array, or JSONL stream")
    if payload.get("schema") == "everest-live-replay/v1":
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError("everest-live-replay/v1 requires a samples array")
        return samples, bool(payload.get("loop", False))
    return [payload], False


class ReplayTelemetryAdapter:
    """Deterministically replay a JSON/JSONL fixture against the wall clock."""

    kind = "replay"

    def __init__(self, path: str | Path, *, speed: float = 1.0) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"LIVE replay missing: {self.path}")
        if speed <= 0.0:
            raise ValueError("replay speed must be positive")
        self.name = self.path.name
        self._samples, self._loop = _load_samples(self.path)
        if not self._samples:
            raise ValueError("LIVE replay contains no samples")
        if not all(isinstance(item, dict) for item in self._samples):
            raise ValueError("every LIVE replay sample must be an object")
        self._speed = float(speed)
        self._offsets = self._derive_offsets(self._samples)
        self._duration = max(self._offsets[-1], 0.001)
        # Start on the first poll, after the comparatively expensive simulator
        # construction. Otherwise a valid first replay sample can already be
        # stale before the WebSocket begins listening.
        self._started_monotonic: float | None = None
        self._started_wall: float | None = None
        self._index = 0
        self._generation = 0
        self._closed = False

    @staticmethod
    def _derive_offsets(samples: list[dict[str, Any]]) -> list[float]:
        explicit = [item.get("offset_s") for item in samples]
        if all(value is not None for value in explicit):
            result = [float(value) for value in explicit]
        else:
            times = [item.get("sample_time") for item in samples]
            if all(isinstance(value, (int, float)) for value in times):
                first = float(times[0])
                result = [float(value) - first for value in times]
            else:
                result = [index / 30.0 for index in range(len(samples))]
        if any(value < 0.0 for value in result):
            raise ValueError("LIVE replay offsets must be non-negative")
        if any(right < left for left, right in zip(result, result[1:])):
            raise ValueError("LIVE replay offsets must be monotonic")
        return result

    def poll(self) -> list[dict[str, Any]]:
        if self._closed:
            return []
        if self._started_monotonic is None:
            self._started_monotonic = time.monotonic()
            self._started_wall = time.time()
        elapsed = (time.monotonic() - self._started_monotonic) * self._speed
        emitted: list[dict[str, Any]] = []
        while self._index < len(self._samples) and self._offsets[self._index] <= elapsed:
            sample = copy.deepcopy(self._samples[self._index])
            offset = self._offsets[self._index]
            recorded = sample.get("sample_time")
            sample["schema"] = LIVE_SAMPLE_SCHEMA
            sample["sample_time"] = float(self._started_wall) + offset / self._speed
            sample["receipt_time"] = time.time()
            sample["adapter_generation"] = self._generation
            if recorded is not None:
                sample.setdefault("metadata", {})["recorded_sample_time"] = recorded
            emitted.append(sample)
            self._index += 1

        if self._loop and self._index >= len(self._samples) and elapsed >= self._duration:
            self._generation += 1
            self._index = 0
            self._started_monotonic = time.monotonic()
            self._started_wall = time.time()
        return emitted

    def health(self) -> dict[str, Any]:
        return {
            "status": "disconnected" if self._closed else "connected",
            "last_error": None,
            "generation": self._generation,
        }

    def close(self) -> None:
        self._closed = True


class JsonFileTelemetryAdapter:
    """Watch an atomically replaced JSON snapshot produced by a sensor gateway."""

    kind = "json_file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = self.path.name
        self._last_mtime_ns: int | None = None
        self._last_error: str | None = None
        self._closed = False

    def poll(self) -> list[dict[str, Any]]:
        if self._closed:
            return []
        try:
            stat = self.path.stat()
            if stat.st_mtime_ns == self._last_mtime_ns:
                return []
            samples, _ = _load_samples(self.path)
            self._last_mtime_ns = stat.st_mtime_ns
            self._last_error = None
            return samples
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._last_error = str(exc)
            return []

    def health(self) -> dict[str, Any]:
        status = "disconnected" if self._closed or self._last_error else "connected"
        return {"status": status, "last_error": self._last_error, "generation": 0}

    def close(self) -> None:
        self._closed = True


class UdpTelemetryAdapter:
    """Receive normalized JSON datagrams from a local read-only sensor bridge."""

    kind = "udp"

    def __init__(self, endpoint: str) -> None:
        host, separator, port_text = endpoint.rpartition(":")
        if not separator or not host or not port_text:
            raise ValueError("UDP LIVE endpoint must use host:port")
        self.name = endpoint
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((host, int(port_text)))
        self._last_error: str | None = None
        self._closed = False

    def poll(self) -> list[dict[str, Any]]:
        if self._closed:
            return []
        samples: list[dict[str, Any]] = []
        while True:
            try:
                raw, _ = self._socket.recvfrom(4 * 1024 * 1024)
            except BlockingIOError:
                break
            except OSError as exc:
                self._last_error = str(exc)
                break
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("UDP telemetry payload must be an object")
                samples.append(payload)
                self._last_error = None
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._last_error = str(exc)
        return samples

    def health(self) -> dict[str, Any]:
        return {
            "status": "disconnected" if self._closed or self._last_error else "connected",
            "last_error": self._last_error,
            "generation": 0,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._socket.close()


class OpenMeteoWeatherAdapter:
    """Fetch the existing simulation-ready Open-Meteo payload off the render loop."""

    kind = "open_meteo"

    def __init__(self, location: str = "south-col", *, interval_s: float = 300.0) -> None:
        from weather.everest_weather import LOCATIONS

        if location not in LOCATIONS:
            raise ValueError(f"unknown Open-Meteo location: {location}")
        if interval_s < 15.0:
            raise ValueError("Open-Meteo interval must be at least 15 seconds")
        self.location = location
        self.name = f"open-meteo:{location}"
        self._interval_s = float(interval_s)
        self._queue: deque[dict[str, Any]] = deque(maxlen=4)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="everest-open-meteo", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from weather.everest_weather import LOCATIONS, build_payload, fetch_weather

        while not self._stop.is_set():
            try:
                payload = build_payload(LOCATIONS[self.location], fetch_weather(LOCATIONS[self.location]))
                sample = {
                    "schema": LIVE_SAMPLE_SCHEMA,
                    "sample_time": time.time(),
                    "weather": {
                        "data": payload,
                        "sample_time": payload.get("fetched_at", time.time()),
                        "provenance": payload.get("source", self.name),
                    },
                    "provenance": {"weather": payload.get("source", self.name)},
                }
                with self._lock:
                    self._queue.append(sample)
                    self._last_error = None
            except Exception as exc:  # keep an optional network source isolated
                with self._lock:
                    self._last_error = str(exc)
            self._stop.wait(self._interval_s)

    def poll(self) -> list[dict[str, Any]]:
        with self._lock:
            result = list(self._queue)
            self._queue.clear()
        return result

    def health(self) -> dict[str, Any]:
        with self._lock:
            error = self._last_error
        return {
            "status": "disconnected" if self._stop.is_set() else "connected",
            "last_error": error,
            "generation": 0,
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class CompositeTelemetryAdapter:
    """Merge independent robot/sensor and weather adapters by channel."""

    kind = "composite"

    def __init__(self, adapters: list[LiveTelemetryAdapter]) -> None:
        if not adapters:
            raise ValueError("composite LIVE adapter requires at least one source")
        self.adapters = adapters
        self.name = "+".join(adapter.name for adapter in adapters)

    def poll(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for adapter in self.adapters:
            samples.extend(adapter.poll())
        return samples

    def health(self) -> dict[str, Any]:
        health = [adapter.health() for adapter in self.adapters]
        errors = [item.get("last_error") for item in health if item.get("last_error")]
        connected = any(item.get("status") == "connected" for item in health)
        return {
            "status": "connected" if connected else "disconnected",
            "last_error": "; ".join(errors) if errors else None,
            "generation": sum(int(item.get("generation", 0)) for item in health),
            "adapters": health,
        }

    def close(self) -> None:
        for adapter in self.adapters:
            adapter.close()
