"""Serve the telemetry-based browser viewer for the Unitree G1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError, URLError

from .engines.mujoco import G1_ASSETS, MuJoCoEngine
from .engines.registry import catalog
from weather.everest_weather import LOCATIONS, build_payload, fetch_weather


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
THREE = STATIC / "vendor" / "three"
EVEREST_TERRAIN = ROOT.parent / "maps" / "everest_terrain.json"
WEATHER_ERRORS = (HTTPError, URLError, TimeoutError, KeyError, ValueError)


def safe_child(root: Path, requested: str) -> Path | None:
    """Resolve a URL path without allowing traversal outside ``root``."""
    candidate = (root / unquote(requested).lstrip("/")).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class WeatherUpdater:
    """Refresh weather periodically while preserving the last valid payload."""

    def __init__(
        self,
        engine: MuJoCoEngine,
        location: dict[str, object],
        timeout: float,
        interval: float,
        *,
        enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.location = location
        self.timeout = timeout
        self.interval = interval
        self.enabled = enabled
        self._stop = threading.Event()
        self._refresh_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_attempt_at: str | None = None
        self._last_success_at: str | None = None
        self._last_error: str | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def refresh(self) -> dict[str, object]:
        """Fetch and atomically apply one update, retaining old state on error."""
        with self._refresh_lock:
            attempted_at = self._now()
            with self._status_lock:
                self._last_attempt_at = attempted_at
            try:
                response = fetch_weather(self.location, self.timeout)
                payload = build_payload(self.location, response)
                self.engine.control("weather", payload)
            except WEATHER_ERRORS as exc:
                with self._status_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                raise
            with self._status_lock:
                self._last_success_at = attempted_at
                self._last_error = None
            return self.engine.weather_state()

    def status(self) -> dict[str, object]:
        with self._status_lock:
            return {
                "enabled": self.enabled,
                "interval_seconds": self.interval,
                "location": self.location["name"],
                "last_attempt_at": self._last_attempt_at,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
            }

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="weather-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.timeout + 1.0, 10.0))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.refresh()
            except WEATHER_ERRORS as exc:
                print(f"Weather refresh failed; keeping last valid payload: {exc}", flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: MuJoCoEngine
    weather_updater: WeatherUpdater

    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers routinely cancel parallel STL requests during reloads.
            return

    def _json(self, value: object, status: int = 200) -> None:
        self._send(json.dumps(value, separators=(",", ":")).encode(), status=status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._file(STATIC / "app.js", "text/javascript; charset=utf-8")
        elif path == "/terrain-renderer.js":
            self._file(STATIC / "terrain-renderer.js", "text/javascript; charset=utf-8")
        elif path == "/debug-controls.js":
            self._file(STATIC / "debug-controls.js", "text/javascript; charset=utf-8")
        elif path == "/editor-ui.js":
            self._file(STATIC / "editor-ui.js", "text/javascript; charset=utf-8")
        elif path == "/weather-effects.js":
            self._file(STATIC / "weather-effects.js", "text/javascript; charset=utf-8")
        elif path == "/everest-terrain.json":
            self._json(self.engine.terrain_tile())
        elif path == "/styles.css":
            self._file(STATIC / "styles.css", "text/css; charset=utf-8")
        elif path == "/api/model" or path == "/api/state":
            self._json(self.engine.state())
        elif path == "/api/frame":
            self._json(self.engine.frame())
        elif path == "/api/terrain/frame":
            self._json(self.engine.terrain_frame())
        elif path == "/api/scene":
            self._json(self.engine.scene_manifest())
        elif path == "/api/weather":
            state = self.engine.weather_state()
            state["refresh"] = self.weather_updater.status()
            self._json(state)
        elif path == "/api/engines":
            self._json(catalog())
        elif path.startswith("/assets/unitree_g1/"):
            relative = path.removeprefix("/assets/unitree_g1/")
            self._asset(G1_ASSETS, relative)
        elif path.startswith("/vendor/three/"):
            relative = path.removeprefix("/vendor/three/")
            content_type = "text/javascript; charset=utf-8" if relative.endswith(".js") else "application/octet-stream"
            self._asset(THREE, relative, content_type)
        else:
            self._send(b"not found", "text/plain; charset=utf-8", 404)

    def do_HEAD(self) -> None:  # noqa: N802
        """Support lightweight asset probes used by operators and health checks."""
        path = urlparse(self.path).path
        if path == "/everest-terrain.json":
            asset = EVEREST_TERRAIN if EVEREST_TERRAIN.is_file() else None
            content_type = "application/json"
        elif path.startswith("/assets/unitree_g1/"):
            root, relative = G1_ASSETS, path.removeprefix("/assets/unitree_g1/")
            content_type = "model/stl"
            asset = safe_child(root, relative)
        elif path.startswith("/vendor/three/"):
            root, relative = THREE, path.removeprefix("/vendor/three/")
            content_type = "text/javascript; charset=utf-8"
            asset = safe_child(root, relative)
        else:
            self.send_response(404)
            self.end_headers()
            return
        if asset is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(asset.stat().st_size))
        self.end_headers()

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        self._send(path.read_bytes(), content_type)

    def _asset(self, root: Path, relative: str, content_type: str | None = None) -> None:
        path = safe_child(root, relative)
        if path is None:
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        if content_type is None:
            content_type = "model/stl" if path.suffix.lower() == ".stl" else "application/octet-stream"
        self._send(path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/weather/refresh":
            try:
                state = self.weather_updater.refresh()
                state["refresh"] = self.weather_updater.status()
                self._json(state)
            except WEATHER_ERRORS as exc:
                self._send(str(exc).encode(), "text/plain; charset=utf-8", 502)
            return
        if path != "/api/control":
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.engine.control(str(payload.get("action", "")), payload.get("value"))
            self._json(self.engine.state())
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(str(exc).encode(), "text/plain; charset=utf-8", 400)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weather-location", choices=LOCATIONS, default="south-col")
    parser.add_argument("--weather-timeout", type=float, default=5.0)
    parser.add_argument("--weather-refresh-seconds", type=float, default=900.0)
    parser.add_argument("--no-weather", action="store_true", help="Disable startup and periodic weather fetches")
    args = parser.parse_args()
    if args.weather_refresh_seconds <= 0:
        parser.error("--weather-refresh-seconds must be positive")
    engine = MuJoCoEngine()
    weather_updater = WeatherUpdater(
        engine,
        LOCATIONS[args.weather_location],
        args.weather_timeout,
        args.weather_refresh_seconds,
        enabled=not args.no_weather,
    )
    if not args.no_weather:
        try:
            weather_updater.refresh()
        except WEATHER_ERRORS as exc:
            print(f"Weather startup fetch failed: {exc}", flush=True)
    engine.start()
    weather_updater.start()
    Handler.engine = engine
    Handler.weather_updater = weather_updater
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Everest telemetry viewer: http://{args.host}:{args.port}/", flush=True)
    print("Engine=mujoco physics=mujoco_menagerie/unitree_g1 policy=unitree_rl_mjlab transport=pose-json", flush=True)
    if weather_updater.enabled:
        print(
            f"Weather={weather_updater.location['name']} refresh={weather_updater.interval:g}s",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        weather_updater.stop()
        engine.stop()


if __name__ == "__main__":
    main()
