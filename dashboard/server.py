"""Serve the browser MuJoCo viewer using an engine-neutral frame contract."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
from urllib.parse import urlparse

from .engines.mujoco import MuJoCoEngine
from .engines.registry import catalog


ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    engine: MuJoCoEngine

    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send((ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/app.js":
            self._send((ROOT / "static" / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/styles.css":
            self._send((ROOT / "static" / "styles.css").read_bytes(), "text/css; charset=utf-8")
        elif path == "/api/model":
            self._send(json.dumps(self.engine.state()).encode())
        elif path == "/api/state":
            self._send(json.dumps(self.engine.state(), separators=(",", ":")).encode())
        elif path == "/api/engines":
            self._send(json.dumps(catalog(), separators=(",", ":")).encode())
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = self.engine.frame()
                    if not frame:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/control":
            self._send(b"not found", "text/plain", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.engine.control(str(payload.get("action", "")), payload.get("value"))
            self._send(json.dumps(self.engine.state()).encode())
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(str(exc).encode(), "text/plain", 400)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    engine = MuJoCoEngine()
    engine.start()
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Everest MuJoCo viewer: http://{args.host}:{args.port}/", flush=True)
    print("Engine=mujoco source=unitreerobotics/unitree_rl_mjlab", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
