"""Serve the local Unitree G1 policy-dashboard shell.

No robot transport is opened here.  Live policy transport is intentionally a
separate follow-up so this demo cannot interact with CyberDog or any hardware.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .g1_model import validate_g1


ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
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
            self._send(json.dumps(validate_g1()).encode())
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Everest G1 dashboard: http://{args.host}:{args.port}/", flush=True)
    print("Hardware telemetry is intentionally disabled.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
