#!/usr/bin/env python3
"""Serve the local Unity WebGL build with headers suitable for development."""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit


class Handler(SimpleHTTPRequestHandler):
    backend_port: int | None = None

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if self.backend_port is not None and request.path == "/":
            host_header = self.headers.get("Host", "127.0.0.1")
            host = urlsplit(f"//{host_header}").hostname or "127.0.0.1"
            backend = f"ws://{host}:{self.backend_port}"
            query = dict(parse_qsl(request.query, keep_blank_values=True))
            # Port-specific demo servers must remain pinned to their backend
            # even when a browser retains cache-busting or diagnostic query
            # parameters. Previously any non-bare URL bypassed the redirect
            # and Unity silently fell back to the main simulation backend.
            if query.get("backend") != backend:
                query["backend"] = backend
                self.send_response(302)
                self.send_header("Location", f"/?{urlencode(query)}")
                self.end_headers()
                return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument(
        "--backend-port",
        type=int,
        help="redirect bare / to a Unity URL targeting this WebSocket port",
    )
    parser.add_argument(
        "--dir",
        default="/mnt/c/Users/auverus/Documents/EverestUnityWeb/Builds/WebGL",
    )
    args = parser.parse_args()
    root = Path(args.dir).resolve()
    if not (root / "index.html").is_file():
        raise SystemExit(f"Unity WebGL build missing: {root / 'index.html'}")
    class ConfiguredHandler(Handler):
        backend_port = args.backend_port

    handler = lambda *hargs, **kwargs: ConfiguredHandler(*hargs, directory=str(root), **kwargs)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Everest Unity WebGL: http://127.0.0.1:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
