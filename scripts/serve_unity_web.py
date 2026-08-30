#!/usr/bin/env python3
"""Serve the local Unity WebGL build with headers suitable for development."""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
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
        "--dir",
        default="/mnt/c/Users/auverus/Documents/EverestUnityWeb/Builds/WebGL",
    )
    args = parser.parse_args()
    root = Path(args.dir).resolve()
    if not (root / "index.html").is_file():
        raise SystemExit(f"Unity WebGL build missing: {root / 'index.html'}")
    handler = lambda *hargs, **kwargs: Handler(*hargs, directory=str(root), **kwargs)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Everest Unity WebGL: http://127.0.0.1:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
