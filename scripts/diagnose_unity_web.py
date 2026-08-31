#!/usr/bin/env python3
"""Print browser console/runtime state for a local Unity WebGL page."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.request import urlopen

import websockets


URL = os.environ.get("EVEREST_BROWSER_URL", "http://127.0.0.1:18888/?diagnose=1")
PORT = int(os.environ.get("EVEREST_CAPTURE_DEBUG_PORT", "9235"))
PROFILE = Path(os.environ.get("EVEREST_CAPTURE_PROFILE", "/tmp/everest-diagnose-profile"))


def launch() -> subprocess.Popen[bytes]:
    shutil.rmtree(PROFILE, ignore_errors=True)
    return subprocess.Popen(
        [
            "/snap/bin/chromium",
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--enable-logging=stderr",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={PROFILE}",
            "--window-size=1600,1000",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def page_socket() -> str:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1) as response:
                for page in json.load(response):
                    if page.get("type") == "page":
                        return page["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("Chromium diagnostic page did not appear")


async def diagnose(socket_url: str) -> None:
    next_id = 0
    events: list[dict] = []
    async with websockets.connect(socket_url, max_size=32 * 1024 * 1024) as socket:
        async def call(method: str, params: dict | None = None) -> dict:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            await socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
            while True:
                message = json.loads(await socket.recv())
                if message.get("id") == request_id:
                    return message.get("result") or message
                events.append(message)

        await call("Runtime.enable")
        await call("Log.enable")
        await call("Page.enable")
        await asyncio.sleep(20)
        expression = r"""
        (() => {
          const canvas = document.getElementById('unity-canvas');
          const gl = canvas && (canvas.getContext('webgl2') || canvas.getContext('webgl'));
          return {
            url: location.href,
            loading: document.getElementById('loading')?.style.display,
            canvasAttr: [canvas?.width, canvas?.height],
            canvasRect: canvas?.getBoundingClientRect()?.toJSON(),
            unity: !!window.everestUnity,
            contextLost: gl?.isContextLost(),
            renderer: gl?.getParameter(gl.RENDERER),
            vendor: gl?.getParameter(gl.VENDOR),
            error: gl?.getError(),
          };
        })()
        """
        result = await call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        print(json.dumps(result, indent=2))
        await asyncio.sleep(2)
        for event in events:
            method = event.get("method")
            if method in {"Runtime.consoleAPICalled", "Runtime.exceptionThrown", "Log.entryAdded"}:
                print(json.dumps(event, separators=(",", ":")))


def main() -> None:
    process = launch()
    try:
        asyncio.run(diagnose(page_socket()))
    finally:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
        print(stderr.decode("utf-8", errors="replace")[-20_000:])
        shutil.rmtree(PROFILE, ignore_errors=True)


if __name__ == "__main__":
    main()
