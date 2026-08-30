from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import shutil
import subprocess
import time

import requests
import websockets

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / ".everest-browser-ready.png"
PROFILE = ROOT / ".everest-chrome-profile"
PORT = 9223
URL = "http://127.0.0.1:18888/?build=20260830-0231"


def launch() -> subprocess.Popen[str]:
    shutil.rmtree(PROFILE, ignore_errors=True)
    SCREENSHOT.unlink(missing_ok=True)
    return subprocess.Popen([
        "/snap/bin/chromium",
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={PROFILE}",
        "--window-size=1600,1000",
        "--hide-scrollbars",
        URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def page_ws_url(timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pages = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page" and "18888" in page.get("url", ""):
                    return page["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("Chrome DevTools page did not appear")


async def capture(ws_url: str) -> None:
    events: list[dict] = []
    next_id = 0
    async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
        async def call(method: str, params: dict | None = None) -> dict:
            nonlocal next_id
            next_id += 1
            ident = next_id
            await ws.send(json.dumps({"id": ident, "method": method, "params": params or {}}))
            while True:
                payload = json.loads(await ws.recv())
                if payload.get("id") == ident:
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    return payload.get("result") or {}
                if payload.get("method") in {"Runtime.exceptionThrown", "Log.entryAdded", "Runtime.consoleAPICalled"}:
                    events.append(payload)

        await call("Runtime.enable")
        await call("Log.enable")
        await call("Page.enable")

        deadline = time.time() + 120.0
        ready = False
        while time.time() < deadline:
            result = await call("Runtime.evaluate", {
                "expression": "(() => { const l=document.getElementById('loading'); const c=document.getElementById('unity-canvas'); return {hidden: !!l && l.style.display==='none', w:c?.width||0, h:c?.height||0}; })()",
                "returnByValue": True,
            })
            value = (((result.get("result") or {}).get("value")) or {})
            if value.get("hidden") and value.get("w", 0) > 0 and value.get("h", 0) > 0:
                ready = True
                break
            await asyncio.sleep(1.0)
        if not ready:
            raise RuntimeError("Unity loading overlay never completed")

        await asyncio.sleep(7.0)
        shot = await call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        SCREENSHOT.write_bytes(base64.b64decode(shot["data"]))

        severe = []
        for event in events:
            method = event.get("method")
            params = event.get("params") or {}
            text = json.dumps(params, separators=(",", ":"))
            if method == "Runtime.exceptionThrown" or any(term in text.lower() for term in ("runtimeerror", "out of bounds", "shader error", "exception")):
                severe.append(text[:800])
        print(f"ready screenshot={SCREENSHOT} bytes={SCREENSHOT.stat().st_size} severe_events={len(severe)}")
        for item in severe[:10]:
            print(item)


proc = launch()
try:
    asyncio.run(capture(page_ws_url()))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(PROFILE, ignore_errors=True)
