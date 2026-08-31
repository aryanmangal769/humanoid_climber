from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import urlparse

import websockets

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = Path(os.environ.get("EVEREST_CAPTURE_SCREENSHOT", ROOT / ".everest-browser-ready.png"))
PROFILE = Path(os.environ.get("EVEREST_CAPTURE_PROFILE", ROOT / ".everest-chrome-profile"))
PORT = int(os.environ.get("EVEREST_CAPTURE_DEBUG_PORT", "9223"))
URL = os.environ.get(
    "EVEREST_BROWSER_URL",
    "http://127.0.0.1:18888/?build=20260830-0231",
)
BACKEND_URL = os.environ.get("EVEREST_CAPTURE_BACKEND")
CAPTURE_WEATHER = os.environ.get("EVEREST_CAPTURE_WEATHER", "").strip().lower()
CLICK_X = os.environ.get("EVEREST_CAPTURE_CLICK_X")
CLICK_Y = os.environ.get("EVEREST_CAPTURE_CLICK_Y")
WAIT_SECONDS = float(os.environ.get("EVEREST_CAPTURE_WAIT_SECONDS", "3.0"))


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
    expected_port = urlparse(URL).port
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1) as response:
                pages = json.load(response)
            for page in pages:
                page_port = urlparse(page.get("url", "")).port
                if page.get("type") == "page" and page_port == expected_port:
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

        if CLICK_X is not None and CLICK_Y is not None:
            point = {"x": float(CLICK_X), "y": float(CLICK_Y), "button": "left", "clickCount": 1}
            await call("Input.dispatchMouseEvent", {"type": "mousePressed", **point})
            await call("Input.dispatchMouseEvent", {"type": "mouseReleased", **point})
            await asyncio.sleep(1.0)

        if BACKEND_URL:
            async with websockets.connect(
                BACKEND_URL,
                max_size=16 * 1024 * 1024,
                compression=None,
            ) as backend:
                initial_time = None
                while initial_time is None:
                    message = json.loads(await asyncio.wait_for(backend.recv(), timeout=20.0))
                    if message.get("type") == "state":
                        initial_time = float(message["data"]["sim_time"])
                if CAPTURE_WEATHER in {"storm", "whiteout"}:
                    whiteout = CAPTURE_WEATHER == "whiteout"
                    await backend.send(json.dumps({
                        "type": "control",
                        "action": "weather",
                        "value": {
                            "temperature_c": -19.0 if whiteout else -22.0,
                            "wind_speed_m_s": 15.0 if whiteout else 24.0,
                            "wind_direction_deg": 250.0,
                            "snowfall_mm_h": 45.0 if whiteout else 22.0,
                            "visibility_scale": 0.10 if whiteout else 0.38,
                            "cloud_density": 0.86 if whiteout else 0.68,
                            "cloud_coverage": 0.96 if whiteout else 0.86,
                            "cloud_radius_m": 170.0,
                            "cloud_altitude_m": 18.0 if whiteout else 26.0,
                            "cloud_thickness_m": 72.0 if whiteout else 54.0,
                            "cloud_speed": 0.40,
                            "cloud_quality": 0.72 if whiteout else 0.68,
                            "movement_allowed": True,
                        },
                    }))
                await backend.send(json.dumps({
                    "type": "control", "action": "command", "value": [0.15, 0.0, 0.0],
                }))
                await backend.send(json.dumps({
                    "type": "control", "action": "pause", "value": False,
                }))
                while True:
                    message = json.loads(await asyncio.wait_for(backend.recv(), timeout=20.0))
                    if (
                        message.get("type") == "state"
                        and float(message["data"]["sim_time"]) >= initial_time + 0.8
                    ):
                        break
                await backend.send(json.dumps({
                    "type": "control", "action": "pause", "value": True,
                }))

        await asyncio.sleep(max(0.0, WAIT_SECONDS))
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
