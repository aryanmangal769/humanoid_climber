from __future__ import annotations

import asyncio
import json
import os

import websockets


async def main() -> None:
    port = int(os.environ.get("EVEREST_BACKEND_PORT", "18765"))
    async with websockets.connect(
        f"ws://127.0.0.1:{port}", max_size=16 * 1024 * 1024, compression=None
    ) as ws:
        while True:
            message = json.loads(await ws.recv())
            if message.get("type") != "scene":
                continue
            scene = message["data"]
            print(f"schema={scene.get('schema')} bodies={len(scene.get('body_names', []))} visuals={len(scene.get('visuals', []))}")
            for visual in scene.get("visuals", [])[:40]:
                print(
                    visual.get("body"),
                    visual.get("mesh"),
                    visual.get("asset"),
                    visual.get("position"),
                    visual.get("scale"),
                )
            return


if __name__ == "__main__":
    asyncio.run(main())
