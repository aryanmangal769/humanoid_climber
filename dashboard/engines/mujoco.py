"""MuJoCo renderer adapter for Unitree RL MjLab's canonical G1 scene."""

from __future__ import annotations

import io
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ..g1_model import G1_XML


ROOT = Path(__file__).resolve().parents[2]
G1_SCENE = G1_XML


class MuJoCoEngine:
    """Owns one model/data/renderer and exposes an engine-neutral frame API."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 20.0):
        self.model = mujoco.MjModel.from_xml_path(str(G1_SCENE))
        self.data = mujoco.MjData(self.model)
        # The Unitree scene keeps MuJoCo's conservative 640x480 offscreen
        # framebuffer. Raise it before creating Renderer for LAN-sized frames.
        self.model.vis.global_.offwidth = max(width, 640)
        self.model.vis.global_.offheight = max(height, 480)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.0, 0.0, 0.82)
        self.camera.distance = 2.9
        self.camera.azimuth = 145.0
        self.camera.elevation = -13.0
        self.width, self.height, self.period = width, height, 1.0 / fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = True
        self._thread: threading.Thread | None = None
        self._jpeg = b""
        self._frames = 0
        self._render_error: str | None = None
        self._started_at = time.time()
        self.reset()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mujoco-renderer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def reset(self) -> None:
        with self._lock:
            self._reset_to_home()

    def _reset_to_home(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.793
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def frame(self) -> bytes:
        with self._lock:
            return self._jpeg

    def state(self) -> dict[str, Any]:
        with self._lock:
            age = time.time() - self._started_at
            return {
                "engine": "mujoco",
                "model": "Unitree G1 / unitree_rl_mjlab",
                "source": "unitreerobotics/unitree_rl_mjlab",
                "source_revision": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
                "scene": str(G1_SCENE.relative_to(ROOT)),
                "bodies": int(self.model.nbody),
                "joints": int(self.model.njnt),
                "actuators": int(self.model.nu),
                "paused": self._paused,
                "frames": self._frames,
                "fps": round(self._frames / age, 1) if age > 0 else 0.0,
                "sim_time": round(float(self.data.time), 3),
                "render_error": self._render_error,
            }

    def control(self, action: str, value: Any = None) -> None:
        with self._lock:
            if action == "reset":
                self._reset_to_home()
            elif action == "pause":
                self._paused = bool(value)
            elif action == "camera" and isinstance(value, str):
                presets = {
                    "front": (180.0, -10.0),
                    "three-quarter": (145.0, -13.0),
                    "side": (90.0, -10.0),
                    "rear": (0.0, -10.0),
                }
                if value in presets:
                    self.camera.azimuth, self.camera.elevation = presets[value]

    def _render(self, renderer: mujoco.Renderer) -> None:
        renderer.update_scene(self.data, camera=self.camera)
        rgb = renderer.render()
        output = io.BytesIO()
        Image.fromarray(np.asarray(rgb)).save(output, format="JPEG", quality=88, optimize=False)
        self._jpeg = output.getvalue()
        self._frames += 1

    def _loop(self) -> None:
        # OpenGL contexts are thread-affine. Create, use, and destroy the
        # renderer in this owner thread; otherwise a healthy process can emit
        # zero frames with EGL_BAD_ACCESS.
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            while not self._stop.is_set():
                start = time.monotonic()
                with self._lock:
                    if not self._paused:
                        mujoco.mj_step(self.model, self.data)
                    self._render(renderer)
                self._stop.wait(max(0.0, self.period - (time.monotonic() - start)))
        except Exception as exc:
            with self._lock:
                self._render_error = f"{type(exc).__name__}: {exc}"
        finally:
            if renderer is not None:
                renderer.close()
