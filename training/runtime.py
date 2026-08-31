"""Process supervisor for the web-exposed Everest PPO trainer."""

from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "everest_rl"
STATUS_FILE = RUN_ROOT / "status.json"
LOG_FILE = RUN_ROOT / "training.log"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "EVEREST_ICE_INCLINE_CHECKPOINT",
        "/home/auverus/git/humanoid_climber_safety_ckpts/ckpt/exported/ice_incline.onnx",
    )
)


class EverestTrainingRuntime:
    """Own at most one local trainer and expose a JSON-safe status snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._adopted_pid: int | None = self._discover_external_trainer()
        self._log_handle = None
        self._last_error: str | None = None
        self._stop_requested = False

    def start(self, value: Any = None) -> dict[str, Any]:
        options = value if isinstance(value, dict) else {}
        with self._lock:
            self._reap_locked()
            if self._process is not None or self._adopted_pid is not None:
                raise ValueError("Everest RL training is already running")
            if not DEFAULT_CHECKPOINT.is_file():
                raise FileNotFoundError(
                    f"Low-friction incline checkpoint is missing: {DEFAULT_CHECKPOINT}"
                )
            iterations = self._bounded_int(options, "iterations", 250, 1, 100_000)
            rollout_steps = self._bounded_int(options, "rollout_steps", 1024, 8, 1_000_000)
            friction = self._bounded_float(options, "friction", 0.15, 0.02, 1.0)
            run_id = time.strftime("everest-web-%Y%m%d-%H%M%S")
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            trainer_python = ROOT / ".venv-rl" / "bin" / "python"
            if not trainer_python.is_file():
                raise FileNotFoundError(
                    f"RL runtime is missing: {trainer_python}. Run scripts/setup-rl-stack.sh."
                )
            command = [
                str(trainer_python),
                "-m",
                "training.everest_ppo",
                "--checkpoint",
                str(DEFAULT_CHECKPOINT),
                "--run-id",
                run_id,
                "--status-file",
                str(STATUS_FILE),
                "--output-dir",
                str(RUN_ROOT),
                "--iterations",
                str(iterations),
                "--rollout-steps",
                str(rollout_steps),
                "--friction",
                str(friction),
            ]
            environment = os.environ.copy()
            environment.setdefault("WARP_CACHE_PATH", str(RUN_ROOT / "warp-cache"))
            warp_runtime = ROOT / ".everest-runtime" / "warp116"
            if warp_runtime.is_dir():
                existing_path = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    f"{warp_runtime}{os.pathsep}{existing_path}"
                    if existing_path
                    else str(warp_runtime)
                )
            self._log_handle = LOG_FILE.open("ab", buffering=0)
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._last_error = None
            self._stop_requested = False
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._reap_locked()
            process = self._process
            if process is not None:
                self._stop_requested = True
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3.0)
                finally:
                    self._reap_locked()
            elif self._adopted_pid is not None:
                self._stop_requested = True
                pid = self._adopted_pid
                try:
                    os.killpg(pid, signal.SIGINT)
                    deadline = time.time() + 5.0
                    while self._pid_alive(pid) and time.time() < deadline:
                        time.sleep(0.05)
                    if self._pid_alive(pid):
                        os.killpg(pid, signal.SIGTERM)
                finally:
                    self._adopted_pid = None
            return self.status()

    def close(self) -> None:
        # A replacement renderer may adopt a trainer that was intentionally
        # launched by the previous backend. Closing the observer must not kill
        # that independent process; the explicit STOP control still does.
        if self._process is not None:
            self.stop()

    def status(self) -> dict[str, Any]:
        # This method is also called while start/stop holds the lock.
        self._reap_locked()
        running_pid = self._process.pid if self._process is not None else self._adopted_pid
        payload: dict[str, Any] = {
            "schema": "everest-rl-training/v1",
            "state": "idle",
            "running": running_pid is not None,
            "pid": running_pid,
            "seed_checkpoint": str(DEFAULT_CHECKPOINT),
            "seed_checkpoint_exists": DEFAULT_CHECKPOINT.is_file(),
            "status_file": str(STATUS_FILE),
            "log_file": str(LOG_FILE),
            "ui_scope": "main_unity_page",
            "demo_enabled": False,
        }
        if STATUS_FILE.is_file():
            try:
                stored = json.loads(STATUS_FILE.read_text())
                if isinstance(stored, dict):
                    payload.update(stored)
            except (OSError, json.JSONDecodeError) as exc:
                payload["status_error"] = f"{type(exc).__name__}: {exc}"
        running_pid = self._process.pid if self._process is not None else self._adopted_pid
        payload["running"] = running_pid is not None
        payload["pid"] = running_pid
        payload["process_ownership"] = "owned" if self._process is not None else ("adopted" if self._adopted_pid is not None else "none")
        payload["demo_enabled"] = False
        if running_pid is None and payload.get("state") in {"starting", "running"}:
            payload["state"] = "stopped" if self._stop_requested else "failed"
        if self._last_error:
            payload["process_error"] = self._last_error
        return payload

    def preview(self) -> dict[str, Any] | None:
        status = self.status()
        path_value = status.get("preview_file")
        if not path_value:
            return None
        path = Path(str(path_value)).resolve()
        try:
            path.relative_to(RUN_ROOT.resolve())
        except ValueError:
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def _reap_locked(self) -> None:
        if self._adopted_pid is not None and not self._pid_alive(self._adopted_pid):
            self._adopted_pid = None
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        if return_code != 0 and not self._stop_requested:
            self._last_error = f"trainer exited with code {return_code}; see {LOG_FILE}"
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @staticmethod
    def _discover_external_trainer() -> int | None:
        run_id = None
        try:
            stored = json.loads(STATUS_FILE.read_text())
            if isinstance(stored, dict) and stored.get("state") in {"starting", "running"}:
                run_id = str(stored.get("run_id") or "")
        except (OSError, json.JSONDecodeError):
            return None
        if not run_id:
            return None
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, PermissionError):
                continue
            if "training.everest_ppo" in command and run_id in command:
                return int(proc.name)
        return None

    @staticmethod
    def _bounded_int(options: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
        value = int(options.get(key, default))
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    @staticmethod
    def _bounded_float(options: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
        value = float(options.get(key, default))
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value
