"""Command-line entry points for project-owned MjLab tasks."""

from importlib import import_module
import os
from pathlib import Path
import sys

TRAINING_ENABLED = False
_WSL_CUDA_LIB_DIR = Path("/usr/lib/wsl/lib")
_WSL_CUDA_REEXEC_GUARD = "HUM_CLIMBER_WSL_CUDA_REEXEC"


def _requested_device() -> str | None:
  """Return an explicitly requested play device without importing MjLab/Warp."""
  for index, arg in enumerate(sys.argv[1:], start=1):
    if arg == "--device" and index + 1 < len(sys.argv):
      return sys.argv[index + 1]
    if arg.startswith("--device="):
      return arg.split("=", 1)[1]
  return None


def _ensure_wsl_cuda_driver_path() -> None:
  """Re-exec the console entrypoint with WSL's real CUDA driver first.

  This host also exposes an older Linux ``libcuda`` which Warp can pick before
  WSL's driver shim. PyTorch still sees the GPU, so MjLab otherwise selects
  ``cuda:0`` and Warp then fails during initialization. The dynamic loader reads
  ``LD_LIBRARY_PATH`` at process startup, hence the one-time re-exec.
  """
  if Path(sys.argv[0]).name != "hum-climber-play":
    return
  requested = _requested_device()
  if requested is not None and requested.startswith("cpu"):
    return
  if not (_WSL_CUDA_LIB_DIR / "libcuda.so.1").exists():
    return
  current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
  if str(_WSL_CUDA_LIB_DIR) in current:
    return
  if os.environ.get(_WSL_CUDA_REEXEC_GUARD) == "1":
    return

  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = ":".join((str(_WSL_CUDA_LIB_DIR), *current))
  env[_WSL_CUDA_REEXEC_GUARD] = "1"
  os.execvpe(sys.argv[0], sys.argv, env)


def play() -> None:
  """Register Humanoid Climber tasks and launch playback with our Viser overlay."""
  _ensure_wsl_cuda_driver_path()
  import_module("mjlab.tasks")
  import_module("humanoid_climber.tasks")

  play_module = import_module("mjlab.scripts.play")
  from humanoid_climber.viewer import HumanoidClimberViserPlayViewer

  # MjLab stays external and untouched. Its play script resolves this global at
  # runtime, so the process-local replacement only affects Hum Climber playback.
  play_module.ViserPlayViewer = HumanoidClimberViserPlayViewer
  play_module.main()


def train() -> None:
  """Reject training until the project explicitly enables that execution path."""
  if not TRAINING_ENABLED:
    raise SystemExit(
      "Humanoid Climber training is disabled. The policy router only emits "
      "fine-tuning templates in the decision log; it does not launch trainers."
    )
  import_module("mjlab.tasks")
  import_module("humanoid_climber.tasks")
  from mjlab.scripts.train import main

  main()
