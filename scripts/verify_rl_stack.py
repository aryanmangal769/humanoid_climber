from __future__ import annotations

import importlib
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
UNITREE = ROOT / "vendor" / "unitree_rl_mjlab"


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    import mujoco
    import mujoco_warp
    import torch
    import warp as wp
    import newton
    import mjlab
    from newton.solvers import SolverImplicitMPM, SolverMuJoCo

    print("Everest Dream RL stack")
    print(f"  Python       {sys.version.split()[0]}")
    print(f"  MuJoCo       {mujoco.__version__}")
    print(f"  MuJoCo Warp  {version('mujoco_warp')}")
    print(f"  Newton       {version('newton')}")
    print(f"  MJLab        {version('mjlab')}")
    print(f"  Torch        {torch.__version__}")
    print(f"  CUDA         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU          {torch.cuda.get_device_name(0)}")

    wp.init()
    print(f"  Warp device  {wp.get_device()}")
    print(f"  Newton MPM   {SolverImplicitMPM.__name__}")
    print(f"  Newton MJ    {SolverMuJoCo.__name__}")

    # Import Unitree task registration from the vendored checkout.  The
    # upstream package is intentionally named `src`, so keep its checkout on
    # sys.path for the same import behavior as the upstream training scripts.
    sys.path.insert(0, str(UNITREE))
    import src.tasks.velocity.config.g1  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = set(list_tasks())
    required = {"Unitree-G1-Flat", "Unitree-G1-Rough"}
    missing = required - tasks
    if missing:
        raise RuntimeError(f"Unitree G1 tasks were not registered: {sorted(missing)}")
    print("  G1 tasks      Unitree-G1-Flat, Unitree-G1-Rough")

    print("\nOK: shared MuJoCo/Newton/Unitree runtime is import-compatible.")


if __name__ == "__main__":
    main()

