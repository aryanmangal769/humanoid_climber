# Everest Unreal Digital Twin

Unreal is the renderer/operator UI. It is intentionally **not** a third physics engine.

- **MuJoCo** owns Unitree G1 articulation, contacts, policy inference, and robot dynamics.
- **Newton `SolverImplicitMPM`** owns multilayer deformable snow and foot/snow reaction forces.
- **Unreal Engine 5.6** renders the true-scale Everest DEM, live Newton snow surface, Unitree link meshes, weather, and operator camera.

## One-command project setup

From the repository root:

```bash
./scripts/setup-unreal.sh
```

This regenerates the measured terrain and G1 render assets, verifies Warp/Newton on CUDA, probes the MuJoCo/Newton bridge, and mirrors the Unreal project to the Windows filesystem when running under WSL.

On this WSL workstation the source project is:

`studio/unreal/EverestSim/EverestSim.uproject`

and the default Windows mirror is:

`C:\Users\auverus\Documents\EverestSim\EverestSim.uproject`

The mirror is deliberate: UnrealBuildTool/MSVC should not build a C++ Unreal project directly from a `\\wsl$` UNC path. The physics process remains in WSL so Newton/Warp gets direct RTX access.

## Windows prerequisites

If Epic Games Launcher / Visual Studio C++ Build Tools are missing, run:

```bash
./scripts/install-unreal-windows-prereqs.sh
```

Windows will show a UAC consent dialog because these are machine-wide installs. The helper installs Epic Games Launcher and Visual Studio 2022 Build Tools with the C++ workload, then opens the Unreal Engine library.

Install **Unreal Engine 5.6** in Epic Games Launcher. Prefer `D:` on this workstation because `C:` has limited free space.

## Run

Once Unreal Engine 5.6 is installed:

```bash
./scripts/start-unreal.sh
```

The launcher auto-detects common Linux/Windows UE installs, refreshes the Windows project mirror, starts the MuJoCo/Newton bridge, waits for Newton CUDA warmup, then opens the project. Set `UNREAL_EDITOR=/path/to/UnrealEditor` only for a custom engine build.

Controls in PIE:

- `W/S`: forward/back command
- `Q/E`: lateral command
- `A/D`: yaw-rate command
- mouse: orbit camera
- wheel: zoom
- `Space`: pause/unpause

## Terrain

`tools/build_true_scale_terrain.py` generates two measured-DEM products from `maps/output_hh.tif`:

- `config/everest_macro_terrain.json`: about 29.5 x 46.8 km true-scale Everest backdrop.
- `config/everest_robot_terrain.json`: 1.2 x 1.2 km, 257 x 257 true-scale MuJoCo contact patch centered on the South Col route start (~5294 m).

The old `maps/everest_terrain.json` compresses the whole mountain into ~12 m and is kept untouched for the legacy viewer. The Unreal bridge overrides that legacy manifest only inside this runtime.

`tools/export_everest_heightmap.py` also emits `EverestSim/SourceData/everest_macro.r16` for later conversion to a native Unreal Landscape.

## Snow

`config/default_snow.json` defines a three-layer static snowpack: fresh powder, wind slab, and dense old snow. Newton owns deformation, sinkage, compaction, and reaction forces. Atmospheric snowfall is rendered in Unreal but does not silently change the selected MPM snowpack mass.

The bridge warms Warp/Newton kernels before accepting Unreal clients. On WSL, `scripts/start-unreal-bridge.sh` prepends `/usr/lib/wsl/lib` so Warp loads the host NVIDIA CUDA driver rather than the stale distro `libcuda.so`.

## Rebuild generated sources manually

```bash
.venv-rl/bin/python studio/unreal/tools/build_true_scale_terrain.py
.venv-rl/bin/python studio/unreal/tools/export_everest_heightmap.py
.venv-rl/bin/python studio/unreal/tools/prepare_g1_assets.py
```
