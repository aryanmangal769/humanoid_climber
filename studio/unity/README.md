# Everest Unity renderer

Unity is the visualization client for the renderer-neutral MuJoCo + Newton backend.
It does not run robot dynamics, contact, snow deformation, sinkage, or policy inference.

## Backend

The current hackathon backend is expected at:

```text
ws://127.0.0.1:18765
```

Override it before launching Unity with the environment variable:

```text
EVEREST_BACKEND_URL=ws://HOST:PORT
```

The client consumes `/everest-scene/v1`, terrain, macro terrain, robot `frame`, Newton `snow`, environment, state, and fault messages described in `../../docs/UNITY_RENDERER_BRIDGE_PROTOCOL.md`.

## Open the project

Open `studio/unity/EverestSim` in Unity Hub. The project is intentionally scene-light: a runtime bootstrap creates the renderer, camera, lighting, HUD, and backend connection when Play starts.

The project targets the 2022.3 LTS API surface and may be upgraded by a newer Unity editor when opened.

### Windows Unity + WSL backend

For a Windows-side Unity Editor, mirror the project onto NTFS first:

```bash
./scripts/sync-unity-windows.sh
```

The current mirror path is:

```text
C:\Users\auverus\Documents\EverestUnity
```

Open that directory in Unity Hub. The generated `Library`, `Temp`, `Obj`, `Logs`, and `UserSettings` directories are intentionally not synced back. Re-run the sync script after source changes made in WSL.

Modern WSL commonly forwards Windows `localhost` to services listening in WSL. If the Unity client cannot reach `ws://127.0.0.1:18765`, set `EVEREST_BACKEND_URL` to an address that reaches the WSL backend.

## Prepare G1 visual assets

The authoritative G1 assets in `vendor/unitree_rl_mjlab` are STL files authored in the backend Z-up/right-handed frame. Run:

```bash
./.venv-rl/bin/python studio/unity/tools/prepare_g1_assets.py
```

This converts them to Unity-ready OBJ assets under `EverestSim/Assets/Resources/G1`, including the required handedness/Y-up conversion. Unity then imports the OBJs as model prefabs and the runtime scene manifest binds each visual to the corresponding MuJoCo body.

## Validate the handoff

With the backend running:

```bash
./.venv-rl/bin/python scripts/verify_unity_renderer.py
```

This verifies the live protocol, G1 visual bindings, terrain dimensions, Newton snow surface arrays, multilayer state, and absence of a physics fault.

## Controls

- `W/S`: forward/back command
- `A/D`: lateral command
- `Q/E`: yaw command
- `Space`: stop
- `P`: pause/unpause
- `R`: reset
- right mouse + mouse: orbit camera
- mouse wheel: zoom

## Rendering ownership

```text
MuJoCo      -> authoritative G1 transforms and telemetry
Newton MPM  -> authoritative local snow surface/compaction
Unity       -> interpolation, terrain/snow meshes, camera, atmosphere, HUD/input
```

The Unity coordinate adapter converts backend `(X east, Y north, Z up)` into Unity `(X right, Y up, Z forward)` and converts `wxyz` quaternions without changing backend data.
