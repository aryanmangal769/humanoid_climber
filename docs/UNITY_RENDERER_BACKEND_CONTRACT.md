# Unity Renderer Backend Contract

We are switching the visualization layer to **Unity**.

## Architecture

```text
MuJoCo
  └── authoritative Unitree G1 robot physics / articulation / controller

Newton
  └── authoritative deformable multilayer snow physics

Unity
  └── rendering + camera + editor UI + user input ONLY
```

Unity must **not** run robot physics, snow physics, collision, locomotion, contact resolution, sinkage estimation, or MPM logic.

The physics/backend agent should expose a renderer-neutral realtime service that Unity can consume without importing MuJoCo or Newton.

## 1. Headless simulation service

Please provide one process that owns:

```text
G1 policy
   ↓
MuJoCo robot dynamics
   ↓ foot poses/contact
Newton MPM snow
   ↓ snow deformation/reaction
MuJoCo
   ↓
renderer state stream
   ↓
Unity
```

Run independently from Unity:

```bash
./scripts/start-simulation-backend.sh
```

MVP transport: WebSocket.

Default endpoint:

```text
ws://127.0.0.1:8765
```

Unity is a pure client.

## 2. Physics ownership

### MuJoCo owns

- G1 rigid-body dynamics
- all G1 joints
- actuators
- policy execution
- robot pose and velocity
- balance
- terrain collision representation
- externally applied Newton reaction forces

### Newton owns

- deformable snow
- compaction
- yielding
- sinkage
- shear deformation
- multilayer material behavior
- foot/snow interaction
- snow surface deformation
- reaction impulses applied back into MuJoCo

Newton should use `SolverImplicitMPM` and CUDA when available.

Expected on this machine:

```text
cuda:0
NVIDIA GeForce RTX 4090
```

WSL CUDA detail:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This must lead the distro NVIDIA libraries so Warp does not load the stale `libcuda.so`.

## 3. Snow is currently a static snowpack

Do **not** simulate long-term snowfall accumulation right now.

Snowfall can exist as atmospheric/rendering state, but should not continuously add Newton MPM mass.

```text
snow parameters
      ↓
construct current snowpack snapshot
      ↓
Newton simulates robot interaction with that snapshot
```

Unity can render visible falling snow independently.

## 4. Multilayer snow state

Preserve 1–6 mechanically distinct layers.

Minimum schema:

```json
{
  "surface_friction": 0.36,
  "snowfall_mm_h": 8.0,
  "wind_speed_m_s": 18.0,
  "wind_direction_deg": 250.0,
  "temperature_c": -18.0,
  "slope_deg": 18.0,
  "layers": [
    {
      "type": "POWDER",
      "label": "Fresh powder",
      "thickness_m": 0.08,
      "density_kg_m3": 120.0,
      "stiffness_pa": 35000.0,
      "compressive_strength_pa": 3500.0,
      "shear_strength_pa": 1800.0,
      "compaction_hardening": 12.0,
      "bond_strength_below_pa": 2500.0
    }
  ]
}
```

These values must map into Newton physics, not just renderer metadata.

## 5. Terrain

Expose two terrain products.

### Local physical terrain

True-scale terrain around G1 used by MuJoCo collision.

Target MVP:

```text
~1.2 km × 1.2 km
257 × 257 height samples
```

Do not compress Everest into a tiny scene.

Payload:

```json
{
  "schema": "everest-terrain/v1",
  "grid_width": 257,
  "grid_height": 257,
  "world_width_m": 1200,
  "world_depth_m": 1200,
  "terrain_center": [0, 0, 0],
  "heights": []
}
```

Heights are metres relative to the simulation origin.

### Macro visual terrain

Also expose the larger Everest DEM for Unity background rendering.

This does not need collision.

A lower-resolution product around `30 km × 47 km` is appropriate.

## 6. Coordinate convention

Use one documented convention across terrain, robot, Newton snow, and all streams.

Backend convention:

```text
units: metres
up: +Z
handedness: right-handed
+X: first horizontal terrain axis / east
+Y: second horizontal terrain axis / north
+Z: up
quaternion order: wxyz
```

Unity will handle conversion to its Y-up/left-handed system.

Do not silently flip axes between subsystems.

## 7. Static robot scene manifest

Send once after connection.

Example:

```json
{
  "type": "scene",
  "data": {
    "schema": "everest-scene/v1",
    "model": "Unitree G1",
    "up_axis": "z",
    "quaternion_order": "wxyz",
    "body_names": ["pelvis", "left_knee_link"],
    "visuals": [
      {
        "body": "pelvis",
        "mesh": "pelvis",
        "asset": "pelvis.obj",
        "position": [0, 0, 0],
        "quaternion": [1, 0, 0, 0],
        "scale": [1, 1, 1]
      }
    ]
  }
}
```

Unity already has/can import the meshes. The backend mainly needs to define which visual mesh belongs to which MuJoCo body and its local transform.

## 8. G1 robot frame stream

Highest-priority stream.

Target: **30–60 Hz**.

```json
{
  "type": "frame",
  "data": {
    "schema": "everest-viewer/v1",
    "sequence": 4839,
    "timestamp": 1788060000.0,
    "sim_time": 19.38,
    "engine": "newton+mujoco",
    "body_names": ["pelvis", "left_hip_pitch_link", "left_knee_link"],
    "body_pos_w": [[0.1,0.2,1.0],[0.1,0.2,0.7],[0.1,0.2,0.35]],
    "body_quat_w": [[1,0,0,0],[1,0,0,0],[1,0,0,0]],
    "paused": false
  }
}
```

Send complete frames, not incremental body deltas. Dropped packets should be harmless.

Do not pre-smooth poses. Unity will interpolate authoritative frames.

## 9. Additional robot telemetry

Expose at least:

```json
{
  "base_linear_velocity": [0,0,0],
  "base_angular_velocity": [0,0,0],
  "joint_names": [],
  "joint_positions": [],
  "joint_velocities": [],
  "joint_torques": [],
  "command": {"forward":0.4,"lateral":0.0,"yaw":0.0}
}
```

This is for Unity debug/editor panels.

## 10. Foot contact telemetry

Expose each foot independently.

```json
{
  "feet": {
    "left": {
      "position": [0,0,0],
      "normal_force_n": 170,
      "tangential_force_n": [4,2],
      "penetration_m": 0.06,
      "slip_speed_m_s": 0.01,
      "contact": true
    },
    "right": {}
  }
}
```

Unity will use this for contact markers, slip indicators, sinkage, loads, footprints, and debug vectors.

## 11. Newton snow surface stream

Unity needs a lightweight deforming surface grid. Do not require Unity to reconstruct the surface from raw MPM particles.

Target: **10–20 Hz**.

```json
{
  "type": "snow",
  "data": {
    "schema": "everest-snow-surface/v1",
    "sequence": 500,
    "sim_time": 10.0,
    "origin": [-0.8,-0.45,0.0],
    "size": [1.6,0.9],
    "resolution": [32,18],
    "heights": [],
    "compaction": [],
    "material_ids": [],
    "surface_depth_m": 0.55,
    "surface_friction": 0.36,
    "layers": []
  }
}
```

Unity will update a procedural mesh from this.

## 12. Compaction is a first-class output

Normalize approximately:

```text
0 = untouched snow
1 = heavily compacted
```

Unity will use compaction for albedo, roughness, normals, footprint darkness, sparkle, and debug overlays.

Do not force Unity to infer compaction from height alone.

## 13. Optional particle debug stream

Raw particles are optional/debug only:

```json
{
  "particles": {
    "positions": [],
    "radii": [],
    "material_ids": []
  }
}
```

Default OFF. Enable only for MPM debugging.

## 14. Environment/weather state

Expose:

```json
{
  "temperature_c": -18,
  "wind_speed_m_s": 18,
  "wind_direction_deg": 250,
  "snowfall_mm_h": 8,
  "visibility_scale": 0.65,
  "movement_allowed": true
}
```

Unity will convert this to visible snowfall, fog, wind-blown particles, sky/visibility effects, etc.

Snowfall remains visual-only with respect to accumulation for now.

## 15. Unity → backend controls

Robot command:

```json
{"type":"control","action":"command","value":[0.4,0.0,0.0]}
```

Interpret as:

```text
[forward_velocity, lateral_velocity, yaw_rate]
```

Also support:

```json
{"type":"control","action":"pause","value":true}
```

```json
{"type":"control","action":"reset","value":null}
```

```json
{"type":"control","action":"snow_parameters","value":{}}
```

```json
{"type":"control","action":"weather","value":{}}
```

## 16. Runtime state/status

Expose one high-level state object:

```json
{
  "schema": "everest-state/v1",
  "engine": "newton+mujoco",
  "mujoco": {"active": true},
  "newton": {
    "active": true,
    "solver": "SolverImplicitMPM",
    "device": "cuda:0",
    "particle_count": 1008
  },
  "policy": {"active": true},
  "paused": false,
  "simulation_fault": null
}
```

Unity must be able to display faults instead of silently freezing.

## 17. Fault handling

Do not kill the backend if physics becomes unstable.

Send:

```json
{
  "type": "fault",
  "data": {
    "source": "mujoco",
    "message": "...",
    "sim_time": 12.3
  }
}
```

Automatically pause and let Unity offer Reset.

## 18. Timing

Suggested internal stepping:

```text
MuJoCo physics:         500–1000 Hz if required
policy:                 ~50 Hz
Newton MPM:             ~50 Hz / dt ~0.02 s
robot render stream:    30–60 Hz
snow surface stream:    10–20 Hz
state/status:            ~2 Hz or on changes
```

Renderer frequency must never control physics stepping.

Always include `sequence`, `timestamp`, and `sim_time` for interpolation and diagnostics.

## 19. Local Newton patch should follow G1

Do not run MPM over the whole mountain.

Use:

```text
global lightweight terrain/snow state
                 ↓
        local active Newton MPM
              around G1
```

Eventually target roughly `5×5 m` to `10×10 m` for the active deformable region.

The patch should recenter/rebuild as G1 travels.

Unity should only need:

```text
origin
size
resolution
heights
compaction
material_ids
```

## 20. Terrain/snow continuity

Avoid a visible rectangular seam around the active Newton patch.

Outside the active patch:

```text
surface = terrain height + nominal snow depth
```

Inside:

```text
surface = Newton deformed height
```

Prefer either one already-composited surface grid or explicit overlap/blend metadata.

## 21. Stream split / performance

Do not send enormous JSON payloads every renderer frame.

Recommended:

```text
scene          once
terrain        once
macro terrain  once
robot frame    30–60 Hz, small
snow surface   10–20 Hz, medium
particles      optional/debug only
state          on change / ~2 Hz
```

JSON + WebSocket is fine for the MVP. MessagePack/FlatBuffers can replace the high-frequency encoding later without changing the conceptual API.

## 22. Backend probe

Provide:

```bash
./scripts/start-simulation-backend.sh --probe
```

Expected output should clearly show:

```text
MuJoCo: OK
G1: OK
policy: OK

Newton: OK
solver: SolverImplicitMPM
device: cuda:0
GPU: RTX 4090

terrain: OK
snow layers: N
MPM particles: N

WebSocket: ws://127.0.0.1:8765
```

## 23. Automated renderer bridge integration test

Provide:

```bash
./scripts/test-renderer-bridge.sh
```

It should:

1. connect to WebSocket
2. receive scene
3. receive local terrain
4. receive macro terrain
5. receive robot frame
6. receive snow frame
7. send forward command
8. unpause
9. wait about one simulated second
10. verify G1 moved
11. verify Newton stepped
12. verify no simulation fault

## 24. Unity-side success criterion

Unity should be able to use the backend approximately like this:

```csharp
backend.OnRobotFrame(frame =>
{
    robot.SetBodyTransforms(frame);
});

backend.OnSnowFrame(frame =>
{
    snow.UpdateMesh(frame);
});

backend.OnTerrain(terrain =>
{
    terrainRenderer.Build(terrain);
});

backend.SendCommand(forward, lateral, yaw);
```

Unity should **not** need to import MuJoCo, import Newton, know MPM internals, calculate foot forces, simulate sinkage, estimate snow deformation, or know how the G1 policy works.

All of that complexity belongs behind the backend API.

## Deliverables required from the physics/backend agent

Please hand the Unity-side agent:

```text
1. start-simulation-backend.sh
2. documented WebSocket protocol
3. scene manifest
4. true-scale local terrain payload
5. macro Everest terrain payload
6. 30–60 Hz G1 body pose stream
7. joint telemetry
8. foot contact telemetry
9. 10–20 Hz Newton snow-surface stream
10. compaction/material maps
11. optional particle debug stream
12. weather/environment state
13. Unity → backend command API
14. reset/pause controls
15. simulation-fault reporting
16. --probe command
17. automated renderer-bridge integration test
```

Final success criterion:

> A Unity developer who knows nothing about MuJoCo/Newton should be able to render Everest, render G1, move G1, see the feet deform the multilayer snow, inspect compaction/contact state, and reset/pause the simulation using only the backend protocol.

## Priority order

Highest priority:

1. complete G1 body transform stream
2. true-scale local terrain
3. Newton snow surface + compaction grid
4. foot contact telemetry
5. Unity control API
6. fault/status reporting
7. automated renderer bridge test

Raw MPM particles and richer diagnostics can come afterward.
