# Everest Unity Snow/Ice/Weather Pass — Handoff

Date: 2026-08-30
Branch: `everest-dream-integration`
Project: `/home/auverus/git/everest-hack`

## Goal

Continue the Unity + MuJoCo/Newton Everest simulator with:
- backend-authoritative multilayer snow / firn / ice,
- terrain-conforming local Newton MPM physics,
- cheap visual LOD outside the physics radius,
- visible simulated weather / clouds,
- an editor-style Unity UI with SIM as the default,
- explicit cheat transport for fast physics inspection.

The exported chat history used as context is the Everest weather/snow research thread. It establishes the intended architecture: weather is a prior/forcing source, Newton owns deformable snow mechanics, MuJoCo owns the robot and rigid terrain/contact, and Unity is renderer/operator UI.

## Changes made in this pass

### 0. Unitree G1 mesh/joint orientation correction

The G1 appeared to have flipped joints and mirrored link geometry in Unity;
the clearest symptom was backward `unitree` lettering on the chest. MuJoCo's
body poses and the WXYZ quaternion conversion were correct. The mismatch was
in the asset boundary:

- `prepare_g1_assets.py` previously baked the final `(x, z, y)` reflection
  into each OBJ;
- Unity's OBJ importer also reflects OBJ X while importing into Unity's
  left-handed coordinate system;
- the resulting runtime mesh vertices were `(-x, z, y)`, while body and geom
  transforms were `(x, z, y)`.

`studio/unity/tools/prepare_g1_assets.py` now bakes the proper rotation
`(-x, z, y)` without reversing triangle winding. After Unity's importer-owned
X reflection, runtime vertices are exactly `(x, z, y)` and align with
`EverestCoordinates` and the authoritative MuJoCo body transforms. All G1 OBJ
resources were regenerated. `scripts/verify_unity_renderer.py` now checks the
asymmetric `logo_link` bounds to prevent this double-handedness conversion
from returning.

### 1. New consolidated editor HUD

Added:

`studio/unity/EverestSim/Assets/Scripts/EverestEditorHud.cs`

and wired it from:

`studio/unity/EverestSim/Assets/Scripts/EverestRuntime.cs`

The new HUD replaces the previous hierarchy + inspector + multiple-toolbar approach with:
- one compact top command bar,
- one persistent right-side simulation controls dock,
- one small bottom status bar.

The dock contains all important tuning controls in one place:
- Environment / Weather
- Physics Window
- Terrain Material
- DEM / Visual LOD
- Robot / Camera
- Editor / System

Current styling uses a restrained cold technical/editor palette (dark neutral panels, cyan selection/accent, orange warning, green status).

UI scale behavior:
- default is now `160%`,
- stored in `PlayerPrefs`,
- supported range is currently `100%–240%`,
- 160 / 180 / 200% presets are exposed.

Important: `EverestHud.cs` still exists in the tree, but `EverestRuntime` now creates `EverestEditorHud`, so the new HUD is the active one.

### 2. Terrain-draped visual snow / ice / rock shell

Added:

`studio/unity/EverestSim/Assets/Scripts/EverestVisualTerrainRenderer.cs`

This fixes the previous “tiny white physics island in an otherwise empty/wireframe world” presentation problem.

Architecture is now:

1. Authoritative Everest DEM
   - lightweight wireframe context renderer
   - Unity LOD groups

2. Visual terrain material shell
   - generated directly from the same DEM heights
   - snow / glacier ice / rock material shader
   - local + macro LOD
   - no physics ownership

3. Active local physical material window
   - backend Newton MPM for snow/firn/ice
   - rigid MuJoCo sampled patch for rigid ice/rock
   - rendered by `EverestSnowRenderer`

The visual shell is deliberately not a second physics simulation. It only makes the terrain outside the expensive Newton window visually continuous.

`EverestRuntime.cs` now routes:
- local terrain -> wire terrain + visual terrain
- macro terrain -> wire terrain + visual terrain
- active snow frame -> active material renderer + visual terrain state
- environment -> atmosphere renderer + visual terrain state

### 3. Backend snow accumulation controls

Backend state now contains:

- `snow_accumulation_enabled`
- `weather_time_scale`

Relevant code is in:

`dashboard/engines/mujoco.py`
`simulation/newton_snow.py`

The Newton patch now receives these settings when built and when settings are changed.

The intent is:
- snowfall remains a backend forcing,
- when accumulation is enabled, the top Newton layer gains mass/depth,
- `weather_time_scale` can accelerate that process for demo/testing,
- Unity particle snowfall is only presentation and does not author snow mass.

This closes the previous gap where Newton had physical deposition code but `accumulation_enabled` was hard-disabled.

### 4. Rock surface support

Added a rigid `rock` material to backend surface definitions in:

`simulation/snow.py`

The UI now exposes:
- Snow / MPM
- Rigid Ice
- Bare Rock

Rock is intentionally rigid MuJoCo DEM contact. There is no fake deformable rock material in Unity.

The backend active surface sampler was extended so both `ice` and `rock` can be rendered from the authoritative terrain surface.

### 5. Physics-radius + adaptive Newton controls remain backend-authoritative

The editor exposes:
- physics radius
- minimum MPM voxel
- target cells per side
- recenter fraction
- accumulation toggle
- weather time scale

These are sent through the existing `simulation_settings` control path.

The backend keeps the expensive Newton region bounded by using:

`effective_voxel = max(min_voxel, diameter / target_cells)`

so increasing radius does not blindly explode the particle count.

The local Newton patch remains terrain-conforming and recenters around the robot.

### 6. Environment / weather editor controls

The consolidated dock exposes:
- temperature
- wind speed
- wind direction
- snowfall rate
- visibility
- cloud density
- cloud coverage
- cloud radius
- cloud altitude
- cloud thickness
- cloud speed
- cloud quality

Presets:
- Clear
- Storm
- Whiteout
- Wind

The intended ownership remains:
- backend: wind force, friction effects, snowfall forcing / accumulation state
- Unity: fog, clouds, sky, snowfall VFX, material appearance

Current Unity atmosphere implementation is in:

`studio/unity/EverestSim/Assets/Scripts/EverestEnvironmentRenderer.cs`
`studio/unity/EverestSim/Assets/Shaders/EverestVolumetricClouds.shader`

### 7. Cheat transport remains explicitly non-physical

The new HUD keeps cheat transport visible and labeled.

Backend cheat mode:
- moves the G1 floating base directly over the DEM,
- preserves the articulated pose,
- recenters/follows the local Newton window,
- is separate from normal MuJoCo locomotion.

Controls remain:
- A/D strafe
- S/F forward/back
- Q/E yaw

Normal manual control remains:
- WASD
- Q/E yaw
- Space stop

## Latest editor interaction request — COMPLETED

Implemented in this pass:

- UI interaction now owns input while the pointer is over editor chrome, while a GUI control is hot/being dragged, while the dock is scrolling, and while the resize gutter is active.
- Orbit/free-camera mouse look, camera wheel zoom/speed, robot manual control, cheat transport, and global P/R shortcuts are suppressed while the editor owns input.
- When UI ownership begins during manual/cheat movement, the runtime sends one backend zero-velocity command so a previously held movement key cannot leave the robot moving while the user edits controls.
- The right controls dock is draggable from its left gutter and persists its width ratio in `PlayerPrefs`.
- Dock width is clamped against both a usable editor minimum and a minimum remaining viewport, so narrow browser windows do not push the viewport completely off-screen.
- Narrow dock mode shortens layer labels and slider/value columns.
- The command bar and status bar progressively collapse optional controls/details at narrow widths; SIM/LIVE and camera mode remain available inside the dock.
- Default UI scale remains `160%` and the width calculation is performed in the scaled virtual coordinate space.

Validation performed after the change:

- JetBrains inspections: no C# errors in `EverestEditorHud.cs`, `EverestCameraController.cs`, or `EverestRuntime.cs`.
- Unity 2022.3.0f1 clean WebGL/IL2CPP build: succeeded.
- Headless Chromium smokes at 1600x1000 and 700x900: both produced rendered screenshots with no WebAssembly out-of-bounds, shader, or runtime fatal errors.

## Files most relevant to continue

### Unity UI / interaction
- `studio/unity/EverestSim/Assets/Scripts/EverestEditorHud.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestCameraController.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestRuntime.cs`

### Terrain / rendering
- `studio/unity/EverestSim/Assets/Scripts/EverestTerrainRenderer.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestVisualTerrainRenderer.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestSnowRenderer.cs`
- `studio/unity/EverestSim/Assets/Shaders/EverestTerrain.shader`
- `studio/unity/EverestSim/Assets/Shaders/EverestSnow.shader`
- `studio/unity/EverestSim/Assets/Shaders/EverestWireTerrain.shader`

### Weather
- `studio/unity/EverestSim/Assets/Scripts/EverestEnvironmentRenderer.cs`
- `studio/unity/EverestSim/Assets/Shaders/EverestVolumetricClouds.shader`

### Backend parity
- `dashboard/engines/mujoco.py`
- `simulation/newton_snow.py`
- `simulation/snow.py`
- `simulation/unity_bridge.py`

## Verification / deployment notes

The repo is currently very dirty with many pre-existing untracked files and renderer work from earlier turns. Do not treat a broad `git diff` as only this pass.

Before continuing, validate specifically:
- Unity C# compile
- Unity shader compile
- backend Python syntax/tests
- `scripts/verify_unity_renderer.py`
- `scripts/verify_unity_sim_controls.py`
- browser/WebGL smoke test if the local Unity WebGL build toolchain is available

The previous project workflow used:
- WebSocket backend port `18765`
- Unity WebGL host port `18888`
- `scripts/build-unity-webgl.sh`
- `scripts/serve_unity_web.py`
- `scripts/sync-unity-windows.sh`

Do not assume the currently running tmux sessions are still alive; check them.

## Design constraints to preserve

- Backend physics parity is the hard constraint.
- Unity must not invent physical deformation.
- Full DEM can be cheap visual LOD/wireframe.
- Only a radius around the robot should use expensive Newton MPM.
- Visual snow/ice outside that radius may be approximated, but must be clearly renderer-only.
- Snow must remain multilayer (1–6 mechanical layers).
- Snow/firn/ice material parameters should continue mapping to Newton’s per-particle attributes.
- SIM is the default mode; LIVE channel remains available but dormant.
- Keep the UI practical/editor-like, not decorative.
- Default UI scale should remain at least 160%.

## 2026-08-30 renderer continuation

The missing clouds and magenta/purple snowfall were traced to WebGL shader
stripping. Renderer materials were constructed from runtime-only `Shader.Find`
calls, so Unity's build dependency scan omitted the custom shaders even though
the player reported a successful build.

The continuation adds build-included Resources materials, a WebGL-safe
`Everest/Snowfall` shader, independent geometry/compaction updates, live Newton
sinkage reporting, and lower-energy snow/terrain shading with higher-contrast
cloud noise. The exact WebGL player served on port `18888` was rebuilt and
loaded in headless Chromium with zero severe console/runtime events. Both
`verify_unity_renderer.py` and `verify_unity_sim_controls.py` pass against CUDA
Newton on the RTX 4090.

The live server serves
`/mnt/c/Users/auverus/Documents/EverestUnityWeb2/Builds/WebGL`, while the build
script defaults to `EverestUnityWeb`. To update that live endpoint, use:

```bash
EVEREST_UNITY_WEB_PROJECT=/mnt/c/Users/auverus/Documents/EverestUnityWeb2 \
  bash scripts/build-unity-webgl.sh
```

### Newton snow fidelity assessment

The current implementation is a useful GPU MPM mechanics prototype and the
Unity surface is synchronized correctly: Unity consumes the backend height,
material-id, and compaction grid and only interpolates between authoritative
targets. It is not yet a calibrated predictive Everest snow model.

Remaining limitations are material: static sinkage has an analytic
pressure/strain assist, pack self-weight is disabled after pre-consolidation,
layer bonds are approximated through tensile-yield ratio, the rendered/contact
height is one top particle per horizontal cell, recentering rebuilds the local
window without route-scale deformation history, and MuJoCo's deformed
heightfield plus forwarded Newton wrenches need a support-force double-counting
audit. The bundled G1 policy was trained on flat terrain and can fall on the
Everest slope independently of snow quality.

Before calling it accurate, add single-foot force-vs-sinkage and shear-box
calibration, reaction-force closure, layer-breakthrough, grid/time-step
convergence, and persistent deformation-transfer tests.

## 2026-08-30 LOD, active-volume, and camera continuation

The local/macro material break and sky-colored far polygon were fixed by using
the same world-space snow/ice/rock texture set and thresholds at every LOD,
cross-fading LOD groups, and retaining partially overlapping macro quads at the
local-tile boundary. The old quad-center carve could remove a large coarse
macro polygon outside the local coverage area.

The active Newton material now uses the same terrain shader as the visual DEM.
It remains a live 50x50 backend-authored 3D height mesh, but no longer reads as
a bright circular decal. The terrain shader has an explicit active-material
path for Newton compaction and exposed layer identity. Snow, rigid ice, and
rigid rock all render through this path.

The backend now also publishes `base_heights` and `layer_heights`: one live
upper-boundary grid for each Newton mechanical layer. Unity uses those grids to
maintain an inset multilayer cutaway skirt under the active top mesh. Normal
views remain seamless; low grazing views can reveal actual layer thickness and
deformation without inventing an independent Unity simulation. This is still a
heightfield representation, so it intentionally cannot show overhangs or
detached chunks.

Camera/UI changes:

- the Unity camera rect ends at the dock and excludes the top/status bars;
- orbit focus is centered in the visible scene viewport;
- left-drag pans the orbit focus and right-drag still orbits;
- `CAM RESET` returns to the robot-centered default orbit;
- free camera accepts arrow keys without the Shift modifier, while
  Shift+WASD/QE remains available;
- scene gestures are cancelled immediately if the pointer enters UI capture.

The snow albedo/roughness were subsequently switched from the close-up
`snow_02` set to `snow_field_aerial` for the macro, local, and active Newton
surfaces. A shared world scale of `0.0075` gives each texture repeat an
approximately 133 m footprint, removing the obvious small dirty-snow tiling
while preserving material continuity across all three renderer layers.

The aerial map is now sampled through a deterministic three-way rotated/offset
stochastic triplanar blend. This keeps the same map and world-space
coordinates at every LOD and on Newton, but hides the recognizable single-tile
repeat that remained visible over the broad DEM.

Interactive streaming was also profiled against the live RTX 4090 backend:
the old default MPM window advanced only about 0.13--0.21 simulated seconds
per wall-clock second and command acknowledgements waited behind the solver.
The default interactive window is now 1.25 m / 0.20 m voxel / 24 target cells,
Newton couples at an operator-adjustable 5--30 Hz (5 Hz interactive default), and the interactive implicit iteration
budget is 12 CUDA (8 CPU). Command and pause controls update immediately without waiting
for a long MPM critical section; the renderer still receives backend-authored
poses/deformation and operators can expand/refine the window for fidelity
tests.

### 2026-08-30 contact-gated deformation correction

A live failure showed the full local surface collapsing by roughly the entire
0.4 m snow depth while Newton reported almost no plastic compaction. The bad
state was genuinely present in the backend height grid; Unity was correctly
rendering it and its multilayer skirt made the failure especially obvious.

Newton deformation is now contact-gated. Each MPM step preserves particle
position, velocity, and plastic-volume state outside a 0.42 m influence disc
around soles that produced an actual non-zero Newton collider impulse. If no
sole interacts, the complete pre-step state is retained. The old analytic
virtual-sole indentation assist was removed, so collider poses are now the
actual MuJoCo foot poses. Exported/coupled surface height is also bounded by
the terrain-conforming snow substrate (`initial surface - column depth`).

In the live regression probe, the old median whole-patch sinkage of 0.369 m
became exactly 0.0 m; the 90th percentile remained numerical zero, maximum
localized contact sinkage was 4.2 mm, and no cell exceeded 5 cm or the snow
depth. Unity still does no snow physics: its top mesh and layer cutaway consume
the backend Newton `heights`, `layer_heights`, and `compaction` grids directly.

### Continuous trail and operator controls

Moving the local Newton window no longer resets snow. Before recentering, the
outgoing patch exports changed particle displacement and plastic volume keyed
to world-space cell/layer/depth coordinates. The new patch rehydrates matching
particles from that history, while the bridge streams completed patch meshes
as `snow_history` frames so Unity keeps the traveled deformation visible.
History is bounded to 250,000 particle records and 256 rendered patches; Reset
intentionally clears it.

Normal CONTROL mode now sends physical `manual_force_mode` input: W/S applies
a forward/backward pelvis nudge force and A/D applies a yaw torque. Q/E are no
longer strafing controls in this mode. Cheat mode remains the explicit direct
transport path.

Orbit camera gestures are now LMB drag; RMB drag pans the orbit focus. Freecam
look uses LMB as well. The UI help/status text reflects these bindings.

### Adaptive contact resolution and performance

Newton `SolverImplicitMPM` exposes one uniform `voxel_size` per solver; it does
not support a genuinely mixed-size grid inside one patch. The backend now uses
the supported adaptive equivalent:

- the patch grid is approximately 10.4 cm (`2.5 m / 24 target cells`);
- all horizontal samples within 0.55 m of either sole are retained;
- untouched background samples use stride 4 and carry proportionally larger
  particle volume/mass, preserving total layer mass;
- Newton `ParticleFlags.ACTIVE` enables only current contact-neighborhood
  particles in the implicit solve;
- airborne or stationary-contact intervals skip the solve, while accumulation
  and simulation time still advance;
- completed path regions remain coarse persistent history rather than active
  MPM particles;
- unchanged terrain snapshots and particle positions are cached to avoid
  repeated CUDA-to-CPU reads, and renderer frames omit raw particles unless
  explicitly launched with `--particles`.

Measured layouts on the RTX 4090: the prior 20 cm uniform default had 507
particles at roughly 0.64x real time; a 20 cm stride-2 adaptive patch had 213
particles and reached about 0.81x; the selected higher-quality 10.4 cm,
stride-4 layout uses roughly 640--785 particles depending on sole position and
runs about 0.67--0.76x when recentering/sliding is included. The UI exposes
Contact refine and Coarse stride so fidelity can be traded deliberately.
