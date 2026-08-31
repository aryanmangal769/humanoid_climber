# Everest Unity Snow/Ice/Weather Pass — Handoff

Date: 2026-08-30
Branch: `everest-policy-retrain-demo`
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
Newton couples at an operator-adjustable 2--30 Hz (3 Hz interactive default), and the interactive implicit iteration
budget is 6 CUDA (5 CPU). Command and pause controls update immediately without waiting
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

## 2026-08-30 Newton 1.5, true-XYZ deformation, and LIVE reconciliation

The combined branch now uses Newton `1.5.1`, Warp `1.16.0`, MuJoCo `3.11.0`,
and MuJoCo-Warp `3.11.0`. Host verification reached `cuda:0`, an NVIDIA RTX
4090 with 24 GiB. The Newton 1.5 API migration includes the renamed joint
target arrays, model-property notifications, collider velocity mode, public
collider projection/impulse APIs, and coordinate-vs-DOF target layout handling.

The snow stream no longer reconstructs a regular XY grid and discard MPM shear.
It publishes actual top-surface `vertices`, per-boundary `layer_vertices`, and
`substrate_vertices`. Unity consumes those XYZ vertices directly and retains
the legacy height grids only for compatibility and MuJoCo heightfield support.
The mechanical-layer volume is an ordered, non-inverting projection derived
from the actual Newton top surface, terrain-conforming substrate, and each
layer's authoritative plastic-volume ratio `Jp`. This is live 3D deformation
within a single-valued surface/volume representation, not a decorative shader
indentation.

The MPM transfer is APIC. Contact activation is selected from current particle
positions, unchanged active masks reuse their cached Newton state, and
no-contact rollback restores GPU-resident position, velocity, APIC gradient,
`Jp`, and stress snapshots. Moving-window history preserves displacement,
plastic volume, and stress while intentionally clearing stale momentum when an
old trail region becomes active again. Coarse background samples retain their
true local cell volume; unsampled background is renderer/contact context rather
than overlapping fake MPM mass.

Support ownership is explicit: MuJoCo's deformed terrain heightfield supports
the production robot, while Newton collider impulses are diagnostic. Applying
those wrenches on top of MuJoCo support was removed because it double-counted
the same ground reaction. The separate direct rigid/MPM two-way diagnostic now
produces non-zero impulses on Newton 1.5, but the robot still sinks/falls and
that path is not production-ready.

The parallel LIVE implementation is reconciled in the same worktree. LIVE is a
read-only replay/JSON/UDP adapter path with source epochs, per-channel
provenance/freshness, stale retention, mutation rejection, and no fallback to
simulated robot data. Selecting LIVE does not arm hardware control. See
`docs/LIVE_MODE_IMPLEMENTATION_HANDOFF_2026-08-30.md` for its adapter contract.

The WebGL cloud shader also now uses a ray-coherent low-frequency bank mask in
addition to its 3D ray march. Previously, long rays averaged unrelated samples
into a uniform grey veil that read as fog. Storm/whiteout browser captures now
exercise backend weather forcing, snowfall, fog, and cloud rendering instead
of checking shader-error absence alone.

Latest sequential host validation:

```text
verify-newton-mujoco: PASS, Newton 1.5.1 on cuda:0
verify-newton-snow-deformation: PASS
  lateral deformation 0.02493 m
  vertical deformation 0.28820 m
  minimum projected layer thickness 0.00590 m
  154 history records restored
  all no-contact rollback errors 0.0
verify-snow-dashboard: PASS with known flat-policy slope falls reported
renderer bridge: PASS
  lateral stream motion 0.06284 m
  vertical stream motion 0.05279 m
  measured stream rate 0.50x realtime
LIVE direct tests: 7 PASS
LIVE WebSocket replay probe: PASS
Unity WebGL/IL2CPP build: PASS
headless Chromium storm smoke: severe_events=0
```

The warmed renderer bridge rate varies with contact/recentering and was only
`0.50x` realtime in the final combined run. Python threading was tested and
reverted because the Warp/Newton work still throttled the process and made
stepping less deterministic. Improving beyond the current rate likely needs a
process-separated MPM worker and/or deeper solver/grid changes.

### Accuracy boundary after this rework

- The implementation is a credible interactive multilayer MPM prototype, not
  yet a calibrated predictive Everest snow model.
- Snow, exposed rigid ice, and rock can each be selected, but simultaneous
  spatial physics/material masks are still missing.
- An `ICE` layer inside snow is stiff MPM material, not brittle/fracturing ice.
- The projected Unity layer volume cannot represent overhangs, detached chunks,
  or arbitrary inter-layer mixing.
- Site snow parameters still need force-vs-sinkage, shear-box,
  layer-breakthrough, reaction-closure, and grid/time-step convergence tests.
- The bundled G1 policy was trained for flat terrain and still falls on the
  Everest slope independently of the deformation renderer.

## 2026-08-30 startup sinkage correction

The startup trace exposed a presentation/controls failure that looked like
full-depth sinking: a zero-command G1 was still free-running the flat-terrain
velocity policy, sliding downhill and losing its feet while the snow surface
remained mostly intact. The reset pose also had only one sole touching the
local DEM because the canonical home keyframe was flat-floor aligned.

The reset path now aligns the base to the local DEM tangent and makes a small
sub-centimetre descent until both soles are initially supported. A zero
command is now an explicit stand/hold request: the authored home joint pose is
held and the base's XY/yaw are locked only until the operator sends a movement,
manual-force, or cheat command. This keeps settlement inspection stationary
without snapping a robot back after it has been intentionally moved.

On the RTX 4090 startup regression, the fixed robot stayed at the spawn XY
(`0.0 m` translation) for 3.0 simulated seconds. The snow column retained
approximately `0.369--0.400 m` of its `0.40 m` depth under the feet; maximum
localized surface sinkage was `0.0305 m`, with p90 sinkage below `0.0001 m`.
The pelvis settled by about `1.4 cm` and did not slide downhill. This is the
expected behavior: localized compression, not a robot disappearing through
the full layer.

The HUD reports this startup state as `SIM HOLD` so an operator can distinguish
intentional stationary settlement inspection from an active locomotion run.

## 2026-08-30 autonomous Unity Everest showcase

The autonomous demo is now a separate launch mode; normal simulator mode and
its full editor dock remain unchanged:

```bash
scripts/start-autonomous-demo.sh --port 18768
```

Persistent deployment keeps the tracks isolated:

- main simulation: WebSocket `18765`, Unity WebGL `18888`;
- autonomous demo: WebSocket `18768`, Unity WebGL `18889`.

The demo web server is launched with `--backend-port 18768`; opening bare
`http://HOST:18889/` redirects to a host-correct Unity backend query instead of
falling back to the main simulator.

The showcase does **not** replace the renderer with a flat MuJoCo scene. The
main viewport remains the existing Unity Everest pipeline: full DEM and macro
terrain, shared snow/ice/rock shading, atmosphere/clouds/snowfall, the G1 mesh,
MuJoCo robot physics, and live Newton multilayer deformation. Backend state
reports this explicitly as `unity_everest+mujoco_robot+newton_snow`.

The backend publishes an `everest-autonomous-showcase/v1` manifest containing:

- the terrain-conforming ascent route;
- an oriented low-friction snow-over-ice region;
- current stage and visible training-attempt count;
- truthful policy/checkpoint-slot status;
- subtle applied wind and scripted-controller force vectors;
- user-facing decision summaries for the chat-style demo transcript.

MuJoCo exposes one friction tuple for the full heightfield rather than one per
heightfield cell. The marked ice region therefore acts as a spatial entry gate:
the low coefficient is applied only while the robot is within the oriented
region, and restored on exit. Unity renders the same backend-authored boundary.

During failure handling, the temporary `RL TRAINING` viewport opens and shows
the raw MuJoCo replay of the captured local Newton terrain state. Main MuJoCo
and Newton physics continue underneath; only the demo narrative advances
through the two visible attempts. The tab closes automatically when the return
stage is reached and Unity resumes the main Everest viewport.

In autonomous-demo mode the right dock intentionally contains only weather
presets and a scrolling `AGENT DECISION STREAM`. Weather presets update clouds,
fog, snowfall, and the physical MuJoCo wind magnitude. The normal mode retains
the complete `ENV SETUP` / `DEMO` dock.

Policy truth boundary:

- `flat` remains the admitted bundled 98-observation / 29-action ONNX policy;
- a separate clone at `/home/auverus/git/humanoid_climber_safety_ckpts` provides
  MjLab 1.6 baseline, incline, and wind actors with a different 99-observation
  ordering; they are exported to `ckpt/exported/*.onnx` and exposed as
  `candidate_available`, not admitted/validated;
- the supplied recovery actor is `160 -> 29` motion tracking and remains
  `incompatible_160_observation` until its reference-motion observation path is
  ported; rough terrain remains `reserved_unavailable`;
- the separate autonomous showcase keeps explicitly labeled deterministic
  controllers until candidate admission is complete. The normal main sim can
  select and test the three 99-observation velocity candidates directly.

Re-export after checkpoint updates with `scripts/export_mjlab_pt_policy.py`.
The exporter uses safe `weights_only=True`, copies G1 joint/action metadata from
the bundled known-good ONNX, and labels the resulting model
`mjlab-1.6-velocity-99/v1`.

The recovery placeholder uses live MuJoCo forces/torques and joint targets, not
a pose teleport. The CUDA regression reached the climb stage with pelvis-up
dot product `0.9967`, terrain clearance `0.765 m`, no simulation fault, and
Newton on `cuda:0`.

An independent launch flag is available for diagnostics:

```bash
scripts/start-simulation-backend.sh --disable-newton
```

It retains MuJoCo robot/DEM physics but disables Newton construction and reports
`disabled_reason=disabled_by_launch_flag`. The autonomous launcher rejects this
flag because the showcase requires live snow deformation.

Validation added/performed:

```text
PYTHONPATH=. .venv-sim/bin/python scripts/verify_autonomous_demo.py: PASS
PYTHONPATH=. .venv-sim/bin/python scripts/verify_candidate_policies.py: PASS
tests/test_policy_supervisor.py: PASS
RTX 4090 Newton lifecycle through recovery/climb: PASS
Unity 2022.3.0f1 WebGL/IL2CPP build: PASS
training-tab Chromium capture: severe_events=0
post-training main-view Chromium capture: severe_events=0
```

## 2026-08-30 policy/retraining demo workflow

The right dock now has two tabs: `ENV SETUP` retains the environment, weather,
physics-window, material, terrain, robot, camera, and system controls; `DEMO`
contains the policy supervisor and retraining workflow.

Failure detection is intentionally truthful and deterministic. It ports the
IMU/contact thresholds and three-frame confirmation from
`/home/auverus/git/humanoid_climber/src/humanoid_climber/safety.py` and is
reported as `deterministic_imu_contact`; there is no learned detector artifact
in that checkout. The route classes are flat, low-friction incline, wind,
rough terrain, recovery, and a combined/new-specialist request. The decision
log is emitted by the same supervisor state that gates the MuJoCo policy step.

When failure is confirmed (or `INJECT DEMO FAILURE` is pressed), the backend:

1. zeros command/manual/cheat inputs and engages an active safety controller,
2. keeps physics running while transitioning into a low, wide four-point
   protective stance,
3. writes `runs/retrain_requests/<request-id>/manifest.json`, and
4. exposes a raw 320x240 native MuJoCo offscreen view of the captured
   Newton-window subset at 1 Hz.

The safety controller is the deterministic 29-joint posture ported from
`humanoid_climber`: a 120 ms aggressive crouch attack blends into a sustained
hands-and-lower-limbs stance. It does not freeze the root or simulation;
gravity, wind, terrain contacts, joint dynamics, and Newton continue advancing.
Planar and angular momentum receive physical damping during the transition.
The posture remains active until reset or a compatible/demo checkpoint return.

The manifest schema is `everest-rl-subset/v1`. A live Newton capture includes
the moving-window vertices, per-layer vertices, MPM solver metadata, weather,
material context, robot `qpos`/`qvel`, command, feet/contact telemetry, and the
detector output. The training status is explicitly
`requested_not_launched` because this repository has no trainer endpoint.

The registry retains the admitted 98-observation `flat` policy and also discovers
the separately cloned MjLab 1.6 candidates. `flat_mjlab_1_6`, `ice_incline`, and
`wind` use an explicit 99-value observation builder (base linear/angular
velocity, projected gravity, relative joint position/velocity, previous action,
and twist command) and are labeled `candidate_available`. Their ONNX outputs
were numerically compared to the original PyTorch actors with maximum errors
below `4e-7`, then each completed 36 inference cycles in the main MuJoCo engine
without a fault. Recovery remains incompatible at 160 observations and rough
terrain remains reserved. A returned checkpoint is accepted only after runtime
schema and actuator-layout validation. Selecting LIVE remains read-only.

Acceptance commands for this branch:

```text
PYTHONPATH=. .venv-sim/bin/python tests/test_policy_supervisor.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=. .venv-sim/bin/python scripts/verify_safe_posture.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=. .venv-sim/bin/python scripts/verify_policy_retrain_demo.py
EVEREST_SIM_CONTROL_TEST_PORT=18767 LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=. .venv-sim/bin/python scripts/verify_unity_sim_controls.py
EVEREST_UNITY_WEB_PROJECT=/mnt/c/Users/auverus/Documents/EverestUnityWeb2 bash scripts/build-unity-webgl.sh
```

The WebGL build was smoke-tested with headless Chromium against a disposable
backend: the DEMO tab rendered, the robot/terrain loaded, and no runtime,
shader, or WebAssembly exception events were observed. Production backend
`18765` and served WebGL assets should only be restarted/replaced after the
branch is committed and pushed.

## 2026-08-30 physical-locomotion continuation

The autonomous showcase now uses real exported MjLab 1.6 checkpoint inference
for both post-recovery locomotion phases:

- Recovery remains a labeled deterministic physical get-up controller.
- Once upright, the deterministic supervisor measures live slope, friction,
  and wind and prepares the compatible checkpoint's actuator gains. Recovery
  retains the validated scene-home target until its load-bearing release; no
  root translation or teleport is used.
- The routed `ice_incline` actor then drives all 29 joint targets. After a
  healthy forward traverse, the demo applies an exact 16 N wind and the same
  supervisor selects and executes the `wind` checkpoint.
- The Unity wind cue uses a larger presentation-only scale (`0.020`) while the
  backend force vector remains exact.

The no-Newton contract now completes:

```text
approach -> baseline_slide -> safety_hold -> training_attempt_1 ->
training_attempt_2 -> specialist_return -> recovery -> climb -> crosswind ->
complete
```

The accepted trace travels 2.72 m forward, finishes with pelvis up-dot 0.999,
has alternating foot contacts, nonzero incline and wind inference counts, and
ends with the router requesting and executing `wind`. These checks are locked
into `scripts/verify_autonomous_demo.py`. Candidate-policy verification and
`tests/test_policy_supervisor.py` also pass.

Deployment state: demo backend `18768` was restarted with Newton on `cuda:0`;
main backend/web ports `18765/18888` remained running. The Unity artifact at
`/mnt/c/Users/auverus/Documents/EverestUnityWeb2/Builds/WebGL` was rebuilt and
is served by demo port `18889`.

Known limitation: live CUDA Newton advances much more slowly than the
no-Newton contract and still needs an extended wall-clock observation to
confirm the full post-recovery route under MPM coupling. The pending Unity
training-subscene replacement remains unchanged from the earlier handoff.
