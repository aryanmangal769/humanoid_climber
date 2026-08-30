# Snow Web Renderer Handoff

Updated: 2026-08-29 (America/Los_Angeles)

## Objective

The web client renders Unitree G1 pose telemetry and a local multilayer
snow/ice path without transporting video. MuJoCo remains authoritative for G1
robot dynamics. Newton terrain telemetry is intentionally not connected yet;
the current right-side controls modify only a clearly labelled browser preview.

## Current live system

- Repo: `/home/auverus/git/everest-hack`
- LAN viewer: `http://172.24.24.81:8765/`
- Alternate LAN viewer: `http://10.36.134.220:8765/`
- Dashboard process: PID `3284194`
- Dashboard tmux pane: `everest_g1:dashboard`
- Native viewer tmux pane: `everest_g1:viewer`
- Bind: `0.0.0.0:8765`
- Start command:

  ```bash
  ./scripts/start-dashboard.sh --host 0.0.0.0 --port 8765 --weather-location south-col
  ```

Live state verified at handoff:

```text
engine              mujoco
paused              true
sim_time             0.0
simulation_fault     null
terrain_collision    everest_hfield
```

The live physics source is currently
`google-deepmind/mujoco_menagerie/unitree_g1`. The visual meshes and policy
interface come from `unitreerobotics/unitree_rl_mjlab`. Do not describe the
active physics scene as Unitree RL MjLab without changing and re-verifying the
model source.

## Browser architecture

```text
MuJoCo G1 state
      |
      v
GET /api/frame -- atomic body names, XYZ positions, WXYZ quaternions
      |
      v
Three.js G1 mesh renderer

Newton MPM (future)
      |
      v
GET /api/terrain/frame -- reduced height/material grid + sparse particles
      |
      v
TerrainRenderer -- interpolate locally and render at display refresh rate
```

There is no MJPEG, WebRTC, or video stream. `/stream` is removed.

## Local path preview

The browser no longer presents the whole compressed Everest DEM as the operator
view. It creates an 8 x 28 metre local path corridor at robot scale:

- Grid: `73 x 225`
- Default slope: `21 degrees`
- Vertical rise at the far end: approximately `9.1 m`
- Underlying rigid/base surface: translucent wireframe
- Snow column: separate translucent PBR surface
- Route centerline: visible line along the corridor
- Wind: debug direction arrow

This fixes the earlier scale error where 6,925 metres of source elevation were
compressed to about 1.78 rendered metres, making the 1.8 metre G1 appear taller
than Everest.

Important boundary: the path corridor is currently a visual sandbox. MuJoCo
still collides against the generated `everest_hfield` from
`maps/everest_terrain.json`. The local preview and MuJoCo collision surface are
not yet the same artifact. The UI explicitly says:

```text
Visual sandbox only - MuJoCo stays authoritative
```

Do not remove that label until the backend publishes a collision-aligned local
patch.

## Debug controls

The right-side panel follows `docs/MULTILAYER_SNOW_MODEL.md`.

Environment controls:

- wind speed, `0-45 m/s`
- meteorological wind direction, `0-359 degrees`
- snowfall, `0-20 mm/h`
- temperature, `-35 to 3 C`
- path slope, `8-32 degrees`
- exposed-surface Coulomb friction, `0.05-0.80`

Per-layer controls:

- `thickness_m`
- `density_kg_m3`
- `stiffness_pa` (Newton `young_modulus`)
- `compressive_strength_pa` (Newton `yield_pressure`)
- `shear_strength_pa` (Newton `yield_stress`)
- `compaction_hardening` (Newton `hardening`)
- `bond_strength_below_pa`

Default four-layer prior, top to bottom:

| Layer | Thickness | Density | Semantic type |
| --- | ---: | ---: | --- |
| Fresh snow | 3 cm | 120 kg/m3 | `POWDER` |
| Wind crust | 6 cm | 380 kg/m3 | `CRUST` |
| Weak layer | 25 cm | 210 kg/m3 | `POWDER` |
| Dense old snow | 40 cm | 520 kg/m3 | `DENSE_SNOW` |

The layer values are stored in Pa internally. UI sliders display kPa/MPa where
appropriate. Input updates are coalesced to one rebuild per animation frame.
They do not call `/api/control` and do not mutate MuJoCo or Newton.

## Renderer behavior

`dashboard/static/terrain-renderer.js` supports:

- dynamic heightfield surfaces
- per-cell material IDs
- per-cell compaction
- multilayer metadata
- loose, packed, crust, and ice appearance
- wireframe base terrain
- smooth height interpolation between physics snapshots
- up to 20,000 instanced sparse particles
- JSON, nested arrays, typed arrays, and little-endian base64 arrays
- static path preview and live Newton frames through the same renderer

The renderer labels terrain `LIVE` only after accepting a monotonic physics
frame. Preview data cannot silently claim to be live physics.

## Newton handoff contract

The complete schema is in `docs/TERRAIN_TELEMETRY.md`. The preferred backend
integration is:

1. Advertise the endpoint from `GET /api/scene`:

   ```json
   {
     "terrain_stream": {
       "schema": "everest-terrain/v1",
       "url": "/api/terrain/frame",
       "rate_hz": 15
     }
   }
   ```

2. Publish one complete, atomic frame containing:

   ```text
   sequence, timestamp, sim_time
   origin, size, resolution
   heights
   material_ids
   compaction
   layers
   optional sparse particles
   ```

3. Keep coordinate conventions identical to MuJoCo/Newton: metres, Z-up, row
   major heights with X changing fastest.
4. Reduce the full MPM particle state to a local surface/material grid at
   10-15 Hz. Do not send every solver particle to the browser.
5. Send only airborne/displaced diagnostic particles in the optional sparse
   particle field.
6. Increment `sequence` only when every array in the frame is complete.

The client accepts terrain embedded in `/api/frame` too, but a dedicated
terrain endpoint is preferred because G1 pose telemetry and Newton surface
sampling have different natural rates.

## Important files

- `dashboard/static/terrain-renderer.js` - terrain rendering and telemetry ingestion
- `dashboard/static/debug-controls.js` - local snow/wind control state
- `dashboard/static/app.js` - G1 renderer and transport integration
- `dashboard/static/index.html` - HUD and debug-panel markup
- `dashboard/static/styles.css` - viewer and panel visual design
- `dashboard/server.py` - static/API routing
- `dashboard/engines/mujoco.py` - active MuJoCo telemetry adapter
- `docs/MULTILAYER_SNOW_MODEL.md` - mechanical parameter specification
- `docs/TERRAIN_TELEMETRY.md` - Newton-to-browser terrain schema
- `docs/VIEWER_ARCHITECTURE.md` - engine-neutral viewer architecture
- `maps/everest_terrain.json` - current full DEM tile
- `maps/build_everest_visual.py` - full DEM generation

## Verification performed

Static checks:

```bash
node --check dashboard/static/app.js
node --check dashboard/static/terrain-renderer.js
node --check dashboard/static/debug-controls.js
python3 -m compileall -q dashboard
git diff --check
```

Browser checks against the live LAN service:

```text
engine label                 MUJOCO
terrain mode                 PATH DEBUG
path detail                  8x28 m path, 74 cm snow column
terrain resolution           73x225
wireframe base               true
default layers               4
environment sliders          6
mechanical layer sliders     7
wind slider mutation         changed rendered height target
page JavaScript errors       none (favicon 404 is harmless)
```

Live artifact checks:

```bash
curl -sS http://127.0.0.1:8765/api/state
curl -sS http://127.0.0.1:8765/api/frame
curl -sS http://127.0.0.1:8765/api/scene
curl -sS http://127.0.0.1:8765/terrain-renderer.js
curl -sS http://127.0.0.1:8765/debug-controls.js
```

## Known issues and cautions

1. The repo worktree is intentionally dirty. Multiple agents contributed
   uncommitted physics, weather, policy, map, viewer, and skill files. Preserve
   all unrelated changes; do not reset or overwrite the worktree.
2. The local browser path is not yet generated from the exact MuJoCo collision
   patch. It is a real-metre procedural debug corridor.
3. Debug values are not persisted across page reloads.
4. The controls display four default layers. The model architecture supports
   1-6 layers, but add/remove/reorder controls are not implemented.
5. The current full Everest heightfield generator intentionally compresses the
   mountain into a small simulation tile. Do not reuse that visual scale for a
   local robot-scale view.
6. The active MuJoCo policy can become numerically unstable on this terrain.
   The engine detects the fault, pauses, and reports `simulation_fault`. At this
   handoff the fault is clear and the simulation is paused at time zero.
7. A browser debug handle is available as `window.__everestViewer`. It exposes
   the Three.js scene, camera, renderer, controls, terrain renderer, and local
   debug-control state for inspection.

## Recommended next steps

1. Backend agent publishes an atomic local Newton patch using
   `everest-terrain/v1`.
2. Make the MuJoCo collision patch and rendered patch derive from the same local
   source artifact.
3. Add optional persistence/export for debug snow-column settings.
4. Add layer creation/removal/reordering, enforcing the documented 1-6 limit.
5. Validate terrain and G1 transforms with a known foot contact at a known grid
   cell before interpreting visual footprints as correct.
6. Measure browser bandwidth and terrain update cost before increasing grid
   resolution or particle count.

## Resume commands

```bash
cd /home/auverus/git/everest-hack
tmux capture-pane -p -t everest_g1:dashboard -S -120
curl -sS http://127.0.0.1:8765/api/state | python3 -m json.tool
```

If the dashboard is not running:

```bash
tmux new-window -t everest_g1 -n dashboard \
  'cd /home/auverus/git/everest-hack && ./scripts/start-dashboard.sh --host 0.0.0.0 --port 8765 --weather-location south-col'
```
