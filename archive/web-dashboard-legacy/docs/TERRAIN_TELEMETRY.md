# Newton terrain telemetry

The browser terrain renderer is implemented in
`dashboard/static/terrain-renderer.js`. It is independent of the robot pose
adapter and accepts a complete surface artifact from either location:

- the static Everest tile advertised by `GET /api/scene` and served at
  `GET /everest-terrain.json`
- `terrain` embedded in `GET /api/frame`
- a dedicated `GET /api/terrain/frame` response, polled at up to 15 Hz

The static tile is also compiled into the dashboard's MuJoCo model as the
`floor` heightfield, so its dimensions, placement, and heights are identical
to the rendered mesh. If no static or live surface is available, the viewer displays a deterministic layered preview
and labels it `PREVIEW`. It labels the terrain `LIVE MPM` only after accepting a
valid physics frame.

The scene manifest should advertise the dedicated endpoint so clients do not
probe optional routes repeatedly:

```json
{"terrain_stream": {"schema": "everest-terrain/v1", "url": "/api/terrain/frame", "rate_hz": 15}}
```

## Schema

```json
{
  "schema": "everest-terrain/v1",
  "sequence": 42,
  "timestamp": 1788039000.25,
  "sim_time": 1.32,
  "origin": [-2.0, -2.0, 0.0],
  "size": [4.0, 4.0],
  "resolution": [129, 129],
  "heights": [0.12, 0.121],
  "material_ids": [0, 0, 1, 2],
  "compaction": [0.0, 0.15, 0.8, 1.0],
  "layers": [
    {"id": 0, "name": "loose snow", "color": [0.86, 0.91, 0.97], "roughness": 0.96, "depth": 0.10},
    {"id": 1, "name": "packed snow", "color": [0.66, 0.76, 0.86], "roughness": 0.78, "depth": 0.08},
    {"id": 2, "name": "ice crust", "color": [0.28, 0.52, 0.68], "roughness": 0.16, "depth": 0.035}
  ],
  "particles": {
    "positions": [0.1, 0.2, 0.3],
    "radii": [0.012],
    "material_ids": [0]
  }
}
```

Coordinates are Newton/MuJoCo world coordinates: Z-up, metres, WXYZ robot
quaternions. `origin` is the lower XY corner and Z reference. Heights are
absolute Z coordinates, row-major with X changing fastest. Material and
compaction arrays have the same element count as heights.

Numeric arrays may be JSON arrays, nested JSON arrays, typed arrays when the
frame is provided directly in JavaScript, or little-endian base64 strings.
Sparse `particles` are optional and intended for ejected snow or diagnostic
sampling, not the full MPM particle set. The viewer caps particles at 20,000.

## Publication discipline

- Increment `sequence` only after every array for the frame is complete.
- Publish one atomic frame; never update heights and materials separately.
- Use simulation time for replay and wall time only for liveness.
- A practical publication rate is 10-15 Hz. The renderer interpolates height
  updates and draws at the display refresh rate.
- Reduce MPM particles to a height/material/compaction grid server-side. Do not
  send the complete solver particle state to every browser.
- Emit sparse particle samples only for airborne or displaced material that a
  heightfield cannot represent.

## Local debug preview

Before live Newton terrain is connected, the browser renders an 8 x 28 metre
path corridor at real robot scale. The underlying rigid terrain is a wireframe;
the snow column is a separate translucent surface. The right-side controls edit
only this browser preview and do not post values to MuJoCo or Newton.

The controls follow `MULTILAYER_SNOW_MODEL.md`: surface friction and 1-6 layers
with thickness, density, stiffness, compressive strength, shear strength,
compaction hardening, and bond strength below. Wind speed/direction, snowfall,
temperature, and slope are visualization inputs so the eventual weather and
Newton feeds have an already-defined presentation boundary.
