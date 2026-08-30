# Viewer architecture

The web viewer transports simulation state, not pixels. MuJoCo publishes one
complete pose snapshot at 30 Hz; the browser loads Unitree's STL meshes once
and renders them locally with Three.js at the display refresh rate.

```text
GET  /api/frame    latest atomic body-pose snapshot
GET  /api/scene    visual mesh manifest and local transforms
GET  /api/state    engine identity, model provenance, and liveness
GET  /api/terrain/frame  optional Newton surface telemetry
POST /api/control  reset and pause
```

`dashboard.engines.protocol.ViewerEngine` is the engine boundary. An adapter
owns its simulator and returns the same engine-neutral artifacts:

```text
Unitree RL MjLab / policy             Newton / future engine
             |                                  |
             v                                  v
       MuJoCoEngine                       NewtonEngine
             |                                  |
             +---- ViewerEngine contract -------+
                              |
                    complete pose snapshot
                              |
                     GET /api/frame (JSON)
                              |
                 Three.js + real Unitree STLs
```

`everest-viewer/v1` frames contain a monotonic sequence, wall timestamp,
simulation time, timestep, reset marker, engine provenance, and matching arrays
of body names, positions, and WXYZ quaternions. Building the dictionary under
the engine lock and replacing it in one assignment makes each frame an atomic
artifact. One request retrieves the complete pose; there are no per-body HTTP
round trips.

This follows `SIM2REAL_PLAYBOOK.md` directly:

- The process being alive is insufficient. `/api/frame.sequence` must advance.
- Sampling and browser polling are independent: the adapter publishes at 30 Hz,
  the browser polls at 25 Hz, and `requestAnimationFrame` renders locally.
- Visual interpolation uses snapshot timestamps while authoritative replay and
  policy timing remain simulation-time based.
- The next frame is fetched while the browser continues rendering the current
  one; network latency does not block camera interaction.
- Reset is an explicit frame marker rather than an inferred visual event.

The viewer does not expose MJPEG, WebRTC, screenshots, or any video endpoint.
Camera orbit and zoom are local, immediate browser interactions. Physics and
policy execution remain entirely server-side.

The optional Newton terrain channel publishes a reduced height/material grid
plus sparse airborne particles. See `docs/TERRAIN_TELEMETRY.md`. The browser
interpolates the surface locally; it never transports rendered terrain pixels.
