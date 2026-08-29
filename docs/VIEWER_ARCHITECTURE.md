# Viewer architecture

The browser is a viewer, not a physics engine. It knows only three endpoints:

```text
GET  /stream       multipart JPEG frames
GET  /api/state    latest engine artifact and liveness
POST /api/control  reset, pause, camera preset
```

`dashboard.engines.protocol.ViewerEngine` is the boundary. The active
`MuJoCoEngine` loads Unitree RL MjLab's canonical G1 scene and owns its model,
data, camera, render loop, and latest complete JPEG. A Newton adapter can
implement the same contract later, without changing the browser or HTTP layer.

```text
Unitree RL MjLab scene/policy
            |
            v
      MuJoCoEngine  ---- future ----> NewtonEngine
            |                         (same contract)
            v
    latest complete frame + state artifact
            |
            v
       HTTP viewer server
            |
            v
      full-screen browser viewer
```

This follows the Sim2Real playbook's verification discipline:

- A running process is not proof of rendering; `/api/state.frames` must advance.
- The browser consumes only complete JPEGs. It never observes a partially
  written frame.
- Physics, capture, transport, and presentation are separate modules, so timing
  can be instrumented independently when performance work begins.
- Engine identity and source model are explicit in `/api/state`; a fallback can
  never silently claim to be the requested backend.

MJPEG is deliberate for the first LAN implementation: it is dependency-light,
works in every browser, reconnects naturally, and provides the exact MuJoCo
render rather than a browser-side approximation. If bandwidth or latency becomes
the measured bottleneck, replace only the transport with WebRTC or WebCodecs;
the engine adapter and viewer state contract stay unchanged.
