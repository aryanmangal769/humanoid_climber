# Policy Router Overlay Contract

The `hum-climber-play` Viser path installs `HumanoidClimberViserPlayViewer`, which adds a movable **Policy Supervisor**
panel over the existing localhost MjLab UI. Scenario code should not import or modify the viewer.

## Scenario -> UI telemetry

For the most accurate overlay, expose this on the unwrapped environment:

```python
def get_policy_router_context(self, env_idx: int) -> dict[str, object]:
  return {
    "terrain_type": "rough stairs",
    "gradient": 0.12,
    "mu": 0.24,
    "roughness": 0.06,
    "step_height": 0.10,
    "wind": [0.0, 8.0, 0.0],
    "slip": 0.17,
    "is_fallen": False,
    "active_policy": "ice_incline",
    "uncertainty": 0.10,
  }
```

Use the **actual sampled/current values for that environment**, not the configuration bounds. The random scenario
especially needs this because MjLab's random terrain generator does not retain the sampled terrain family in a public
per-environment name.

The viewer also accepts a mapping or per-environment sequence under `policy_router_context`, `scenario_state`, or
`terrain_context`.

Canonical fields are:

- `slope_gradient`: rise/run gradient, not degrees
- `friction`: estimated contact friction coefficient
- `roughness_m`: terrain relief in meters
- `step_height_m`: step/obstacle height in meters
- `wind_force_n`: force magnitude in newtons; a 3-vector is also accepted through the `wind` alias
- `slip_ratio`: normalized slip indicator
- `fallen`: current fall state
- `torso_height_m`: optional posture signal
- `active_policy`: currently deployed policy/checkpoint label
- `terrain_label`: short human-readable terrain name
- `uncertainty`: 0-1 estimate uncertainty

Aliases accepted by the UI include `gradient`, `slope`, `mu`, `roughness`, `step_height`, `wind`, `slip`, `is_fallen`,
`height`, `policy`, and `terrain_type`.

## Current heuristic behavior

The overlay is intentionally a demo supervisor, not the final safety gate. It compares current terrain criteria against
the current policy portfolio and emits one of three live outcomes:

- **POLICY SELECTED / POLICY SWAP** for the available flat or low-friction incline specialists.
- **FINE TUNING NEW POLICY** when a matching checkpoint is unavailable or a
  condition is outside every loaded baseline envelope.

Unsupported combinations generate a concise fine-tuning template entry instead
of silently choosing the least-bad policy. The template has no execution path:
no queue, upload, subprocess, trainer, or weight update is invoked.

The panel updates at roughly 4 Hz and keeps a short decision history whenever the selected route/action changes. When no
scenario telemetry hook exists, it now reads the current randomized foot friction from MjLab and probes the local MuJoCo
terrain geometry around the robot to estimate slope, relief, and step height. Fixed task configuration, live
external-wrench, and posture state fill the remaining gaps. Exact scenario telemetry still wins whenever it is supplied.

That geometry fallback is intentionally local: it describes the patch under/near the robot, not an entire future route.
The hard-coded scenario should therefore publish its known upcoming terrain through the hook when it wants the UI to
reason ahead of the robot; the randomized feasibility scenario can work without that extra coupling.

## Policy swapping

The overlay currently recommends the route; it does **not** load networks itself. A scenario/supervisor agent that
performs real checkpoint swaps should publish the resulting `active_policy` value through the same context hook so the
UI can distinguish `Keep`, `Select`, and `Switch to` decisions.
