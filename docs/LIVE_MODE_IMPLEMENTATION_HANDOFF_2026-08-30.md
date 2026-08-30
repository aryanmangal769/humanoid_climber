# Everest LIVE Mode Implementation Handoff

Date: 2026-08-30  
Branch: `everest-dream-integration`  
Parallel-work baseline: `8827468` (`Build Everest Unity snow and simulation stack`)  
Repository: `/home/auverus/git/everest-hack`

## Assignment

Turn the Unity editor's existing `LIVE` selector into a truthful, fail-safe live
data path. Keep this work separate from the concurrent Newton 1.5 snow-physics
migration. The live path must never silently present simulated MuJoCo/Newton
state as physical-robot telemetry.

This handoff is an implementation contract, not a claim that a physical robot
or live weather endpoint is already configured. No concrete Unitree/ROS or
weather-station endpoint exists in this checkout yet, so transport-specific
code must sit behind an adapter and be testable with a replay/fake source.

## Current truth

The `SIM` / `LIVE` UI exists, but `LIVE` is currently cosmetic:

- `EverestEditorHud.SetDataMode()` sends `{action: "mode", value: "live"}`.
- `UnityRendererBridge.control("mode", ...)` only changes `self.data_mode` and
  `environment["data_mode"]`.
- `UnityRendererBridge` constructs one local `MuJoCoEngine` in `__init__` and
  all robot frames continue to come from it in both modes.
- The environment panel becomes read-only in `LIVE`, but the displayed values
  are still the last local dictionary values.
- There is no Unitree SDK, ROS 2 subscriber, log/replay source, live weather
  client, clock synchronizer, source-health monitor, or stale-data policy.
- There is no safe command-routing contract for a physical robot.

Relevant current files:

- `simulation/unity_bridge.py`
- `studio/unity/EverestSim/Assets/Scripts/EverestBackendClient.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestEditorHud.cs`
- `studio/unity/EverestSim/Assets/Scripts/EverestRuntime.cs`
- `docs/UNITY_RENDERER_BACKEND_CONTRACT.md`
- `docs/UNITY_RENDERER_BRIDGE_PROTOCOL.md`

## Non-negotiable behavior

1. `SIM` continues to use the local MuJoCo/Newton engine and stays the default.
2. `LIVE` uses an explicit external/replay adapter. It must not fall through to
   the local simulator when the live source is missing or stale.
3. Missing live data produces a visible `DISCONNECTED` or `STALE` state while
   retaining the last sample only as visibly stale presentation data.
4. LIVE is read-only by default. Do not forward movement, reset, pause, surface,
   snow, cheat, or weather-edit commands to physical hardware.
5. Hardware control, if added later, requires a separate opt-in arming
   handshake, an e-stop path, command expiry/deadman behavior, and an explicit
   allowlist. Merely selecting `LIVE` must never arm control.
6. Preserve the renderer-neutral WebSocket schema and coordinate contract:
   metres, right-handed backend coordinates, Z up, and WXYZ quaternions.
7. Report provenance and age for every live channel. Do not label inferred or
   visually generated snow deformation as measured live snow state.

## Recommended architecture

Add a small source interface instead of branching throughout the WebSocket
loop. Suggested shape (names may vary):

```python
class RendererDataSource(Protocol):
    def frame(self) -> dict: ...
    def environment(self) -> dict: ...
    def snow(self) -> dict | None: ...
    def health(self) -> dict: ...
    def close(self) -> None: ...
```

Implement two sources:

- `SimDataSource`: a thin owner/wrapper around the existing `MuJoCoEngine`.
- `LiveDataSource`: consumes a configured adapter, normalizes samples, keeps a
  bounded last-known sample, and exposes connection/age/error metadata.

The first live adapter should be a deterministic replay or local JSON/UDP/ROS
fixture, not hard-wired vendor code. Put the transport behind a protocol such
as `LiveTelemetryAdapter`, then add Unitree/ROS 2 as a separate adapter once
the actual topic/API contract is supplied.

Mode switching should be atomic:

1. Validate that the requested source is configured.
2. Stop accepting SIM-only controls.
3. Switch the publisher to the selected source.
4. Increment a source epoch/revision so Unity discards interpolation across
   the SIM/LIVE boundary.
5. Immediately publish state and source health.

Do not tear down the local simulator merely to view LIVE. It may remain paused
in the background so switching back to SIM is quick, but it must not publish
through the live path.

## Protocol additions

Extend the `state` message compatibly with an optional block similar to:

```json
{
  "data_mode": "live",
  "source": {
    "kind": "unitree_ros2",
    "status": "connected",
    "epoch": 3,
    "sample_time": 1788134400.125,
    "age_ms": 18.4,
    "stale_after_ms": 250,
    "last_error": null,
    "channels": {
      "robot": {"status": "connected", "age_ms": 18.4},
      "weather": {"status": "stale", "age_ms": 4200.0},
      "snow": {"status": "unavailable", "age_ms": null}
    }
  },
  "control_authority": "read_only"
}
```

Use source timestamps where available and keep receipt time separately.
Reject future samples and non-monotonic samples beyond a small documented
tolerance. A source epoch must change after reconnect or replay seek so Unity
does not interpolate between unrelated poses.

The live robot frame should preserve the existing keys consumed by Unity:

- base position and WXYZ orientation
- base linear/angular velocity
- ordered joint names, positions, velocities, and torques
- foot poses/contact data when genuinely available
- `paused` may be false, but must not imply that LIVE can be paused

If the live source uses a different joint order or coordinate system, normalize
at the adapter boundary and add fixture tests. Missing joints must be reported,
not silently filled from the simulated G1.

## Snow, ice, rock, and weather in LIVE

Keep source classes honest:

- If a live/replay source provides a measured or reconstructed terrain/snow
  surface, publish it with provenance and its own timestamp.
- If only static DEM/material priors are available, render those as a static
  context layer and mark snow deformation `unavailable`.
- Do not run local Newton using the live robot pose and then call the result
  measured snow. If that predictive/digital-twin mode is useful, name it
  separately (for example `twin`) and label its snow as predicted.
- Snow, exposed glacier ice, and rock need spatial material masks. The present
  physics backend instead selects one global active surface (`snow`, rigid
  `ice`, or rigid `rock`), so do not claim simultaneous live material physics.
- An `ICE` layer inside the current snow column is still an MPM material with
  stiffer parameters; it is not a brittle/fracturing glacier-ice model.
- Weather values need per-channel timestamps and provenance. The UI must not
  allow local weather sliders to overwrite a live feed.

## UI work

Update the dock/status bar to show, at minimum:

- `LIVE CONNECTED`, `LIVE STALE`, or `LIVE DISCONNECTED`
- live source kind/name
- sample age
- `READ ONLY` unless a future explicit hardware-control system is armed
- per-channel warning when robot, weather, or snow data is unavailable

When entering LIVE:

- send one zero/stop command to the SIM engine before changing source;
- clear held manual/cheat input locally;
- disable all simulation mutation controls, not only weather controls;
- clear interpolation buffers when the source epoch changes;
- keep camera controls available.

When returning to SIM, resynchronize all UI drafts from the SIM backend rather
than sending stale LIVE values into it.

## Suggested file ownership for parallel work

Prefer adding files so this work does not collide with the Newton migration:

- add `simulation/data_sources.py`
- add `simulation/live_telemetry.py` (adapter protocol and replay/fake adapter)
- edit `simulation/unity_bridge.py`
- edit `EverestBackendClient.cs`, `EverestEditorHud.cs`, and
  `EverestRuntime.cs` only for source health, interpolation reset, and control
  lockout
- extend bridge/control tests
- update the renderer bridge protocol documentation

Avoid editing these concurrently owned Newton files unless absolutely needed:

- `simulation/newton_mujoco.py`
- `simulation/newton_snow.py`
- `simulation/static_snow_coupling.py`
- `scripts/setup-sim-stack.sh`
- `scripts/verify-newton-mujoco.sh`

Coordinate before changing `dashboard/engines/mujoco.py`; the Newton migration
and live adapter can otherwise remain independent.

## Configuration

Add explicit CLI/environment configuration. Suggested initial flags:

```text
--live-adapter disabled|replay|ros2|unitree
--live-endpoint <adapter-specific endpoint>
--live-replay <path>
--live-stale-ms 250
```

Default must be `disabled`. Selecting LIVE while disabled should return a
structured error and leave the current SIM source active, while the UI reports
that LIVE is not configured.

Do not add secrets, robot credentials, DDS private configuration, or recorded
operator data to Git.

## Tests and acceptance gates

Add a fake/replay source so all behavior can be verified without hardware.
Minimum automated coverage:

1. SIM remains the default and existing bridge tests still pass.
2. Selecting configured LIVE changes frame provenance and source epoch.
3. A known replay pose maps to the correct Unity position, quaternion, and
   named joint ordering.
4. Stopping replay updates changes health to `STALE` within the configured
   threshold; it never falls back to SIM frames.
5. Disconnect/reconnect increments epoch and does not interpolate across it.
6. Mutation commands in LIVE return a structured read-only rejection and do
   not reach MuJoCo or the adapter.
7. Switching back to SIM restores local control without applying stale LIVE
   values.
8. Missing snow/weather channels render static context plus an explicit UI
   warning rather than fabricated measurements.
9. WebSocket reconnect and malformed/out-of-order samples do not crash the
   backend.

Run at least:

```bash
PYTHONPATH=. .venv-sim/bin/python scripts/test_renderer_bridge.py
PYTHONPATH=. .venv-sim/bin/python scripts/verify_unity_sim_controls.py
PYTHONPATH=. .venv-sim/bin/python tests/test_live_data_sources.py
PYTHONPATH=. .venv-sim/bin/python scripts/verify_live_mode.py
```

The live-source test module has a built-in direct runner because the simulation
environment intentionally does not install pytest. It remains pytest-compatible
when pytest is available in a development environment.

Then build and smoke the Unity WebGL player. Verify both source switching and
stale/disconnected presentation in the browser console and screenshot.

## Live deployment/cutover

Do not replace the persistent backend until tests pass.

1. Start a disposable backend on a non-production port:

   ```bash
   scripts/start-simulation-backend.sh --port 18766 \
     --live-adapter replay --live-replay <fixture>
   ```

2. Connect Unity/test client, switch SIM -> LIVE -> SIM, and inspect source
   epoch, sample age, command rejection, and pose continuity.
3. Verify the real adapter separately with read-only telemetry.
4. Capture the current persistent process/tmux command before restarting it.
5. Restart the persistent backend only after the disposable probe is clean.
6. Verify the actual listening process, WebSocket state, and Unity status bar;
   do not infer success from a launcher exiting normally.

The current bridge default port in `simulation/unity_bridge.py` is `8765`.
Some older handoff text mentions `18765`; always inspect the live process and
launcher arguments instead of assuming either port.

## Definition of done

LIVE is done only when an external or deterministic replay source—not the
local `MuJoCoEngine`—drives the displayed robot frame; provenance, freshness,
and channel availability are visible; stale/disconnected behavior is tested;
simulation mutations are safely rejected; and SIM remains fully functional.

Do not call LIVE complete merely because the button changes color or the
environment sliders are disabled.

## Implementation result (2026-08-30)

The handoff is implemented by `simulation/data_sources.py` and
`simulation/live_telemetry.py`. Replay, watched JSON, local UDP, composited
Open-Meteo weather, and generic robot sensor channels now drive the renderer
through a normalized read-only contract. `simulation/unity_bridge.py` owns
atomic SIM/LIVE switching, source epochs, per-channel freshness/provenance,
stale retention, structured mutation rejection, and SIM restoration.

Unity displays LIVE source status/age/channel warnings, clears robot and snow
interpolation across source epochs, hides stale cross-source geometry, and
locks simulation/robot mutation controls while keeping the camera available.
`tests/fixtures/live_replay.json`, `tests/test_live_data_sources.py`, and
`scripts/verify_live_mode.py` provide the hardware-free acceptance path.

No physical Unitree command transport was added. A real ROS 2/Unitree process
should normalize telemetry into the watched JSON or UDP schema; adding command
authority still requires the separate arming, e-stop, allowlist, and deadman
design described above.
