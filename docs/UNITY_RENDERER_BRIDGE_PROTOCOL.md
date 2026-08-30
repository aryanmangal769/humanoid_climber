# Unity renderer bridge protocol

The canonical requirements are in `UNITY_RENDERER_BACKEND_CONTRACT.md`. This
document records the concrete MVP transport exposed by the backend.

Start the renderer-neutral physics process with:

```bash
./scripts/start-simulation-backend.sh
```

The default endpoint is `ws://127.0.0.1:8765`. Every server message is a JSON
object with `type` and `data`. On connection the backend sends `scene`,
`terrain`, `macro_terrain`, `environment`, and `state`. It then streams `frame`
at up to 60 Hz, `snow` at up to 15 Hz, and `state`/`environment` at 2 Hz.

Coordinates are metres, right-handed, +X east, +Y north, +Z up. Quaternions are
`wxyz`. Unity alone converts into its left-handed Y-up convention.

`frame` contains complete G1 body transforms plus base velocity, joint
position/velocity/torque, command state, and independent left/right foot
contact telemetry. `snow` contains the Newton surface grid (`origin`, `size`,
`resolution`, `heights`, `compaction`, `material_ids`) and layer metadata. Raw
particles are omitted unless the backend is launched with `--particles`.

Client controls use:

```json
{"type":"control","action":"command","value":[0.4,0.0,0.0]}
{"type":"control","action":"pause","value":false}
{"type":"control","action":"reset","value":null}
{"type":"control","action":"snow_parameters","value":{}}
{"type":"control","action":"weather","value":{}}
```

Every control receives a `control_ack`. Physics faults are also emitted as a
`fault` message and remain visible in subsequent `state` messages.

Probe and integration test:

```bash
./scripts/start-simulation-backend.sh --probe
./scripts/test-renderer-bridge.sh
```

Both terrain products are sourced from
`HMA2_DCG_SMB_KHU_ANNUAL_DEM_20151102T0517Z_V01.0.tif`, whose GeoTIFF pixel
scale is 2 m x 2 m in UTM zone 45N. The local terrain product is 1.2 km square
at 257x257 transport samples and is the same true-scale product compiled into
MuJoCo. The macro visual product is a downsampled view of the complete 2 m
raster footprint, approximately 13.77 x 10.40 km, and has no physics ownership.
