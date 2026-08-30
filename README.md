# Everest Dream — Unitree G1 + Newton simulation

Everest Dream is moving to a native Omniverse/Isaac Sim editor instead of a
browser renderer. The active UI direction lives in `studio/omniverse`: a USD/
RTX scene editor for Everest terrain, Unitree G1, and static Newton MPM snow.

The retired Three.js/browser dashboard is preserved under
`archive/web-dashboard-legacy/` for reference only. It is no longer an active
runtime or product entrypoint.

```text
physics: vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/scene_mjx.xml
visual assets: vendor/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml
upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
```

No hardware connection or robot-control path is part of this phase.

## Setup

```bash
git submodule update --init --recursive
./scripts/setup-rl-stack.sh
./scripts/setup-playground.sh  # fetches Playground's pinned Menagerie checkout
./scripts/verify-g1.sh
./scripts/verify-newton-mujoco.sh
./scripts/verify_static_snow_twoway.py
```

`verify-g1.sh` constructs the exact Unitree RL MjLab scene and reports its
body, joint, actuator, and MuJoCo metadata.

## Use / editor direction

```bash
# Native desktop MuJoCo viewer.
./scripts/view-g1.sh
```

The rigid-body Newton integration is available as
`scripts/verify-newton-mujoco.sh`: it imports the G1 MJCF into Newton, advances
it with `SolverMuJoCo`, and exercises the body-wrench boundary used by MPM
impulses. `simulation/newton_snow_twoway.py` is the replacement path for static
snow: it follows Newton's direct two-way rigid/MPM coupling architecture rather
than treating a browser heightfield as the snow contact surface.

`maps/everest_terrain.json` remains the shared physical terrain source. Terrain
crop/scale operations must change the physical artifact (and therefore the
collider), not merely stretch a rendered mesh. Snow is currently treated as a
static initial multilayer snapshot; snowfall accumulation is intentionally out
of scope for this phase.

The Omniverse extension scaffold is in
`studio/omniverse/exts/everest.studio/`. An Isaac Sim/Kit runtime is required to
launch the RTX editor and is not vendored in this repository.

## Weather parameters

`weather/everest_weather.py` can derive environment and snowpack priors from
South Col forecast data:

- surface pressure and temperature derive air density; gust speed, estimated
  G1 frontal area, and drag coefficient derive a bounded pelvis wind force;
- snowfall, precipitation, temperature, sustained wind, and gusts derive a
  surface snow type, density, stiffness, compressive/shear strength, hardening,
  bond strength, fresh-layer thickness, and traction prior;
- the weather snow prior can initialize a static multilayer/MPM snapshot unless
  operator-provided or sensor-corrected parameters take ownership.

These are forecast-derived demo parameters, not calibrated mountain control
limits. Weather cannot observe buried snow layers, so operator or robot-derived
column parameters take precedence over later forecast priors.

## Training

Unitree RL MjLab is the canonical task and policy interface:

```bash
./scripts/train-unitree-g1.sh
NUM_ENVS=256 ./scripts/train-unitree-g1.sh
TASK=Unitree-G1-Rough ./scripts/train-unitree-g1.sh
```

## Pinned upstreams

- Unitree RL MjLab: `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- Newton: `d6046f187f1f6c6b8f8da98c5d0f93b8944eb5f0`
- MuJoCo Playground: `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`
- Native editor target: Omniverse Kit / Isaac Sim 6.x (runtime external)

The current snow work uses Newton Implicit MPM on CUDA when available. The
target architecture is direct Newton two-way coupling between the G1 rigid
model and a static, already-deposited snowpack. Material parameters still need
site-specific system identification before being called Everest-calibrated.
