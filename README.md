# Everest Dream — Unitree G1 viewer and RL stack

The LAN site is a pure MuJoCo viewer. MuJoCo renders Unitree RL MjLab's
canonical G1 scene server-side and streams complete frames to an engine-neutral
browser client.

```text
vendor/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml
```

No hardware connection or robot-control path is part of this phase.

## Setup

```bash
git submodule update --init --recursive
./scripts/setup-rl-stack.sh
./scripts/verify-g1.sh
```

`verify-g1.sh` constructs the exact Unitree RL MjLab scene and reports its
body, joint, actuator, and MuJoCo metadata.

## Use

```bash
# Native desktop MuJoCo viewer.
./scripts/view-g1.sh

# Full-screen web viewer; add --host 0.0.0.0 for LAN access.
./scripts/start-dashboard.sh --host 0.0.0.0 --port 8765
```

The viewer exposes separate frame, state, and control contracts. Physics,
capture, transport, and presentation remain independent, so Newton can later
implement the same adapter without changing the browser UI. See
`docs/VIEWER_ARCHITECTURE.md` and `docs/STACK.md`.

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
