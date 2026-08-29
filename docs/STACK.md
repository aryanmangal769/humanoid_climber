# Simulation and RL stack

Everest Dream keeps **MuJoCo as the robot physics backend** and uses Newton only
for coupled material simulation such as granular/deformable snow.

## Pinned upstreams

- `unitreerobotics/unitree_rl_mjlab` at `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- `newton-physics/newton` at `v1.0.0` / `d6046f187f1f6c6b8f8da98c5d0f93b8944eb5f0`
- `google-deepmind/mujoco_playground` at `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`; it owns the dashboard's Menagerie dependency
- `google-deepmind/mujoco_menagerie` at `1b86ece576591213e2b666ebf59508454200ca97` (pinned and fetched by Playground)

The Unitree stack provides the canonical G1 tasks and RSL-RL/PPO training path.
Newton v1.0.0 was chosen deliberately because it is on the same MuJoCo 3.5
generation and already contains `SolverMuJoCo`, `SolverImplicitMPM`, two-way
rigid/MPM coupling, and the MPM ANYmal walking example.

## Runtime compatibility

The shared environment pins:

```text
mujoco         3.5.0
mujoco-warp    3.5.0.2
warp-lang      1.12.0
mjlab          1.2.0
newton         v1.0.0 checkout
unitree_rl_mjlab editable checkout
```

`unitree_rl_mjlab/setup.py` pins `mujoco-warp==3.5.0`, but `mjlab==1.2.0`
declares `mujoco-warp>=3.5.0`. Newton v1.0.0 requires `3.5.0.2`, so the setup
script installs the common runtime explicitly and installs Unitree with
`--no-deps`. This avoids a needless downgrade while staying in the supported
MJLab 1.2 / MuJoCo 3.5 API family.

Warp is pinned to `1.12.0`: MJLab 1.2 still accesses the deprecated
`wp.context` API, which was removed in Warp 1.13+. Newton v1.0.0 supports
Warp 1.12, so this is the common compatible version.

The Unitree wrapper defaults to TensorBoard logging. MJLab 1.2 requires a
modern W&B release, while the RSL-RL 5.0.1 W&B writer still references the
removed `start_method` setting. This is only a logging integration issue and
does not affect simulation or PPO, so TensorBoard is the stable default.

MJLab 1.2 imports SciPy from its terrain stack but does not currently pull it
in through the dependency path used here, so the bootstrap script installs it
explicitly. On WSL2 the run wrappers also add `/usr/lib/wsl/lib` to
`LD_LIBRARY_PATH` so NVIDIA Warp can resolve the host `libcuda` driver.

Newton's packaged G1 example uses USD assets and its ANYmal MPM example uses
Collada meshes, so the bootstrap also installs `usd-core` and `pycollada`.

## Setup

```bash
./scripts/setup-rl-stack.sh
./scripts/verify-rl-stack.sh
```

Train the upstream Unitree G1 policy with:

```bash
./scripts/train-unitree-g1.sh

# smaller smoke run configuration
NUM_ENVS=256 ./scripts/train-unitree-g1.sh

# upstream rough-terrain task
TASK=Unitree-G1-Rough ./scripts/train-unitree-g1.sh
```

## Integration boundary

### G1 model provenance

The browser dashboard and `scripts/view-g1.sh` load
`vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/scene_mjx.xml`
directly with native MuJoCo. This is Menagerie's official G1 MJX scene,
including its collision pairs, position actuators, and keyframes.

`scripts/train-unitree-g1.sh` intentionally remains on Unitree MJLab's adapted
G1 XML because the training task's actuator and observation conventions are
coupled to it. Its meshes are byte-identical to Menagerie, but it is a separate
training representation rather than the canonical dashboard scene.

The intended architecture is:

```text
Unitree MJLab / RSL-RL
        |
        | G1 policy + task/reward conventions
        v
MuJoCo / MuJoCo Warp 3.5  <---- robot dynamics
        ^
        | two-way forces
        v
Newton Implicit MPM       <---- snow / granular incident physics
```

We will keep the Unitree task observation/action ordering as the canonical
policy interface. Everest-specific incident environments should wrap that
interface rather than fork the G1 policy contract. That lets policies trained
or exported through `unitree_rl_mjlab` remain deployable through its existing
ONNX/sim2real path while Newton supplies additional environment forces during
incident reconstruction and adaptation.
