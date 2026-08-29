# Humanoid Climber

Repository-owned MjLab tasks for adapting a Unitree G1 walking policy to
low-friction terrain. MjLab remains an external Python dependency; its source is
not copied into this repository.

## Setup

```bash
uv venv --python 3.12 .venv-rl
UV_PROJECT_ENVIRONMENT=.venv-rl uv sync
```

Downloaded checkpoints belong in `ckpt/` and are ignored by Git.

## Evaluate the stock walker on ice

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run hum-climber-play \
  HumClimber-Velocity-Ice-Unitree-G1 \
  --checkpoint-file ./ckpt/g1_velocity_model_final.pt \
  --num-envs 1 \
  --viewer viser
```

The task uses the stock G1 velocity observation and action interfaces, allowing
the unchanged flat-ground checkpoint to load. Training friction is randomized
from 0.1 to 1.0; evaluation friction is fixed at 0.2.

## Baseline result

On August 29, 2026, the stock policy could walk sideways at low speed and made
visible balance corrections at friction 0.2. It fell when given a fast forward
command. This establishes the baseline before ice-specific finetuning.

## Tests

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run pytest -q
```