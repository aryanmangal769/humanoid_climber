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
from 0.1 to 1.0. Evaluation uses friction 0.1 on a fixed uphill slope with a 0.2
gradient (approximately 11.3 degrees). The 16-by-16-meter terrain provides four
times the surface area of the earlier test. Playback uses the stock randomized
velocity commands, including forward, lateral, and turning movement.

## Fine-tune on ice

Fine-tuning starts from the existing flat-walking checkpoint. Training uses a
terrain curriculum from flat ground through a 0.2 gradient while randomizing
foot friction from 0.1 to 1.0. The actor and critic interfaces remain identical
to the stock G1 task.

Place the pretrained checkpoint where MjLab's resume loader can find it:

```bash
mkdir -p logs/rsl_rl/g1_ice/pretrained
cp ckpt/g1_velocity_model_final.pt logs/rsl_rl/g1_ice/pretrained/model_0.pt
```

Then launch fine-tuning on a Linux NVIDIA GPU:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run hum-climber-train \
  HumClimber-Velocity-Ice-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run pretrained \
  --agent.load-checkpoint model_0.pt \
  --agent.max-iterations 5000 \
  --agent.run-name slope-0.2-friction-0.1 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Training outputs are written under `logs/rsl_rl/g1_ice/`. Checkpoint resume and
one PPO iteration were smoke-tested locally on August 29, 2026. Full training
should run on Hugging Face Jobs rather than macOS because MjLab training is
intended for Linux with NVIDIA acceleration.

The Hugging Face job entry point is `scripts/hf_train.sh`. Its `NUM_ENVS`,
`MAX_ITERATIONS`, `RUN_NAME`, and `HF_OUTPUT_REPO` environment variables allow
the same immutable project snapshot to run a short smoke test or the full job.

On the training Mac, `scripts/poll_hf_job.sh` is installed as the launchd agent
`com.humclimber.hf-job-poll`. It checks the active job every 20 minutes, writes
status updates to `logs/job-monitor/hf-job.log`, and shows a macOS notification
when the job completes or stops with an error.

## Baseline result

On August 29, 2026, the stock policy could walk sideways at low speed and made
visible balance corrections at friction 0.2. It fell when given a fast forward
command. This establishes the baseline before ice-specific finetuning.

## Tests

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run pytest -q
```