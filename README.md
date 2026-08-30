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

## Evaluate a checkpoint headlessly

The deterministic evaluator runs matched parallel episodes without opening a
viewer and writes per-episode JSON under the ignored `logs/` directory:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run python scripts/evaluate_policy.py \
  ckpt/recovered/model_34400.pt \
  --episodes 16 \
  --steps 500 \
  --forward-speed 0.5 \
  --friction 0.15 \
  --seed 42 \
  --output logs/evaluation/model_34400_friction_0.15.json
```

## Results

The stock policy and recovered `model_34400.pt` checkpoint were tested with
matched seeds on the fixed `0.2`-gradient slope. Each condition used 16
episodes, a constant `0.5 m/s` forward command, and a 10-second limit.

| Friction | Policy | Fall rate | Mean survival | Max. height gain | Mean reward |
|---:|---|---:|---:|---:|---:|
| 0.10 | Stock | 93.75% | 4.31 s | 0.03 m | -0.98 |
| 0.10 | Fine-tuned | 100% | 5.63 s | 0.07 m | 16.20 |
| 0.15 | Stock | 75% | 5.28 s | 0.03 m | -4.42 |
| 0.15 | Fine-tuned | **68.75%** | **7.70 s** | **0.14 m** | **23.22** |

At friction `0.15`, fine-tuning increased mean survival by `2.42 s`; the paired
95% confidence interval was `+0.81` to `+4.03 s`. Balance, height, and reward
also improved, but the policy still fell in most episodes. It is a stronger
intermediate policy, not yet a reliable climber. See [evaluation.md](evaluation.md)
for the complete metrics and confidence intervals.

## Development log

### August 29, 2026 — environment and baseline

- Created an isolated Python 3.12 environment with `uv`, MjLab 1.6.0, MuJoCo,
  Torch, and Warp.
- Loaded and played the stock Unitree G1 velocity checkpoint without changing
  its 99-observation/29-action policy interface.
- Implemented the project-owned ice task with friction randomized from `0.1`
  to `1.0` during training and fixed at `0.1` during evaluation.
- Added a deterministic `16 × 16 m`, `0.2`-gradient evaluation slope and a
  training curriculum from flat ground to gradient `0.2`.
- Added configuration tests for interface compatibility, friction, commands,
  slope generation, curriculum behavior, and PPO configuration.

### August 29, 2026 — training and stabilization

- Verified checkpoint resume with one local CPU PPO iteration.
- Completed a 10-iteration A10G cloud smoke test.
- Diagnosed a simulation NaN in the initial 4,096-environment run and removed
  out-of-scope random robot pushes from the low-friction task.
- Reduced the stable cloud configuration to 1,024 environments and completed a
  200-iteration stability run.
- Continued fine-tuning from iteration 30,000 through iteration 34,400 on an
  A10G before stopping the Job for organization-storage privacy cleanup.
- Recovered `model_34400.pt` locally and backed it up to a private personal
  Hugging Face model repository.

### August 29, 2026 — evaluation and privacy

- Built `scripts/evaluate_policy.py` for reproducible headless evaluation with
  fixed commands, seeds, friction, and slope conditions.
- Compared the stock and fine-tuned policies at friction `0.1` and `0.15`.
- Confirmed statistically supported improvements in survival, balance, height,
  and reward at friction `0.15`, while documenting the remaining fall rate.
- Removed Hum Climber source snapshots and training outputs from shared
  organization storage after securing the checkpoint.
- Added a privacy-first Hugging Face Jobs workflow that rejects plaintext
  secrets, local-directory uploads to organization buckets, and unverified
  checkpoint destinations.

## Current status

- Latest recovered checkpoint: `model_34400.pt` (kept outside Git).
- Best measured condition: friction `0.15`, gradient `0.2`, with 31.25% of
  episodes completing the 10-second benchmark.
- Next work: improve reliable forward ascent, add disturbances and friction
  patches, and train fall recovery. See [ideas.md](ideas.md).

## Tests

```bash
UV_PROJECT_ENVIRONMENT=.venv-rl uv run pytest -q
```