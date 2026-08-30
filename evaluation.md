# Policy Evaluation

## Fixed ice-slope benchmark

The recovered `model_34400.pt` policy and the stock G1 policy were evaluated on August 29, 2026 with identical conditions:

- 16 matched parallel episodes
- Seed `42`
- Friction `0.1`
- Slope gradient `0.2`
- Constant forward command `0.5 m/s`
- 500 control steps (`10 s`) per episode

| Metric | Stock policy | `model_34400.pt` | Result |
|---|---:|---:|---|
| Fall rate | 93.75% | 100% | Worse by one episode |
| Mean survival | 4.31 s | 5.63 s | +1.33 s |
| Mean maximum forward progress | 0.65 m | 0.57 m | -0.07 m |
| Mean maximum height gain | 0.03 m | 0.07 m | +0.04 m |
| Mean terminal height change | -0.47 m | -0.10 m | +0.37 m |
| Mean episode reward | -0.98 | 16.20 | +17.18 |

Paired 95% confidence intervals for the fine-tuned-minus-stock differences:

- Survival: `+0.22` to `+2.44 s`
- Maximum forward progress: `-0.22` to `+0.08 m`
- Maximum height gain: `-0.01` to `+0.09 m`
- Terminal height change: `+0.32` to `+0.42 m`
- Episode reward: `+10.47` to `+23.88`

## Conclusion

Fine-tuning produced a real improvement in balance, survival, and task reward. It did not produce reliable climbing at the target condition: all 16 fine-tuned episodes fell before 10 seconds, and forward progress was not significantly better than the stock policy. The checkpoint is a stronger intermediate policy, not a successful final climber.

## Evaluation at friction 0.15

Both policies were evaluated over 16 matched episodes with all conditions unchanged except friction increased from `0.1` to `0.15`.

| Metric | Stock policy | `model_34400.pt` | Fine-tuned change |
|---|---:|---:|---:|
| Fall rate | 75% | 68.75% | -6.25 points |
| Completed 10 seconds | 25% | 31.25% | +6.25 points |
| Mean survival | 5.28 s | 7.70 s | +2.42 s |
| Mean maximum forward progress | 0.56 m | 0.75 m | +0.18 m |
| Mean maximum height gain | 0.03 m | 0.14 m | +0.11 m |
| Mean terminal height change | -0.43 m | +0.01 m | +0.43 m |
| Mean episode reward | -4.42 | 23.22 | +27.64 |

Paired 95% confidence intervals support improvements in survival (`+0.81` to `+4.03 s`), maximum height gain (`+0.07` to `+0.15 m`), terminal height (`+0.31` to `+0.55 m`), and reward (`+18.32` to `+36.96`). The forward-progress difference (`-0.13` to `+0.50 m`) remains inconclusive.

The fine-tuned policy is substantially more stable than the stock policy at friction `0.15`, with five rather than four episodes surviving the full benchmark. It still falls in most episodes and therefore remains unreliable at this slope and command speed.

Raw per-episode results are stored locally under `logs/evaluation/`. The reusable benchmark is `scripts/evaluate_policy.py`.