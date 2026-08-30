# Hum Climber — Winning Ideas

## Objective

Build the most complete and measurable single-robot mountain-survival system rather than copying competitors' features.

## 1. Prove the current result

Evaluate `model_34400.pt` against the original policy on an identical benchmark:

- At least 100 episodes per policy
- Fixed gradient `0.2`
- Fixed friction `0.1`
- Identical commands and random seeds
- Measure:
  - Distance climbed
  - Maximum height reached
  - Fall rate
  - Survival time
  - Velocity-tracking error
  - Energy consumption
  - Mean values and 95% confidence intervals

Produce a synchronized before-and-after video comparing the original and fine-tuned policies. This is stronger evidence than training curves alone.

## 2. Adaptive mountain curriculum

Expand the current slope-and-friction curriculum progressively with:

1. Random lateral and backward disturbances
2. Crosswind and gusts
3. Uneven friction patches
4. Rough or stepped slopes
5. Payload-mass variation
6. Sensor and action latency
7. Joint-strength variation

Promote the robot to harder conditions only after it consistently succeeds. Keep a deterministic evaluation environment completely separate from training randomization.

## 3. Fall prevention and recovery

Create a two-policy system:

- **Locomotion policy:** handles normal climbing.
- **Recovery policy:** activates after a fall or unrecoverable tilt.
- **Supervisor:** switches policies using torso height, orientation, and angular velocity.

This would provide a more complete system than traction without recovery or recovery infrastructure without a trained policy.

## 4. Climbing intelligence

Add higher-level terrain and gait adaptation:

- Detect unsafe ascent directions
- Traverse diagonally when direct ascent is unstable
- Reduce speed automatically as friction decreases
- Adapt gait using observed foot slip
- Add hand contacts later for true four-contact climbing

A learned or heuristic route-and-gait supervisor would make the project autonomous mountaineering rather than ordinary uphill walking.

## 5. Himalayan Gauntlet demonstration

Build one reproducible continuous challenge:

1. Flat ice start
2. Gradient increasing to `0.2`
3. Alternating low-friction patches
4. Wind-gust section
5. Small rough-terrain section
6. Forced disturbance
7. Recovery followed by continued ascent

Display a live overlay containing:

- Current slope
- Estimated friction or slip
- Height gained
- Wind force
- Active policy mode
- Energy consumed

## Recommended implementation order

1. Evaluate `model_34400.pt`
2. Build the automated 100-episode benchmark
3. Add wind and friction patches
4. Resume privacy-preserving fine-tuning from `model_34400.pt`
5. Add fall detection and recovery
6. Produce the Gauntlet video and quantitative comparison

## Target claim

> A Unitree G1 that adapts to ice, climbs a progressively steeper mountain, survives disturbances, recovers after falling, and continues its mission—with statistically reproducible results.
