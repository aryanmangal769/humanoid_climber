#!/usr/bin/env python3
"""Headless, deterministic evaluation for Hum Climber checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--episodes", type=int, default=32)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--forward-speed", type=float, default=0.5)
  parser.add_argument("--friction", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--output", type=Path, default=Path("logs/evaluation/policy.json"))
  return parser.parse_args()


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
  mean = float(values.mean())
  if values.size < 2:
    return mean, mean
  margin = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
  return mean - margin, mean + margin


def summary(values: np.ndarray) -> dict[str, float | list[float]]:
  low, high = confidence_interval_95(values)
  return {
    "mean": float(values.mean()),
    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    "median": float(np.median(values)),
    "min": float(values.min()),
    "max": float(values.max()),
    "ci95": [low, high],
  }


def main() -> None:
  args = parse_args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if args.episodes < 1 or args.steps < 1:
    raise ValueError("episodes and steps must be positive")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  import mjlab.tasks  # noqa: F401
  import humanoid_climber.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  task_id = "HumClimber-Velocity-Ice-Unitree-G1"
  configure_torch_backends()
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = args.episodes
  env_cfg.events["foot_friction"].params["ranges"] = (
    args.friction,
    args.friction,
  )
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  command = env.command_manager.get_command("twist")
  fixed_command = torch.tensor(
    [args.forward_speed, 0.0, 0.0], device=env.device, dtype=command.dtype
  )
  command[:] = fixed_command
  observations = wrapped.get_observations()

  robot = env.scene["robot"]
  start_pos = robot.data.root_link_pos_w.clone()
  last_pos = start_pos.clone()
  max_height = start_pos[:, 2].clone()
  max_forward = torch.zeros(args.episodes, device=env.device)
  reward_sum = torch.zeros(args.episodes, device=env.device)
  survival_steps = torch.full(
    (args.episodes,), args.steps, dtype=torch.long, device=env.device
  )
  active = torch.ones(args.episodes, dtype=torch.bool, device=env.device)

  with torch.inference_mode():
    for step in range(args.steps):
      pre_pos = robot.data.root_link_pos_w.clone()
      actions = policy(observations)
      observations, rewards, dones, _ = wrapped.step(actions)
      post_pos = robot.data.root_link_pos_w.clone()

      reward_sum[active] += rewards[active]
      last_pos[active] = post_pos[active]
      max_height[active] = torch.maximum(max_height[active], post_pos[active, 2])
      max_forward[active] = torch.maximum(
        max_forward[active], post_pos[active, 0] - start_pos[active, 0]
      )

      newly_done = active & dones.bool()
      if newly_done.any():
        # Auto-reset has already replaced post_pos, so retain the final pre-reset
        # position from the preceding control step for displacement reporting.
        last_pos[newly_done] = pre_pos[newly_done]
        survival_steps[newly_done] = step + 1
        active[newly_done] = False
      if not active.any():
        break

      command[:] = fixed_command
      observations = wrapped.get_observations()

  elapsed = survival_steps.float() * env.step_dt
  displacement = last_pos - start_pos
  result = {
    "checkpoint": str(args.checkpoint.resolve()),
    "condition": {
      "friction": args.friction,
      "slope_gradient": 0.2,
      "forward_speed_command_mps": args.forward_speed,
      "episode_limit_steps": args.steps,
      "step_dt_seconds": env.step_dt,
      "seed": args.seed,
      "episodes": args.episodes,
    },
    "fall_rate": float((survival_steps < args.steps).float().mean().item()),
    "completed_rate": float((survival_steps == args.steps).float().mean().item()),
    "survival_seconds": summary(elapsed.cpu().numpy()),
    "forward_displacement_m": summary(displacement[:, 0].cpu().numpy()),
    "maximum_forward_progress_m": summary(max_forward.cpu().numpy()),
    "height_change_m": summary(displacement[:, 2].cpu().numpy()),
    "maximum_height_gain_m": summary((max_height - start_pos[:, 2]).cpu().numpy()),
    "episode_reward": summary(reward_sum.cpu().numpy()),
    "per_episode": [
      {
        "survival_steps": int(survival_steps[i].item()),
        "survival_seconds": float(elapsed[i].item()),
        "fell": bool(survival_steps[i].item() < args.steps),
        "forward_displacement_m": float(displacement[i, 0].item()),
        "height_change_m": float(displacement[i, 2].item()),
        "maximum_forward_progress_m": float(max_forward[i].item()),
        "maximum_height_gain_m": float((max_height[i] - start_pos[i, 2]).item()),
        "reward": float(reward_sum[i].item()),
      }
      for i in range(args.episodes)
    ],
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({key: value for key, value in result.items() if key != "per_episode"}, indent=2))
  print(f"Wrote {args.output}")
  wrapped.close()


if __name__ == "__main__":
  main()
