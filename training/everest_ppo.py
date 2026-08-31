#!/usr/bin/env python3
"""Fine-tune a G1 velocity actor on the live Everest Newton/MuJoCo stack.

This is intentionally a small, single-environment PPO implementation.  It is
not a visual replay: every collected transition advances the same Everest DEM,
MuJoCo robot, and optional Newton snow patch used by the Unity backend.  The
result is exported as a 99-observation/29-action ONNX candidate that the
existing policy registry can validate before activation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import mujoco
import numpy as np
import onnx
import torch
from torch import nn
from torch.distributions import Normal

from dashboard.engines.mujoco import MuJoCoEngine
from dashboard.policy import G1VelocityPolicy
from simulation.unity_bridge import DEFAULT_SNOW


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CHECKPOINT = Path(
    os.environ.get(
        "EVEREST_ICE_INCLINE_CHECKPOINT",
        "/home/auverus/git/humanoid_climber_safety_ckpts/ckpt/exported/ice_incline.onnx",
    )
)


def _atomic_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class ObservationNormalizer(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        # Keep these names aligned with the G1VelocityPolicy ONNX loader.
        self.register_buffer("_mean", torch.from_numpy(mean.copy()).reshape(1, -1))
        self.register_buffer("_std", torch.from_numpy(std.copy()).reshape(1, -1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self._mean) / torch.clamp(self._std, min=1.0e-6)


class Actor(nn.Module):
    """Torch equivalent of the exported MjLab actor, initialized from ONNX."""

    def __init__(self, policy: G1VelocityPolicy) -> None:
        super().__init__()
        self.obs_normalizer = ObservationNormalizer(policy.mean, policy.std)
        layers: list[nn.Module] = []
        for index, (weight, bias) in enumerate(zip(policy.weights, policy.biases)):
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(torch.from_numpy(weight))
            linear.bias.data.copy_(torch.from_numpy(bias))
            layers.append(linear)
            if index < len(policy.weights) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((29,), -1.25))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(observation))

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self(observation)
        std = self.log_std.clamp(-4.0, 0.5).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1.0e-6)
        return action, raw, log_probability.sum(-1)

    def log_probability(self, observation: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1.0e-6)
        return log_probability.sum(-1), distribution.entropy().sum(-1)


class ValueFunction(nn.Module):
    def __init__(self, policy: G1VelocityPolicy) -> None:
        super().__init__()
        self.register_buffer("obs_mean", torch.from_numpy(policy.mean.copy()))
        self.register_buffer("obs_std", torch.from_numpy(policy.std.copy()))
        self.network = nn.Sequential(
            nn.Linear(policy.mean.size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.obs_mean) / torch.clamp(self.obs_std, min=1.0e-6)
        return self.network(normalized).squeeze(-1)


class EverestTrainingEnvironment:
    """Gym-shaped adapter over the authoritative Everest engine.

    MuJoCo owns the G1 and DEM contacts.  When enabled, the engine's Newton
    patch advances at its configured coupling rate and writes deformation back
    into the MuJoCo heightfield before the next rigid-body step.
    """

    def __init__(
        self,
        seed_checkpoint: Path,
        *,
        enable_newton: bool,
        friction: float,
        command_speed: float,
        episode_seconds: float,
    ) -> None:
        self.policy_adapter = G1VelocityPolicy(seed_checkpoint)
        if self.policy_adapter.mean.size != 99:
            raise ValueError(
                f"Everest adaptation expects the MjLab 1.6 99-observation actor, got {self.policy_adapter.mean.size}"
            )
        self.engine = MuJoCoEngine(
            telemetry_hz=50.0,
            checkpoint=seed_checkpoint,
            enable_newton=enable_newton,
        )
        snow = copy.deepcopy(DEFAULT_SNOW)
        snow["surface_friction"] = float(friction)
        self.engine.control("snow_parameters", snow)
        self.command = (float(command_speed), 0.0, 0.0)
        self.control_period = 1.0 / 50.0
        self.episode_seconds = float(episode_seconds)
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.episode_start_time = 0.0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.pelvis = mujoco.mj_name2id(
            self.engine.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

    @property
    def newton_active(self) -> bool:
        return self.engine._snow_patch is not None

    def close(self) -> None:
        self.engine.stop()

    def reset(self) -> np.ndarray:
        self.engine._reset_to_home()
        self.engine._stand_lock_enabled = False
        self.engine._surface_friction_override = float(
            self.engine._surface_friction_override or DEFAULT_SNOW["surface_friction"]
        )
        self.engine._apply_surface_friction()
        if self.engine._snow_patch is None and self.engine._newton_enabled:
            self.engine._rebuild_snow_patch()
        self.policy_adapter.last_action.fill(0.0)
        self.previous_action.fill(0.0)
        self.episode_start_time = float(self.engine.data.time)
        self.episode_return = 0.0
        self.episode_steps = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        return self.policy_adapter.observation(
            self.engine.data,
            self.engine.model,
            command=self.command,
        )

    def _terrain_normal(self) -> np.ndarray:
        x, y = (float(value) for value in self.engine.data.qpos[:2])
        sample = 0.08
        dz_dx = (
            self.engine._terrain_height(x + sample, y)
            - self.engine._terrain_height(x - sample, y)
        ) / (2.0 * sample)
        dz_dy = (
            self.engine._terrain_height(x, y + sample)
            - self.engine._terrain_height(x, y - sample)
        ) / (2.0 * sample)
        normal = np.asarray((-dz_dx, -dz_dy, 1.0), dtype=np.float64)
        return normal / np.linalg.norm(normal)

    def _fallen(self) -> tuple[bool, float, float]:
        data = self.engine.data
        rotation = np.asarray(data.xmat[self.pelvis]).reshape(3, 3)
        upright = float(np.dot(rotation[:, 2], self._terrain_normal()))
        terrain = self.engine._terrain_height(float(data.qpos[0]), float(data.qpos[1]))
        clearance = float(data.qpos[2] - terrain)
        invalid = not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        return bool(invalid or upright < 0.45 or clearance < 0.34), upright, clearance

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        data = self.engine.data
        model = self.engine.model
        target = self.policy_adapter.target_positions(action)
        data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        target_time = float(data.time) + self.control_period
        while data.time + model.opt.timestep * 0.5 < target_time:
            self.engine._maybe_recenter_snow_patch()
            self.engine._step_mpm_if_due()
            data.xfrc_applied[:] = 0.0
            if self.engine._wind_force_n > 0.0:
                direction = math.radians(self.engine._wind_direction_deg)
                data.xfrc_applied[self.pelvis, 0] = -self.engine._wind_force_n * math.sin(direction)
                data.xfrc_applied[self.pelvis, 1] = -self.engine._wind_force_n * math.cos(direction)
            mujoco.mj_step(model, data)

        rotation = np.asarray(data.xmat[self.pelvis]).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(data.qvel[:3], dtype=np.float64)
        fallen, upright, clearance = self._fallen()
        speed_error = (float(body_velocity[0]) - self.command[0]) / 0.35
        velocity_reward = math.exp(-(speed_error * speed_error))
        lateral_penalty = 0.12 * float(body_velocity[1] ** 2)
        action_rate_penalty = 0.018 * float(np.mean((action - self.previous_action) ** 2))
        effort_penalty = 2.0e-6 * float(
            np.mean(np.abs(np.asarray(data.qfrc_actuator[6:]) * np.asarray(data.qvel[6:])))
        )
        reward = (
            0.20
            + 1.35 * velocity_reward
            + 0.75 * max(0.0, upright)
            - lateral_penalty
            - action_rate_penalty
            - effort_penalty
            - (5.0 if fallen else 0.0)
        )
        self.previous_action[:] = action
        self.policy_adapter.last_action[:] = action
        self.episode_return += reward
        self.episode_steps += 1
        timed_out = float(data.time) - self.episode_start_time >= self.episode_seconds
        done = bool(fallen or timed_out)
        observation = self._observation()
        return observation, float(reward), done, {
            "fallen": fallen,
            "timeout": timed_out,
            "upright": upright,
            "clearance_m": clearance,
            "forward_speed_m_s": float(body_velocity[0]),
            "episode_return": self.episode_return,
            "episode_steps": self.episode_steps,
            "newton_active": self.newton_active,
            "sim_time": float(data.time),
        }


def _export_actor(actor: Actor, seed_checkpoint: Path, output: Path, iteration: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    export_model = actor.cpu().eval()
    torch.onnx.export(
        export_model,
        torch.zeros(1, 99),
        str(output),
        input_names=["observation"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load(str(output))
    source = onnx.load(str(seed_checkpoint))
    metadata = {item.key: item.value for item in source.metadata_props}
    metadata.update({
        "source_checkpoint": str(seed_checkpoint.resolve()),
        "policy_role": "everest_newton_mujoco_ice_incline",
        "training_environment": "everest-dem+newton-mpm+mujoco/v1",
        "training_iteration": str(iteration),
        "validation_state": "candidate_unvalidated",
    })
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = str(key)
        item.value = str(value)
    onnx.save(model, str(output))


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_checkpoint = args.checkpoint.expanduser().resolve()
    if not seed_checkpoint.is_file():
        raise FileNotFoundError(f"Low-friction incline checkpoint is missing: {seed_checkpoint}")
    device = torch.device(args.device)
    policy_adapter = G1VelocityPolicy(seed_checkpoint)
    actor = Actor(policy_adapter).to(device)
    value = ValueFunction(policy_adapter).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()), lr=args.learning_rate
    )
    environment = EverestTrainingEnvironment(
        seed_checkpoint,
        enable_newton=not args.disable_newton,
        friction=args.friction,
        command_speed=args.command_speed,
        episode_seconds=args.episode_seconds,
    )
    started_at = time.time()
    run_id = args.run_id or time.strftime("everest-ice-%Y%m%d-%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "schema": "everest-rl-training/v1",
        "run_id": run_id,
        "state": "starting",
        "seed_checkpoint": str(seed_checkpoint),
        "physics": "newton+mujoco" if not args.disable_newton else "mujoco",
        "terrain": str(ROOT / "maps/everest_terrain.json"),
        "friction": args.friction,
        "iteration": 0,
        "iterations": args.iterations,
        "started_at": started_at,
        "output_dir": str(output_dir.resolve()),
    }
    _atomic_json(args.status_file, status)
    try:
        observation = environment.reset()
        if not args.disable_newton and not environment.newton_active:
            raise RuntimeError("Newton was requested but the live MPM patch is not active")
        if args.dry_run:
            observation, reward, done, info = environment.step(np.zeros(29, dtype=np.float32))
            status.update({
                "state": "dry_run_complete",
                "observation_size": int(observation.size),
                "reward": reward,
                "done": done,
                "environment": info,
                "finished_at": time.time(),
            })
            _atomic_json(args.status_file, status)
            return status

        completed_returns: list[float] = []
        for iteration in range(1, args.iterations + 1):
            observations: list[np.ndarray] = []
            raw_actions: list[np.ndarray] = []
            old_log_probabilities: list[float] = []
            rewards: list[float] = []
            dones: list[float] = []
            values: list[float] = []
            last_info: dict[str, Any] = {}
            for _ in range(args.rollout_steps):
                obs_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, raw_tensor, log_probability = actor.sample(obs_tensor)
                    state_value = value(obs_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
                next_observation, reward, done, last_info = environment.step(action)
                observations.append(observation.copy())
                raw_actions.append(raw_tensor.squeeze(0).cpu().numpy())
                old_log_probabilities.append(float(log_probability.item()))
                rewards.append(reward)
                dones.append(float(done))
                values.append(float(state_value.item()))
                observation = next_observation
                if done:
                    completed_returns.append(float(last_info["episode_return"]))
                    observation = environment.reset()

            with torch.no_grad():
                next_value = float(
                    value(torch.as_tensor(observation, device=device).unsqueeze(0)).item()
                )
            advantages = np.zeros(len(rewards), dtype=np.float32)
            gae = 0.0
            for step in reversed(range(len(rewards))):
                continuation = 1.0 - dones[step]
                bootstrap = next_value if step == len(rewards) - 1 else values[step + 1]
                delta = rewards[step] + args.gamma * bootstrap * continuation - values[step]
                gae = delta + args.gamma * args.gae_lambda * continuation * gae
                advantages[step] = gae
            returns = advantages + np.asarray(values, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-6)

            observation_batch = torch.as_tensor(np.asarray(observations), device=device)
            raw_action_batch = torch.as_tensor(np.asarray(raw_actions), device=device)
            old_log_batch = torch.as_tensor(old_log_probabilities, device=device)
            advantage_batch = torch.as_tensor(advantages, device=device)
            return_batch = torch.as_tensor(returns, device=device)
            indices = np.arange(len(rewards))
            losses: dict[str, float] = {}
            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), args.minibatch_size):
                    selected = torch.as_tensor(
                        indices[start : start + args.minibatch_size], device=device
                    )
                    log_probability, entropy = actor.log_probability(
                        observation_batch[selected], raw_action_batch[selected]
                    )
                    ratio = torch.exp(log_probability - old_log_batch[selected])
                    unclipped = ratio * advantage_batch[selected]
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                    ) * advantage_batch[selected]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (
                        value(observation_batch[selected]) - return_batch[selected]
                    ).square().mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_coefficient * value_loss - args.entropy_coefficient * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(value.parameters()), args.max_grad_norm
                    )
                    optimizer.step()
                    losses = {
                        "policy": float(policy_loss.item()),
                        "value": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                    }

            candidate = output_dir / f"ice_incline_everest_{iteration:05d}.onnx"
            if iteration % args.export_every == 0 or iteration == args.iterations:
                _export_actor(actor, seed_checkpoint, candidate, iteration)
                actor.to(device).train()
            preview_file = output_dir / "preview.json"
            environment.engine._subset_preview_enabled = True
            preview = environment.engine.subset_preview()
            if preview is not None:
                preview.update({
                    "source": "everest_ppo_trainer",
                    "run_id": run_id,
                    "iteration": iteration,
                    "iterations": args.iterations,
                    "caption": (
                        f"Live PPO MuJoCo rollout · iteration {iteration}/{args.iterations} · "
                        f"Newton {'active' if environment.newton_active else 'off'}"
                    ),
                })
                _atomic_json(preview_file, preview)
            status.update({
                "state": "running" if iteration < args.iterations else "complete",
                "iteration": iteration,
                "environment_steps": iteration * args.rollout_steps,
                "mean_episode_return": (
                    float(np.mean(completed_returns[-20:])) if completed_returns else None
                ),
                "episodes_completed": len(completed_returns),
                "loss": losses,
                "latest_environment": last_info,
                "latest_candidate": str(candidate.resolve()) if candidate.is_file() else None,
                "preview_file": str(preview_file.resolve()) if preview_file.is_file() else None,
                "updated_at": time.time(),
            })
            _atomic_json(args.status_file, status)
            print(json.dumps(status, separators=(",", ":")), flush=True)
        status["finished_at"] = time.time()
        _atomic_json(args.status_file, status)
        return status
    except Exception as exc:
        status.update({
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": time.time(),
        })
        _atomic_json(args.status_file, status)
        raise
    finally:
        environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SEED_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/everest_rl")
    parser.add_argument("--status-file", type=Path, default=ROOT / "runs/everest_rl/status.json")
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.15)
    parser.add_argument("--command-speed", type=float, default=0.35)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--export-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-newton", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("iterations", "rollout_steps", "epochs", "minibatch_size", "export_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.friction <= 1.0:
        parser.error("--friction must be in (0, 1]")
    return args


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
