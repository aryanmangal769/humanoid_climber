#!/usr/bin/env python3
"""Export an RSL-RL MjLab actor checkpoint to an Everest-loadable ONNX file.

PyTorch checkpoint deserialization uses ``weights_only=True``. The exporter
copies robot/action metadata from the known-good bundled G1 ONNX artifact; the
actor and normalizer tensors always come from the supplied checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch


def mjlab_1_6_default_joint_positions(joint_names: tuple[str, ...]) -> list[float]:
    """Resolve the neutral G1 pose used by MjLab 1.6 velocity training.

    The bundled Everest ONNX is from the older Unitree RL MjLab deployment and
    has a different neutral crouch.  Copying that field onto a current MjLab
    actor makes otherwise valid checkpoints drive the hips, knees, and ankles
    around the wrong offsets and immediately collapse in MuJoCo.

    These values mirror ``get_g1_robot_cfg().init_state.joint_pos`` in MjLab
    1.6.0.  Keeping the resolver name-based also makes the exported vector
    follow the checkpoint's explicit actuator order.
    """
    result: list[float] = []
    for name in joint_names:
        value = 0.0
        if "hip_pitch_joint" in name:
            value = -0.312
        elif "knee_joint" in name:
            value = 0.669
        elif "ankle_pitch_joint" in name:
            value = -0.363
        elif "elbow_joint" in name:
            value = 0.6
        elif name == "left_shoulder_roll_joint":
            value = 0.2
        elif name == "right_shoulder_roll_joint":
            value = -0.2
        elif "shoulder_pitch_joint" in name:
            value = 0.2
        result.append(value)
    return result


class ObservationNormalizer(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_mean", mean.reshape(1, -1).float())
        self.register_buffer("_std", std.reshape(1, -1).float())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean) / self._std


class Actor(torch.nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        mean = state["obs_normalizer._mean"]
        std = state["obs_normalizer._std"]
        self.obs_normalizer = ObservationNormalizer(mean, std)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(state["mlp.0.weight"].shape[1], state["mlp.0.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(state["mlp.2.weight"].shape[1], state["mlp.2.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(state["mlp.4.weight"].shape[1], state["mlp.4.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(state["mlp.6.weight"].shape[1], state["mlp.6.weight"].shape[0]),
        )
        self.mlp.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")})

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(observation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--metadata-from", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload["actor_state_dict"]
    actor = Actor(state).eval()
    input_size = int(state["obs_normalizer._mean"].numel())
    action_size = int(state["mlp.6.bias"].numel())
    if action_size != 29:
        raise ValueError(f"Expected 29 G1 actions, got {action_size}")
    if input_size != 99:
        raise ValueError(
            f"Everest's velocity-policy exporter currently supports the MjLab 1.6 99-observation actor, got {input_size}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        torch.zeros(1, input_size, dtype=torch.float32),
        str(args.output),
        input_names=["observation"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load(str(args.output))
    source = onnx.load(str(args.metadata_from))
    metadata = {item.key: item.value for item in source.metadata_props}
    joint_names = tuple(name for name in metadata.get("joint_names", "").split(",") if name)
    if len(joint_names) != action_size:
        raise ValueError(
            f"Metadata source has {len(joint_names)} joint names for {action_size} actions"
        )
    metadata["default_joint_pos"] = ",".join(
        f"{value:.9g}" for value in mjlab_1_6_default_joint_positions(joint_names)
    )
    metadata.update({
        "source_checkpoint": str(args.checkpoint.resolve()),
        "policy_role": args.role,
        "observation_layout": "mjlab-1.6-velocity-99/v1",
        "robot_neutral_pose": "mjlab-1.6-unitree-g1/v1",
        "init_base_height_m": "0.76",
        "validation_state": "candidate_unvalidated",
    })
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = str(key)
        item.value = str(value)
    onnx.save(model, str(args.output))
    print(f"exported {args.output} ({input_size} -> {action_size}, role={args.role})")


if __name__ == "__main__":
    main()
