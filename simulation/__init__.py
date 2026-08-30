"""Everest simulation integration boundaries with optional Newton imports."""

from typing import Any

__all__ = ["NewtonMuJoCoBridge"]


def __getattr__(name: str) -> Any:
    if name == "NewtonMuJoCoBridge":
        from .newton_mujoco import NewtonMuJoCoBridge

        return NewtonMuJoCoBridge
    raise AttributeError(name)
