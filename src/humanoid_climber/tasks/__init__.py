"""Register project-owned MjLab tasks."""

from humanoid_climber.tasks.g1_all_conditions import (
	CONTROLLED_TASK_ID as G1_CONTROLLED_TASK_ID,
	RANDOMIZED_TASK_ID as G1_RANDOMIZED_TASK_ID,
)
from humanoid_climber.tasks.g1_flat_wind import TASK_ID as G1_FLAT_WIND_TASK_ID
from humanoid_climber.tasks.g1_ice import TASK_ID as G1_ICE_TASK_ID
from humanoid_climber.tasks.g1_recovery import TASK_ID as G1_RECOVERY_TASK_ID

__all__ = [
	"G1_CONTROLLED_TASK_ID",
	"G1_FLAT_WIND_TASK_ID",
	"G1_ICE_TASK_ID",
	"G1_RANDOMIZED_TASK_ID",
	"G1_RECOVERY_TASK_ID",
]
