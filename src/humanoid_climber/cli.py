"""Command-line entry points for project-owned MjLab tasks."""

from importlib import import_module


def play() -> None:
  """Register Humanoid Climber tasks and delegate to MjLab playback."""
  import_module("mjlab.tasks")
  import_module("humanoid_climber.tasks")
  from mjlab.scripts.play import main

  main()


def train() -> None:
  """Register Humanoid Climber tasks and delegate to MjLab training."""
  import_module("mjlab.tasks")
  import_module("humanoid_climber.tasks")
  from mjlab.scripts.train import main

  main()
