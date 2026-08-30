"""Controlled and fully randomized Unitree G1 velocity environments."""

from dataclasses import replace
import math
from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg

from humanoid_climber.mountain_ridges import (
  MOUNTAIN_RANGE_SPECS,
  add_mountain_enclosure,
)
from humanoid_climber import mdp as climber_mdp
from humanoid_climber.orchestrator import (
  HIGH_WIND_FORCE_RANGES,
  INCLINE_FRICTION_RANGE,
  INCLINE_GRADIENT_RANGE,
  POLICY_ANNOUNCEMENT_DELAY_SECONDS,
  STAGE_DURATION_SECONDS,
)
from humanoid_climber.safety import CENTERLINE_MAX_OFFSET_M
from humanoid_climber.trail import (
  TRAIL_HALF_WIDTH_M,
  TRAIL_MARKER_STRIDE,
  TRAIL_SEGMENT_OVERLAP_M,
  TRAIL_SEGMENTS,
)

CONTROLLED_TASK_ID = "HumClimber-Velocity-Controlled-Unitree-G1"
RANDOMIZED_TASK_ID = "HumClimber-Velocity-Randomized-Unitree-G1"

# The controlled task is the reproducible admission benchmark. Every range is
# collapsed to one value and the terrain seed is fixed.
CONTROLLED_SEED = 42
CONTROLLED_FRICTION = 0.6
CONTROLLED_FORWARD_SPEED = 1.0
CONTROLLED_WIND_FORCE_RANGES = {
  "x": (0.0, 0.0),
  "y": (0.0, 0.0),
  "z": (0.0, 0.0),
}

# These bounds are intentionally explicit. "Randomized" means every condition
# below has a non-zero sampling interval; it does not mean unbounded physics.
NORMAL_FRICTION_RANGE = (0.65, 1.0)
RANDOM_ICE_FRICTION_RANGE = (0.008, 0.035)
# Kept as the public aggregate random-friction range for callers which inspect
# the scenario bounds. Runtime phases use the two explicit ranges above.
RANDOM_FRICTION_RANGE = RANDOM_ICE_FRICTION_RANGE
RANDOM_WIND_FORCE_RANGES = {
  # Keep the worst-case vector magnitude below the router's 18 N envelope:
  # sqrt(4^2 + 16^2 + 2^2) ~= 16.6 N.
  "x": (-4.0, 4.0),
  "y": (-16.0, 16.0),
  "z": (-2.0, 2.0),
}
RANDOM_EVENT_DURATION_S = (3.0, 8.0)
RANDOM_EVENT_BREAK_DURATION_S = (2.0, 5.0)
RANDOM_CONDITION_INTERVAL_S = RANDOM_EVENT_DURATION_S
RANDOM_SLOPE_GRADIENT_RANGE = (-0.20, 0.20)
RANDOM_ACTIVE_SLOPE_MAGNITUDE_RANGE = (0.06, 0.20)
RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M = (0.008, 0.10)
RANDOM_EVENT_PATCH_AHEAD_RANGE_M = (1.8, 2.3)
ICE_PATCH_LENGTH_M = 2.2
SLOPE_PATCH_LENGTH_M = 3.2
SLOPE_PATCH_SEGMENT_COUNT = 4
EVENT_PATCH_HALF_WIDTH_M = TRAIL_HALF_WIDTH_M - 0.03
HAZARD_WIDTH_MULTIPLIER = 8.0
ICE_PATCH_HALF_WIDTH_M = EVENT_PATCH_HALF_WIDTH_M * HAZARD_WIDTH_MULTIPLIER
SLOPE_PATCH_HALF_WIDTH_M = EVENT_PATCH_HALF_WIDTH_M * HAZARD_WIDTH_MULTIPLIER
RANDOM_SLOPE_OUTER_FRACTION_RANGE = (0.45, 0.75)
RANDOM_SLOPE_INNER_FRACTION_RANGE = (0.85, 1.0)
RANDOM_ROUGH_FORWARD_RANGE_M = (-1.25, 1.25)
RANDOM_ROUGH_LATERAL_RANGE_M = (-13.12, 13.12)
RANDOM_ROCK_HEIGHT_RANGE_M = (0.035, 0.10)
TREADMILL_SLOPE_DELTA_PER_STEP = 0.0015
TREADMILL_ROUGH_CLEAR_DELTA_PER_STEP_M = 0.005
TREADMILL_FORWARD_SPEED = 1.0
# The visual/safety trail itself is procedural and recycled forever.  The flat
# collision substrate is one cheap box, so make it extremely long as a guard
# against the robot ever reaching a physical edge during an interactive run.
TREADMILL_SIZE = (20_000.0, 60.0)
TREADMILL_ROUGH_ROWS = 6
TREADMILL_ROUGH_COLS = 80
TREADMILL_ROUGH_MOUND_COUNT = TREADMILL_ROUGH_ROWS * TREADMILL_ROUGH_COLS
TREADMILL_ROCK_COUNT = 128
MOUNTAIN_RIDGE_GEOM_NAMES = tuple(
  name
  for index in range(len(MOUNTAIN_RANGE_SPECS))
  for name in (
    f"mountain_range_front_{index:02d}",
    f"mountain_range_rear_{index:02d}",
    f"mountain_side_left_{index:02d}",
    f"mountain_side_right_{index:02d}",
  )
)
MOUNTAIN_COUNT = len(MOUNTAIN_RIDGE_GEOM_NAMES)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNOW_SURFACE_TILE = PROJECT_ROOT / "assets" / "textures" / "snow_surface_tile.png"
# Neutral ground is cool mineral frost rather than fresh snow. A future snow
# event can therefore use the brighter reserved color and remain unambiguous.
WINTER_GROUND_BASE_COLOR = (0.48, 0.55, 0.60)
WINTER_GROUND_FLECK_COLOR = (0.68, 0.73, 0.76)
FUTURE_SNOW_EVENT_COLOR = (0.93, 0.97, 1.0, 1.0)
WINTER_GROUND_TEXTURE_NAME = "winter_alpine_ground_texture"
WINTER_GROUND_MATERIAL_NAME = "winter_alpine_ground_material"
HIDDEN_TRAIL_RGBA = (*WINTER_GROUND_BASE_COLOR, 0.0)
RANDOM_ROOT_POSE_RANGES = {
  "x": (-0.35, 0.35),
  "y": (-0.35, 0.35),
  "z": (0.0, 0.08),
  "roll": (-0.10, 0.10),
  "pitch": (-0.10, 0.10),
  "yaw": (-math.pi, math.pi),
}
RANDOM_ROOT_VELOCITY_RANGES = {
  "x": (-0.25, 0.25),
  "y": (-0.25, 0.25),
  "z": (-0.10, 0.10),
  "roll": (-0.15, 0.15),
  "pitch": (-0.15, 0.15),
  "yaw": (-0.20, 0.20),
}
# Dashboard playback should start from a settled, centered stance instead of
# sampling the training-time root/joint perturbations.  The stock G1 neutral
# reset leaves the foot collision capsules ~3.5 mm above the flat treadmill on
# this model. A 3.5 mm downward offset leaves only ~0.04-0.25 mm clearance in
# the measured reset pose: visually seated, but still not interpenetrating.
PLAYBACK_ROOT_POSE_RANGES = {"z": (-0.0035, -0.0035)}
PLAYBACK_ROOT_VELOCITY_RANGES: dict[str, tuple[float, float]] = {}
PLAYBACK_JOINT_POSITION_RANGE = (0.0, 0.0)
PLAYBACK_JOINT_VELOCITY_RANGE = (0.0, 0.0)
RANDOM_JOINT_POSITION_RANGE = (-0.08, 0.08)
RANDOM_JOINT_VELOCITY_RANGE = (-0.15, 0.15)
RANDOM_ENCODER_BIAS_RANGE = (-0.02, 0.02)
RANDOM_DENSITY_LOG_SCALE_RANGE = (-0.08, 0.08)
RANDOM_COM_OFFSET_RANGE = (-0.02, 0.02)
RANDOM_JOINT_DAMPING_SCALE_RANGE = (0.8, 1.2)
RANDOM_JOINT_FRICTION_SCALE_RANGE = (0.8, 1.2)
RANDOM_ARMATURE_SCALE_RANGE = (0.9, 1.1)
RANDOM_PD_GAIN_SCALE_RANGE = (0.85, 1.15)
RANDOM_ACTION_DELAY_STEPS = (0, 2)

_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def _treadmill_terrain() -> TerrainGeneratorCfg:
  """One long flat substrate carrying the recycled procedural trail window."""
  return TerrainGeneratorCfg(
    seed=CONTROLLED_SEED,
    curriculum=False,
    size=TREADMILL_SIZE,
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    difficulty_range=(0.0, 0.0),
    color_scheme="none",
    sub_terrains={"treadmill": BoxFlatTerrainCfg(proportion=1.0)},
  )


def _make_treadmill_dynamic(cfg: ManagerBasedRlEnvCfg) -> None:
  """Make the terrain movable and allocate the infinite trail's geom pool."""
  terrain_cfg = cfg.scene.terrain
  if terrain_cfg is None:
    raise ValueError("The treadmill terrain configuration is missing.")
  # Tile the branch-provided snow surface across the complete substrate without
  # visually revealing the steering route.
  terrain_cfg.textures = (
    spec_cfg.TextureCfg(
      name=WINTER_GROUND_TEXTURE_NAME,
      type="2d",
      file=str(SNOW_SURFACE_TILE),
    ),
  )
  terrain_cfg.materials = (
    spec_cfg.MaterialCfg(
      name=WINTER_GROUND_MATERIAL_NAME,
      texture=WINTER_GROUND_TEXTURE_NAME,
      texuniform=True,
      texrepeat=(24.0, 12.0),
      reflectance=0.08,
      geom_names_expr=(r"terrain_0$",),
    ),
  )
  previous_spec_fn = cfg.scene.spec_fn

  def configure_treadmill(spec: mujoco.MjSpec) -> None:
    if previous_spec_fn is not None:
      previous_spec_fn(spec)
    body = spec.body("terrain")
    if body is None:
      raise ValueError("The treadmill terrain body was not generated.")

    # MuJoCo and MuJoCo-Warp both update mocap descendants through normal
    # kinematics. This keeps contacts, ray sensors, and Viser on one transform.
    body.mocap = True

    # Keep the fixed route pool for recycling and controller bookkeeping, but
    # render it transparent so the mathematical steering route is never shown.
    for segment in TRAIL_SEGMENTS:
      half_yaw = 0.5 * segment.yaw
      body.add_geom(
        name=f"treadmill_trail_surface_{segment.index:03d}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          segment.length * 0.5 + TRAIL_SEGMENT_OVERLAP_M,
          TRAIL_HALF_WIDTH_M,
          0.003,
        ),
        pos=(segment.center[0], segment.center[1], 0.004),
        quat=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        rgba=HIDDEN_TRAIL_RGBA,
        contype=0,
        conaffinity=0,
        group=1,
      )

    # Markers remain addressable by the recycler but are intentionally hidden.
    marker_stride = TRAIL_MARKER_STRIDE
    for marker_index, segment in enumerate(TRAIL_SEGMENTS[::marker_stride]):
      half_yaw = 0.5 * segment.yaw
      body.add_geom(
        name=f"treadmill_pitch_marker_{marker_index}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, TRAIL_HALF_WIDTH_M - 0.08, 0.002),
        pos=(segment.center[0], segment.center[1], 0.010),
        quat=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        rgba=HIDDEN_TRAIL_RGBA,
        contype=0,
        conaffinity=0,
        group=1,
      )

    # Safety boundaries remain computationally available but are not rendered.
    for segment in TRAIL_SEGMENTS:
      half_yaw = 0.5 * segment.yaw
      quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
      for side, sign in (("left", 1.0), ("right", -1.0)):
        x = segment.center[0] + sign * CENTERLINE_MAX_OFFSET_M * segment.normal[0]
        y = segment.center[1] + sign * CENTERLINE_MAX_OFFSET_M * segment.normal[1]
        body.add_geom(
          name=f"treadmill_centerline_safety_{side}_{segment.index:03d}",
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(segment.length * 0.5 + TRAIL_SEGMENT_OVERLAP_M, 0.018, 0.006),
          pos=(x, y, 0.014),
          quat=quat,
          rgba=HIDDEN_TRAIL_RGBA,
          contype=0,
          conaffinity=0,
          group=1,
        )

    # Terrain events live on dedicated mocap bodies. Their meshes, sizes and
    # colors never change after compilation; runtime only updates body poses.
    # This matters for Viser: changing geom_pos/quat/size/rgba forces expensive
    # baked-mesh rebuilds and caused the short freeze on each random event.
    event_hidden_z = -0.80

    ice_body = spec.worldbody.add_body(name="treadmill_ice_patch_body")
    ice_body.mocap = True
    ice_body.pos = (0.0, 0.0, event_hidden_z)
    ice_body.add_geom(
      name="treadmill_ice_patch",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(ICE_PATCH_LENGTH_M * 0.5, ICE_PATCH_HALF_WIDTH_M, 0.004),
      pos=(0.0, 0.0, 0.0),
      rgba=(0.42, 0.82, 1.0, 0.82),
      friction=(RANDOM_ICE_FRICTION_RANGE[1], 0.0005, 0.00001),
      priority=2,
      contype=1,
      conaffinity=1,
      condim=3,
      group=1,
    )

    # Four fixed-size mocap boxes form one continuous piecewise hill. Runtime
    # only moves/rotates the bodies, so each event can have several different
    # local grades without changing baked Viser mesh fields.
    slope_half_segment = SLOPE_PATCH_LENGTH_M / (2.0 * SLOPE_PATCH_SEGMENT_COUNT)
    slope_colors = (
      (0.24, 0.80, 0.98, 0.92),
      (0.12, 0.66, 0.96, 0.94),
      (0.12, 0.56, 0.92, 0.94),
      (0.24, 0.72, 0.94, 0.92),
    )
    for piece in range(SLOPE_PATCH_SEGMENT_COUNT):
      slope_body = spec.worldbody.add_body(
        name=f"treadmill_slope_patch_{piece:02d}_body"
      )
      slope_body.mocap = True
      slope_body.pos = (0.0, 0.0, event_hidden_z)
      slope_body.add_geom(
        name=f"treadmill_slope_patch_{piece:02d}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(slope_half_segment, SLOPE_PATCH_HALF_WIDTH_M, 0.018),
        pos=(0.0, 0.0, 0.0),
        rgba=slope_colors[piece % len(slope_colors)],
        contype=1,
        conaffinity=1,
        condim=3,
        group=1,
      )

    # Keep one empty aggregate mocap body for backwards-compatible scene
    # bookkeeping, but give every rounded mound and angular rock its own mocap
    # body. That lets every rough-terrain primitive randomize XY/yaw/exposed
    # height without touching baked geom fields and rebuilding the Viser scene.
    rough_body = spec.worldbody.add_body(name="treadmill_rough_patch_body")
    rough_body.mocap = True
    rough_body.pos = (0.0, 0.0, event_hidden_z)
    for index in range(TREADMILL_ROUGH_MOUND_COUNT):
      radius_x = 0.34 + 0.025 * ((index * 3) % 4)
      radius_y = 0.22 + 0.018 * ((index * 5) % 4)
      radius_z = 0.105 + 0.012 * ((index * 7) % 4)
      mound_body = spec.worldbody.add_body(
        name=f"treadmill_rough_mound_{index:03d}_body"
      )
      mound_body.mocap = True
      mound_body.pos = (0.0, 0.0, event_hidden_z)
      mound_body.add_geom(
        name=f"treadmill_rough_mound_{index:03d}",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=(radius_x, radius_y, radius_z),
        pos=(0.0, 0.0, 0.0),
        rgba=(0.40 + 0.015 * (index % 5), 0.36, 0.30, 1.0),
        contype=1,
        conaffinity=1,
        condim=3,
        group=1,
      )

    max_rock_half_height = 0.5 * RANDOM_ROCK_HEIGHT_RANGE_M[1]
    for index in range(TREADMILL_ROCK_COUNT):
      half_x = 0.055 + 0.018 * ((index * 3) % 5)
      half_y = 0.040 + 0.014 * ((index * 7) % 5)
      rock_body = spec.worldbody.add_body(
        name=f"treadmill_rough_rock_{index:02d}_body"
      )
      rock_body.mocap = True
      rock_body.pos = (0.0, 0.0, event_hidden_z)
      rock_body.add_geom(
        name=f"treadmill_rough_rock_{index:02d}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_x, half_y, max_rock_half_height),
        pos=(0.0, 0.0, 0.0),
        rgba=(0.34, 0.33, 0.31, 1.0),
        contype=1,
        conaffinity=1,
        condim=3,
        group=1,
      )

    # Procedural mesh ridges from the snow/mountain asset branch enclose both
    # sides and both longitudinal ends without affecting physics or probes.
    add_mountain_enclosure(spec)

  cfg.scene.spec_fn = configure_treadmill


def _set_treadmill_contact_capacity(cfg: ManagerBasedRlEnvCfg) -> None:
  """Leave headroom for robot/terrain contacts during live slope changes."""
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)


def _set_controlled_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (20.0, 20.0)
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 0.0
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 1.0
  command.init_velocity_prob = 0.0
  command.heading_command = False
  command.ranges.lin_vel_x = (
    CONTROLLED_FORWARD_SPEED,
    CONTROLLED_FORWARD_SPEED,
  )
  # MjLab's Viser joystick requires a positive configured maximum. With every
  # environment marked forward-only below, the command sampler still forces
  # lateral and yaw commands to exactly zero.
  command.ranges.lin_vel_y = (-0.1, 0.1)
  command.ranges.ang_vel_z = (-0.1, 0.1)
  command.ranges.heading = None


def _set_treadmill_command(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep the robot moving forward while external conditions change."""
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (20.0, 20.0)
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 0.0
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 1.0
  command.init_velocity_prob = 0.0
  command.heading_command = False
  command.ranges.lin_vel_x = (
    TREADMILL_FORWARD_SPEED,
    TREADMILL_FORWARD_SPEED,
  )
  command.ranges.lin_vel_y = (-0.1, 0.1)
  command.ranges.ang_vel_z = (-0.1, 0.1)
  command.ranges.heading = None


def _set_action_latency(cfg: ManagerBasedRlEnvCfg) -> None:
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(
      replace(
        actuator,
        delay_min_lag=RANDOM_ACTION_DELAY_STEPS[0],
        delay_max_lag=RANDOM_ACTION_DELAY_STEPS[1],
        delay_hold_prob=0.75,
        delay_update_period=20,
        delay_per_env_phase=True,
      )
      for actuator in robot.articulation.actuators
    ),
  )


def _set_controlled_events(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events = {
    "reset_base": replace(
      cfg.events["reset_base"],
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": replace(
      cfg.events["reset_robot_joints"],
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": replace(
      cfg.events["foot_friction"],
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=_FOOT_GEOM_NAMES
        ),
        "operation": "abs",
        "ranges": (CONTROLLED_FRICTION, CONTROLLED_FRICTION),
        "shared_random": True,
      },
    ),
    "wind": EventTermCfg(
      func=climber_mdp.apply_wind_force,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_ranges": CONTROLLED_WIND_FORCE_RANGES,
      },
    ),
    "advance_infinite_trail": EventTermCfg(
      func=climber_mdp.advance_infinite_trail,
      mode="step",
      params={
        "terrain_cfg": SceneEntityCfg("terrain"),
        "robot_cfg": SceneEntityCfg("robot"),
        "centerline_max_offset_m": CENTERLINE_MAX_OFFSET_M,
      },
    ),
  }


def _set_showcase_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep only reset, trail, and deterministic showcase orchestration events."""
  cfg.events["reset_base"].params = {
    "pose_range": PLAYBACK_ROOT_POSE_RANGES,
    "velocity_range": PLAYBACK_ROOT_VELOCITY_RANGES,
  }
  cfg.events["reset_robot_joints"].params = {
    "position_range": PLAYBACK_JOINT_POSITION_RANGE,
    "velocity_range": PLAYBACK_JOINT_VELOCITY_RANGE,
    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
  }
  cfg.events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": cfg.events["reset_robot_joints"],
    "advance_infinite_trail": EventTermCfg(
    func=climber_mdp.advance_infinite_trail,
    mode="step",
    params={
      "terrain_cfg": SceneEntityCfg("terrain"),
      "robot_cfg": SceneEntityCfg("robot"),
      "centerline_max_offset_m": CENTERLINE_MAX_OFFSET_M,
    },
    ),
    "orchestrated_policy_sequence": EventTermCfg(
    func=climber_mdp.orchestrated_policy_sequence,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "friction_asset_cfg": SceneEntityCfg(
        "robot", geom_names=_FOOT_GEOM_NAMES
      ),
      "terrain_cfg": SceneEntityCfg("terrain"),
      "robot_cfg": SceneEntityCfg("robot"),
      "normal_friction_range": (0.8, 0.8),
      "ice_friction_range": INCLINE_FRICTION_RANGE,
      "wind_force_ranges": HIGH_WIND_FORCE_RANGES,
      "slope_gradient_range": INCLINE_GRADIENT_RANGE,
      "active_slope_magnitude_range": INCLINE_GRADIENT_RANGE,
      "slope_piece_count": SLOPE_PATCH_SEGMENT_COUNT,
      "slope_outer_fraction_range": RANDOM_SLOPE_OUTER_FRACTION_RANGE,
      "slope_inner_fraction_range": RANDOM_SLOPE_INNER_FRACTION_RANGE,
      "event_patch_ahead_range_m": (2.0, 2.0),
      "rough_surface_height_range_m": RANDOM_ROUGH_SURFACE_HEIGHT_RANGE_M,
      "rough_forward_range_m": RANDOM_ROUGH_FORWARD_RANGE_M,
      "rough_lateral_range_m": RANDOM_ROUGH_LATERAL_RANGE_M,
      "rock_height_range_m": RANDOM_ROCK_HEIGHT_RANGE_M,
      "rock_count": TREADMILL_ROCK_COUNT,
      "rough_rows": TREADMILL_ROUGH_ROWS,
      "rough_cols": TREADMILL_ROUGH_COLS,
      "event_duration_range_s": (STAGE_DURATION_SECONDS, STAGE_DURATION_SECONDS),
      "break_duration_range_s": (STAGE_DURATION_SECONDS, STAGE_DURATION_SECONDS),
      "max_delta_per_step": TREADMILL_SLOPE_DELTA_PER_STEP,
      "rough_clear_delta_per_step_m": TREADMILL_ROUGH_CLEAR_DELTA_PER_STEP_M,
      "stage_duration_s": STAGE_DURATION_SECONDS,
      "policy_announcement_delay_s": POLICY_ANNOUNCEMENT_DELAY_SECONDS,
    },
    ),
  }

def unitree_g1_controlled_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a deterministic reference environment with pinned conditions."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.seed = CONTROLLED_SEED
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = _treadmill_terrain()
  cfg.scene.terrain.max_init_terrain_level = 0
  _make_treadmill_dynamic(cfg)
  _set_treadmill_contact_capacity(cfg)
  cfg.events.pop("randomize_terrain", None)
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  _set_controlled_commands(cfg)
  _set_controlled_events(cfg)
  if play:
    # Playback never consumes reward or metric values. Keeping the training
    # managers active costs substantial wall time for a single interactive
    # environment, especially on contact-heavy reward terms.
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.terminations = {}
  return cfg


def unitree_g1_randomized_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build the deterministic four-specialist showcase environment."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = _treadmill_terrain()
  cfg.scene.terrain.max_init_terrain_level = 0
  _make_treadmill_dynamic(cfg)
  _set_treadmill_contact_capacity(cfg)
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum = {}
  _set_treadmill_command(cfg)
  _set_showcase_events(cfg)
  if play:
    cfg.rewards = {}
    cfg.metrics = {}
    # Viewer playback is intentionally continuous. MjLab's stock time-out and
    # fell-over terminations auto-reset an episode behind the viewer's back,
    # which makes the "Reset robot after safety" checkbox ineffective and can
    # carry a latched protective-sit command into the freshly spawned episode.
    # The viewer's own safety controller remains active and owns any optional
    # reset requested by the user.
    cfg.terminations = {}
  return cfg


def unitree_g1_controlled_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_controlled"
  cfg.max_iterations = 5_000
  return cfg


def unitree_g1_randomized_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_randomized"
  cfg.max_iterations = 10_000
  return cfg


register_mjlab_task(
  task_id=CONTROLLED_TASK_ID,
  env_cfg=unitree_g1_controlled_env_cfg(),
  play_env_cfg=unitree_g1_controlled_env_cfg(play=True),
  rl_cfg=unitree_g1_controlled_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=RANDOMIZED_TASK_ID,
  env_cfg=unitree_g1_randomized_env_cfg(),
  play_env_cfg=unitree_g1_randomized_env_cfg(play=True),
  rl_cfg=unitree_g1_randomized_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
