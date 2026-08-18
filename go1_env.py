"""Chrono Gymnasium environment for the default Unitree Go1 locomotion stack.

Chrono runs here in a Y-up world, so the imported Z-up URDF is rotated at the
root. Planar command tracking uses the robot body X/Z axes, and yaw is about Y.

Observation, 48 float32 values:
    body-frame linear velocity, body-frame angular velocity, projected gravity,
    [command_vx, command_vz, command_yaw_rate], 12 joint position errors,
    12 joint velocities, and the previous executed 12D action.

Action, 12 float32 values in [-100, 100]:
    joint-position target offsets scaled by 0.25 rad before the default
    actuator-net torque model applies URDF-limited motor torques.
"""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pychrono as chrono
import pychrono.irrlicht as irr
import pychrono.parsers as parsers
import torch
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).parent
_URDF = _ROOT / "models/go1/go1_chrono.urdf"
_MESH_OBJ_DIR = _ROOT / "models/go1/meshes_obj"
_MESH_OBJ_LOD50_DIR = _ROOT / "models/go1/meshes_obj_lod50"
_ACTUATOR_NET_PATH = _ROOT / "resources/actuator_nets/unitree_go1.pt"

_PHYSICS_TIME_STEP = 5e-3
_FORCE_ACTUATOR_UPDATE_TIME_STEP = _PHYSICS_TIME_STEP
_CONTROL_FREQUENCY = 50.0
_TIME_STEP = 1.0 / _CONTROL_FREQUENCY
_PHYSICS_SUBSTEPS = int(round(_TIME_STEP / _PHYSICS_TIME_STEP))
if not np.isclose(_PHYSICS_SUBSTEPS * _PHYSICS_TIME_STEP, _TIME_STEP):
    raise ValueError("Control timestep must be an integer multiple of the physics timestep.")
_GROUND_RESTITUTION = 0.0
_GROUND_ROLLING_FRICTION = 0.0001
_FLAT_GROUND_LENGTH = 100.0
_FLAT_GROUND_WIDTH = 100.0
_FLAT_GROUND_THICKNESS = 0.2
_FLAT_GROUND_GRID_SPACING = 1.0
_FLAT_GROUND_GRID_LINE_WIDTH = 0.015
_FLAT_GROUND_GRID_LINE_HEIGHT = 0.002
_FOOT_FRICTION = 2.0
_FOOT_RESTITUTION = 0.0
_FOOT_COLLISION_RADIUS = 0.02
_CONTACT_METHOD = "NSC"
_SOLVER_TYPE = "BARZILAIBORWEIN"
_SOLVER_MAX_ITERATIONS = 60
_MATERIAL_COMPOSITION_RULE = "min"

_SPAWN_HEIGHT = 0.34
_TERM_RELATIVE_HEIGHT = 0.22
_MIN_UPRIGHT_ALIGNMENT = 0.85
_TRACKING_SIGMA = 0.25
_TRACKING_MIN_NORMALIZE_SCALE = 0.2
_BASE_HEIGHT_TARGET = 0.34
_ZERO_COMMAND = np.zeros(3, dtype=np.float32)
DEFAULT_COMMAND_INTERVAL_TIME = 10.0
_DEFAULT_COMMAND_ZERO_PROBABILITY = 0.10
_DEFAULT_COMMAND_VX_RANGE = (-1.0, 1.0)
_DEFAULT_COMMAND_VZ_RANGE = (-0.6, 0.6)
_DEFAULT_COMMAND_YAW_RATE_RANGE = (-1.0, 1.0)
_DEFAULT_ACTION_CLIP = 100.0
_DEFAULT_FRICTION = 0.8
_DEFAULT_ROOT_POSITION_XZ_RANGE = (-0.5, 0.5)
_DEFAULT_ROOT_YAW_RANGE = (-math.pi, math.pi)
_DEFAULT_BASE_MASS_ADD_RANGE = (-1.0, 3.0)
_DEFAULT_TRUNK_CONTACT_TERMINATION_FORCE = 1.0
_DEFAULT_OBS_NOISE_SCALE = 1.0
_DEFAULT_OBS_NOISE_BASE_HEIGHT = 0.10
_DEFAULT_OBS_NOISE_BASE_LIN_VEL = 0.10
_DEFAULT_OBS_NOISE_BASE_ANG_VEL = 0.20
_DEFAULT_OBS_NOISE_PROJECTED_GRAVITY = 0.05
_DEFAULT_OBS_NOISE_JOINT_POS = 0.01
_DEFAULT_OBS_NOISE_JOINT_VEL = 1.50
_REWARD_TRACKING_LIN_VEL_WEIGHT = 1.5
_REWARD_TRACKING_ANG_VEL_WEIGHT = 0.75
_REWARD_TORQUES_WEIGHT = -0.0002
_REWARD_LIN_VEL_Y_WEIGHT = -2.0
_REWARD_ANG_VEL_XZ_WEIGHT = -0.05
_REWARD_FLAT_ORIENTATION_L2_WEIGHT = -2.5
_REWARD_DOF_ACC_WEIGHT = -2.5e-7
_REWARD_ACTION_RATE_WEIGHT = -0.01
_REWARD_FEET_AIR_TIME_WEIGHT = 0.25
_REWARD_TERMINATION_WEIGHT = 0.0
_FEET_AIR_TIME_CONTACT_FORCE = 1.0
_FEET_AIR_TIME_COMMAND_MIN_SPEED = 0.1
_FEET_AIR_TIME_MIN = 0.18
_FEET_AIR_TIME_MAX_BONUS = 0.25
_CONTACT_SWITCH_ON_LOAD = 22.0
_CONTACT_SWITCH_OFF_LOAD = 18.0
_MIN_FOOT_LOAD = 20.0
_CONTACT_DIAGNOSTIC_FORCE_LIMIT = 1.0
_RESET_FOOT_CLEARANCE = 0.0005

# Zero action keeps the default home control pose.
# Joint order is [FR, FL, RR, RL], each with [hip, thigh, calf].
_HOME_JOINT_ANGLES = np.array(
    [
        -0.05, 0.75, -1.30,  # FR
         0.05, 0.75, -1.30,  # FL
        -0.05, 0.85, -1.30,  # RR
         0.05, 0.85, -1.30,  # RL
    ],
    dtype=np.float32,
)
_ACTION_SCALE = 0.25
_HIP_ACTION_SCALE_MULTIPLIER = 1.0
_HIP_ACTION_INDICES = np.array([0, 3, 6, 9], dtype=np.int32)
_ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
_ACTUATOR_MODEL = "actuator_net"
_FORCE_ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
_PD_KP = 20.0
_PD_KD = 0.5

# Joint limits from go1_chrono.urdf, in _JOINT_NAMES order.
_JOINT_LOW = np.tile([-0.863, -0.686, -2.818], 4).astype(np.float32)
_JOINT_HIGH = np.tile([0.863, 4.501, -0.888], 4).astype(np.float32)
_JOINT_VELOCITY_LIMIT = np.tile([30.1, 30.1, 20.06], 4).astype(np.float32)
_JOINT_EFFORT_LIMIT = np.tile([23.7, 23.7, 35.55], 4).astype(np.float32)

# Joint order is shared by actions, observations, limits, and home targets.
# The axis/sign arrays convert Chrono motor-frame rotation vectors back to
# URDF joint angles. After the imported robot's spawn-frame transform, all
# actuated Go1 joints read cleanly on Chrono Z in this PyChrono build.
# The imported motor-frame Z rotation has the opposite sign from the policy
# joint coordinate for every actuated joint, so sign=-1 maps it back.
_JOINT_AXES = np.array(
    [2, 2, 2,   # FR
     2, 2, 2,   # FL
     2, 2, 2,   # RR
     2, 2, 2],  # RL
    dtype=np.int32,
)
_JOINT_AXIS_SIGN = -np.ones(12, dtype=np.float32)
_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
_JOINT_BODY_PAIR_NAMES = [
    ("trunk", "FR_hip"), ("FR_hip", "FR_thigh"), ("FR_thigh", "FR_calf"),
    ("trunk", "FL_hip"), ("FL_hip", "FL_thigh"), ("FL_thigh", "FL_calf"),
    ("trunk", "RR_hip"), ("RR_hip", "RR_thigh"), ("RR_thigh", "RR_calf"),
    ("trunk", "RL_hip"), ("RL_hip", "RL_thigh"), ("RL_thigh", "RL_calf"),
]

# Self-collision filtering is disabled; robot-ground contacts are kept for
# trunk, feet, thighs, and calves.
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_thigh", "FR_calf",
    "FL_thigh", "FL_calf",
    "RR_thigh", "RR_calf",
    "RL_thigh", "RL_calf",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
_FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
_LEG_PREFIXES = ("FR", "FL", "RR", "RL")
_VISUAL_MESH_FORMATS = ("urdf", "obj", "obj_lod50", "none")
_OBJ_VISUAL_MESHES = {
    "trunk": "trunk.obj",
    "FR_hip": "hip.obj",
    "FL_hip": "hip.obj",
    "RR_hip": "hip.obj",
    "RL_hip": "hip.obj",
    "FR_thigh": "thigh_mirror.obj",
    "RR_thigh": "thigh_mirror.obj",
    "FL_thigh": "thigh.obj",
    "RL_thigh": "thigh.obj",
    "FR_calf": "calf.obj",
    "FL_calf": "calf.obj",
    "RR_calf": "calf.obj",
    "RL_calf": "calf.obj",
}
_OBJ_VISUAL_ORIGIN_RPY = {
    "FR_hip": (math.pi, 0.0, 0.0),
    "RR_hip": (math.pi, math.pi, 0.0),
    "RL_hip": (0.0, math.pi, 0.0),
}
_VIEWER_HIDDEN_SENSOR_BODIES = (
    "imu_link",
    "camera_face",
    "camera_chin",
    "camera_left",
    "camera_right",
    "camera_rearDown",
    "ultraSound_left",
    "ultraSound_right",
    "ultraSound_face",
)
_COLLISION_FILTER_BODIES = (
    "trunk",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
    "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
)


def standing_env_metadata(actuator_model: str = _ACTUATOR_MODEL) -> dict:
    """Return the public metadata needed to reproduce the default stack."""
    if actuator_model not in _ACTUATOR_MODELS:
        raise ValueError(f"actuator_model must be one of {_ACTUATOR_MODELS}")
    reward_weights = {
        "tracking_lin_vel": _REWARD_TRACKING_LIN_VEL_WEIGHT,
        "tracking_ang_vel": _REWARD_TRACKING_ANG_VEL_WEIGHT,
        "lin_vel_z": _REWARD_LIN_VEL_Y_WEIGHT,
        "ang_vel_xy": _REWARD_ANG_VEL_XZ_WEIGHT,
        "torques": _REWARD_TORQUES_WEIGHT,
        "dof_acc": _REWARD_DOF_ACC_WEIGHT,
        "flat_orientation_l2": _REWARD_FLAT_ORIENTATION_L2_WEIGHT,
        "feet_air_time": _REWARD_FEET_AIR_TIME_WEIGHT,
        "action_rate": _REWARD_ACTION_RATE_WEIGHT,
        "termination": _REWARD_TERMINATION_WEIGHT,
    }
    return {
        "stack": "default",
        "time_step": _TIME_STEP,
        "control_time_step": _TIME_STEP,
        "control_frequency": _CONTROL_FREQUENCY,
        "physics_time_step": _PHYSICS_TIME_STEP,
        "physics_frequency": 1.0 / _PHYSICS_TIME_STEP,
        "physics_substeps": _PHYSICS_SUBSTEPS,
        "force_actuator_update_time_step": _FORCE_ACTUATOR_UPDATE_TIME_STEP,
        "force_actuator_update_frequency": 1.0 / _FORCE_ACTUATOR_UPDATE_TIME_STEP,
        "observation_dimension": 48,
        "observation_layout": {
            "base_linear_velocity": [0, 3],
            "base_angular_velocity": [3, 6],
            "projected_gravity": [6, 9],
            "command": [9, 12],
            "relative_joint_positions": [12, 24],
            "joint_velocities": [24, 36],
            "previous_executed_action": [36, 48],
        },
        "command_observation_indices": [9, 12],
        "previous_action_observation_indices": [36, 48],
        "command_dimension": 3,
        "command_mode": "default",
        "command_interval_time": DEFAULT_COMMAND_INTERVAL_TIME,
        "command_sampler": {
            "zero_probability": _DEFAULT_COMMAND_ZERO_PROBABILITY,
            "moving_probability": 1.0 - _DEFAULT_COMMAND_ZERO_PROBABILITY,
            "vx_range": list(_DEFAULT_COMMAND_VX_RANGE),
            "vz_range": list(_DEFAULT_COMMAND_VZ_RANGE),
            "yaw_rate_range": list(_DEFAULT_COMMAND_YAW_RATE_RANGE),
        },
        "randomization": {
            "ground_friction": _DEFAULT_FRICTION,
            "root_position_xz_range": list(_DEFAULT_ROOT_POSITION_XZ_RANGE),
            "root_yaw_range": list(_DEFAULT_ROOT_YAW_RANGE),
            "root_linear_velocity": [0.0, 0.0, 0.0],
            "root_angular_velocity": [0.0, 0.0, 0.0],
            "joint_reset_multiplier": [1.0, 1.0],
            "obs_noise_scale": _DEFAULT_OBS_NOISE_SCALE,
            "base_mass_add_range": list(_DEFAULT_BASE_MASS_ADD_RANGE),
            "base_com_offset": [0.0, 0.0, 0.0],
            "pushes": False,
        },
        "spawn_height": _SPAWN_HEIGHT,
        "home_joint_angles": _HOME_JOINT_ANGLES.tolist(),
        "action_scale": _ACTION_SCALE,
        "action_clip": _DEFAULT_ACTION_CLIP,
        "hip_action_scale_multiplier": _HIP_ACTION_SCALE_MULTIPLIER,
        "hip_action_indices": _HIP_ACTION_INDICES.astype(int).tolist(),
        "actuator_model": actuator_model,
        "actuator_models": list(_ACTUATOR_MODELS),
        "torque_backend": "force_motor",
        "actuator_net_path": str(_ACTUATOR_NET_PATH),
        "actuator_net_input": [
            "joint_pos_error_t",
            "joint_pos_error_t_minus_1",
            "joint_pos_error_t_minus_2",
            "joint_velocity_t",
            "joint_velocity_t_minus_1",
            "joint_velocity_t_minus_2",
        ],
        "actuator_net_history": "repeat_current",
        "actuator_net_update_frequency": 1.0 / _FORCE_ACTUATOR_UPDATE_TIME_STEP,
        "pd_kp": _PD_KP,
        "pd_kd": _PD_KD,
        "joint_effort_limits": _JOINT_EFFORT_LIMIT.tolist(),
        "joint_velocity_limits": _JOINT_VELOCITY_LIMIT.tolist(),
        "collision_bodies": list(_ROBOT_COLLISION_BODIES),
        "foot_collision_radius": _FOOT_COLLISION_RADIUS,
        "reward_weights": reward_weights,
        "reward_dt_scaled_coefficients": {
            name: _TIME_STEP * weight for name, weight in reward_weights.items()
        },
        "reward_notes": {
            "dt_scaled": True,
            "positive_rewards_only": False,
            "command_tracking_frame": "imported Go1 trunk body frame: local X forward, local Y lateral, local Z vertical/up; public command_vz names the lateral planar command for compatibility",
            "feet_air_time_contact_force_limit": _FEET_AIR_TIME_CONTACT_FORCE,
            "feet_air_time_command_min_speed": _FEET_AIR_TIME_COMMAND_MIN_SPEED,
            "tracking_lin_vel_formula": "exp(-((command_vx-body_vx)^2 + (command_vz-body_vz)^2) / 0.25)",
            "tracking_ang_vel_formula": "exp(-(command_yaw_rate - body_yaw_rate)^2 / 0.25)",
            "feet_air_time_formula": "moving planar command gate times sum(first_contact_i * (previous_air_time_i - 0.5))",
        },
        "minimum_foot_load": _MIN_FOOT_LOAD,
        "contact_switch_on_load": _CONTACT_SWITCH_ON_LOAD,
        "contact_switch_off_load": _CONTACT_SWITCH_OFF_LOAD,
        "solver": {
            "type": _SOLVER_TYPE,
            "max_iterations": _SOLVER_MAX_ITERATIONS,
        },
        "contact_materials": {
            "contact_method": _CONTACT_METHOD,
            "composition_rule": _MATERIAL_COMPOSITION_RULE,
            "effective_friction": "min(flat_ground.friction, feet.friction)",
            "static_sliding_note": "SetFriction sets Chrono's friction value used for both static/sliding contact in this setup.",
            "flat_ground": {
                "friction": _DEFAULT_FRICTION,
                "restitution": _GROUND_RESTITUTION,
                "rolling_friction": _GROUND_ROLLING_FRICTION,
            },
            "feet": {
                "friction": _FOOT_FRICTION,
                "restitution": _FOOT_RESTITUTION,
            },
        },
    }


def _set_visual_color(body, color: chrono.ChColor) -> None:
    """Apply one color to all visual shapes attached to a Chrono body."""
    visual_model = body.GetVisualModel()
    if visual_model is None:
        return

    for index in range(visual_model.GetNumShapes()):
        visual_model.GetShape(index).SetColor(color)


def _contact_material(mu: float, restitution: float = 0.0):
    """Create parser contact material data for imported URDF bodies."""
    material = chrono.ChContactMaterialData()
    material.mu = mu
    material.cr = restitution
    return material


def _rigid_contact_material(
    friction: float,
    restitution: float,
    gn: float | None = None,
    kn: float | None = None,
    contact_method: str = _CONTACT_METHOD,
):
    """Create Chrono rigid contact material for foot/ground contact."""
    if contact_method == _CONTACT_METHOD:
        material = chrono.ChContactMaterialNSC()
    elif contact_method == "SMC":
        material = chrono.ChContactMaterialSMC()
        material.SetGn(60.0 if gn is None else float(gn))
        material.SetKn(2e5 if kn is None else float(kn))
    else:
        raise ValueError(f"unsupported contact method: {contact_method}")
    material.SetFriction(friction)
    material.SetRestitution(restitution)
    return material


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class Go1Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: str = None,
        enable_motors: bool = True,
        spawn_x: float = 0.0,
        spawn_z: float = 0.0,
        spawn_height: float = _SPAWN_HEIGHT,
        ground_height_offset: float = 0.0,
        fixed_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
        command_seed: int | None = None,
        env_rank: int = 0,
        command_interval_time: float = DEFAULT_COMMAND_INTERVAL_TIME,
        actuator_model: str = _ACTUATOR_MODEL,
        pd_kp: float | None = None,
        pd_kd: float | None = None,
        observation_mode: str = "current",
        default_randomization: bool = True,
        observation_noise: bool = True,
        actuator_net_substep_recompute: bool = False,
        dynamic_spawn_height: bool = True,
        show_collision_shapes: bool = False,
        show_urdf_collision_visuals: bool = False,
        visual_mesh_format: str = "urdf",
    ):
        super().__init__()
        if len(fixed_command) != 3:
            raise ValueError("fixed_command must contain (vx, vz, yaw_rate)")
        if observation_mode != "current":
            raise ValueError("only the current 48D observation layout is supported")
        if actuator_model not in _ACTUATOR_MODELS:
            raise ValueError(f"actuator_model must be one of {_ACTUATOR_MODELS}")
        if visual_mesh_format not in _VISUAL_MESH_FORMATS:
            raise ValueError(f"visual_mesh_format must be one of {_VISUAL_MESH_FORMATS}")
        if pd_kp is not None and float(pd_kp) <= 0.0:
            raise ValueError("pd_kp must be > 0 when provided")
        if pd_kd is not None and float(pd_kd) < 0.0:
            raise ValueError("pd_kd must be >= 0 when provided")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.env_backend = getattr(self, "_env_backend", "flat")
        self.terrain_type = getattr(self, "_terrain_type", "flat")
        self.enable_motors = enable_motors
        self.friction_range = getattr(self, "_friction_range", (_DEFAULT_FRICTION, _DEFAULT_FRICTION))
        self.physics_time_step = float(getattr(self, "_physics_time_step", _PHYSICS_TIME_STEP))
        self.physics_substeps = int(getattr(self, "_physics_substeps", _PHYSICS_SUBSTEPS))
        self.synchronize_terrain = bool(getattr(self, "_synchronize_terrain", True))
        self.update_force_actuator_each_substep = bool(
            getattr(self, "_update_force_actuator_each_substep", True)
        )
        self.force_actuator_update_time_step = float(
            getattr(self, "_force_actuator_update_time_step", _FORCE_ACTUATOR_UPDATE_TIME_STEP)
        )
        self.force_actuator_update_interval = max(
            1,
            int(round(self.force_actuator_update_time_step / self.physics_time_step)),
        )
        actual_force_dt = self.force_actuator_update_interval * self.physics_time_step
        if not np.isclose(actual_force_dt, self.force_actuator_update_time_step, rtol=0.0, atol=1e-12):
            raise ValueError("Force actuator update timestep must be an integer multiple of physics timestep.")
        self.spawn_xz = np.array([float(spawn_x), float(spawn_z)], dtype=np.float32)
        self.spawn_height = float(spawn_height)
        self.ground_height_offset = float(ground_height_offset)
        self.ground_friction = None
        self.home_joint_angles = _HOME_JOINT_ANGLES.copy()
        self.fixed_command = np.asarray(fixed_command, dtype=np.float32)
        self.fixed_command_enabled = bool(np.linalg.norm(self.fixed_command.astype(np.float32)) > 1e-6)
        self.command_seed = int(command_seed if command_seed is not None else 1 + 1000 * int(env_rank))
        self.env_rank = int(env_rank)
        self.dynamic_spawn_height = bool(dynamic_spawn_height)
        self._command_resample_count = 0
        self._command_bucket = "default"
        self.command_interval_time = float(command_interval_time)
        self.actuator_model = str(actuator_model)
        self.pd_kp = float(_PD_KP if pd_kp is None else pd_kp)
        self.pd_kd = float(_PD_KD if pd_kd is None else pd_kd)
        self.reset_settle_physics_steps = 0
        self.observation_mode = observation_mode
        self.default_randomization = bool(default_randomization)
        self.observation_noise = bool(observation_noise)
        self.actuator_net_substep_recompute = bool(actuator_net_substep_recompute)
        self.show_collision_shapes = bool(show_collision_shapes)
        self.show_urdf_collision_visuals = bool(show_urdf_collision_visuals)
        self.visual_mesh_format = str(visual_mesh_format)
        self.contact_method = getattr(self, "_contact_method", _CONTACT_METHOD)
        self._actuator_net = None
        if self.actuator_model == "actuator_net":
            self._actuator_net = self._load_actuator_net()
        self.step_count = 0
        self.command = _ZERO_COMMAND.copy()
        self.command_target = _ZERO_COMMAND.copy()
        self._push_velocity_vector = np.zeros(3, dtype=np.float32)
        self._push_velocity_magnitude = 0.0
        self._push_direction_xz = np.zeros(2, dtype=np.float32)
        self._push_interval = 0.0
        self._push_next_time = float("inf")
        self._push_end_time = -float("inf")
        self._push_active = False
        self._push_count = 0
        self.command_diag = self._command_diagnostics()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32,
        )
        self.action_clip = _DEFAULT_ACTION_CLIP
        self.action_space = spaces.Box(
            low=-self.action_clip, high=self.action_clip, shape=(12,), dtype=np.float32
        )

        self._system = None
        self._terrain = None
        self._trunk = None
        self._trunk_nominal_mass = None
        self._base_com_ballast = None
        self._base_com_ballast_link = None
        self._motors = []
        self._motor_funcs = []
        self._drive_links = []
        self._drive_clutches = []
        self._drive_motors = []
        self._drive_shafts = []
        self._last_motor_targets = self.home_joint_angles.copy()
        self._last_motor_torques = np.zeros(12, dtype=np.float32)
        self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
        self._last_joint_pos = self.home_joint_angles.copy()
        self._last_joint_vel = np.zeros(12, dtype=np.float32)
        self._prev_joint_vel_for_reward = np.zeros(12, dtype=np.float32)
        self._actuator_net_pos_err_last = np.zeros(12, dtype=np.float32)
        self._actuator_net_pos_err_last_last = np.zeros(12, dtype=np.float32)
        self._actuator_net_vel_last = np.zeros(12, dtype=np.float32)
        self._actuator_net_vel_last_last = np.zeros(12, dtype=np.float32)
        self._joint_body_pairs = []
        self._vis = None
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._reset_base_xz = np.zeros(2, dtype=np.float32)
        self._reward_contact_active = np.zeros(len(_FOOT_BODY_NAMES), dtype=bool)
        self._foot_air_times = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        self._prev_air_time_contact = np.zeros(len(_FOOT_BODY_NAMES), dtype=bool)
        self._reset_noise_sample = self._zero_reset_noise_sample()
        self._reset_joint_targets = self.home_joint_angles.copy()
        self.step_count = 0

        self._build_sim()

    def _load_actuator_net(self):
        if not _ACTUATOR_NET_PATH.exists():
            raise FileNotFoundError(
                "actuator_net model not found. Expected default TorchScript file at "
                f"{_ACTUATOR_NET_PATH}"
            )
        model = torch.jit.load(str(_ACTUATOR_NET_PATH), map_location="cpu")
        model.eval()
        return model

    def set_global_step(self, global_step: int) -> None:
        self.command_diag = self._command_diagnostics()

    def set_fixed_command(self, vx: float, vz: float, yaw_rate: float) -> None:
        self.fixed_command = np.array([vx, vz, yaw_rate], dtype=np.float32)
        self.fixed_command_enabled = bool(np.linalg.norm(self.fixed_command) > 1e-6)
        self.command = self.fixed_command.copy()
        self.command_target = self.fixed_command.copy()
        self._command_bucket = "fixed" if self.fixed_command_enabled else "zero"
        self.command_diag = self._command_diagnostics()

    def _sample_default_command(self) -> None:
        if self.fixed_command_enabled:
            self.command_target = self.fixed_command.copy()
            self._command_bucket = "fixed"
            return
        if float(self.np_random.random()) < _DEFAULT_COMMAND_ZERO_PROBABILITY:
            self.command_target = _ZERO_COMMAND.copy()
            self._command_bucket = "zero"
            return
        self.command_target = np.array(
            [
                float(self.np_random.uniform(*_DEFAULT_COMMAND_VX_RANGE)),
                float(self.np_random.uniform(*_DEFAULT_COMMAND_VZ_RANGE)),
                float(self.np_random.uniform(*_DEFAULT_COMMAND_YAW_RATE_RANGE)),
            ],
            dtype=np.float32,
        )
        self._command_bucket = "moving"

    def _reset_command(self) -> None:
        self._command_resample_count = 0
        self._sample_default_command()
        self.command = self.command_target.copy()
        self.command_diag = self._command_diagnostics()

    def _advance_command_after_reward(self) -> None:
        interval_steps = int(round(self.command_interval_time / _TIME_STEP))
        if interval_steps > 0 and self.step_count % interval_steps == 0:
            self._sample_default_command()
            self._command_resample_count += 1
        self.command = self.command_target.copy()
        self.command_diag = self._command_diagnostics()

    def _command_diagnostics(self) -> dict:
        interval_steps = int(round(self.command_interval_time / _TIME_STEP))
        fixed = bool(self.fixed_command_enabled)
        return {
            "mode": "default",
            "phase": "default",
            "bucket": "fixed" if fixed else self._command_bucket,
            "cmd_vx": float(self.command[0]),
            "cmd_vz": float(self.command[1]),
            "cmd_yaw": float(self.command[2]),
            "target_vx": float(self.command_target[0]),
            "target_vz": float(self.command_target[1]),
            "target_yaw": float(self.command_target[2]),
            "range_vx_min": float(self.fixed_command[0]) if fixed else _DEFAULT_COMMAND_VX_RANGE[0],
            "range_vx_max": float(self.fixed_command[0]) if fixed else _DEFAULT_COMMAND_VX_RANGE[1],
            "range_vz_min": float(self.fixed_command[1]) if fixed else _DEFAULT_COMMAND_VZ_RANGE[0],
            "range_vz_max": float(self.fixed_command[1]) if fixed else _DEFAULT_COMMAND_VZ_RANGE[1],
            "range_yaw_min": float(self.fixed_command[2]) if fixed else _DEFAULT_COMMAND_YAW_RATE_RANGE[0],
            "range_yaw_max": float(self.fixed_command[2]) if fixed else _DEFAULT_COMMAND_YAW_RATE_RANGE[1],
            "friction_min": _DEFAULT_FRICTION,
            "friction_max": _DEFAULT_FRICTION,
            "obs_noise_scale": _DEFAULT_OBS_NOISE_SCALE,
            "push_enabled": False,
            "episode_step": int(self.step_count),
            "command_interval_steps": interval_steps,
            "command_resample_count": int(self._command_resample_count),
        }

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _build_sim(self):
        """Tear down and rebuild the Chrono sim from scratch."""
        self.ground_friction = self._sample_ground_friction()

        if self.enable_motors:
            assembly_system, _, assembly_parser = self._build_imported_system(
                parsers.ChParserURDF.ActuationType_POSITION,
                include_terrain=getattr(self, "_include_terrain_in_assembly", True),
            )
            assembly_motors = [assembly_parser.GetChMotor(name) for name in _JOINT_NAMES]
            reset_targets = np.clip(self._reset_joint_targets, _JOINT_LOW, _JOINT_HIGH)
            for motor, target in zip(assembly_motors, reset_targets):
                function = chrono.ChFunctionConst(float(target))
                motor.SetMotorFunction(function)
            assembly_trunk = assembly_parser.GetChBody("trunk")
            assembly_feet = [assembly_parser.GetChBody(name) for name in _FOOT_BODY_NAMES]
            assembly_trunk.SetFixed(True)
            assembly_system.DoAssembly(1)
            assembly_trunk.SetFixed(False)
            if self.dynamic_spawn_height:
                self._apply_dynamic_spawn_height(system=assembly_system, feet=assembly_feet)
            else:
                before = self._min_foot_clearance(assembly_feet)
                self._reset_noise_sample["dynamic_spawn_height_enabled"] = False
                self._reset_noise_sample["dynamic_spawn_height_target_clearance"] = _RESET_FOOT_CLEARANCE
                self._reset_noise_sample["dynamic_spawn_height_shift"] = 0.0
                self._reset_noise_sample["dynamic_spawn_height_min_clearance_before_shift"] = before
                self._reset_noise_sample["dynamic_spawn_height_min_clearance_after_shift"] = before
                self._reset_noise_sample["dynamic_spawn_height_effective_root_y"] = float(assembly_trunk.GetPos().y)
            self._apply_reset_contact_safety_lift(system=assembly_system, feet=assembly_feet)

            if self.actuator_model in _FORCE_ACTUATOR_MODELS:
                runtime_actuation = parsers.ChParserURDF.ActuationType_FORCE
            else:
                runtime_actuation = None
            system, terrain, parser = self._build_imported_system(runtime_actuation)
            self._copy_body_states(assembly_system, system)
            self._refresh_runtime_state_after_copy(
                system,
                run_assembly=True,
            )
        else:
            system, terrain, parser = self._build_imported_system(None)

        self._cache_robot_handles(system, terrain, parser)
        self._apply_default_base_mass_randomization()
        self._last_motor_targets = self._reset_joint_targets.copy()
        self._last_motor_torques = np.zeros(12, dtype=np.float32)
        self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
        self._record_post_refresh_reset_diagnostics()
        self._apply_reset_velocity_noise()
        self._sync_joint_state_cache(use_actual_velocity=True)
        self._reset_actuator_net_history(self._reset_joint_targets)

        if self.render_mode == "human":
            if self.show_collision_shapes:
                self._add_collision_visual_overlays(parser)
            self._create_visualizer(system)

    def _new_system(self):
        system = chrono.ChSystemNSC()
        system.SetGravityY()
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        solver_types = {
            "MINRES": chrono.ChSolver.Type_MINRES,
            "BARZILAIBORWEIN": chrono.ChSolver.Type_BARZILAIBORWEIN,
            "APGD": chrono.ChSolver.Type_APGD,
            "PSOR": chrono.ChSolver.Type_PSOR,
            "PSSOR": chrono.ChSolver.Type_PSSOR,
            "PJACOBI": chrono.ChSolver.Type_PJACOBI,
            "BICGSTAB": chrono.ChSolver.Type_BICGSTAB,
            "GMRES": chrono.ChSolver.Type_GMRES,
        }
        system.SetSolverType(solver_types.get(_SOLVER_TYPE, chrono.ChSolver.Type_MINRES))
        system.GetSolver().AsIterative().SetMaxIterations(_SOLVER_MAX_ITERATIONS)
        return system

    def _build_imported_system(self, actuation_type, include_terrain: bool = True):
        system = self._new_system()
        terrain = None
        if include_terrain:
            self._add_flat_ground(system)

        parser = self._create_robot_parser(actuation_type)
        parser.PopulateSystem(system)
        self._configure_imported_bodies(system, parser)
        if self.visual_mesh_format == "none":
            self._clear_robot_visuals(parser)
        elif self.visual_mesh_format == "urdf":
            self._hide_viewer_sensor_visuals(parser)
        elif self.visual_mesh_format in ("obj", "obj_lod50"):
            self._replace_body_visuals_with_obj(parser)
            self._hide_viewer_sensor_visuals(parser)
        return system, terrain, parser

    def _sync_terrain_before_step(self) -> None:
        if self._terrain is None or not self.synchronize_terrain:
            return
        sync = getattr(self._terrain, "Synchronize", None)
        if sync is not None:
            sync(self._system.GetChTime())

    def _advance_terrain(self, dt: float) -> None:
        if self._terrain is None:
            return
        advance = getattr(self._terrain, "Advance", None)
        if advance is not None:
            advance(float(dt))

    def _advance_system_and_terrain(self, dt: float) -> None:
        self._system.DoStepDynamics(float(dt))
        self._advance_terrain(dt)

    def _copy_body_states(self, source_system, target_system) -> None:
        source_bodies = {body.GetName(): body for body in source_system.GetBodies()}
        for target in target_system.GetBodies():
            if target.IsFixed():
                continue
            source = source_bodies.get(target.GetName())
            if source is None:
                continue
            target.SetPos(source.GetPos())
            target.SetRot(source.GetRot())
            target.SetLinVel(source.GetPosDt())
            target.SetAngVelParent(source.GetAngVelParent())

    def _refresh_runtime_state_after_copy(self, system, run_assembly: bool = True) -> None:
        diag = self._reset_noise_sample
        diag["runtime_refresh_update_ran"] = False
        diag["runtime_refresh_assembly_ran"] = False
        diag["runtime_refresh_collision_ran"] = False
        diag["runtime_refresh_assembly_status"] = "not_run"
        try:
            result = system.Update()
            diag["runtime_refresh_update_ran"] = True
            diag["runtime_refresh_update_status"] = "ok" if result is None else str(result)
        except Exception as exc:
            diag["runtime_refresh_update_status"] = f"{type(exc).__name__}: {exc}"
        if run_assembly:
            try:
                result = system.DoAssembly(1)
                diag["runtime_refresh_assembly_ran"] = True
                diag["runtime_refresh_assembly_status"] = "ok" if result is None else str(result)
            except Exception as exc:
                diag["runtime_refresh_assembly_status"] = f"{type(exc).__name__}: {exc}"
        else:
            diag["runtime_refresh_assembly_status"] = "skipped_for_passive_force_runtime"
        try:
            result = system.ComputeCollisions()
            diag["runtime_refresh_collision_ran"] = True
            diag["runtime_refresh_collision_status"] = "ok" if result is None else str(result)
        except Exception as exc:
            diag["runtime_refresh_collision_status"] = f"{type(exc).__name__}: {exc}"

    def _zero_reset_noise_sample(self) -> dict:
        zeros12 = [0.0] * 12
        return {
            "enabled": False,
            "level": "clean",
            "components": "combined",
            "rn2_scale": 0.0,
            "base_position_offset_x": 0.0,
            "base_position_offset_z": 0.0,
            "base_position_offset": [0.0, 0.0],
            "base_height_offset": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "joint_position_offsets": zeros12.copy(),
            "joint_velocity_offsets": zeros12.copy(),
            "base_linear_velocity": [0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "base_mass_added": 0.0,
            "base_mass_add_range": [0.0, 0.0],
            "base_mass_nominal": 0.0,
            "base_mass_actual": 0.0,
            "base_mass_randomization_applied": False,
            "base_com_offset": [0.0, 0.0, 0.0],
            "base_com_offset_range": [0.0, 0.0],
            "base_com_randomization_applied": False,
            "base_com_implementation": "none",
            "base_com_ballast_mass": 0.0,
            "base_com_ballast_local_pos": [0.0, 0.0, 0.0],
            "base_com_effective_offset": [0.0, 0.0, 0.0],
            "base_com_status": "not_applied",
            "contact_safety_clearance": _RESET_FOOT_CLEARANCE,
            "contact_safety_min_clearance_before_lift": 0.0,
            "contact_safety_lift": 0.0,
            "contact_safety_min_clearance_after_lift": 0.0,
            "dynamic_spawn_height_enabled": True,
            "dynamic_spawn_height_target_clearance": _RESET_FOOT_CLEARANCE,
            "dynamic_spawn_height_shift": 0.0,
            "dynamic_spawn_height_min_clearance_before_shift": 0.0,
            "dynamic_spawn_height_min_clearance_after_shift": 0.0,
            "dynamic_spawn_height_effective_root_y": 0.0,
            "runtime_refresh_update_ran": False,
            "runtime_refresh_update_status": "not_run",
            "runtime_refresh_assembly_ran": False,
            "runtime_refresh_assembly_status": "not_run",
            "runtime_refresh_collision_ran": False,
            "runtime_refresh_collision_status": "not_run",
            "post_refresh_min_foot_clearance": 0.0,
            "post_refresh_foot_loads": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "post_settle_base_height": 0.0,
            "post_settle_foot_loads": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "post_settle_joint_vel_max_abs": 0.0,
            "initial_foot_y": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "initial_min_foot_y": 0.0,
            "initial_max_foot_y": 0.0,
            "initial_foot_loads": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "initial_max_foot_load": 0.0,
        }

    def _sample_reset_noise(self) -> None:
        if not self.default_randomization:
            self._reset_joint_targets = self.home_joint_angles.copy()
            self._reset_noise_sample = self._zero_reset_noise_sample()
            return

        sample = self._zero_reset_noise_sample()
        sample["enabled"] = True
        sample["level"] = "default"
        sample["components"] = "default"
        sample["rn2_scale"] = _DEFAULT_OBS_NOISE_SCALE
        sample["base_position_offset_x"] = float(self.np_random.uniform(*_DEFAULT_ROOT_POSITION_XZ_RANGE))
        sample["base_position_offset_z"] = float(self.np_random.uniform(*_DEFAULT_ROOT_POSITION_XZ_RANGE))
        sample["base_position_offset"] = [
            sample["base_position_offset_x"],
            sample["base_position_offset_z"],
        ]
        sample["yaw"] = float(self.np_random.uniform(*_DEFAULT_ROOT_YAW_RANGE))
        sample["joint_position_multipliers"] = [1.0] * 12
        sample["joint_position_offsets"] = [0.0] * 12
        sample["joint_velocity_offsets"] = [0.0] * 12
        sample["base_linear_velocity"] = [0.0, 0.0, 0.0]
        sample["base_angular_velocity"] = [0.0, 0.0, 0.0]
        sample["ground_friction"] = _DEFAULT_FRICTION
        sample["ground_friction_range"] = [_DEFAULT_FRICTION, _DEFAULT_FRICTION]
        sample["root_position_xz_range"] = list(_DEFAULT_ROOT_POSITION_XZ_RANGE)
        sample["root_yaw_range"] = list(_DEFAULT_ROOT_YAW_RANGE)
        sample["obs_noise_scale"] = _DEFAULT_OBS_NOISE_SCALE
        sample["base_mass_add_range"] = list(_DEFAULT_BASE_MASS_ADD_RANGE)
        sample["base_mass_added"] = float(self.np_random.uniform(*_DEFAULT_BASE_MASS_ADD_RANGE))
        sample["base_com_offset_range"] = [0.0, 0.0]
        sample["base_com_offset"] = [0.0, 0.0, 0.0]
        self._reset_joint_targets = self.home_joint_angles.copy()
        self._reset_noise_sample = sample

    def _sample_ground_friction(self) -> float:
        return _DEFAULT_FRICTION

    def _reset_push_schedule(self) -> None:
        self._push_velocity_vector = np.zeros(3, dtype=np.float32)
        self._push_velocity_magnitude = 0.0
        self._push_direction_xz = np.zeros(2, dtype=np.float32)
        self._push_interval = 0.0
        self._push_next_time = float("inf")
        self._push_end_time = -float("inf")
        self._push_active = False
        self._push_count = 0

    def _apply_scheduled_push(self) -> None:
        return

    def _push_diagnostics(self) -> dict[str, float]:
        return {
            "push_enabled": 0.0,
            "push_active": 0.0,
            "push_count": float(self._push_count),
            "push_velocity": 0.0,
            "push_velocity_sampled": 0.0,
            "push_velocity_x": 0.0,
            "push_velocity_z": 0.0,
            "push_direction_x": 0.0,
            "push_direction_z": 0.0,
            "push_next_time": float(self._push_next_time),
            "push_end_time": float(self._push_end_time),
            "push_interval": 0.0,
        }

    def _add_flat_ground(self, system) -> None:
        # Ground friction is fixed at 0.8; higher foot friction lets ground
        # friction determine effective contact friction.
        ground_mat = _rigid_contact_material(
            friction=self.ground_friction,
            restitution=_GROUND_RESTITUTION,
            contact_method=self.contact_method,
        )
        if hasattr(ground_mat, "SetRollingFriction"):
            ground_mat.SetRollingFriction(_GROUND_ROLLING_FRICTION)

        ground = chrono.ChBodyEasyBox(
            _FLAT_GROUND_LENGTH,
            _FLAT_GROUND_THICKNESS,
            _FLAT_GROUND_WIDTH,
            1000,
            True,
            True,
            ground_mat,
        )
        ground.SetName("ground")
        ground.SetFixed(True)
        ground.SetPos(chrono.ChVector3d(0, -0.5 * _FLAT_GROUND_THICKNESS + self.ground_height_offset, 0))
        _set_visual_color(ground, chrono.ChColor(0.34, 0.34, 0.34))
        system.AddBody(ground)
        self._add_flat_ground_grid(system)

    def _add_flat_ground_grid(self, system) -> None:
        y = self.ground_height_offset + 0.5 * _FLAT_GROUND_GRID_LINE_HEIGHT
        half_x = 0.5 * _FLAT_GROUND_LENGTH
        half_z = 0.5 * _FLAT_GROUND_WIDTH
        count_x = int(round(_FLAT_GROUND_LENGTH / _FLAT_GROUND_GRID_SPACING))
        count_z = int(round(_FLAT_GROUND_WIDTH / _FLAT_GROUND_GRID_SPACING))
        grid_color = chrono.ChColor(0.56, 0.56, 0.56)
        for i in range(count_z + 1):
            z = -half_z + i * _FLAT_GROUND_GRID_SPACING
            line = chrono.ChBodyEasyBox(
                _FLAT_GROUND_LENGTH,
                _FLAT_GROUND_GRID_LINE_HEIGHT,
                _FLAT_GROUND_GRID_LINE_WIDTH,
                1000,
                True,
                False,
            )
            line.SetName(f"ground_grid_x_{i:03d}")
            line.SetFixed(True)
            line.SetPos(chrono.ChVector3d(0, y, z))
            _set_visual_color(line, grid_color)
            system.AddBody(line)
        for i in range(count_x + 1):
            x = -half_x + i * _FLAT_GROUND_GRID_SPACING
            line = chrono.ChBodyEasyBox(
                _FLAT_GROUND_GRID_LINE_WIDTH,
                _FLAT_GROUND_GRID_LINE_HEIGHT,
                _FLAT_GROUND_WIDTH,
                1000,
                True,
                False,
            )
            line.SetName(f"ground_grid_z_{i:03d}")
            line.SetFixed(True)
            line.SetPos(chrono.ChVector3d(x, y, 0))
            _set_visual_color(line, grid_color)
            system.AddBody(line)

    def _ground_top_y(self) -> float:
        return float(self.ground_height_offset)

    def _base_world_pos(self) -> np.ndarray:
        pos = self._trunk.GetPos()
        return np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float32)

    def _base_relative_height(self) -> float:
        return float(self._trunk.GetPos().y - self._ground_top_y())

    def _trunk_yaw(self) -> float:
        forward = self._trunk.GetRot().Rotate(chrono.ChVector3d(1, 0, 0))
        return float(math.atan2(float(forward.z), float(forward.x)))

    def _foot_loads(self) -> np.ndarray:
        return np.array(
            [abs(float(foot.GetContactForce().y)) for foot in self._feet],
            dtype=np.float32,
        )

    def _max_nonfoot_load(self) -> float:
        if self._system is None:
            return 0.0
        bodies = {body.GetName(): body for body in self._system.GetBodies()}
        max_load = 0.0
        for leg in ("FR", "FL", "RR", "RL"):
            for group in ("calf", "thigh", "hip"):
                body = bodies.get(f"{leg}_{group}")
                if body is not None:
                    max_load = max(max_load, abs(float(body.GetContactForce().y)))
        return float(max_load)

    def _contact_diagnostic_count(self) -> tuple[float, dict[str, float]]:
        if self._system is None:
            return 0.0, {}
        bodies = {body.GetName(): body for body in self._system.GetBodies()}
        count = 0.0
        diagnostics: dict[str, float] = {}
        for leg in ("FR", "FL", "RR", "RL"):
            for group in ("thigh", "calf"):
                name = f"{leg}_{group}"
                body = bodies.get(name)
                if body is None:
                    diagnostics[f"collision_force_norm_{name}"] = 0.0
                    continue
                force = body.GetContactForce()
                force_norm = math.sqrt(
                    float(force.x) ** 2
                    + float(force.y) ** 2
                    + float(force.z) ** 2
                )
                diagnostics[f"collision_force_norm_{name}"] = float(force_norm)
                if force_norm > _CONTACT_DIAGNOSTIC_FORCE_LIMIT:
                    count += 1.0
        diagnostics["contact_diagnostic_force_limit"] = _CONTACT_DIAGNOSTIC_FORCE_LIMIT
        return float(count), diagnostics

    def _trunk_contact_force_norm(self) -> float:
        if self._trunk is None:
            return 0.0
        force = self._trunk.GetContactForce()
        return float(math.sqrt(float(force.x) ** 2 + float(force.y) ** 2 + float(force.z) ** 2))

    def _create_robot_parser(self, actuation_type=None):
        parser = parsers.ChParserURDF(str(_URDF))
        if self.show_urdf_collision_visuals:
            parser.EnableCollisionVisualization()
        reset_roll = float(self._reset_noise_sample["roll"])
        reset_pitch = float(self._reset_noise_sample["pitch"])
        reset_yaw = float(self._reset_noise_sample["yaw"])
        reset_height = (
            self._ground_top_y()
            + self.spawn_height
            + float(self._reset_noise_sample["base_height_offset"])
        )
        root_rot = (
            chrono.QuatFromAngleY(reset_yaw)
            * chrono.QuatFromAngleX(reset_roll)
            * chrono.QuatFromAngleZ(reset_pitch)
            * chrono.QuatFromAngleX(-math.pi / 2)
        )
        reset_x = float(self.spawn_xz[0]) + float(self._reset_noise_sample["base_position_offset_x"])
        reset_z = float(self.spawn_xz[1]) + float(self._reset_noise_sample["base_position_offset_z"])
        parser.SetRootInitPose(
            chrono.ChFramed(
                chrono.ChVector3d(reset_x, reset_height, reset_z),
                root_rot,
            )
        )
        parser.SetAllBodiesMeshCollisionType(
            parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
        )

        if self.enable_motors and actuation_type is not None:
            parser.SetAllJointsActuationType(actuation_type)

        parser.SetDefaultContactMaterial(_contact_material(mu=0.6))
        foot_mat = _contact_material(mu=0.8)
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            parser.SetBodyContactMaterial(name, foot_mat)

        return parser

    def _configure_imported_bodies(self, system, parser) -> None:
        for body in system.GetBodies():
            if not body.IsFixed():
                body.EnableCollision(False)
                _set_visual_color(body, chrono.ChColor(0.02, 0.02, 0.02))

        for name in _ROBOT_COLLISION_BODIES:
            body = parser.GetChBody(name)
            if body is not None:
                body.EnableCollision(True)

        foot_mat = _rigid_contact_material(
            friction=_FOOT_FRICTION,
            restitution=_FOOT_RESTITUTION,
            contact_method=self.contact_method,
        )
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            body = parser.GetChBody(name)
            if body is not None:
                body.GetCollisionModel().SetAllShapesMaterial(foot_mat)

        self._configure_robot_collision_filters(parser)

    def _configure_robot_collision_filters(self, parser) -> None:
        """Disable robot self-collision while keeping robot-ground collision."""
        bodies = {
            name: parser.GetChBody(name)
            for name in _COLLISION_FILTER_BODIES
        }
        robot_family = 1
        for name, body in bodies.items():
            if body is None:
                continue
            collision_model = body.GetCollisionModel()
            collision_model.SetFamily(robot_family)
            collision_model.DisallowCollisionsWith(robot_family)

    def _replace_body_visuals_with_obj(self, parser) -> None:
        """Use OBJ mesh files for Irrlicht, whose Collada loader is unreliable here."""
        if self.visual_mesh_format == "obj_lod50":
            mesh_dir = _MESH_OBJ_LOD50_DIR
        else:
            mesh_dir = _MESH_OBJ_DIR
        for body_name, filename in _OBJ_VISUAL_MESHES.items():
            body = parser.GetChBody(body_name)
            if body is None:
                continue
            obj_path = mesh_dir / filename
            if not obj_path.exists():
                continue
            visual_model = body.GetVisualModel()
            if visual_model is not None:
                visual_model.Clear()
            shape = chrono.ChVisualShapeModelFile(str(obj_path))
            shape.SetColor(chrono.ChColor(0.02, 0.02, 0.02))
            rpy = _OBJ_VISUAL_ORIGIN_RPY.get(body_name)
            if rpy is None:
                body.AddVisualShape(shape)
                continue
            roll, pitch, yaw = rpy
            frame = chrono.ChFramed(
                chrono.ChVector3d(0.0, 0.0, 0.0),
                chrono.QuatFromAngleX(roll)
                * chrono.QuatFromAngleY(pitch)
                * chrono.QuatFromAngleZ(yaw),
            )
            body.AddVisualShape(shape, frame)

    def _clear_robot_visuals(self, parser) -> None:
        """Hide imported robot visuals so diagnostic overlays can be viewed alone."""
        for body_name in (*_COLLISION_FILTER_BODIES, *_VIEWER_HIDDEN_SENSOR_BODIES):
            body = parser.GetChBody(body_name)
            if body is None:
                continue
            visual_model = body.GetVisualModel()
            if visual_model is not None:
                visual_model.Clear()

    def _hide_viewer_sensor_visuals(self, parser) -> None:
        """Hide decorative sensor meshes in the clean robot viewer path."""
        for body_name in _VIEWER_HIDDEN_SENSOR_BODIES:
            body = parser.GetChBody(body_name)
            if body is None:
                continue
            visual_model = body.GetVisualModel()
            if visual_model is not None:
                visual_model.Clear()

    def _cache_robot_handles(self, system, terrain, parser) -> None:
        self._system = system
        self._terrain = terrain
        self._trunk = parser.GetChBody("trunk")
        self._feet = [parser.GetChBody(name) for name in _FOOT_BODY_NAMES]
        self._drive_links = []
        self._drive_clutches = []
        self._drive_motors = []
        self._drive_shafts = []
        if self.enable_motors:
            if self.actuator_model in _FORCE_ACTUATOR_MODELS:
                self._motors = [parser.GetChMotor(name) for name in _JOINT_NAMES]
                if any(motor is None for motor in self._motors):
                    missing = [
                        name for name, motor in zip(_JOINT_NAMES, self._motors) if motor is None
                    ]
                    raise RuntimeError(f"missing torque motor links: {missing}")
                self._motor_funcs = []
                for motor in self._motors:
                    function = chrono.ChFunctionConst(0.0)
                    motor.SetMotorFunction(function)
                    self._motor_funcs.append(function)
            else:
                self._motors = [parser.GetChMotor(name) for name in _JOINT_NAMES]
                if any(motor is None for motor in self._motors):
                    missing = [name for name, motor in zip(_JOINT_NAMES, self._motors) if motor is None]
                    raise RuntimeError(f"missing torque motor links: {missing}")
                self._motor_funcs = []
                for motor in self._motors:
                    function = chrono.ChFunctionConst(0.0)
                    motor.SetMotorFunction(function)
                    self._motor_funcs.append(function)
        else:
            self._motors = []
            self._motor_funcs = []

        if self.enable_motors:
            bodies_by_name = {body.GetName(): body for body in system.GetBodies()}
            self._joint_body_pairs = [
                (bodies_by_name[parent], bodies_by_name[child])
                for parent, child in _JOINT_BODY_PAIR_NAMES
            ]
        else:
            self._joint_body_pairs = []

    def _apply_default_base_mass_randomization(self) -> None:
        if self._trunk is None or self._system is None:
            return

        diag = self._reset_noise_sample
        if self._trunk_nominal_mass is None:
            self._trunk_nominal_mass = float(self._trunk.GetMass())
        nominal_mass = float(self._trunk_nominal_mass)
        added_mass = float(diag.get("base_mass_added", 0.0))
        target_total_mass = max(0.10, nominal_mass + added_mass)
        com_offset = np.asarray(diag.get("base_com_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        if com_offset.shape != (3,):
            com_offset = np.zeros(3, dtype=np.float64)
        randomization_enabled = bool(diag.get("enabled", False))

        diag["base_mass_nominal"] = nominal_mass
        diag["base_mass_actual"] = float(target_total_mass)
        diag["base_mass_randomization_applied"] = False
        diag["base_com_effective_offset"] = [0.0, 0.0, 0.0]
        diag["base_com_ballast_mass"] = 0.0
        diag["base_com_ballast_local_pos"] = [0.0, 0.0, 0.0]
        diag["base_com_randomization_applied"] = False
        diag["base_com_implementation"] = "direct trunk mass only"

        if not randomization_enabled:
            self._trunk.SetMass(nominal_mass)
            diag["base_mass_actual"] = nominal_mass
            diag["base_com_status"] = "disabled"
            return

        diag["base_mass_randomization_applied"] = True
        if float(np.linalg.norm(com_offset)) <= 1e-9:
            self._trunk.SetMass(target_total_mass)
            diag["base_com_status"] = "zero_offset"
            return

        # This PyChrono URDF path imports the trunk as ChBody, which has no COM
        # setter. Approximate a COM shift with a collisionless ballast body fixed
        # to the trunk; the combined trunk+ballast mass and COM match the sample.
        ballast_mass = max(0.05, 0.50 * target_total_mass)
        trunk_mass = max(0.05, target_total_mass - ballast_mass)
        actual_total_mass = trunk_mass + ballast_mass
        ballast_local_pos = com_offset * actual_total_mass / ballast_mass
        self._trunk.SetMass(trunk_mass)

        ballast = chrono.ChBody()
        ballast.SetName("default_base_com_ballast")
        ballast.SetMass(float(ballast_mass))
        inertia = max(1e-6, float(ballast_mass) * 1e-4)
        ballast.SetInertiaXX(chrono.ChVector3d(inertia, inertia, inertia))
        ballast.EnableCollision(False)

        trunk_pos = self._trunk.GetPos()
        trunk_rot = self._trunk.GetRot()
        local_vec = chrono.ChVector3d(
            float(ballast_local_pos[0]),
            float(ballast_local_pos[1]),
            float(ballast_local_pos[2]),
        )
        world_offset = trunk_rot.Rotate(local_vec)
        ballast.SetPos(
            chrono.ChVector3d(
                float(trunk_pos.x + world_offset.x),
                float(trunk_pos.y + world_offset.y),
                float(trunk_pos.z + world_offset.z),
            )
        )
        ballast.SetRot(trunk_rot)
        ballast.SetPosDt(self._trunk.GetPosDt())
        ballast.SetAngVelParent(self._trunk.GetAngVelParent())
        self._system.AddBody(ballast)

        link = chrono.ChLinkMateFix()
        link.SetName("default_base_com_ballast_lock")
        link.Initialize(ballast, self._trunk, chrono.ChFramed(ballast.GetPos(), ballast.GetRot()))
        self._system.AddLink(link)

        self._base_com_ballast = ballast
        self._base_com_ballast_link = link
        diag["base_mass_actual"] = float(actual_total_mass)
        diag["base_com_randomization_applied"] = True
        diag["base_com_implementation"] = "collisionless fixed ballast body"
        diag["base_com_ballast_mass"] = float(ballast_mass)
        diag["base_com_ballast_local_pos"] = ballast_local_pos.astype(float).tolist()
        diag["base_com_effective_offset"] = com_offset.astype(float).tolist()
        diag["base_com_status"] = "applied_ballast_approximation"

    def _read_joint_angles(self) -> np.ndarray:
        if not self._motors:
            return np.zeros(12, dtype=np.float32)
        return np.array(
            [
                self._joint_angle(motor, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                for i, motor in enumerate(self._motors)
            ],
            dtype=np.float32,
        )

    def _sync_joint_state_cache(
        self,
        reset_velocity: bool = False,
        dt: float = _TIME_STEP,
        use_actual_velocity: bool = False,
    ) -> None:
        joint_pos = self._read_joint_angles()
        if reset_velocity:
            joint_vel = np.zeros(12, dtype=np.float32)
        elif use_actual_velocity:
            joint_vel = self._read_joint_velocities()
        else:
            joint_vel = ((joint_pos - self._last_joint_pos) / max(float(dt), 1e-9)).astype(np.float32)
        self._last_joint_pos = joint_pos
        self._last_joint_vel = joint_vel

    def _run_reset_settle(self) -> None:
        if self.reset_settle_physics_steps <= 0 or not self.enable_motors:
            return
        settle_targets = np.clip(self._reset_joint_targets, _JOINT_LOW, _JOINT_HIGH).astype(np.float32)
        self._last_motor_targets = settle_targets.copy()
        for _ in range(self.reset_settle_physics_steps):
            if self.actuator_model == "actuator_net":
                self._apply_actuator_net(settle_targets)
            else:
                self._apply_torque_limited_pd(settle_targets)
            self._sync_terrain_before_step()
            self._advance_system_and_terrain(self.physics_time_step)
            self._update_actuator_load_cache()
            self._sync_joint_state_cache(use_actual_velocity=True)

    def _min_foot_clearance(self, feet=None) -> float:
        feet = self._feet if feet is None else feet
        if not feet:
            return 0.0
        ground_y = self._ground_top_y()
        return float(min(float(foot.GetPos().y) - _FOOT_COLLISION_RADIUS - ground_y for foot in feet))

    def _apply_dynamic_spawn_height(self, system=None, feet=None) -> None:
        system = self._system if system is None else system
        before = self._min_foot_clearance(feet)
        shift = float(_RESET_FOOT_CLEARANCE - before)
        self._reset_noise_sample["dynamic_spawn_height_min_clearance_before_shift"] = before
        self._reset_noise_sample["dynamic_spawn_height_shift"] = shift
        if abs(shift) > 1e-9:
            for body in system.GetBodies():
                if body.IsFixed():
                    continue
                pos = body.GetPos()
                body.SetPos(chrono.ChVector3d(float(pos.x), float(pos.y) + shift, float(pos.z)))
        after = self._min_foot_clearance(feet)
        self._reset_noise_sample["dynamic_spawn_height_min_clearance_after_shift"] = after
        if system is not None:
            trunk = next((body for body in system.GetBodies() if body.GetName() == "trunk"), None)
            if trunk is not None:
                self._reset_noise_sample["dynamic_spawn_height_effective_root_y"] = float(trunk.GetPos().y)

    def _apply_reset_contact_safety_lift(self, system=None, feet=None) -> None:
        if not self._reset_noise_sample["enabled"]:
            return
        system = self._system if system is None else system
        before = self._min_foot_clearance(feet)
        lift = max(0.0, _RESET_FOOT_CLEARANCE - before)
        self._reset_noise_sample["contact_safety_min_clearance_before_lift"] = before
        self._reset_noise_sample["contact_safety_lift"] = float(lift)
        if lift > 0.0:
            for body in system.GetBodies():
                if body.IsFixed():
                    continue
                pos = body.GetPos()
                body.SetPos(chrono.ChVector3d(float(pos.x), float(pos.y) + lift, float(pos.z)))
        self._reset_noise_sample["contact_safety_min_clearance_after_lift"] = self._min_foot_clearance(feet)

    def _apply_reset_velocity_noise(self) -> None:
        lin_vel = self._reset_noise_sample["base_linear_velocity"]
        ang_vel = self._reset_noise_sample["base_angular_velocity"]
        self._trunk.SetLinVel(chrono.ChVector3d(float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])))
        self._trunk.SetAngVelParent(chrono.ChVector3d(float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])))

        if not self._joint_body_pairs:
            return
        joint_vel_offsets = self._reset_noise_sample["joint_velocity_offsets"]
        for index, (_, child_body) in enumerate(self._joint_body_pairs):
            axis_idx = int(_JOINT_AXES[index])
            sign = float(_JOINT_AXIS_SIGN[index])
            velocity_component = sign * float(joint_vel_offsets[index])
            current = child_body.GetAngVelParent()
            updated = [float(current.x), float(current.y), float(current.z)]
            updated[axis_idx] += velocity_component
            child_body.SetAngVelParent(chrono.ChVector3d(updated[0], updated[1], updated[2]))

    def _scaled_action_offsets(self, executed_action: np.ndarray) -> np.ndarray:
        offsets = (_ACTION_SCALE * executed_action).astype(np.float32)
        offsets[_HIP_ACTION_INDICES] *= _HIP_ACTION_SCALE_MULTIPLIER
        return offsets

    def _motor_targets_from_action(self, executed_action: np.ndarray) -> np.ndarray:
        return np.clip(
            self.home_joint_angles + self._scaled_action_offsets(executed_action),
            _JOINT_LOW,
            _JOINT_HIGH,
        ).astype(np.float32)

    def _apply_torque_pd(self, desired_targets: np.ndarray, *, clip_to_effort_limit: bool) -> None:
        joint_pos = self._last_joint_pos
        joint_vel = self._last_joint_vel
        pd_torque = (self.pd_kp * (desired_targets - joint_pos) - self.pd_kd * joint_vel).astype(np.float32)
        if clip_to_effort_limit:
            torques = np.clip(pd_torque, -_JOINT_EFFORT_LIMIT, _JOINT_EFFORT_LIMIT).astype(np.float32)
        else:
            torques = pd_torque.astype(np.float32)
        for function, torque in zip(self._motor_funcs, torques):
            function.SetConstant(float(torque))
        self._last_motor_torques = torques.copy()
        self._last_torque_limit_fraction = (
            np.abs(torques) / np.maximum(_JOINT_EFFORT_LIMIT, 1e-6)
        ).astype(np.float32)

    def _apply_torque_limited_pd(self, desired_targets: np.ndarray) -> None:
        self._apply_torque_pd(desired_targets, clip_to_effort_limit=True)

    def _reset_actuator_net_history(self, desired_targets: np.ndarray | None = None) -> None:
        if desired_targets is None:
            desired_targets = self._last_motor_targets
        pos_err = (self._last_joint_pos - desired_targets).astype(np.float32)
        joint_vel = self._last_joint_vel.astype(np.float32)
        self._actuator_net_pos_err_last = pos_err.copy()
        self._actuator_net_pos_err_last_last = pos_err.copy()
        self._actuator_net_vel_last = joint_vel.copy()
        self._actuator_net_vel_last_last = joint_vel.copy()

    def _apply_actuator_net(self, desired_targets: np.ndarray, *, update_history: bool = True) -> None:
        if self._actuator_net is None:
            raise RuntimeError("actuator_net selected but model is not loaded")
        joint_pos = self._last_joint_pos.astype(np.float32)
        joint_vel = self._last_joint_vel.astype(np.float32)
        joint_pos_err = (joint_pos - desired_targets).astype(np.float32)
        features = np.stack(
            [
                joint_pos_err,
                self._actuator_net_pos_err_last,
                self._actuator_net_pos_err_last_last,
                joint_vel,
                self._actuator_net_vel_last,
                self._actuator_net_vel_last_last,
            ],
            axis=1,
        ).astype(np.float32)
        with torch.no_grad():
            torque_pred = self._actuator_net(torch.from_numpy(features)).detach().cpu().numpy()
        torques = np.asarray(torque_pred, dtype=np.float32).reshape(-1)[:12]
        torques = np.clip(torques, -_JOINT_EFFORT_LIMIT, _JOINT_EFFORT_LIMIT).astype(np.float32)
        for function, torque in zip(self._motor_funcs, torques):
            function.SetConstant(float(torque))
        self._last_motor_torques = torques.copy()
        self._last_torque_limit_fraction = (
            np.abs(torques) / np.maximum(_JOINT_EFFORT_LIMIT, 1e-6)
        ).astype(np.float32)
        if update_history:
            self._actuator_net_pos_err_last_last = self._actuator_net_pos_err_last.copy()
            self._actuator_net_pos_err_last = joint_pos_err.copy()
            self._actuator_net_vel_last_last = self._actuator_net_vel_last.copy()
            self._actuator_net_vel_last = joint_vel.copy()

    def _apply_motor_targets(self, executed_action: np.ndarray) -> np.ndarray:
        if not self.enable_motors:
            self._last_motor_targets = np.zeros(12, dtype=np.float32)
            self._last_motor_torques = np.zeros(12, dtype=np.float32)
            self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
            return self._last_motor_targets.copy()

        desired_targets = self._motor_targets_from_action(executed_action)
        if self.actuator_model == "actuator_net":
            self._apply_actuator_net(desired_targets)
        else:
            self._apply_torque_limited_pd(desired_targets)
        self._last_motor_targets = desired_targets.copy()
        return desired_targets

    def _update_actuator_load_cache(self) -> None:
        if not self.enable_motors or not self._drive_links:
            if self.actuator_model in _FORCE_ACTUATOR_MODELS:
                return
            self._last_motor_torques = np.zeros(12, dtype=np.float32)
            self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
            return
        torques = np.array(
            [float(drive.GetMotorTorque()) for drive in self._drive_links],
            dtype=np.float32,
        )
        torques = np.clip(torques, -_JOINT_EFFORT_LIMIT, _JOINT_EFFORT_LIMIT)
        self._last_motor_torques = torques
        self._last_torque_limit_fraction = (
            np.abs(torques) / np.maximum(_JOINT_EFFORT_LIMIT, 1e-6)
        ).astype(np.float32)

    def _actuator_diagnostic_terms(self) -> dict[str, float | str]:
        saturation_fraction = float(np.mean(self._last_torque_limit_fraction >= 0.999))
        joint_frame_separations = []
        if self._motors:
            for motor in self._motors:
                try:
                    p1 = motor.GetFrame1Abs().GetPos()
                    p2 = motor.GetFrame2Abs().GetPos()
                    dx = float(p2.x - p1.x)
                    dy = float(p2.y - p1.y)
                    dz = float(p2.z - p1.z)
                    joint_frame_separations.append(float(math.sqrt(dx * dx + dy * dy + dz * dz)))
                except Exception:
                    joint_frame_separations.append(float("nan"))
        finite_separations = [
            value for value in joint_frame_separations if math.isfinite(float(value))
        ]
        terms: dict[str, float | str] = {
            "actuator_model": self.actuator_model,
            "torque_backend": "force_motor",
            "pd_kp": self.pd_kp,
            "pd_kd": self.pd_kd,
            "action_scale": _ACTION_SCALE,
            "hip_action_scale_multiplier": _HIP_ACTION_SCALE_MULTIPLIER,
            "mean_abs_motor_target": float(np.mean(np.abs(self._last_motor_targets))),
            "max_abs_motor_target": float(np.max(np.abs(self._last_motor_targets))),
            "mean_abs_motor_torque": float(np.mean(np.abs(self._last_motor_torques))),
            "max_abs_motor_torque": float(np.max(np.abs(self._last_motor_torques))),
            "mean_torque_limit_fraction": float(np.mean(self._last_torque_limit_fraction)),
            "max_torque_limit_fraction": float(np.max(self._last_torque_limit_fraction)),
            "fraction_torque_saturated": saturation_fraction,
            "mean_joint_frame_separation": float(np.mean(finite_separations)) if finite_separations else 0.0,
            "max_joint_frame_separation": float(max(finite_separations)) if finite_separations else 0.0,
        }
        for name, target, torque, limit in zip(
            _JOINT_NAMES,
            self._last_motor_targets,
            self._last_motor_torques,
            _JOINT_EFFORT_LIMIT,
        ):
            key = name.removesuffix("_joint")
            terms[f"motor_target_{key}"] = float(target)
            terms[f"motor_torque_{key}"] = float(torque)
            terms[f"motor_torque_limit_{key}"] = float(limit)
        for name, separation in zip(_JOINT_NAMES, joint_frame_separations):
            key = name.removesuffix("_joint")
            terms[f"joint_frame_separation_{key}"] = float(separation)
        return terms

    def _reset_foot_load_dict(self) -> dict[str, float]:
        return {
            name.split("_")[0]: abs(float(foot.GetContactForce().y))
            for name, foot in zip(_FOOT_BODY_NAMES, self._feet)
        }

    def _record_post_refresh_reset_diagnostics(self) -> None:
        self._reset_noise_sample["post_refresh_min_foot_clearance"] = self._min_foot_clearance()
        self._reset_noise_sample["post_refresh_foot_loads"] = self._reset_foot_load_dict()

    def _record_post_settle_reset_diagnostics(self) -> None:
        self._reset_noise_sample["post_settle_base_height"] = self._base_relative_height()
        self._reset_noise_sample["post_settle_foot_loads"] = self._reset_foot_load_dict()
        self._reset_noise_sample["post_settle_joint_vel_max_abs"] = float(
            np.max(np.abs(self._last_joint_vel)) if self._last_joint_vel.size else 0.0
        )

    def _record_initial_reset_diagnostics(self) -> None:
        foot_y = {}
        foot_clearance = {}
        foot_loads = {}
        for name, foot in zip(_FOOT_BODY_NAMES, self._feet):
            leg = name.split("_")[0]
            foot_y[leg] = float(foot.GetPos().y)
            foot_clearance[leg] = float(foot.GetPos().y - _FOOT_COLLISION_RADIUS - self._ground_top_y())
            foot_loads[leg] = abs(float(foot.GetContactForce().y))
        self._reset_noise_sample["initial_foot_y"] = foot_y
        self._reset_noise_sample["initial_foot_clearance"] = foot_clearance
        self._reset_noise_sample["initial_min_foot_y"] = float(min(foot_y.values(), default=0.0))
        self._reset_noise_sample["initial_max_foot_y"] = float(max(foot_y.values(), default=0.0))
        self._reset_noise_sample["initial_min_foot_clearance"] = float(min(foot_clearance.values(), default=0.0))
        self._reset_noise_sample["initial_max_foot_clearance"] = float(max(foot_clearance.values(), default=0.0))
        self._reset_noise_sample["initial_foot_loads"] = foot_loads
        self._reset_noise_sample["initial_max_foot_load"] = float(max(foot_loads.values(), default=0.0))

    def _create_visualizer(self, system) -> None:
        # Always create a fresh visualizer. Reusing an initialized Irrlicht
        # device and calling AttachSystem again can crash after reset.
        self._vis = None
        vis = irr.ChVisualSystemIrrlicht()
        vis.AttachSystem(system)
        vis.SetWindowSize(1280, 720)
        vis.SetWindowTitle("Chrono Go1 Env")
        vis.Initialize()
        vis.AddSkyBox()
        vis.AddCamera(
            chrono.ChVector3d(2.5, 1.5, 2.5),
            chrono.ChVector3d(0, 0.4, 0),
        )
        vis.AddTypicalLights()
        self._vis = vis

    def _add_collision_visual_overlays(self, parser) -> None:
        """Show simplified thigh/calf collision boxes on top of mesh visuals."""
        specs = {
            "thigh": ((0.213, 0.0245, 0.034), chrono.ChColor(1.0, 0.55, 0.05), 0.35),
            "calf": ((0.193, 0.016, 0.016), chrono.ChColor(1.0, 0.05, 0.05), 0.45),
        }
        for leg in _LEG_PREFIXES:
            for group, (size, color, opacity) in specs.items():
                body = parser.GetChBody(f"{leg}_{group}")
                if body is None:
                    continue
                shape = chrono.ChVisualShapeBox(*size)
                shape.SetColor(color)
                shape.SetOpacity(opacity)
                z_offset = -0.0965 if group == "calf" else -0.1065
                frame = chrono.ChFramed(
                    chrono.ChVector3d(0.0, 0.0, z_offset),
                    chrono.QuatFromAngleY(math.pi / 2.0),
                )
                body.AddVisualShape(shape, frame)

    def _joint_angle(self, motor, axis_idx: int, sign: float) -> float:
        """Revolute joint angle from relative motor-frame rotation.

        Reads the component of the motor-frame rotation vector selected by
        _JOINT_AXES, then applies _JOINT_AXIS_SIGN to map it into the
        policy/observation joint coordinate.
        """
        frame1 = motor.GetFrame1Abs()
        frame2 = motor.GetFrame2Abs()
        q_rel = frame1.GetRot().GetInverse() * frame2.GetRot()
        rv = q_rel.GetRotVec()
        components = (rv.x, rv.y, rv.z)
        return sign * float(components[axis_idx])

    def _joint_vel(self, b1, b2, axis_idx: int, sign: float) -> float:
        """Relative angular velocity along the joint's rotation axis."""
        w1 = b1.GetAngVelParent()
        w2 = b2.GetAngVelParent()
        dw_world = chrono.ChVector3d(w2.x - w1.x, w2.y - w1.y, w2.z - w1.z)
        dw_local = b1.GetRot().GetInverse().Rotate(dw_world)
        components = (dw_local.x, dw_local.y, dw_local.z)
        return sign * float(components[axis_idx])

    def _read_joint_velocities(self) -> np.ndarray:
        if not self._joint_body_pairs:
            return np.zeros(12, dtype=np.float32)
        return np.array(
            [
                self._joint_vel(b1, b2, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                for i, (b1, b2) in enumerate(self._joint_body_pairs)
            ],
            dtype=np.float32,
        )

    def _projected_gravity_body(self) -> np.ndarray:
        gravity_world = chrono.ChVector3d(0, -1, 0)
        gravity_body = self._trunk.GetRot().GetInverse().Rotate(gravity_world)
        return np.array([gravity_body.x, gravity_body.y, gravity_body.z], dtype=np.float32)

    def _trunk_linear_velocity_body(self) -> np.ndarray:
        if self._trunk is None:
            return np.zeros(3, dtype=np.float32)
        lin_world = self._trunk.GetPosDt()
        lin_body = self._trunk.GetRot().GetInverse().Rotate(lin_world)
        return np.array([lin_body.x, lin_body.y, lin_body.z], dtype=np.float32)

    def _trunk_angular_velocity_body(self) -> np.ndarray:
        if self._trunk is None:
            return np.zeros(3, dtype=np.float32)
        ang_world = self._trunk.GetAngVelParent()
        ang_body = self._trunk.GetRot().GetInverse().Rotate(ang_world)
        return np.array([ang_body.x, ang_body.y, ang_body.z], dtype=np.float32)

    def _get_default_obs48(self) -> np.ndarray:
        lin_vel = self._trunk_linear_velocity_body()
        ang_vel = self._trunk_angular_velocity_body()
        joint_pos = self._last_joint_pos if self._motors else np.zeros(12, dtype=np.float32)
        joint_vel = self._last_joint_vel if self._motors else np.zeros(12, dtype=np.float32)
        return np.concatenate([
            lin_vel,
            ang_vel,
            self._projected_gravity_body(),
            self.command,
            joint_pos - self.home_joint_angles,
            joint_vel,
            self._prev_action,
        ]).astype(np.float32)

    def _get_obs(self, apply_noise: bool = True) -> np.ndarray:
        obs = self._get_default_obs48()
        if apply_noise:
            obs = self._apply_default_obs_noise(obs)
        return obs.astype(np.float32)

    def _apply_default_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        if not self.observation_noise:
            return obs.astype(np.float32)
        noise_scale = _DEFAULT_OBS_NOISE_SCALE
        if noise_scale <= 0.0:
            return obs.astype(np.float32)
        noisy = obs.copy()
        noisy[0:3] += self.np_random.uniform(
            -noise_scale * _DEFAULT_OBS_NOISE_BASE_LIN_VEL,
            noise_scale * _DEFAULT_OBS_NOISE_BASE_LIN_VEL,
            size=3,
        ).astype(np.float32)
        noisy[3:6] += self.np_random.uniform(
            -noise_scale * _DEFAULT_OBS_NOISE_BASE_ANG_VEL,
            noise_scale * _DEFAULT_OBS_NOISE_BASE_ANG_VEL,
            size=3,
        ).astype(np.float32)
        noisy[6:9] += self.np_random.uniform(
            -noise_scale * _DEFAULT_OBS_NOISE_PROJECTED_GRAVITY,
            noise_scale * _DEFAULT_OBS_NOISE_PROJECTED_GRAVITY,
            size=3,
        ).astype(np.float32)
        noisy[12:24] += self.np_random.uniform(
            -noise_scale * _DEFAULT_OBS_NOISE_JOINT_POS,
            noise_scale * _DEFAULT_OBS_NOISE_JOINT_POS,
            size=12,
        ).astype(np.float32)
        noisy[24:36] += self.np_random.uniform(
            -noise_scale * _DEFAULT_OBS_NOISE_JOINT_VEL,
            noise_scale * _DEFAULT_OBS_NOISE_JOINT_VEL,
            size=12,
        ).astype(np.float32)
        return noisy.astype(np.float32)

    def _trunk_axis_alignments(self) -> dict[str, float]:
        """Return each trunk local axis alignment with Chrono world Y-up."""
        rot = self._trunk.GetRot()
        return {
            "trunk_x_up": float(np.clip(rot.Rotate(chrono.ChVector3d(1, 0, 0)).y, -1.0, 1.0)),
            "trunk_y_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 1, 0)).y, -1.0, 1.0)),
            "trunk_z_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 0, 1)).y, -1.0, 1.0)),
        }

    def _flat_orientation_l2(self) -> float:
        """Projected-gravity penalty adapted to Chrono Y-up."""
        gravity_body = self._projected_gravity_body()
        # The world is Y-up, but this imported Go1 trunk's local Z axis is the
        # upright/down axis: a level reset has projected gravity ~= [0, 0, -1].
        # Penalize the two components perpendicular to that local upright axis.
        return float(gravity_body[0] ** 2 + gravity_body[1] ** 2)

    def _reward_obs_components(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        trunk_lin_vel = obs[0:3]
        trunk_ang_vel = obs[3:6]
        command = obs[9:12]
        joint_pos = obs[12:24] + self.home_joint_angles
        joint_vel = obs[24:36]
        return (
            trunk_lin_vel.astype(np.float32),
            trunk_ang_vel.astype(np.float32),
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            command.astype(np.float32),
        )

    def _default_reward_terms(
        self,
        obs: np.ndarray,
        policy_action: np.ndarray,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        raw_action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        _, _, _, joint_vel, command = self._reward_obs_components(obs)
        trunk_lin_vel = self._trunk_linear_velocity_body()
        trunk_ang_vel = self._trunk_angular_velocity_body()

        command_vx, command_vz, command_yaw_rate = (float(value) for value in command)
        vx_body = float(trunk_lin_vel[0])
        vz_body = float(trunk_lin_vel[1])
        vertical_body_vel = float(trunk_lin_vel[2])
        body_yaw_rate = float(trunk_ang_vel[2])
        lin_vel_error_x = float(command_vx - vx_body)
        lin_vel_error_z = float(command_vz - vz_body)
        yaw_rate_error = float(command_yaw_rate - body_yaw_rate)
        lin_err_sq = float(lin_vel_error_x ** 2 + lin_vel_error_z ** 2)
        yaw_err_sq = float(yaw_rate_error ** 2)

        tracking_lin_vel = float(math.exp(-lin_err_sq / _TRACKING_SIGMA))
        tracking_ang_vel = float(math.exp(-yaw_err_sq / _TRACKING_SIGMA))
        lin_cmd = float(math.sqrt(command_vx ** 2 + command_vz ** 2))
        lin_vel_z = float(vertical_body_vel ** 2)
        ang_vel_xy = float(trunk_ang_vel[0] ** 2 + trunk_ang_vel[1] ** 2)
        torques = float(np.sum(self._last_motor_torques ** 2))
        dof_acc = float(np.sum(((joint_vel - self._prev_joint_vel_for_reward) / _PHYSICS_TIME_STEP) ** 2))
        action_rate = float(np.sum(raw_action_delta ** 2))
        flat_orientation_l2 = self._flat_orientation_l2()

        foot_loads = self._foot_loads()
        foot_contact = foot_loads > _FEET_AIR_TIME_CONTACT_FORCE
        previous_air_times = self._foot_air_times.copy()
        first_contact = foot_contact & (previous_air_times > 0.0)
        moving_cmd_active = lin_cmd > _FEET_AIR_TIME_COMMAND_MIN_SPEED
        feet_air_time = float(
            float(moving_cmd_active)
            * np.sum(first_contact.astype(np.float32) * (previous_air_times - 0.5))
        )
        self._foot_air_times = np.where(
            foot_contact,
            0.0,
            self._foot_air_times + _TIME_STEP,
        ).astype(np.float32)

        raw_terms = {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "lin_vel_z": lin_vel_z,
            "ang_vel_xy": ang_vel_xy,
            "torques": torques,
            "dof_acc": dof_acc,
            "flat_orientation_l2": flat_orientation_l2,
            "feet_air_time": feet_air_time,
            "action_rate": action_rate,
        }
        weights = {
            "tracking_lin_vel": _REWARD_TRACKING_LIN_VEL_WEIGHT,
            "tracking_ang_vel": _REWARD_TRACKING_ANG_VEL_WEIGHT,
            "lin_vel_z": _REWARD_LIN_VEL_Y_WEIGHT,
            "ang_vel_xy": _REWARD_ANG_VEL_XZ_WEIGHT,
            "torques": _REWARD_TORQUES_WEIGHT,
            "dof_acc": _REWARD_DOF_ACC_WEIGHT,
            "flat_orientation_l2": _REWARD_FLAT_ORIENTATION_L2_WEIGHT,
            "feet_air_time": _REWARD_FEET_AIR_TIME_WEIGHT,
            "action_rate": _REWARD_ACTION_RATE_WEIGHT,
        }
        weighted_terms = {name: weights[name] * value for name, value in raw_terms.items()}
        raw_reward = float(sum(weighted_terms.values()))
        reward_dt_scaled = float(_TIME_STEP * raw_reward)

        load_shares = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        foot_load_sum = float(np.sum(np.maximum(foot_loads, 0.0)))
        if foot_load_sum > 1e-6:
            load_shares = np.maximum(foot_loads, 0.0).astype(np.float32) / foot_load_sum
        contact_switches_this_step = float(np.sum(foot_contact != self._reward_contact_active))
        self._reward_contact_active = foot_contact.copy()
        collision_count, contact_diagnostics = self._contact_diagnostic_count()
        normalized_torques = self._last_motor_torques / np.maximum(_JOINT_EFFORT_LIMIT, 1e-6)

        terms = {
            "reward_raw_sum": raw_reward,
            "reward_dt_scaled": reward_dt_scaled,
            "reward_clipped": reward_dt_scaled,
            "reward_unclipped": reward_dt_scaled,
            "positive_rewards_only": 0.0,
            "tracking_sigma": _TRACKING_SIGMA,
            "command_vx": command_vx,
            "command_vz": command_vz,
            "command_yaw_rate": command_yaw_rate,
            "body_lin_vel_x": vx_body,
            "body_lin_vel_z": vz_body,
            "body_vertical_vel_z": vertical_body_vel,
            "body_yaw_rate": body_yaw_rate,
            "trunk_yaw": self._trunk_yaw(),
            "lin_vel_error_x": lin_vel_error_x,
            "lin_vel_error_z": lin_vel_error_z,
            "lin_err_sq": lin_err_sq,
            "yaw_rate_error": yaw_rate_error,
            "yaw_err_sq": yaw_err_sq,
            "lin_cmd": lin_cmd,
            "yaw_cmd_abs": abs(command_yaw_rate),
            "moving_cmd_active": float(moving_cmd_active),
            "flat_orientation_l2_error": flat_orientation_l2,
            "lin_vel_z_error": lin_vel_z,
            "ang_vel_xy_error": ang_vel_xy,
            "torques_error": torques,
            "raw_torques_l2_sum": torques,
            "normalized_torques_l2_mean": float(np.mean(normalized_torques ** 2)),
            "torque_saturation_error": float(np.mean((np.maximum(np.abs(normalized_torques) - 0.85, 0.0) / 0.15) ** 2)),
            "dof_acc_error": dof_acc,
            "action_rate_error": action_rate,
            "feet_air_time_error": feet_air_time,
            "feet_air_time_bonus_raw": feet_air_time,
            "mean_foot_air_time": float(np.mean(self._foot_air_times)),
            "max_foot_air_time": float(np.max(self._foot_air_times)),
            "contact_switches_this_step": contact_switches_this_step,
            "contact_switch_count_step": contact_switches_this_step,
            "contact_diagnostic_count": collision_count,
            "preclip_policy_action_abs_mean": float(np.mean(np.abs(policy_action))),
            "preclip_policy_action_abs_max": float(np.max(np.abs(policy_action))),
            "mean_abs_action": float(np.mean(np.abs(executed_action))),
            "max_abs_action": float(np.max(np.abs(executed_action))),
            "reward_load_share_FR": float(load_shares[0]),
            "reward_load_share_FL": float(load_shares[1]),
            "reward_load_share_RR": float(load_shares[2]),
            "reward_load_share_RL": float(load_shares[3]),
            "reward_front_load_share": float(load_shares[0] + load_shares[1]),
            "reward_rear_load_share": float(load_shares[2] + load_shares[3]),
            "reward_left_load_share": float(load_shares[1] + load_shares[3]),
            "reward_right_load_share": float(load_shares[0] + load_shares[2]),
        }
        for leg_name, is_contact, air_time in zip(_FOOT_BODY_NAMES, foot_contact, self._foot_air_times):
            leg = leg_name.split("_", 1)[0]
            terms[f"foot_contact_{leg}"] = float(is_contact)
            terms[f"foot_air_time_{leg}"] = float(air_time)
        terms.update(contact_diagnostics)
        for name, raw_value in raw_terms.items():
            terms[f"{name}_raw"] = float(raw_value)
            terms[f"{name}_weighted"] = float(weighted_terms[name])
            terms[f"{name}_reward"] = float(_TIME_STEP * weighted_terms[name])
        self._prev_joint_vel_for_reward = joint_vel.copy()
        return reward_dt_scaled, terms

    def _standing_reward(
        self,
        obs: np.ndarray,
        policy_action: np.ndarray,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        executed_action_delta: np.ndarray,
        raw_action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        if not np.all(np.isfinite(obs)):
            return 0.0, {
                "invalid_obs": 1.0,
                "reward_raw_sum": 0.0,
                "reward_dt_scaled": 0.0,
                "reward_clipped": 0.0,
            }
        return self._default_reward_terms(
            obs,
            policy_action,
            raw_action,
            executed_action,
            raw_action_delta,
        )

    def _termination_reason(self, obs: np.ndarray, reward_terms: dict) -> str | None:
        if not np.all(np.isfinite(obs)):
            return "invalid_obs"
        if self._trunk_contact_force_norm() > _DEFAULT_TRUNK_CONTACT_TERMINATION_FORCE:
            return "trunk_contact"
        if self.env_backend == "scm":
            base_height = float(reward_terms.get("base_relative_height", self._base_relative_height()))
            if base_height < _TERM_RELATIVE_HEIGHT:
                return "base_height"
        return None

    # ---------------------------------------------------------------------- #
    # Gymnasium interface
    # ---------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.command_seed = int(seed) + 1000 * self.env_rank
        self._sample_reset_noise()
        self._build_sim()
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._foot_air_times.fill(0.0)
        self._run_reset_settle()
        self._sync_joint_state_cache(use_actual_velocity=True)
        self._prev_joint_vel_for_reward = self._last_joint_vel.copy()
        foot_loads_after_settle = self._foot_loads()
        self._reward_contact_active = foot_loads_after_settle >= _MIN_FOOT_LOAD
        self._prev_air_time_contact = foot_loads_after_settle > _FEET_AIR_TIME_CONTACT_FORCE
        self._record_post_settle_reset_diagnostics()
        self.step_count = 0
        self._reset_command()
        self._reset_push_schedule()
        obs = self._get_obs()
        self._record_initial_reset_diagnostics()
        base_world = self._base_world_pos()
        self._reset_base_xz = np.array([base_world[0], base_world[2]], dtype=np.float32)
        return obs, self._info()

    def step(self, action: np.ndarray):
        self.step_count += 1

        command_before_step = self.command.copy()
        command_diag_before_step = self.command_diag.copy()
        policy_action = np.asarray(action, dtype=np.float32)
        raw_action = np.clip(policy_action, -self.action_clip, self.action_clip).astype(np.float32)
        executed_action = raw_action.copy()
        action_delta = executed_action - self._prev_action
        raw_action_delta = raw_action - self._prev_raw_action
        if self.actuator_model in _FORCE_ACTUATOR_MODELS and self.enable_motors:
            targets = self._motor_targets_from_action(executed_action)
            self._last_motor_targets = targets.copy()
        else:
            targets = self._apply_motor_targets(executed_action)

        for substep_index in range(self.physics_substeps):
            if self.actuator_model in _FORCE_ACTUATOR_MODELS and self.enable_motors:
                if self.update_force_actuator_each_substep or (
                    substep_index % self.force_actuator_update_interval == 0
                ):
                    if self.actuator_model == "actuator_net":
                        self._apply_actuator_net(targets)
                    else:
                        self._apply_torque_limited_pd(targets)
            self._sync_terrain_before_step()
            self._apply_scheduled_push()
            self._advance_system_and_terrain(self.physics_time_step)
            if self.actuator_model in _FORCE_ACTUATOR_MODELS and self.enable_motors:
                self._sync_joint_state_cache(use_actual_velocity=True)
        self._update_actuator_load_cache()
        if self.actuator_model not in _FORCE_ACTUATOR_MODELS or not self.enable_motors:
            self._sync_joint_state_cache(reset_velocity=False)

        self._prev_raw_action = raw_action.copy()
        self._prev_action = executed_action.copy()
        obs_for_reward = self._get_obs(apply_noise=False)
        truncated = self.step_count >= self.max_steps
        reward, reward_terms = self._standing_reward(
            obs_for_reward,
            policy_action,
            raw_action,
            executed_action,
            action_delta,
            raw_action_delta,
        )
        base_world = self._base_world_pos()
        reward_terms.update({
            "base_world_x": float(base_world[0]),
            "base_world_y": float(base_world[1]),
            "base_world_z": float(base_world[2]),
            "base_relative_height": self._base_relative_height(),
            "base_height_termination_threshold": _TERM_RELATIVE_HEIGHT,
            "ground_top_y": self._ground_top_y(),
            "ground_height_offset": self.ground_height_offset,
            "trunk_contact_force_norm": self._trunk_contact_force_norm(),
            "trunk_contact_termination_force": _DEFAULT_TRUNK_CONTACT_TERMINATION_FORCE,
            "default_phase": command_diag_before_step.get("phase", ""),
            "default_friction": _DEFAULT_FRICTION,
            "default_obs_noise_scale": _DEFAULT_OBS_NOISE_SCALE,
            "default_push_enabled": 0.0,
            "reset_noise_enabled": float(self._reset_noise_sample["enabled"]),
            "reset_noise_rn2_scale": float(self._reset_noise_sample["rn2_scale"]),
            "reset_noise_base_position_offset_x": float(self._reset_noise_sample["base_position_offset_x"]),
            "reset_noise_base_position_offset_z": float(self._reset_noise_sample["base_position_offset_z"]),
            "reset_noise_base_height_offset": float(self._reset_noise_sample["base_height_offset"]),
            "reset_noise_roll": float(self._reset_noise_sample["roll"]),
            "reset_noise_pitch": float(self._reset_noise_sample["pitch"]),
            "reset_noise_yaw": float(self._reset_noise_sample["yaw"]),
            "reset_noise_base_linear_velocity_x": float(self._reset_noise_sample["base_linear_velocity"][0]),
            "reset_noise_base_linear_velocity_y": float(self._reset_noise_sample["base_linear_velocity"][1]),
            "reset_noise_base_linear_velocity_z": float(self._reset_noise_sample["base_linear_velocity"][2]),
            "reset_noise_base_angular_velocity_x": float(self._reset_noise_sample["base_angular_velocity"][0]),
            "reset_noise_base_angular_velocity_y": float(self._reset_noise_sample["base_angular_velocity"][1]),
            "reset_noise_base_angular_velocity_z": float(self._reset_noise_sample["base_angular_velocity"][2]),
            "reset_noise_contact_safety_clearance": float(self._reset_noise_sample["contact_safety_clearance"]),
            "reset_noise_contact_safety_min_clearance_before_lift": float(self._reset_noise_sample["contact_safety_min_clearance_before_lift"]),
            "reset_noise_contact_safety_lift": float(self._reset_noise_sample["contact_safety_lift"]),
            "reset_noise_contact_safety_min_clearance_after_lift": float(self._reset_noise_sample["contact_safety_min_clearance_after_lift"]),
            "reset_dynamic_spawn_height_enabled": float(self._reset_noise_sample["dynamic_spawn_height_enabled"]),
            "reset_dynamic_spawn_height_target_clearance": float(self._reset_noise_sample["dynamic_spawn_height_target_clearance"]),
            "reset_dynamic_spawn_height_shift": float(self._reset_noise_sample["dynamic_spawn_height_shift"]),
            "reset_dynamic_spawn_height_min_clearance_before_shift": float(self._reset_noise_sample["dynamic_spawn_height_min_clearance_before_shift"]),
            "reset_dynamic_spawn_height_min_clearance_after_shift": float(self._reset_noise_sample["dynamic_spawn_height_min_clearance_after_shift"]),
            "reset_dynamic_spawn_height_effective_root_y": float(self._reset_noise_sample["dynamic_spawn_height_effective_root_y"]),
            "reset_noise_joint_pos_offset_rms": float(np.sqrt(np.mean(np.square(self._reset_noise_sample["joint_position_offsets"])))),
            "reset_noise_joint_vel_offset_rms": float(np.sqrt(np.mean(np.square(self._reset_noise_sample["joint_velocity_offsets"])))),
            "reset_noise_base_linear_velocity_norm": float(np.linalg.norm(self._reset_noise_sample["base_linear_velocity"])),
            "reset_noise_base_angular_velocity_norm": float(np.linalg.norm(self._reset_noise_sample["base_angular_velocity"])),
            "reset_noise_base_mass_added": float(self._reset_noise_sample["base_mass_added"]),
            "reset_noise_base_mass_nominal": float(self._reset_noise_sample["base_mass_nominal"]),
            "reset_noise_base_mass_actual": float(self._reset_noise_sample["base_mass_actual"]),
            "reset_noise_base_com_offset_x": float(self._reset_noise_sample["base_com_offset"][0]),
            "reset_noise_base_com_offset_y": float(self._reset_noise_sample["base_com_offset"][1]),
            "reset_noise_base_com_offset_z": float(self._reset_noise_sample["base_com_offset"][2]),
            "reset_noise_base_com_randomization_applied": float(
                self._reset_noise_sample["base_com_randomization_applied"]
            ),
            "reset_noise_initial_min_foot_y": float(self._reset_noise_sample["initial_min_foot_y"]),
            "reset_noise_initial_max_foot_load": float(self._reset_noise_sample["initial_max_foot_load"]),
            "mean_abs_raw_action": float(np.mean(np.abs(raw_action))),
            "max_abs_raw_action": float(np.max(np.abs(raw_action))),
            "mean_abs_policy_action": float(np.mean(np.abs(policy_action))),
            "max_abs_policy_action": float(np.max(np.abs(policy_action))),
            "mean_abs_raw_action_delta": float(np.mean(np.abs(raw_action_delta))),
            "max_abs_raw_action_delta": float(np.max(np.abs(raw_action_delta))),
            "mean_abs_executed_action": float(np.mean(np.abs(executed_action))),
            "max_abs_executed_action": float(np.max(np.abs(executed_action))),
            "mean_abs_executed_action_delta": float(np.mean(np.abs(action_delta))),
            "max_abs_executed_action_delta": float(np.max(np.abs(action_delta))),
            "command_mode": "default",
            "command_phase": command_diag_before_step.get("phase", ""),
            "command_bucket": command_diag_before_step.get("bucket", ""),
            "command_vx": float(command_before_step[0]),
            "command_vz": float(command_before_step[1]),
            "command_yaw_rate": float(command_before_step[2]),
            "command_target_vx": float(command_diag_before_step.get("target_vx", command_before_step[0])),
            "command_target_vz": float(command_diag_before_step.get("target_vz", command_before_step[1])),
            "command_target_yaw_rate": float(command_diag_before_step.get("target_yaw", command_before_step[2])),
            "command_range_vx_min": float(command_diag_before_step.get("range_vx_min", 0.0)),
            "command_range_vx_max": float(command_diag_before_step.get("range_vx_max", 0.0)),
            "command_range_vz_min": float(command_diag_before_step.get("range_vz_min", 0.0)),
            "command_range_vz_max": float(command_diag_before_step.get("range_vz_max", 0.0)),
            "command_range_yaw_min": float(command_diag_before_step.get("range_yaw_min", 0.0)),
            "command_range_yaw_max": float(command_diag_before_step.get("range_yaw_max", 0.0)),
            "command_resample_count": float(command_diag_before_step.get("command_resample_count", 0)),
        })
        reward_terms.update(self._actuator_diagnostic_terms())
        reward_terms.update(self._push_diagnostics())
        termination_reason = self._termination_reason(obs_for_reward, reward_terms)
        terminated = termination_reason is not None
        terminal_reward = float(_REWARD_TERMINATION_WEIGHT if terminated else 0.0)
        reward += terminal_reward
        reward_terms["termination_raw"] = float(terminated)
        reward_terms["termination_weighted"] = terminal_reward
        reward_terms["termination_reward"] = terminal_reward
        reward_terms["reward_with_termination"] = float(reward)
        self._advance_command_after_reward()
        obs = self._get_obs()

        info = self._info()
        info["target_joint_angles"] = targets
        info["raw_action"] = raw_action
        info["executed_action"] = executed_action
        info["raw_action_delta"] = raw_action_delta
        info["executed_action_delta"] = action_delta
        info["reward_terms"] = reward_terms
        info["termination_reason"] = termination_reason
        info["command_sampler"] = self.command_diag
        info["default_randomization"] = {
            "phase": command_diag_before_step.get("phase", ""),
            "friction": _DEFAULT_FRICTION,
            "obs_noise_scale": _DEFAULT_OBS_NOISE_SCALE,
            "pushes": False,
        }
        return obs, reward, terminated, truncated, info

    def _material_info(self) -> dict:
        effective_friction = None
        if self.ground_friction is not None:
            effective_friction = min(float(self.ground_friction), _FOOT_FRICTION)
        return {
            "contact_method": self.contact_method,
            "material_composition_rule": _MATERIAL_COMPOSITION_RULE,
            "composition_strategy": "ChContactMaterialCompositionStrategy::CombineFriction",
            "configured_ground_friction": None if self.ground_friction is None else float(self.ground_friction),
            "configured_foot_friction": _FOOT_FRICTION,
            "effective_friction": effective_friction,
            "static_friction": effective_friction,
            "sliding_friction": effective_friction,
            "static_sliding_note": "Current SetFriction calls use the same Chrono friction value for static/sliding contact.",
            "ground": {
                "friction": None if self.ground_friction is None else float(self.ground_friction),
                "restitution": _GROUND_RESTITUTION,
                "rolling_friction": _GROUND_ROLLING_FRICTION,
            },
            "feet": {
                "friction": _FOOT_FRICTION,
                "restitution": _FOOT_RESTITUTION,
            },
        }

    def _info(self) -> dict:
        material_info = self._material_info()
        return {
            "env_backend": self.env_backend,
            "terrain": self.terrain_type,
            "actuator_model": self.actuator_model,
            "torque_backend": "force_motor",
            "ground_friction": self.ground_friction,
            "foot_friction": _FOOT_FRICTION,
            "effective_friction": material_info["effective_friction"],
            "friction_range": self.friction_range,
            "spawn_x": float(self.spawn_xz[0]),
            "spawn_z": float(self.spawn_xz[1]),
            "spawn_height": float(self.spawn_height),
            "reset_settle_physics_steps": int(self.reset_settle_physics_steps),
            "reset_settle_seconds": float(self.reset_settle_physics_steps * self.physics_time_step),
            "actuator_net_history": "repeat_current",
            "force_actuator_update_time_step": float(self.force_actuator_update_time_step),
            "force_actuator_update_frequency": float(1.0 / self.force_actuator_update_time_step),
            "force_actuator_update_interval": int(self.force_actuator_update_interval),
            "ground_height_offset": self.ground_height_offset,
            "ground_top_y": self._ground_top_y(),
            "command_mode": "default",
            "command": self.command.astype(float).tolist(),
            "command_target": self.command_target.astype(float).tolist(),
            "command_sampler": self.command_diag,
            "default_randomization": {
                "phase": self.command_diag.get("phase", ""),
                "friction": _DEFAULT_FRICTION,
                "obs_noise_scale": _DEFAULT_OBS_NOISE_SCALE,
                "pushes": False,
                **self._push_diagnostics(),
            },
            "reset_noise": self._reset_noise_sample,
            "material": {
                "ground_friction": material_info["ground"]["friction"],
                "foot_friction": material_info["feet"]["friction"],
                "effective_friction": material_info["effective_friction"],
            },
            "material_properties": material_info,
            "collision_bodies": list(_ROBOT_COLLISION_BODIES),
            "actuator_model": self.actuator_model,
        }

    def render(self) -> bool:
        """Render one frame. Returns False when the window has been closed."""
        if self._vis is None or not self._vis.Run():
            return False

        self._vis.BeginScene()
        self._vis.Render()
        self._vis.EndScene()
        return True

    def close(self):
        self._vis = None
        self._system = None
        self._terrain = None
        self._trunk = None
        self._motors = []
        self._motor_funcs = []
        self._joint_body_pairs = []
