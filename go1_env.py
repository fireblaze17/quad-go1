"""Chrono Gymnasium environment for a Unitree Go1-style quadruped.

The project uses Chrono as the simulator and MuJoCo Menagerie only as a source
of model/reference values that are known to be sane for Go1. Chrono runs here in
a Y-up world, so the imported ROS-style Z-up URDF is rotated at the root.

Observation, 48 float32 values:
    trunk height relative to support/ground, trunk quaternion, trunk linear
    velocity, trunk angular velocity, 12 joint angles, 12 joint velocities,
    support-relative standing errors, and 3 command values
    [command_vx, command_vz, command_yaw_rate].

Action, 12 float32 values in [-1, 1]:
    normalized joint-position offsets around the nominal standing pose.
"""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pychrono as chrono
import pychrono.irrlicht as irr
import pychrono.parsers as parsers
import pychrono.vehicle as veh
from gymnasium import spaces


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_URDF = Path(__file__).parent / "models/go1/go1_chrono.urdf"

_PHYSICS_TIME_STEP = 5e-3
_CONTROL_FREQUENCY = 50.0
_TIME_STEP = 1.0 / _CONTROL_FREQUENCY
_PHYSICS_SUBSTEPS = int(round(_TIME_STEP / _PHYSICS_TIME_STEP))
if not np.isclose(_PHYSICS_SUBSTEPS * _PHYSICS_TIME_STEP, _TIME_STEP):
    raise ValueError("Control timestep must be an integer multiple of the physics timestep.")
_TERRAIN_LENGTH = 6.0
_TERRAIN_WIDTH = 4.0
_TERRAIN_DELTA = 0.04
_GROUND_RESTITUTION = 0.1
_GROUND_KN = 2e5
_GROUND_GN = 60.0
_GROUND_ROLLING_FRICTION = 0.0001
_FOOT_FRICTION = 2.0
_FOOT_RESTITUTION = 0.01
_FOOT_GN = 60.0
_CONTACT_METHOD = "SMC"
_MATERIAL_COMPOSITION_RULE = "min"

# Zero-action diagnostics showed the original Menagerie crouch
# (hip=0, thigh=0.9, calf=-1.8 at y=0.27) slowly sank in Chrono. This less
# crouched pose starts at its natural support height and holds with zero action.
_SPAWN_HEIGHT = 0.34  # trunk root height; DoAssembly drives legs to home before first step
_TERM_RELATIVE_HEIGHT = 0.22
_MIN_UPRIGHT_ALIGNMENT = 0.85
_TRACKING_SIGMA = 0.25
_BASE_HEIGHT_TARGET = 0.34
_ZERO_COMMAND = np.zeros(3, dtype=np.float32)
_REWARD_TRACKING_LIN_VEL_WEIGHT = 1.0
_REWARD_TRACKING_ANG_VEL_WEIGHT = 0.5
_REWARD_LIN_VEL_Y_WEIGHT = -2.0
_REWARD_ANG_VEL_XZ_WEIGHT = -0.05
_REWARD_ORIENTATION_WEIGHT = -5.0
_REWARD_BASE_HEIGHT_WEIGHT = -30.0
_REWARD_TORQUES_WEIGHT = -0.0001
_REWARD_DOF_ACC_WEIGHT = -2.5e-7
_REWARD_ACTION_RATE_WEIGHT = -0.01
_REWARD_DOF_POS_LIMITS_WEIGHT = -10.0
_REWARD_COLLISION_WEIGHT = -1.0
_REWARD_TERMINATION_WEIGHT = -0.0
_UPRIGHT_REWARD_WEIGHT = 0.0
_ALIVE_BONUS = 0.0
_POSE_PENALTY_WEIGHT = 0.0
_CONTROL_PENALTY_WEIGHT = 0.0
_ANG_VEL_PENALTY_WEIGHT = 0.0
_XZ_VEL_PENALTY_WEIGHT = 0.0
_JOINT_VEL_PENALTY_WEIGHT = 0.0
_ACTION_RATE_PENALTY_WEIGHT = 0.0
_RAW_ACTION_RATE_PENALTY_WEIGHT = 0.0
_TILT_PENALTY_WEIGHT = 0.0
_FOOT_CONTACT_PENALTY_WEIGHT = 0.0
_FOOT_CONTACT_MEAN_WEIGHT = 1.00
_FOOT_CONTACT_WORST_WEIGHT = 2.00
_TARGET_FOOT_LOAD = 25.0
_FOOT_SLIP_PENALTY_WEIGHT = 0.0
_FOOT_SLIP_GATE_TOTAL = 0.03
_FOOT_SLIP_STEP_NOISE_FLOOR = 1e-5
_FOOT_ANCHOR_PENALTY_WEIGHT = 0.0
_FOOT_ANCHOR_DEADBAND = 0.005
_FOOT_ANCHOR_CONTACT_ON_LOAD = 15.0
_FOOT_ANCHOR_CONTACT_OFF_LOAD = 5.0
_FOOT_ANCHOR_CONTACT_OFF_FRAMES = 5
_BASE_DRIFT_PENALTY_WEIGHT = 0.0
_BASE_DRIFT_DEADBAND = 0.01
_CONTACT_SWITCH_PENALTY_WEIGHT = 0.0
_CONTACT_SWITCH_ON_LOAD = 22.0
_CONTACT_SWITCH_OFF_LOAD = 18.0
_ANCHOR_RESET_PENALTY_WEIGHT = 0.0
_ANCHOR_DEACTIVATION_PENALTY_WEIGHT = 0.0
_OBS_ERROR_SCALE = 0.03
_STANDING_QUALITY_START_STEP = 100
_LOAD_QUALITY_RAMP_STEPS = 50
_STANCE_QUALITY_RAMP_STEPS = 100
_MIN_FOOT_LOAD = 20.0
_ANCHOR_LOAD_THRESHOLDS = (20.0, 15.0, 8.0, 5.0)
_RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
_RESET_FOOT_CLEARANCE = 0.005
_RESET_NOISE_COMPONENTS = (
    "combined",
    "joint_pos",
    "joint_vel",
    "roll_pitch",
    "yaw",
    "base_height",
    "base_position",
    "base_velocity",
)
_RESET_NOISE_SPECS = {
    "clean": {
        "base_position_xz": 0.0,
        "base_height": 0.0,
        "roll_pitch": 0.0,
        "yaw": 0.0,
        "joint_pos_by_type": (0.0, 0.0, 0.0),
        "joint_vel": 0.0,
        "base_linear_xz": 0.0,
        "base_linear_y": 0.0,
        "base_angular_xz": 0.0,
        "base_angular_y": 0.0,
    },
    "rn1": {
        "base_position_xz": 0.03,
        "base_height": 0.015,
        "roll_pitch": 0.05,
        "yaw": math.pi,
        "joint_pos_by_type": (0.04, 0.08, 0.10),
        "joint_vel": 0.20,
        "base_linear_xz": 0.10,
        "base_linear_y": 0.03,
        "base_angular_xz": 0.15,
        "base_angular_y": 0.20,
    },
    "rn2": {
        "base_position_xz": 0.10,
        "base_height": 0.030,
        "roll_pitch": 0.12,
        "yaw": math.pi,
        "joint_pos_by_type": (0.10, 0.12, 0.15),
        "joint_vel": 0.50,
        "base_linear_xz": 0.25,
        "base_linear_y": 0.05,
        "base_angular_xz": 0.40,
        "base_angular_y": 0.50,
    },
    "rn3": {
        "base_position_xz": 0.10,
        "base_height": 0.030,
        "roll_pitch": 0.12,
        "yaw": math.pi,
        "joint_pos_by_type": (0.10, 0.12, 0.15),
        "joint_vel": 0.50,
        "base_linear_xz": 0.25,
        "base_linear_y": 0.05,
        "base_angular_xz": 0.40,
        "base_angular_y": 0.50,
    },
}

# Zero action holds this Go1-style mirrored home control pose.
# Joint order is [FR, FL, RR, RL], each with [hip, thigh, calf].
_HOME_JOINT_ANGLES = np.array(
    [
        -0.1, 0.8, -1.5,  # FR
         0.1, 0.8, -1.5,  # FL
        -0.1, 1.0, -1.5,  # RR
         0.1, 1.0, -1.5,  # RL
    ],
    dtype=np.float32,
)
_ACTION_SCALE = 0.25
_HIP_ACTION_SCALE_MULTIPLIER = 0.5
_HIP_ACTION_INDICES = np.array([0, 3, 6, 9], dtype=np.int32)
_ACTUATOR_MODEL = "implicit_limited_drive"
_PD_KP = 20.0
_PD_KD = 0.5
_DRIVE_SHAFT_INERTIA = 0.01

# Joint limits from go1_chrono.urdf, in _JOINT_NAMES order.
_JOINT_LOW = np.tile([-0.863, -0.686, -2.818], 4).astype(np.float32)
_JOINT_HIGH = np.tile([0.863, 4.501, -0.888], 4).astype(np.float32)
_JOINT_VELOCITY_LIMIT = np.tile([30.1, 30.1, 20.06], 4).astype(np.float32)
_JOINT_EFFORT_LIMIT = np.tile([23.7, 23.7, 35.55], 4).astype(np.float32)

# Joint order is shared by actions, observations, limits, and home targets.
# The axis/sign arrays convert Chrono motor-frame rotation vectors back to
# URDF joint angles. After the imported robot's spawn-frame transform, all
# actuated Go1 joints read cleanly on Chrono Z in this PyChrono build.
# At the mirrored Go1 home pose, e.g. FR hip=-0.1 produces rotvec.z=+0.1,
# FL hip=+0.1 produces rotvec.z=-0.1, thigh=+0.8 produces rotvec.z=-0.8,
# and calf=-1.5 produces rotvec.z=+1.5, so sign=-1 maps motor-frame Z back
# to the policy/observation joint coordinate for all joints.
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

# See docs/collision_debug_log.md for why only trunk and feet collide.
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
_FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


def standing_env_metadata() -> dict:
    """Return the public training metadata needed to reproduce standing runs."""
    return {
        "time_step": _TIME_STEP,
        "control_time_step": _TIME_STEP,
        "control_frequency": _CONTROL_FREQUENCY,
        "physics_time_step": _PHYSICS_TIME_STEP,
        "physics_frequency": 1.0 / _PHYSICS_TIME_STEP,
        "physics_substeps": _PHYSICS_SUBSTEPS,
        "observation_dimension": 48,
        "observation_terms": [
            "base_relative_height",
            "trunk_quaternion",
            "base_linear_velocity",
            "base_angular_velocity",
            "joint_positions",
            "joint_velocities",
            "support_frame_base_xz_error",
            "support_frame_foot_anchor_xz_errors",
            "command_vx",
            "command_vz",
            "command_yaw_rate",
        ],
        "command_dimension": 3,
        "command_values": _ZERO_COMMAND.tolist(),
        "command_notes": "Standing uses zero commands; future locomotion will sample nonzero planar velocity and yaw-rate commands.",
        "spawn_height": _SPAWN_HEIGHT,
        "termination_relative_height": _TERM_RELATIVE_HEIGHT,
        "home_joint_angles": _HOME_JOINT_ANGLES.tolist(),
        "action_scale": _ACTION_SCALE,
        "hip_action_scale_multiplier": _HIP_ACTION_SCALE_MULTIPLIER,
        "hip_action_indices": _HIP_ACTION_INDICES.astype(int).tolist(),
        "actuator_model": _ACTUATOR_MODEL,
        "pd_kp": _PD_KP,
        "pd_kd": _PD_KD,
        "drive_shaft_inertia": _DRIVE_SHAFT_INERTIA,
        "joint_effort_limits": _JOINT_EFFORT_LIMIT.tolist(),
        "joint_velocity_limits": _JOINT_VELOCITY_LIMIT.tolist(),
        "collision_bodies": list(_ROBOT_COLLISION_BODIES),
        "reward_weights": {
            "tracking_lin_vel_zero": _REWARD_TRACKING_LIN_VEL_WEIGHT,
            "tracking_ang_vel_zero": _REWARD_TRACKING_ANG_VEL_WEIGHT,
            "lin_vel_y": _REWARD_LIN_VEL_Y_WEIGHT,
            "ang_vel_xz": _REWARD_ANG_VEL_XZ_WEIGHT,
            "orientation": _REWARD_ORIENTATION_WEIGHT,
            "base_height": _REWARD_BASE_HEIGHT_WEIGHT,
            "torques": _REWARD_TORQUES_WEIGHT,
            "dof_acc": _REWARD_DOF_ACC_WEIGHT,
            "action_rate": _REWARD_ACTION_RATE_WEIGHT,
            "dof_pos_limits": _REWARD_DOF_POS_LIMITS_WEIGHT,
            "collision": _REWARD_COLLISION_WEIGHT,
            "termination": _REWARD_TERMINATION_WEIGHT,
            "alive_bonus_diagnostic_only": _ALIVE_BONUS,
            "upright_diagnostic_only": _UPRIGHT_REWARD_WEIGHT,
            "pose_diagnostic_only": _POSE_PENALTY_WEIGHT,
            "control_diagnostic_only": _CONTROL_PENALTY_WEIGHT,
            "joint_velocity_diagnostic_only": _JOINT_VEL_PENALTY_WEIGHT,
            "raw_action_rate_diagnostic_only": _RAW_ACTION_RATE_PENALTY_WEIGHT,
            "tilt_diagnostic_only": _TILT_PENALTY_WEIGHT,
            "angular_velocity_diagnostic_only": _ANG_VEL_PENALTY_WEIGHT,
            "xz_velocity_diagnostic_only": _XZ_VEL_PENALTY_WEIGHT,
            "foot_contact_diagnostic_only": _FOOT_CONTACT_PENALTY_WEIGHT,
            "foot_slip_diagnostic_only": _FOOT_SLIP_PENALTY_WEIGHT,
            "foot_anchor_diagnostic_only": _FOOT_ANCHOR_PENALTY_WEIGHT,
            "base_drift_diagnostic_only": _BASE_DRIFT_PENALTY_WEIGHT,
            "contact_switch_diagnostic_only": _CONTACT_SWITCH_PENALTY_WEIGHT,
            "anchor_reset_diagnostic_only": _ANCHOR_RESET_PENALTY_WEIGHT,
            "anchor_deactivation_diagnostic_only": _ANCHOR_DEACTIVATION_PENALTY_WEIGHT,
        },
        "reward_notes": {
            "style": "legged_gym/walk-these-ways command-conditioned standing adaptation",
            "dt_scaled": True,
            "positive_rewards_only": True,
            "alive_reward": False,
            "command_velocity_sampler": False,
            "command_velocity_inputs": True,
            "old_custom_terms": "diagnostics only with zero reward weights",
        },
        "minimum_foot_load": _MIN_FOOT_LOAD,
        "target_foot_load": _TARGET_FOOT_LOAD,
        "foot_slip_gate_total": _FOOT_SLIP_GATE_TOTAL,
        "foot_slip_step_noise_floor": _FOOT_SLIP_STEP_NOISE_FLOOR,
        "foot_anchor_deadband": _FOOT_ANCHOR_DEADBAND,
        "foot_anchor_contact_on_load": _FOOT_ANCHOR_CONTACT_ON_LOAD,
        "foot_anchor_contact_off_load": _FOOT_ANCHOR_CONTACT_OFF_LOAD,
        "foot_anchor_contact_off_frames": _FOOT_ANCHOR_CONTACT_OFF_FRAMES,
        "base_drift_deadband": _BASE_DRIFT_DEADBAND,
        "contact_switch_on_load": _CONTACT_SWITCH_ON_LOAD,
        "contact_switch_off_load": _CONTACT_SWITCH_OFF_LOAD,
        "observation_error_scale": _OBS_ERROR_SCALE,
        "standing_quality_start_step": _STANDING_QUALITY_START_STEP,
        "load_quality_ramp_steps": _LOAD_QUALITY_RAMP_STEPS,
        "stance_quality_ramp_steps": _STANCE_QUALITY_RAMP_STEPS,
        "reset_noise_levels": _RESET_NOISE_SPECS,
        "reset_noise_components": list(_RESET_NOISE_COMPONENTS),
        "reset_noise_notes": {
            "base_xz_translation": "enabled for RN1/RN2 as a small flat-ground spawn offset",
            "yaw": "enabled for RN1/RN2 as full yaw about Chrono world Y/gravity",
            "joint_velocity": "implemented as small child-link angular velocity perturbations in Chrono",
        },
        "solver": {
            "type": "BARZILAIBORWEIN",
            "max_iterations": 60,
        },
        "contact_materials": {
            "contact_method": _CONTACT_METHOD,
            "composition_rule": _MATERIAL_COMPOSITION_RULE,
            "effective_friction": "min(flat_ground.friction, feet.friction)",
            "static_sliding_note": "SetFriction sets Chrono's friction value used for both static/sliding contact in this setup.",
            "flat_ground": {
                "friction": "sampled from friction_range",
                "restitution": _GROUND_RESTITUTION,
                "Kn": _GROUND_KN,
                "Gn": _GROUND_GN,
                "rolling_friction": _GROUND_ROLLING_FRICTION,
            },
            "feet": {
                "friction": _FOOT_FRICTION,
                "restitution": _FOOT_RESTITUTION,
                "Gn": _FOOT_GN,
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


def _magic_contact_material(
    friction: float,
    restitution: float,
    gn: float,
    kn: float | None = None,
):
    """Create MaGIC-style SMC material for rigid foot/ground contact."""
    material = chrono.ChContactMaterialSMC()
    material.SetFriction(friction)
    material.SetRestitution(restitution)
    material.SetGn(gn)
    if kn is not None:
        material.SetKn(kn)
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
        terrain: str = "flat",
        enable_motors: bool = True,
        friction_range: tuple[float, float] = (0.8, 0.8),
        reset_noise_level: str = "clean",
        reset_noise_components: str = "combined",
        spawn_x: float = 0.0,
        spawn_z: float = 0.0,
        ground_height_offset: float = 0.0,
    ):
        super().__init__()
        if terrain not in ("flat", "scm"):
            raise ValueError("terrain must be 'flat' or 'scm'")
        if reset_noise_level not in _RESET_NOISE_LEVELS:
            raise ValueError(f"reset_noise_level must be one of {_RESET_NOISE_LEVELS}")
        if reset_noise_components not in _RESET_NOISE_COMPONENTS:
            raise ValueError(f"reset_noise_components must be one of {_RESET_NOISE_COMPONENTS}")
        if len(friction_range) != 2:
            raise ValueError("friction_range must be a (min, max) pair")
        friction_min, friction_max = friction_range
        if friction_min <= 0 or friction_max <= 0 or friction_min > friction_max:
            raise ValueError("friction_range must satisfy 0 < min <= max")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terrain_type = terrain
        self.enable_motors = enable_motors
        self.friction_range = (float(friction_min), float(friction_max))
        self.reset_noise_level = reset_noise_level
        self.reset_noise_components = reset_noise_components
        self.spawn_xz = np.array([float(spawn_x), float(spawn_z)], dtype=np.float32)
        self.ground_height_offset = float(ground_height_offset)
        self.ground_friction = None
        self.home_joint_angles = _HOME_JOINT_ANGLES.copy()
        self.command = _ZERO_COMMAND.copy()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._system = None
        self._terrain = None
        self._trunk = None
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
        self._joint_body_pairs = []
        self._vis = None
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._reset_base_xz = np.zeros(2, dtype=np.float32)
        self._base_anchor_xz = np.zeros(2, dtype=np.float32)
        self._base_anchor_yaw = 0.0
        self._foot_anchor_xz = np.zeros((len(_FOOT_BODY_NAMES), 2), dtype=np.float32)
        self._foot_anchor_active = np.zeros(len(_FOOT_BODY_NAMES), dtype=bool)
        self._foot_anchor_off_frames = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._foot_anchor_reset_counts = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._foot_anchor_deactivate_counts = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._prev_foot_xz = np.zeros((len(_FOOT_BODY_NAMES), 2), dtype=np.float32)
        self._reward_contact_active = np.zeros(len(_FOOT_BODY_NAMES), dtype=bool)
        self._foot_load_below_counts = {
            threshold: np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
            for threshold in _ANCHOR_LOAD_THRESHOLDS
        }
        self._standing_reference_captured = False
        self._reset_noise_sample = self._zero_reset_noise_sample()
        self._reset_joint_targets = self.home_joint_angles.copy()
        self.step_count = 0

        self._build_sim()

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _build_sim(self):
        """Tear down and rebuild the entire Chrono sim from scratch.

        Full rebuild is required on every reset because SCMTerrain accumulates
        deformation state that cannot be cleared any other way.
        """
        if self.terrain_type == "flat":
            self.ground_friction = self._sample_ground_friction()

        if self.enable_motors:
            assembly_system, _, assembly_parser = self._build_imported_system(
                parsers.ChParserURDF.ActuationType_POSITION
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
            self._apply_reset_contact_safety_lift(system=assembly_system, feet=assembly_feet)

            system, terrain, parser = self._build_imported_system(None)
            self._copy_body_states(assembly_system, system)
        else:
            system, terrain, parser = self._build_imported_system(None)

        self._cache_robot_handles(system, terrain, parser)
        self._last_motor_targets = self._reset_joint_targets.copy()
        self._last_motor_torques = np.zeros(12, dtype=np.float32)
        self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
        self._apply_reset_velocity_noise()
        self._sync_joint_state_cache(reset_velocity=True)
        self._prev_joint_vel_for_reward = self._last_joint_vel.copy()
        self._prev_foot_xz = self._foot_xz_positions()
        self._reward_contact_active = self._foot_loads() >= _MIN_FOOT_LOAD

        if self.render_mode == "human":
            self._create_visualizer(system)

    def _new_system(self):
        system = chrono.ChSystemSMC()
        system.SetGravityY()
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(60)
        return system

    def _build_imported_system(self, actuation_type):
        system = self._new_system()
        if self.terrain_type == "scm":
            terrain = self._create_scm_terrain(system)
        else:
            terrain = None
            self._add_flat_ground(system)

        parser = self._create_robot_parser(actuation_type)
        parser.PopulateSystem(system)
        self._configure_imported_bodies(system, parser)
        return system, terrain, parser

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

    def _zero_reset_noise_sample(self) -> dict:
        zeros12 = [0.0] * 12
        return {
            "enabled": False,
            "level": "clean",
            "components": "combined",
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
            "contact_safety_clearance": _RESET_FOOT_CLEARANCE,
            "contact_safety_min_clearance_before_lift": 0.0,
            "contact_safety_lift": 0.0,
            "contact_safety_min_clearance_after_lift": 0.0,
            "initial_foot_y": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "initial_min_foot_y": 0.0,
            "initial_max_foot_y": 0.0,
            "initial_foot_loads": {name.split("_")[0]: 0.0 for name in _FOOT_BODY_NAMES},
            "initial_max_foot_load": 0.0,
        }

    def _component_enabled(self, component: str) -> bool:
        return self.reset_noise_components in ("combined", component)

    def _sample_reset_uniform(self, limit: float, size=None):
        if limit <= 0.0:
            return np.zeros(size, dtype=np.float32) if size is not None else 0.0
        return self.np_random.uniform(-limit, limit, size=size)

    def _sample_reset_noise(self) -> None:
        spec = _RESET_NOISE_SPECS[self.reset_noise_level]
        sample = self._zero_reset_noise_sample()
        sample["enabled"] = self.reset_noise_level != "clean"
        sample["level"] = self.reset_noise_level
        sample["components"] = self.reset_noise_components

        if self._component_enabled("base_position"):
            limit = spec["base_position_xz"]
            if self.reset_noise_level == "rn3":
                sample["base_position_offset_x"] = float(limit)
                sample["base_position_offset_z"] = float(limit)
            else:
                sample["base_position_offset_x"] = float(self._sample_reset_uniform(limit))
                sample["base_position_offset_z"] = float(self._sample_reset_uniform(limit))
            sample["base_position_offset"] = [
                sample["base_position_offset_x"],
                sample["base_position_offset_z"],
            ]
        if self._component_enabled("base_height"):
            if self.reset_noise_level == "rn3":
                sample["base_height_offset"] = float(spec["base_height"])
            else:
                sample["base_height_offset"] = float(self._sample_reset_uniform(spec["base_height"]))
        if self._component_enabled("roll_pitch"):
            limit = spec["roll_pitch"]
            if self.reset_noise_level == "rn3":
                sample["roll"] = float(limit)
                sample["pitch"] = float(limit)
            else:
                sample["roll"] = float(self._sample_reset_uniform(limit))
                sample["pitch"] = float(self._sample_reset_uniform(limit))
        if self._component_enabled("yaw"):
            if self.reset_noise_level == "rn3":
                sample["yaw"] = float(spec["yaw"])
            else:
                sample["yaw"] = float(self._sample_reset_uniform(spec["yaw"]))
        if self._component_enabled("joint_pos"):
            hip, thigh, calf = spec["joint_pos_by_type"]
            per_joint_limits = np.tile([hip, thigh, calf], 4).astype(np.float32)
            if self.reset_noise_level == "rn3":
                offsets = per_joint_limits.copy()
            else:
                offsets = self.np_random.uniform(
                    -per_joint_limits,
                    per_joint_limits,
                ).astype(np.float32)
            targets = self.home_joint_angles + offsets
            targets = np.clip(targets, _JOINT_LOW, _JOINT_HIGH)
            offsets = targets - self.home_joint_angles
            sample["joint_position_offsets"] = offsets.astype(float).tolist()
            self._reset_joint_targets = targets.astype(np.float32)
        else:
            self._reset_joint_targets = self.home_joint_angles.copy()
        if self._component_enabled("joint_vel"):
            limit = spec["joint_vel"]
            if self.reset_noise_level == "rn3":
                sample["joint_velocity_offsets"] = [float(limit)] * 12
            else:
                sample["joint_velocity_offsets"] = (
                    self._sample_reset_uniform(limit, size=12).astype(float).tolist()
                )
        if self._component_enabled("base_velocity"):
            xz = spec["base_linear_xz"]
            y = spec["base_linear_y"]
            angular = spec["base_angular_xz"]
            angular_y = spec["base_angular_y"]
            if self.reset_noise_level == "rn3":
                sample["base_linear_velocity"] = [float(xz), float(y), float(xz)]
                sample["base_angular_velocity"] = [float(angular), float(angular_y), float(angular)]
            else:
                sample["base_linear_velocity"] = [
                    float(self._sample_reset_uniform(xz)),
                    float(self._sample_reset_uniform(y)),
                    float(self._sample_reset_uniform(xz)),
                ]
                sample["base_angular_velocity"] = [
                    float(self._sample_reset_uniform(angular)),
                    float(self._sample_reset_uniform(angular_y)),
                    float(self._sample_reset_uniform(angular)),
                ]
        self._reset_noise_sample = sample

    def _sample_ground_friction(self) -> float:
        friction_min, friction_max = self.friction_range
        return float(self.np_random.uniform(friction_min, friction_max))

    def _create_scm_terrain(self, system):
        self.ground_friction = None
        terrain = veh.SCMTerrain(system)
        # SCMTerrain's native frame is Z-up. Rotate it to match the Y-up
        # robot/world convention used throughout this repo.
        terrain.SetReferenceFrame(
            chrono.ChCoordsysd(
                chrono.ChVector3d(0, self.ground_height_offset, 0),
                chrono.QuatFromAngleX(-math.pi / 2),
            )
        )
        terrain.SetSoilParameters(
            0.2e6,  # Bekker Kphi
            0,      # Bekker Kc
            1.1,    # Bekker n
            0,      # Mohr cohesion (Pa)
            30,     # Mohr friction (deg)
            0.01,   # Janosi shear coeff (m)
            4e7,    # elastic stiffness (Pa/m)
            3e4,    # damping (Pa s/m)
        )
        terrain.Initialize(_TERRAIN_LENGTH, _TERRAIN_WIDTH, _TERRAIN_DELTA)
        return terrain

    def _add_flat_ground(self, system) -> None:
        # Sampled flat-ground friction is the domain-randomization knob. Foot
        # friction is set high enough that it does not cap the target range.
        ground_mat = _magic_contact_material(
            friction=self.ground_friction,
            restitution=_GROUND_RESTITUTION,
            gn=_GROUND_GN,
            kn=_GROUND_KN,
        )
        ground_mat.SetRollingFriction(_GROUND_ROLLING_FRICTION)

        ground = chrono.ChBodyEasyBox(10, 0.2, 10, 1000, True, True, ground_mat)
        ground.SetFixed(True)
        ground.SetPos(chrono.ChVector3d(0, -0.1 + self.ground_height_offset, 0))
        _set_visual_color(ground, chrono.ChColor(0.05, 0.05, 0.05))
        system.AddBody(ground)

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

    def _to_support_frame(self, vectors_xz: np.ndarray) -> np.ndarray:
        """Rotate world X/Z vectors into the captured standing support frame."""
        vectors = np.asarray(vectors_xz, dtype=np.float32)
        yaw = -float(self._base_anchor_yaw)
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return vectors @ rot.T

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

    def _create_robot_parser(self, actuation_type=None):
        parser = parsers.ChParserURDF(str(_URDF))
        parser.EnableCollisionVisualization()
        reset_roll = float(self._reset_noise_sample["roll"])
        reset_pitch = float(self._reset_noise_sample["pitch"])
        reset_yaw = float(self._reset_noise_sample["yaw"])
        reset_height = (
            self._ground_top_y()
            + _SPAWN_HEIGHT
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
                _set_visual_color(body, chrono.ChColor(0.2, 0.45, 0.85))

        for name in _ROBOT_COLLISION_BODIES:
            body = parser.GetChBody(name)
            if body is not None:
                body.EnableCollision(True)

        foot_mat = _magic_contact_material(
            friction=_FOOT_FRICTION,
            restitution=_FOOT_RESTITUTION,
            gn=_FOOT_GN,
        )
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            body = parser.GetChBody(name)
            if body is not None:
                body.GetCollisionModel().SetAllShapesMaterial(foot_mat)

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
            links_by_name = {link.GetName(): chrono.CastToChLink(link) for link in system.GetLinks()}
            self._motors = [links_by_name[name] for name in _JOINT_NAMES]
            self._create_implicit_limited_drives(system)
        else:
            self._motors = []
            self._motor_funcs = []

        # ChLinkMotor's direct angle accessors are not exposed in this PyChrono
        # build. Motor frame rotation still gives the assembled joint angle;
        # linked body pairs are kept only for the approximate velocity term.
        self._joint_body_pairs = (
            [(motor.GetBody1(), motor.GetBody2()) for motor in self._motors]
            if self.enable_motors else []
        )

    def _create_implicit_limited_drives(self, system) -> None:
        self._motor_funcs = []
        for index, (name, joint_link, effort_limit) in enumerate(
            zip(_JOINT_NAMES, self._motors, _JOINT_EFFORT_LIMIT)
        ):
            drive = chrono.ChLinkMotorRotationDriveline()
            drive.SetName(f"{name}_implicit_drive")
            drive.SetSpindleConstraint(chrono.ChLinkMotorRotation.SpindleConstraint_FREE)
            drive.Initialize(
                joint_link.GetBody1(),
                joint_link.GetBody2(),
                joint_link.GetFrame1Abs(),
            )
            system.Add(drive)

            drive_shaft = chrono.ChShaft()
            drive_shaft.SetName(f"{name}_drive_shaft")
            drive_shaft.SetInertia(_DRIVE_SHAFT_INERTIA)
            system.Add(drive_shaft)

            clutch = chrono.ChShaftsClutch()
            clutch.SetName(f"{name}_torque_limit")
            if not clutch.Initialize(drive.GetInnerShaft1(), drive_shaft):
                raise RuntimeError(f"failed to initialize clutch for {name}")
            clutch.SetTorqueLimit(float(effort_limit))
            system.Add(clutch)

            speed_motor = chrono.ChShaftsMotorSpeed()
            speed_motor.SetName(f"{name}_speed_drive")
            if not speed_motor.Initialize(drive_shaft, drive.GetInnerShaft2()):
                raise RuntimeError(f"failed to initialize speed drive for {name}")
            function = chrono.ChFunctionConst(0.0)
            speed_motor.SetSpeedFunction(function)
            system.Add(speed_motor)

            self._drive_links.append(drive)
            self._drive_shafts.append(drive_shaft)
            self._drive_clutches.append(clutch)
            self._drive_motors.append(speed_motor)
            self._motor_funcs.append(function)

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

    def _sync_joint_state_cache(self, reset_velocity: bool = False) -> None:
        joint_pos = self._read_joint_angles()
        if reset_velocity:
            joint_vel = np.zeros(12, dtype=np.float32)
        else:
            joint_vel = ((joint_pos - self._last_joint_pos) / _TIME_STEP).astype(np.float32)
        self._last_joint_pos = joint_pos
        self._last_joint_vel = joint_vel

    def _min_foot_clearance(self, feet=None) -> float:
        feet = self._feet if feet is None else feet
        if not feet:
            return 0.0
        ground_y = self._ground_top_y()
        return float(min(float(foot.GetPos().y) - ground_y for foot in feet))

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

    def _apply_motor_targets(self, executed_action: np.ndarray) -> np.ndarray:
        if not self.enable_motors:
            self._last_motor_targets = np.zeros(12, dtype=np.float32)
            self._last_motor_torques = np.zeros(12, dtype=np.float32)
            self._last_torque_limit_fraction = np.zeros(12, dtype=np.float32)
            return self._last_motor_targets.copy()

        desired_targets = np.clip(
            self.home_joint_angles + self._scaled_action_offsets(executed_action),
            _JOINT_LOW,
            _JOINT_HIGH,
        ).astype(np.float32)
        joint_pos = self._last_joint_pos
        joint_vel = self._last_joint_vel
        desired_speed = (_PD_KP * (desired_targets - joint_pos) - _PD_KD * joint_vel).astype(np.float32)
        for function, speed in zip(self._motor_funcs, desired_speed):
            function.SetConstant(float(speed))
        self._last_motor_targets = desired_targets.copy()
        return desired_targets

    def _update_actuator_load_cache(self) -> None:
        if not self.enable_motors or not self._drive_links:
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
        terms: dict[str, float | str] = {
            "actuator_model": _ACTUATOR_MODEL,
            "pd_kp": _PD_KP,
            "pd_kd": _PD_KD,
            "action_scale": _ACTION_SCALE,
            "hip_action_scale_multiplier": _HIP_ACTION_SCALE_MULTIPLIER,
            "mean_abs_motor_target": float(np.mean(np.abs(self._last_motor_targets))),
            "max_abs_motor_target": float(np.max(np.abs(self._last_motor_targets))),
            "mean_abs_motor_torque": float(np.mean(np.abs(self._last_motor_torques))),
            "max_abs_motor_torque": float(np.max(np.abs(self._last_motor_torques))),
            "mean_torque_limit_fraction": float(np.mean(self._last_torque_limit_fraction)),
            "max_torque_limit_fraction": float(np.max(self._last_torque_limit_fraction)),
            "fraction_torque_saturated": saturation_fraction,
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
        return terms

    def _record_initial_reset_diagnostics(self) -> None:
        foot_y = {}
        foot_loads = {}
        for name, foot in zip(_FOOT_BODY_NAMES, self._feet):
            leg = name.split("_")[0]
            foot_y[leg] = float(foot.GetPos().y)
            foot_loads[leg] = abs(float(foot.GetContactForce().y))
        self._reset_noise_sample["initial_foot_y"] = foot_y
        self._reset_noise_sample["initial_min_foot_y"] = float(min(foot_y.values(), default=0.0))
        self._reset_noise_sample["initial_max_foot_y"] = float(max(foot_y.values(), default=0.0))
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

    def _get_obs(self) -> np.ndarray:
        rot = self._trunk.GetRot()  # Chrono stores w, x, y, z as e0..e3.
        lin_vel = self._trunk.GetPosDt()
        ang_vel = self._trunk.GetAngVelParent()

        joint_pos = self._last_joint_pos if self._motors else np.zeros(12, dtype=np.float32)
        joint_vel = self._last_joint_vel if self._motors else np.zeros(12, dtype=np.float32)

        if self._standing_reference_captured:
            base_world = self._base_world_pos()
            base_xz = np.array([base_world[0], base_world[2]], dtype=np.float32)
            base_error = self._to_support_frame(base_xz - self._base_anchor_xz) / _OBS_ERROR_SCALE
            foot_errors = self._to_support_frame(self._foot_xz_positions() - self._foot_anchor_xz)
            foot_errors[~self._foot_anchor_active] = 0.0
            foot_errors = foot_errors.reshape(-1) / _OBS_ERROR_SCALE
        else:
            base_error = np.zeros(2, dtype=np.float32)
            foot_errors = np.zeros(8, dtype=np.float32)

        return np.concatenate([
            [self._base_relative_height()],
            [rot.e0, rot.e1, rot.e2, rot.e3],
            [lin_vel.x, lin_vel.y, lin_vel.z],
            [ang_vel.x, ang_vel.y, ang_vel.z],
            joint_pos,
            joint_vel,
            base_error,
            foot_errors,
            self.command,
        ]).astype(np.float32)

    def _trunk_axis_alignments(self) -> dict[str, float]:
        """Return each trunk local axis alignment with Chrono world Y-up."""
        rot = self._trunk.GetRot()
        return {
            "trunk_x_up": float(np.clip(rot.Rotate(chrono.ChVector3d(1, 0, 0)).y, -1.0, 1.0)),
            "trunk_y_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 1, 0)).y, -1.0, 1.0)),
            "trunk_z_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 0, 1)).y, -1.0, 1.0)),
        }

    def _trunk_reward_terms(self, obs: np.ndarray) -> tuple[float, dict]:
        axis_alignments = self._trunk_axis_alignments()
        upright_score = max(0.0, axis_alignments["trunk_z_up"])
        upright_reward = _UPRIGHT_REWARD_WEIGHT * upright_score
        tilt_error = (
            axis_alignments["trunk_x_up"] ** 2
            + axis_alignments["trunk_y_up"] ** 2
        )
        tilt_penalty = _TILT_PENALTY_WEIGHT * float(tilt_error)
        terms = {
            "alive_bonus": _ALIVE_BONUS,
            "upright_score": float(upright_score),
            "upright_reward": float(upright_reward),
            "trunk_y": float(self._trunk.GetPos().y),
            "base_world_y": float(self._trunk.GetPos().y),
            "base_relative_height": float(obs[0]),
            "ground_top_y": self._ground_top_y(),
            "tilt_error": float(tilt_error),
            "tilt_penalty": float(tilt_penalty),
        }
        terms.update(axis_alignments)
        penalty = tilt_penalty
        reward = _ALIVE_BONUS + upright_reward - penalty
        return float(reward), terms

    def _pose_reward_terms(self, obs: np.ndarray) -> tuple[float, dict]:
        joint_pos = obs[11:23]
        pose_error = joint_pos - self.home_joint_angles
        pose_mse = float(np.mean(pose_error ** 2))
        pose_penalty = _POSE_PENALTY_WEIGHT * pose_mse

        fr = joint_pos[0:3]
        fl = joint_pos[3:6]
        rr = joint_pos[6:9]
        rl = joint_pos[9:12]
        leg_symmetry_error = 0.5 * (
            float(np.mean((fr - fl) ** 2))
            + float(np.mean((rr - rl) ** 2))
        )
        terms = {
            "pose_penalty": float(pose_penalty),
            "pose_error": pose_mse,
            "leg_symmetry_error": float(leg_symmetry_error),
        }
        return -float(pose_penalty), terms

    def _foot_contact_terms(self) -> tuple[float, dict]:
        foot_loads = self._foot_loads()
        load_quality_scale = float(np.clip(self.step_count / _LOAD_QUALITY_RAMP_STEPS, 0.0, 1.0))
        missing_contact = np.maximum(0.0, _TARGET_FOOT_LOAD - foot_loads) / _TARGET_FOOT_LOAD
        missing_squared = missing_contact ** 2
        foot_contact_mean_error = float(np.mean(missing_squared))
        foot_contact_worst_error = float(np.max(missing_squared))
        foot_contact_error = (
            _FOOT_CONTACT_MEAN_WEIGHT * foot_contact_mean_error
            + _FOOT_CONTACT_WORST_WEIGHT * foot_contact_worst_error
        )
        foot_contact_penalty = _FOOT_CONTACT_PENALTY_WEIGHT * load_quality_scale * foot_contact_error
        terms = {
            "foot_contact_error": float(foot_contact_error),
            "foot_contact_mean_error": float(foot_contact_mean_error),
            "foot_contact_worst_error": float(foot_contact_worst_error),
            "foot_contact_penalty": float(foot_contact_penalty),
            "load_quality_scale": float(load_quality_scale),
            "min_foot_load": float(np.min(foot_loads)),
            "mean_foot_load": float(np.mean(foot_loads)),
        }
        return -float(foot_contact_penalty), terms

    def _foot_xz_positions(self) -> np.ndarray:
        return np.array(
            [
                [float(foot.GetPos().x), float(foot.GetPos().z)]
                for foot in self._feet
            ],
            dtype=np.float32,
        )

    def _capture_standing_reference(self, obs: np.ndarray, foot_loads: np.ndarray) -> None:
        base_world = self._base_world_pos()
        self._base_anchor_xz = np.array([base_world[0], base_world[2]], dtype=np.float32)
        self._base_anchor_yaw = self._trunk_yaw()
        self._foot_anchor_xz = self._foot_xz_positions()
        self._foot_anchor_active = foot_loads >= _FOOT_ANCHOR_CONTACT_ON_LOAD
        self._foot_anchor_off_frames.fill(0)
        self._foot_anchor_reset_counts += self._foot_anchor_active.astype(np.int32)
        self._standing_reference_captured = True

    def _anchor_diagnostic_terms(
        self,
        foot_loads: np.ndarray,
        foot_displacements: np.ndarray,
        foot_errors: np.ndarray,
        base_xz: np.ndarray,
    ) -> dict:
        terms = {
            "standing_reference_captured": float(self._standing_reference_captured),
            "base_ref_x": float(self._base_anchor_xz[0]),
            "base_ref_z": float(self._base_anchor_xz[1]),
            "base_reset_x": float(self._reset_base_xz[0]),
            "base_reset_z": float(self._reset_base_xz[1]),
            "base_drift_from_reset": float(np.linalg.norm(base_xz - self._reset_base_xz)),
            "foot_anchor_total_resets": float(np.sum(self._foot_anchor_reset_counts)),
            "foot_anchor_total_deactivations": float(np.sum(self._foot_anchor_deactivate_counts)),
        }
        for index, name in enumerate(_FOOT_BODY_NAMES):
            leg = name.split("_")[0]
            terms[f"foot_anchor_active_{leg}"] = float(self._foot_anchor_active[index])
            terms[f"foot_anchor_load_{leg}"] = float(foot_loads[index])
            terms[f"foot_anchor_displacement_{leg}"] = float(foot_displacements[index])
            terms[f"foot_anchor_error_{leg}"] = float(foot_errors[index])
            terms[f"foot_anchor_reset_count_{leg}"] = float(self._foot_anchor_reset_counts[index])
            terms[f"foot_anchor_deactivate_count_{leg}"] = float(self._foot_anchor_deactivate_counts[index])
            terms[f"foot_anchor_off_frames_{leg}"] = float(self._foot_anchor_off_frames[index])
            terms[f"foot_anchor_ref_x_{leg}"] = float(self._foot_anchor_xz[index, 0])
            terms[f"foot_anchor_ref_z_{leg}"] = float(self._foot_anchor_xz[index, 1])
            for threshold in _ANCHOR_LOAD_THRESHOLDS:
                label = str(int(threshold))
                terms[f"foot_load_below_{label}n_frames_{leg}"] = float(
                    self._foot_load_below_counts[threshold][index]
                )
        return terms

    def _settled_standing_quality_terms(self, obs: np.ndarray) -> tuple[float, dict]:
        foot_loads = self._foot_loads()
        for threshold in _ANCHOR_LOAD_THRESHOLDS:
            self._foot_load_below_counts[threshold] += (foot_loads < threshold).astype(np.int32)

        base_world = self._base_world_pos()
        base_xz = np.array([base_world[0], base_world[2]], dtype=np.float32)
        foot_xz_positions = self._foot_xz_positions()
        foot_step_displacements = np.linalg.norm(foot_xz_positions - self._prev_foot_xz, axis=1)
        loaded_step_slip = (
            (foot_loads >= _MIN_FOOT_LOAD).astype(np.float32)
            * np.maximum(0.0, foot_step_displacements - _FOOT_SLIP_STEP_NOISE_FLOOR)
        )
        foot_slip_step = float(np.sum(loaded_step_slip))
        foot_slip_error = float(foot_slip_step / _FOOT_SLIP_GATE_TOTAL)
        foot_slip_penalty_unscaled = _FOOT_SLIP_PENALTY_WEIGHT * foot_slip_error

        contact_switches_this_step = 0
        next_contact_active = self._reward_contact_active.copy()
        for index, load in enumerate(foot_loads):
            if next_contact_active[index]:
                if load <= _CONTACT_SWITCH_OFF_LOAD:
                    next_contact_active[index] = False
                    contact_switches_this_step += 1
            elif load >= _CONTACT_SWITCH_ON_LOAD:
                next_contact_active[index] = True
                contact_switches_this_step += 1
        self._reward_contact_active = next_contact_active

        foot_displacements = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        foot_errors = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        foot_anchor_excess = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        new_anchor_resets = 0
        new_anchor_deactivations = 0
        stance_quality_scale = float(np.clip(self.step_count / _STANCE_QUALITY_RAMP_STEPS, 0.0, 1.0))

        if (
            not self._standing_reference_captured
            and self.step_count >= _STANDING_QUALITY_START_STEP
        ):
            self._capture_standing_reference(obs, foot_loads)

        foot_anchor_max_displacement = 0.0
        if self._standing_reference_captured:
            for index, (foot_xz, load) in enumerate(zip(foot_xz_positions, foot_loads)):
                if self._foot_anchor_active[index]:
                    if load <= _FOOT_ANCHOR_CONTACT_OFF_LOAD:
                        self._foot_anchor_off_frames[index] += 1
                        if self._foot_anchor_off_frames[index] >= _FOOT_ANCHOR_CONTACT_OFF_FRAMES:
                            self._foot_anchor_active[index] = False
                            self._foot_anchor_deactivate_counts[index] += 1
                            new_anchor_deactivations += 1
                    else:
                        self._foot_anchor_off_frames[index] = 0
                    displacement = float(np.linalg.norm(foot_xz - self._foot_anchor_xz[index]))
                    foot_displacements[index] = displacement
                    foot_anchor_max_displacement = max(foot_anchor_max_displacement, displacement)
                    raw_error = max(0.0, displacement - _FOOT_ANCHOR_DEADBAND)
                    foot_errors[index] = raw_error
                    foot_anchor_excess[index] = max(0.0, displacement / _FOOT_ANCHOR_DEADBAND - 1.0)
                elif load >= _FOOT_ANCHOR_CONTACT_ON_LOAD:
                    self._foot_anchor_xz[index] = foot_xz
                    self._foot_anchor_active[index] = True
                    self._foot_anchor_off_frames[index] = 0
                    self._foot_anchor_reset_counts[index] += 1
                    new_anchor_resets += 1

        foot_anchor_error = float(np.mean(foot_anchor_excess ** 2))
        foot_anchor_penalty_unscaled = _FOOT_ANCHOR_PENALTY_WEIGHT * foot_anchor_error

        if self._standing_reference_captured:
            base_drift = float(np.linalg.norm(base_xz - self._base_anchor_xz))
        else:
            base_drift = 0.0
        base_drift_error = max(0.0, base_drift - _BASE_DRIFT_DEADBAND)
        base_drift_normalized_error = max(0.0, base_drift / _BASE_DRIFT_DEADBAND - 1.0)
        base_drift_penalty_unscaled = _BASE_DRIFT_PENALTY_WEIGHT * float(base_drift_normalized_error ** 2)
        contact_switch_penalty_unscaled = _CONTACT_SWITCH_PENALTY_WEIGHT * contact_switches_this_step
        anchor_reset_penalty_unscaled = _ANCHOR_RESET_PENALTY_WEIGHT * new_anchor_resets
        anchor_deactivation_penalty_unscaled = _ANCHOR_DEACTIVATION_PENALTY_WEIGHT * new_anchor_deactivations

        foot_slip_penalty = stance_quality_scale * foot_slip_penalty_unscaled
        foot_anchor_penalty = stance_quality_scale * foot_anchor_penalty_unscaled
        base_drift_penalty = stance_quality_scale * base_drift_penalty_unscaled
        contact_switch_penalty = stance_quality_scale * contact_switch_penalty_unscaled
        anchor_reset_penalty = stance_quality_scale * anchor_reset_penalty_unscaled
        anchor_deactivation_penalty = stance_quality_scale * anchor_deactivation_penalty_unscaled

        terms = {
            "stance_quality_scale": float(stance_quality_scale),
            "foot_slip_step": float(foot_slip_step),
            "foot_slip_error": float(foot_slip_error),
            "foot_slip_penalty": float(foot_slip_penalty),
            "foot_slip_penalty_unscaled": float(foot_slip_penalty_unscaled),
            "base_drift": float(base_drift),
            "base_drift_error": float(base_drift_error),
            "base_drift_normalized_error": float(base_drift_normalized_error),
            "base_drift_penalty": float(base_drift_penalty),
            "base_drift_penalty_unscaled": float(base_drift_penalty_unscaled),
            "foot_anchor_error": float(foot_anchor_error),
            "foot_anchor_normalized_error": float(foot_anchor_error),
            "foot_anchor_penalty": float(foot_anchor_penalty),
            "foot_anchor_penalty_unscaled": float(foot_anchor_penalty_unscaled),
            "foot_anchor_active_count": float(np.sum(self._foot_anchor_active)),
            "foot_anchor_max_displacement": float(foot_anchor_max_displacement),
            "contact_switches_this_step": float(contact_switches_this_step),
            "contact_switch_penalty": float(contact_switch_penalty),
            "contact_switch_penalty_unscaled": float(contact_switch_penalty_unscaled),
            "anchor_resets_this_step": float(new_anchor_resets),
            "anchor_deactivations_this_step": float(new_anchor_deactivations),
            "anchor_reset_penalty": float(anchor_reset_penalty),
            "anchor_deactivation_penalty": float(anchor_deactivation_penalty),
        }
        terms.update(self._anchor_diagnostic_terms(foot_loads, foot_displacements, foot_errors, base_xz))
        self._prev_foot_xz = foot_xz_positions.copy()
        penalty = (
            foot_slip_penalty
            + foot_anchor_penalty
            + base_drift_penalty
            + contact_switch_penalty
            + anchor_reset_penalty
            + anchor_deactivation_penalty
        )
        return -float(penalty), terms

    def _motion_reward_terms(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        executed_action_delta: np.ndarray,
        raw_action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        trunk_lin_vel = obs[5:8]
        trunk_ang_vel = obs[8:11]
        joint_vel = obs[23:35]
        joint_vel_penalty = _JOINT_VEL_PENALTY_WEIGHT * float(np.mean(joint_vel ** 2))
        action_rate_penalty = _ACTION_RATE_PENALTY_WEIGHT * float(np.mean(executed_action_delta ** 2))
        raw_action_rate_penalty = _RAW_ACTION_RATE_PENALTY_WEIGHT * float(np.mean(raw_action_delta ** 2))
        control_penalty = _CONTROL_PENALTY_WEIGHT * float(np.mean(executed_action ** 2))
        ang_vel_penalty = _ANG_VEL_PENALTY_WEIGHT * float(np.mean(trunk_ang_vel ** 2))
        xz_vel = trunk_lin_vel[[0, 2]]
        xz_vel_penalty = _XZ_VEL_PENALTY_WEIGHT * float(np.mean(xz_vel ** 2))

        terms = {
            "mean_abs_joint_vel": float(np.mean(np.abs(joint_vel))),
            "max_abs_joint_vel": float(np.max(np.abs(joint_vel))),
            "joint_vel_penalty": float(joint_vel_penalty),
            "action_rate_penalty": float(action_rate_penalty),
            "raw_action_rate_penalty": float(raw_action_rate_penalty),
            "mean_abs_action_delta": float(np.mean(np.abs(executed_action_delta))),
            "max_abs_action_delta": float(np.max(np.abs(executed_action_delta))),
            "control_penalty": float(control_penalty),
            "mean_abs_action": float(np.mean(np.abs(executed_action))),
            "max_abs_action": float(np.max(np.abs(executed_action))),
            "ang_vel_penalty": float(ang_vel_penalty),
            "mean_abs_ang_vel": float(np.mean(np.abs(trunk_ang_vel))),
            "max_abs_ang_vel": float(np.max(np.abs(trunk_ang_vel))),
            "ang_vel_x": float(trunk_ang_vel[0]),
            "ang_vel_y": float(trunk_ang_vel[1]),
            "ang_vel_z": float(trunk_ang_vel[2]),
            "xz_vel_penalty": float(xz_vel_penalty),
            "mean_abs_xz_vel": float(np.mean(np.abs(xz_vel))),
            "max_abs_xz_vel": float(np.max(np.abs(xz_vel))),
            "lin_vel_x": float(trunk_lin_vel[0]),
            "lin_vel_z": float(trunk_lin_vel[2]),
        }
        penalty = (
            joint_vel_penalty
            + action_rate_penalty
            + raw_action_rate_penalty
            + control_penalty
            + ang_vel_penalty
            + xz_vel_penalty
        )
        return -float(penalty), terms

    def _source_style_reward_terms(
        self,
        obs: np.ndarray,
        executed_action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        trunk_lin_vel = obs[5:8]
        trunk_ang_vel = obs[8:11]
        joint_pos = obs[11:23]
        joint_vel = obs[23:35]
        axis_alignments = self._trunk_axis_alignments()

        command_vx, command_vz, command_yaw_rate = (float(value) for value in obs[45:48])
        lin_vel_error_x = float(trunk_lin_vel[0] - command_vx)
        lin_vel_error_z = float(trunk_lin_vel[2] - command_vz)
        yaw_rate_error = float(trunk_ang_vel[1] - command_yaw_rate)

        tracking_lin_vel_zero = float(
            math.exp(-float(lin_vel_error_x ** 2 + lin_vel_error_z ** 2) / _TRACKING_SIGMA)
        )
        tracking_ang_vel_zero = float(
            math.exp(-float(yaw_rate_error ** 2) / _TRACKING_SIGMA)
        )
        lin_vel_y = float(trunk_lin_vel[1] ** 2)
        ang_vel_xz = float(trunk_ang_vel[0] ** 2 + trunk_ang_vel[2] ** 2)
        orientation = float(axis_alignments["trunk_x_up"] ** 2 + axis_alignments["trunk_y_up"] ** 2)
        base_height = float((obs[0] - _BASE_HEIGHT_TARGET) ** 2)
        torques = float(np.sum(self._last_motor_torques ** 2))
        dof_acc = float(np.sum(((joint_vel - self._prev_joint_vel_for_reward) / _TIME_STEP) ** 2))
        action_rate = float(np.sum(executed_action_delta ** 2))
        lower_violation = np.maximum(_JOINT_LOW - joint_pos, 0.0)
        upper_violation = np.maximum(joint_pos - _JOINT_HIGH, 0.0)
        dof_pos_limits = float(np.sum(lower_violation + upper_violation))
        collision = float(self._max_nonfoot_load() > 1e-6)

        weighted_terms = {
            "tracking_lin_vel_zero": _REWARD_TRACKING_LIN_VEL_WEIGHT * tracking_lin_vel_zero,
            "tracking_ang_vel_zero": _REWARD_TRACKING_ANG_VEL_WEIGHT * tracking_ang_vel_zero,
            "lin_vel_y": _REWARD_LIN_VEL_Y_WEIGHT * lin_vel_y,
            "ang_vel_xz": _REWARD_ANG_VEL_XZ_WEIGHT * ang_vel_xz,
            "orientation": _REWARD_ORIENTATION_WEIGHT * orientation,
            "base_height": _REWARD_BASE_HEIGHT_WEIGHT * base_height,
            "torques": _REWARD_TORQUES_WEIGHT * torques,
            "dof_acc": _REWARD_DOF_ACC_WEIGHT * dof_acc,
            "action_rate": _REWARD_ACTION_RATE_WEIGHT * action_rate,
            "dof_pos_limits": _REWARD_DOF_POS_LIMITS_WEIGHT * dof_pos_limits,
            "collision": _REWARD_COLLISION_WEIGHT * collision,
        }
        raw_reward = float(sum(weighted_terms.values()))
        reward = float(max(_TIME_STEP * raw_reward, 0.0))
        terms = {
            "reward_raw_sum": raw_reward,
            "reward_dt_scaled": float(_TIME_STEP * raw_reward),
            "tracking_sigma": _TRACKING_SIGMA,
            "base_height_target": _BASE_HEIGHT_TARGET,
            "tracking_lin_vel_zero": tracking_lin_vel_zero,
            "tracking_ang_vel_zero": tracking_ang_vel_zero,
            "command_vx": command_vx,
            "command_vz": command_vz,
            "command_yaw_rate": command_yaw_rate,
            "lin_vel_error_x": lin_vel_error_x,
            "lin_vel_error_z": lin_vel_error_z,
            "yaw_rate_error": yaw_rate_error,
            "lin_vel_y_error": lin_vel_y,
            "ang_vel_xz_error": ang_vel_xz,
            "orientation_error": orientation,
            "base_height_error": base_height,
            "torques_error": torques,
            "dof_acc_error": dof_acc,
            "action_rate_error": action_rate,
            "dof_pos_limits_error": dof_pos_limits,
            "collision_error": collision,
            "nonfoot_collision": collision,
        }
        for name, weighted_value in weighted_terms.items():
            terms[f"{name}_reward"] = float(_TIME_STEP * weighted_value)
            terms[f"{name}_weighted"] = float(weighted_value)
        terms.update(axis_alignments)
        self._prev_joint_vel_for_reward = joint_vel.copy()
        return reward, terms

    def _standing_reward(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        executed_action_delta: np.ndarray,
        raw_action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        """Standing reward: survive, stay upright, and keep four-foot support."""
        if not np.all(np.isfinite(obs)):
            return -10.0, {"invalid_obs": 1.0}

        terms = {}
        reward, source_terms = self._source_style_reward_terms(obs, executed_action_delta)
        terms.update(source_terms)
        for _delta, delta_terms in (
            self._trunk_reward_terms(obs),
            self._pose_reward_terms(obs),
            self._foot_contact_terms(),
            self._settled_standing_quality_terms(obs),
            self._motion_reward_terms(obs, raw_action, executed_action, executed_action_delta, raw_action_delta),
        ):
            terms.update(delta_terms)
        return float(reward), terms

    def _termination_reason(self, obs: np.ndarray, reward_terms: dict) -> str | None:
        if not np.all(np.isfinite(obs)):
            return "invalid_obs"
        if float(reward_terms.get("base_relative_height", obs[0])) < _TERM_RELATIVE_HEIGHT:
            return "height"
        # Upright termination stays outside the reward so tipping is still a
        # failure while the reward baseline remains terrain-agnostic.
        if reward_terms.get("upright_score", 1.0) < _MIN_UPRIGHT_ALIGNMENT:
            return "tip"
        return None

    # ---------------------------------------------------------------------- #
    # Gymnasium interface
    # ---------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_reset_noise()
        self._build_sim()
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._foot_anchor_xz.fill(0.0)
        self._foot_anchor_active.fill(False)
        self._foot_anchor_off_frames.fill(0)
        self._foot_anchor_reset_counts.fill(0)
        self._foot_anchor_deactivate_counts.fill(0)
        for counts in self._foot_load_below_counts.values():
            counts.fill(0)
        self._standing_reference_captured = False
        self.step_count = 0
        obs = self._get_obs()
        self._record_initial_reset_diagnostics()
        base_world = self._base_world_pos()
        self._reset_base_xz = np.array([base_world[0], base_world[2]], dtype=np.float32)
        self._base_anchor_xz = self._reset_base_xz.copy()
        self._base_anchor_yaw = self._trunk_yaw()
        return obs, self._info()

    def step(self, action: np.ndarray):
        self.step_count += 1

        raw_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        executed_action = raw_action.copy()
        action_delta = executed_action - self._prev_action
        raw_action_delta = raw_action - self._prev_raw_action
        targets = self._apply_motor_targets(executed_action)

        for _ in range(_PHYSICS_SUBSTEPS):
            if self._terrain is not None:
                self._terrain.Synchronize(self._system.GetChTime())
            self._system.DoStepDynamics(_PHYSICS_TIME_STEP)
            if self._terrain is not None:
                self._terrain.Advance(_PHYSICS_TIME_STEP)
        self._update_actuator_load_cache()
        self._sync_joint_state_cache(reset_velocity=False)

        self._prev_raw_action = raw_action.copy()
        self._prev_action = executed_action.copy()
        obs = self._get_obs()
        truncated = self.step_count >= self.max_steps
        reward, reward_terms = self._standing_reward(obs, raw_action, executed_action, action_delta, raw_action_delta)
        base_world = self._base_world_pos()
        reward_terms.update({
            "base_world_x": float(base_world[0]),
            "base_world_y": float(base_world[1]),
            "base_world_z": float(base_world[2]),
            "base_relative_height": self._base_relative_height(),
            "ground_top_y": self._ground_top_y(),
            "ground_height_offset": self.ground_height_offset,
            "reset_noise_enabled": float(self._reset_noise_sample["enabled"]),
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
            "reset_noise_joint_pos_offset_rms": float(np.sqrt(np.mean(np.square(self._reset_noise_sample["joint_position_offsets"])))),
            "reset_noise_joint_vel_offset_rms": float(np.sqrt(np.mean(np.square(self._reset_noise_sample["joint_velocity_offsets"])))),
            "reset_noise_base_linear_velocity_norm": float(np.linalg.norm(self._reset_noise_sample["base_linear_velocity"])),
            "reset_noise_base_angular_velocity_norm": float(np.linalg.norm(self._reset_noise_sample["base_angular_velocity"])),
            "reset_noise_initial_min_foot_y": float(self._reset_noise_sample["initial_min_foot_y"]),
            "reset_noise_initial_max_foot_load": float(self._reset_noise_sample["initial_max_foot_load"]),
            "mean_abs_raw_action": float(np.mean(np.abs(raw_action))),
            "max_abs_raw_action": float(np.max(np.abs(raw_action))),
            "mean_abs_raw_action_delta": float(np.mean(np.abs(raw_action_delta))),
            "max_abs_raw_action_delta": float(np.max(np.abs(raw_action_delta))),
            "mean_abs_executed_action": float(np.mean(np.abs(executed_action))),
            "max_abs_executed_action": float(np.max(np.abs(executed_action))),
            "mean_abs_executed_action_delta": float(np.mean(np.abs(action_delta))),
            "max_abs_executed_action_delta": float(np.max(np.abs(action_delta))),
        })
        reward_terms.update(self._actuator_diagnostic_terms())
        termination_reason = self._termination_reason(obs, reward_terms)
        terminated = termination_reason is not None
        if terminated:
            reward += _TIME_STEP * _REWARD_TERMINATION_WEIGHT

        info = self._info()
        info["target_joint_angles"] = targets
        info["raw_action"] = raw_action
        info["executed_action"] = executed_action
        info["raw_action_delta"] = raw_action_delta
        info["executed_action_delta"] = action_delta
        info["reward_terms"] = reward_terms
        info["termination_reason"] = termination_reason
        return obs, reward, terminated, truncated, info

    def _material_info(self) -> dict:
        effective_friction = None
        if self.ground_friction is not None:
            effective_friction = min(float(self.ground_friction), _FOOT_FRICTION)
        return {
            "contact_method": _CONTACT_METHOD,
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
                "Kn": _GROUND_KN,
                "Gn": _GROUND_GN,
                "rolling_friction": _GROUND_ROLLING_FRICTION,
            },
            "feet": {
                "friction": _FOOT_FRICTION,
                "restitution": _FOOT_RESTITUTION,
                "Gn": _FOOT_GN,
            },
        }

    def _info(self) -> dict:
        material_info = self._material_info()
        return {
            "terrain": self.terrain_type,
            "ground_friction": self.ground_friction,
            "foot_friction": _FOOT_FRICTION,
            "effective_friction": material_info["effective_friction"],
            "friction_range": self.friction_range,
            "spawn_x": float(self.spawn_xz[0]),
            "spawn_z": float(self.spawn_xz[1]),
            "ground_height_offset": self.ground_height_offset,
            "ground_top_y": self._ground_top_y(),
            "reset_noise_level": self.reset_noise_level,
            "reset_noise_components": self.reset_noise_components,
            "reset_noise": self._reset_noise_sample,
            "material_properties": material_info,
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
