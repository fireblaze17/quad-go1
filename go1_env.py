"""Chrono Gymnasium environment for a Unitree Go1-style quadruped.

The project uses Chrono as the simulator and MuJoCo Menagerie only as a source
of model/reference values that are known to be sane for Go1. Chrono runs here in
a Y-up world, so the imported ROS-style Z-up URDF is rotated at the root.

Observation, 37 float32 values:
    trunk position, trunk quaternion, trunk linear velocity,
    trunk angular velocity, 12 joint angles, 12 joint velocities.

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

_TIME_STEP = 0.002
_TERRAIN_LENGTH = 6.0
_TERRAIN_WIDTH = 4.0
_TERRAIN_DELTA = 0.04

# Zero-action diagnostics showed the original Menagerie crouch
# (hip=0, thigh=0.9, calf=-1.8 at y=0.27) slowly sank in Chrono. This less
# crouched pose starts at its natural support height and holds with zero action.
_SPAWN_HEIGHT = 0.34  # trunk root height; DoAssembly drives legs to home before first step
_TERM_HEIGHT = 0.22
_MIN_UPRIGHT_ALIGNMENT = 0.85
_UPRIGHT_REWARD_WEIGHT = 0.15
_ALIVE_BONUS = 1.0  # reward per surviving step; terrain-agnostic
_POSE_PENALTY_WEIGHT = 0.30
_CONTROL_PENALTY_WEIGHT = 0.03
_ANG_VEL_PENALTY_WEIGHT = 0.01
_XZ_VEL_PENALTY_WEIGHT = 1.00
_JOINT_VEL_PENALTY_WEIGHT = 0.02
_ACTION_RATE_PENALTY_WEIGHT = 0.05
_TILT_PENALTY_WEIGHT = 0.25
_FOOT_CONTACT_PENALTY_WEIGHT = 2.00
_FOOT_SLIP_PENALTY_WEIGHT = 0.00
_FOOT_ANCHOR_PENALTY_WEIGHT = 5.00
_FOOT_ANCHOR_DEADBAND = 0.005
_FOOT_ANCHOR_CONTACT_ON_LOAD = 15.0
_FOOT_ANCHOR_CONTACT_OFF_LOAD = 5.0
_FOOT_ANCHOR_CONTACT_OFF_FRAMES = 5
_BASE_DRIFT_PENALTY_WEIGHT = 2.00
_BASE_DRIFT_DEADBAND = 0.01
_STANDING_QUALITY_START_STEP = 100
_MIN_FOOT_LOAD = 20.0
_ANCHOR_LOAD_THRESHOLDS = (20.0, 15.0, 8.0, 5.0)

# Zero action holds this home control pose.
_HOME_JOINT_ANGLES = np.tile([0.0, 0.7, -1.4], 4).astype(np.float32)
_ACTION_SCALE = 0.20

# Joint limits from go1_chrono.urdf, in _JOINT_NAMES order.
_JOINT_LOW = np.tile([-0.863, -0.686, -2.818], 4).astype(np.float32)
_JOINT_HIGH = np.tile([0.863, 4.501, -0.888], 4).astype(np.float32)

# Joint order is shared by actions, observations, limits, and home targets.
# The axis/sign arrays convert Chrono motor-frame rotation vectors back to
# URDF joint angles; see docs/chrono_port_notes.md ADR-008 for the derivation.
# the -90° spawn rotation → read component 0, sign=+1.
# Thigh/calf rotate about URDF +Y (axis="0 1 0") → Chrono -Z after spawn.
# For rotation θ about -Z: GetRotVec().z = -θ.
# At home (thigh=0.7): GetRotVec().z = -0.7.
# sign=-1 corrects this: reading = -1*(-0.7) = +0.7, matching _HOME_JOINT_ANGLES.
_JOINT_AXES = np.array(
    [0, 2, 2,   # FR: hip=X, thigh=Z, calf=Z
     0, 2, 2,   # FL
     0, 2, 2,   # RR
     0, 2, 2],  # RL
    dtype=np.int32,
)
# Hips read Chrono X directly; thigh/calf URDF Y maps to Chrono -Z.
_JOINT_AXIS_SIGN = np.where(_JOINT_AXES == 0, 1.0, -1.0).astype(np.float32)
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
        "spawn_height": _SPAWN_HEIGHT,
        "home_joint_angles": _HOME_JOINT_ANGLES.tolist(),
        "action_scale": _ACTION_SCALE,
        "collision_bodies": list(_ROBOT_COLLISION_BODIES),
        "reward_weights": {
            "alive_bonus": _ALIVE_BONUS,
            "upright": _UPRIGHT_REWARD_WEIGHT,
            "pose": _POSE_PENALTY_WEIGHT,
            "control": _CONTROL_PENALTY_WEIGHT,
            "joint_velocity": _JOINT_VEL_PENALTY_WEIGHT,
            "action_rate": _ACTION_RATE_PENALTY_WEIGHT,
            "tilt": _TILT_PENALTY_WEIGHT,
            "angular_velocity": _ANG_VEL_PENALTY_WEIGHT,
            "xz_velocity": _XZ_VEL_PENALTY_WEIGHT,
            "foot_contact": _FOOT_CONTACT_PENALTY_WEIGHT,
            "foot_slip": _FOOT_SLIP_PENALTY_WEIGHT,
            "foot_anchor": _FOOT_ANCHOR_PENALTY_WEIGHT,
            "base_drift": _BASE_DRIFT_PENALTY_WEIGHT,
        },
        "minimum_foot_load": _MIN_FOOT_LOAD,
        "foot_anchor_deadband": _FOOT_ANCHOR_DEADBAND,
        "foot_anchor_contact_on_load": _FOOT_ANCHOR_CONTACT_ON_LOAD,
        "foot_anchor_contact_off_load": _FOOT_ANCHOR_CONTACT_OFF_LOAD,
        "foot_anchor_contact_off_frames": _FOOT_ANCHOR_CONTACT_OFF_FRAMES,
        "base_drift_deadband": _BASE_DRIFT_DEADBAND,
        "standing_quality_start_step": _STANDING_QUALITY_START_STEP,
        "solver": {
            "type": "BARZILAIBORWEIN",
            "max_iterations": 60,
        },
        "contact_materials": {
            "flat_ground": {
                "friction": "sampled from friction_range",
                "restitution": 0.1,
                "Kn": 2e5,
                "Gn": 60.0,
            },
            "feet": {
                "friction": 0.9,
                "restitution": 0.01,
                "Gn": 60.0,
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
        action_filter_tau: float | None = None,
    ):
        super().__init__()
        if terrain not in ("flat", "scm"):
            raise ValueError("terrain must be 'flat' or 'scm'")
        if len(friction_range) != 2:
            raise ValueError("friction_range must be a (min, max) pair")
        friction_min, friction_max = friction_range
        if friction_min <= 0 or friction_max <= 0 or friction_min > friction_max:
            raise ValueError("friction_range must satisfy 0 < min <= max")
        if action_filter_tau is not None and action_filter_tau <= 0.0:
            raise ValueError("action_filter_tau must be positive when provided")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terrain_type = terrain
        self.enable_motors = enable_motors
        self.friction_range = (float(friction_min), float(friction_max))
        self.action_filter_tau = None if action_filter_tau is None else float(action_filter_tau)
        self.action_filter_alpha = (
            None
            if self.action_filter_tau is None
            else float(_TIME_STEP / (self.action_filter_tau + _TIME_STEP))
        )
        self.ground_friction = None
        self.home_joint_angles = _HOME_JOINT_ANGLES.copy()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._system = None
        self._terrain = None
        self._trunk = None
        self._motors = []
        self._motor_funcs = []
        self._joint_body_pairs = []
        self._vis = None
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._action_filter_initialized = False
        self._reset_base_xz = np.zeros(2, dtype=np.float32)
        self._base_anchor_xz = np.zeros(2, dtype=np.float32)
        self._foot_anchor_xz = np.zeros((len(_FOOT_BODY_NAMES), 2), dtype=np.float32)
        self._foot_anchor_active = np.zeros(len(_FOOT_BODY_NAMES), dtype=bool)
        self._foot_anchor_off_frames = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._foot_anchor_reset_counts = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._foot_anchor_deactivate_counts = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
        self._foot_load_below_counts = {
            threshold: np.zeros(len(_FOOT_BODY_NAMES), dtype=np.int32)
            for threshold in _ANCHOR_LOAD_THRESHOLDS
        }
        self._standing_reference_captured = False
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
        system = chrono.ChSystemSMC()
        system.SetGravityY()
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(60)

        if self.terrain_type == "scm":
            terrain = self._create_scm_terrain(system)
        else:
            terrain = None
            self.ground_friction = self._sample_ground_friction()
            self._add_flat_ground(system)

        parser = self._create_robot_parser()
        parser.PopulateSystem(system)
        self._configure_imported_bodies(system, parser)
        self._cache_robot_handles(system, terrain, parser)

        # Zero-overhead home-pose init: fix the trunk so it cannot drift, then
        # run Chrono's kinematic assembly solver (pure constraint satisfaction,
        # no forces, no time integration). This drives every position-motor
        # constraint to its target (home angle) in one call, placing all leg
        # bodies in the correct standing pose before the first DoStepDynamics().
        # AssemblyAnalysis.POSITION = 1.
        self._trunk.SetFixed(True)
        system.DoAssembly(1)
        self._trunk.SetFixed(False)

        if self.render_mode == "human":
            self._create_visualizer(system)

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
                chrono.ChVector3d(0, 0, 0),
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
        # Sampled flat-ground friction is the first domain-randomization knob.
        # Foot friction stays at the Go1 reference value; only the floor material
        # changes from episode to episode.
        ground_mat = _magic_contact_material(
            friction=self.ground_friction,
            restitution=0.1,
            gn=60.0,
            kn=2e5,
        )
        ground_mat.SetRollingFriction(0.0001)

        ground = chrono.ChBodyEasyBox(10, 0.2, 10, 1000, True, True, ground_mat)
        ground.SetFixed(True)
        ground.SetPos(chrono.ChVector3d(0, -0.1, 0))
        _set_visual_color(ground, chrono.ChColor(0.05, 0.05, 0.05))
        system.AddBody(ground)

    def _create_robot_parser(self):
        parser = parsers.ChParserURDF(str(_URDF))
        parser.EnableCollisionVisualization()
        parser.SetRootInitPose(
            chrono.ChFramed(
                chrono.ChVector3d(0, _SPAWN_HEIGHT, 0),
                chrono.QuatFromAngleX(-math.pi / 2),
            )
        )
        parser.SetAllBodiesMeshCollisionType(
            parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
        )

        if self.enable_motors:
            parser.SetAllJointsActuationType(
                parsers.ChParserURDF.ActuationType_POSITION
            )

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
            friction=0.9,
            restitution=0.01,
            gn=60.0,
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
        self._motors = (
            [parser.GetChMotor(name) for name in _JOINT_NAMES]
            if self.enable_motors else []
        )

        # Pre-allocate one constant target function per position motor so we can
        # update desired joint angles in-place each step.
        # Initialise to home angles so joints are at the correct pose from step 0
        # — no ramp needed, matching SBEL's Go2 actuate() pattern.
        self._motor_funcs = []
        for i, motor in enumerate(self._motors):
            function = chrono.ChFunctionConst(float(self.home_joint_angles[i]))
            motor.SetMotorFunction(function)
            self._motor_funcs.append(function)

        # ChLinkMotor's direct angle accessors are not exposed in this PyChrono
        # build. Motor frame rotation still gives the assembled joint angle;
        # linked body pairs are kept only for the approximate velocity term.
        self._joint_body_pairs = (
            [(motor.GetBody1(), motor.GetBody2()) for motor in self._motors]
            if self.enable_motors else []
        )

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

        Reads the component of the rotation vector along the joint's actual
        rotation axis in Chrono world space:
          axis_idx=0 (X) for hip abduction joints (URDF axis="1 0 0")
          axis_idx=2 (Z) for thigh/calf joints    (URDF axis="0 1 0" → Chrono -Z)
        sign corrects for URDF Y mapping to Chrono -Z.
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
        pos = self._trunk.GetPos()
        rot = self._trunk.GetRot()  # Chrono stores w, x, y, z as e0..e3.
        lin_vel = self._trunk.GetPosDt()
        ang_vel = self._trunk.GetAngVelParent()

        if self._motors:
            joint_pos = np.array(
                [
                    self._joint_angle(motor, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                    for i, motor in enumerate(self._motors)
                ],
                dtype=np.float32,
            )
        else:
            joint_pos = np.zeros(12, dtype=np.float32)

        if self._joint_body_pairs:
            joint_vel = np.array(
                [
                    self._joint_vel(b1, b2, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                    for i, (b1, b2) in enumerate(self._joint_body_pairs)
                ],
                dtype=np.float32,
            )
        else:
            joint_vel = np.zeros(12, dtype=np.float32)

        return np.concatenate([
            [pos.x, pos.y, pos.z],
            [rot.e0, rot.e1, rot.e2, rot.e3],
            [lin_vel.x, lin_vel.y, lin_vel.z],
            [ang_vel.x, ang_vel.y, ang_vel.z],
            joint_pos,
            joint_vel,
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
            "trunk_y": float(obs[1]),
            "tilt_error": float(tilt_error),
            "tilt_penalty": float(tilt_penalty),
        }
        terms.update(axis_alignments)
        penalty = tilt_penalty
        reward = _ALIVE_BONUS + upright_reward - penalty
        return float(reward), terms

    def _pose_reward_terms(self, obs: np.ndarray) -> tuple[float, dict]:
        joint_pos = obs[13:25]
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
        foot_loads = np.array(
            [abs(float(foot.GetContactForce().y)) for foot in self._feet],
            dtype=np.float32,
        )
        missing_contact = np.maximum(0.0, _MIN_FOOT_LOAD - foot_loads) / _MIN_FOOT_LOAD
        foot_contact_error = float(np.mean(missing_contact ** 2))
        foot_contact_penalty = _FOOT_CONTACT_PENALTY_WEIGHT * foot_contact_error
        terms = {
            "foot_contact_error": float(foot_contact_error),
            "foot_contact_penalty": float(foot_contact_penalty),
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
        self._base_anchor_xz = np.array([float(obs[0]), float(obs[2])], dtype=np.float32)
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
        foot_loads = np.array(
            [abs(float(foot.GetContactForce().y)) for foot in self._feet],
            dtype=np.float32,
        )
        for threshold in _ANCHOR_LOAD_THRESHOLDS:
            self._foot_load_below_counts[threshold] += (foot_loads < threshold).astype(np.int32)

        base_xz = np.array([float(obs[0]), float(obs[2])], dtype=np.float32)
        foot_displacements = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        foot_errors = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)

        if self.step_count <= _STANDING_QUALITY_START_STEP:
            terms = {
                "foot_slip_error": 0.0,
                "foot_slip_penalty": 0.0,
                "base_drift": 0.0,
                "base_drift_error": 0.0,
                "base_drift_penalty": 0.0,
                "foot_anchor_error": 0.0,
                "foot_anchor_penalty": 0.0,
                "foot_anchor_active_count": 0.0,
                "foot_anchor_max_displacement": 0.0,
            }
            terms.update(self._anchor_diagnostic_terms(foot_loads, foot_displacements, foot_errors, base_xz))
            return 0.0, terms

        if not self._standing_reference_captured:
            self._capture_standing_reference(obs, foot_loads)

        foot_slip_error = 0.0
        foot_anchor_error = 0.0
        foot_anchor_max_displacement = 0.0
        for index, (foot, load) in enumerate(zip(self._feet, foot_loads)):
            pos = foot.GetPos()
            foot_xz = np.array([float(pos.x), float(pos.z)], dtype=np.float32)
            if self._foot_anchor_active[index]:
                if load <= _FOOT_ANCHOR_CONTACT_OFF_LOAD:
                    self._foot_anchor_off_frames[index] += 1
                    if self._foot_anchor_off_frames[index] >= _FOOT_ANCHOR_CONTACT_OFF_FRAMES:
                        self._foot_anchor_active[index] = False
                        self._foot_anchor_deactivate_counts[index] += 1
                else:
                    self._foot_anchor_off_frames[index] = 0
                displacement = float(np.linalg.norm(foot_xz - self._foot_anchor_xz[index]))
                foot_displacements[index] = displacement
                foot_anchor_max_displacement = max(foot_anchor_max_displacement, displacement)
                error = max(0.0, displacement - _FOOT_ANCHOR_DEADBAND)
                foot_errors[index] = error
                foot_anchor_error += float(error ** 2)
            elif load >= _FOOT_ANCHOR_CONTACT_ON_LOAD:
                self._foot_anchor_xz[index] = foot_xz
                self._foot_anchor_active[index] = True
                self._foot_anchor_off_frames[index] = 0
                self._foot_anchor_reset_counts[index] += 1

        foot_slip_penalty = _FOOT_SLIP_PENALTY_WEIGHT * foot_slip_error
        foot_anchor_penalty = _FOOT_ANCHOR_PENALTY_WEIGHT * foot_anchor_error

        base_drift = float(np.linalg.norm(base_xz - self._base_anchor_xz))
        base_drift_error = max(0.0, base_drift - _BASE_DRIFT_DEADBAND)
        base_drift_penalty = _BASE_DRIFT_PENALTY_WEIGHT * float(base_drift_error ** 2)

        terms = {
            "foot_slip_error": float(foot_slip_error),
            "foot_slip_penalty": float(foot_slip_penalty),
            "base_drift": float(base_drift),
            "base_drift_error": float(base_drift_error),
            "base_drift_penalty": float(base_drift_penalty),
            "foot_anchor_error": float(foot_anchor_error),
            "foot_anchor_penalty": float(foot_anchor_penalty),
            "foot_anchor_active_count": float(np.sum(self._foot_anchor_active)),
            "foot_anchor_max_displacement": float(foot_anchor_max_displacement),
        }
        terms.update(self._anchor_diagnostic_terms(foot_loads, foot_displacements, foot_errors, base_xz))
        penalty = foot_slip_penalty + foot_anchor_penalty + base_drift_penalty
        return -float(penalty), terms

    def _motion_reward_terms(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        trunk_lin_vel = obs[7:10]
        trunk_ang_vel = obs[10:13]
        joint_vel = obs[25:37]
        joint_vel_penalty = _JOINT_VEL_PENALTY_WEIGHT * float(np.mean(joint_vel ** 2))
        action_rate_penalty = _ACTION_RATE_PENALTY_WEIGHT * float(np.mean(action_delta ** 2))
        control_penalty = _CONTROL_PENALTY_WEIGHT * float(np.mean(action ** 2))
        ang_vel_penalty = _ANG_VEL_PENALTY_WEIGHT * float(np.mean(trunk_ang_vel ** 2))
        xz_vel = trunk_lin_vel[[0, 2]]
        xz_vel_penalty = _XZ_VEL_PENALTY_WEIGHT * float(np.mean(xz_vel ** 2))

        terms = {
            "mean_abs_joint_vel": float(np.mean(np.abs(joint_vel))),
            "max_abs_joint_vel": float(np.max(np.abs(joint_vel))),
            "joint_vel_penalty": float(joint_vel_penalty),
            "action_rate_penalty": float(action_rate_penalty),
            "mean_abs_action_delta": float(np.mean(np.abs(action_delta))),
            "max_abs_action_delta": float(np.max(np.abs(action_delta))),
            "control_penalty": float(control_penalty),
            "mean_abs_action": float(np.mean(np.abs(action))),
            "max_abs_action": float(np.max(np.abs(action))),
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
            + control_penalty
            + ang_vel_penalty
            + xz_vel_penalty
        )
        return -float(penalty), terms

    def _standing_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        action_delta: np.ndarray,
    ) -> tuple[float, dict]:
        """Standing reward: survive, stay upright, and keep four-foot support."""
        if not np.all(np.isfinite(obs)):
            return -10.0, {"invalid_obs": 1.0}

        reward = 0.0
        terms = {}
        for delta, delta_terms in (
            self._trunk_reward_terms(obs),
            self._pose_reward_terms(obs),
            self._foot_contact_terms(),
            self._settled_standing_quality_terms(obs),
            self._motion_reward_terms(obs, action, action_delta),
        ):
            reward += delta
            terms.update(delta_terms)
        return float(reward), terms

    def _termination_reason(self, obs: np.ndarray, reward_terms: dict) -> str | None:
        if not np.all(np.isfinite(obs)):
            return "invalid_obs"
        if float(obs[1]) < _TERM_HEIGHT:
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
        self._build_sim()
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_raw_action = np.zeros(12, dtype=np.float32)
        self._action_filter_initialized = False
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
        self._reset_base_xz = np.array([float(obs[0]), float(obs[2])], dtype=np.float32)
        self._base_anchor_xz = self._reset_base_xz.copy()
        return obs, self._info()

    def step(self, action: np.ndarray):
        self.step_count += 1

        raw_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.action_filter_alpha is None:
            executed_action = raw_action.copy()
        elif not self._action_filter_initialized:
            executed_action = raw_action.copy()
            self._action_filter_initialized = True
        else:
            executed_action = self._prev_action + self.action_filter_alpha * (raw_action - self._prev_action)
            executed_action = executed_action.astype(np.float32)
        action_delta = executed_action - self._prev_action
        raw_action_delta = raw_action - self._prev_raw_action
        if self.enable_motors:
            desired_targets = self.home_joint_angles + _ACTION_SCALE * executed_action
            targets = np.clip(desired_targets, _JOINT_LOW, _JOINT_HIGH)
            for function, target in zip(self._motor_funcs, targets):
                function.SetConstant(float(target))
        else:
            targets = np.zeros(12, dtype=np.float32)

        if self._terrain is not None:
            self._terrain.Synchronize(self._system.GetChTime())
        self._system.DoStepDynamics(_TIME_STEP)
        if self._terrain is not None:
            self._terrain.Advance(_TIME_STEP)

        obs = self._get_obs()
        truncated = self.step_count >= self.max_steps
        reward, reward_terms = self._standing_reward(obs, executed_action, action_delta)
        reward_terms.update({
            "action_filter_tau": 0.0 if self.action_filter_tau is None else float(self.action_filter_tau),
            "action_filter_alpha": 0.0 if self.action_filter_alpha is None else float(self.action_filter_alpha),
            "mean_abs_raw_action": float(np.mean(np.abs(raw_action))),
            "max_abs_raw_action": float(np.max(np.abs(raw_action))),
            "mean_abs_raw_action_delta": float(np.mean(np.abs(raw_action_delta))),
            "max_abs_raw_action_delta": float(np.max(np.abs(raw_action_delta))),
            "mean_abs_executed_action": float(np.mean(np.abs(executed_action))),
            "max_abs_executed_action": float(np.max(np.abs(executed_action))),
            "mean_abs_executed_action_delta": float(np.mean(np.abs(action_delta))),
            "max_abs_executed_action_delta": float(np.max(np.abs(action_delta))),
        })
        termination_reason = self._termination_reason(obs, reward_terms)
        terminated = termination_reason is not None
        if terminated:
            reward -= 5.0
        self._prev_raw_action = raw_action.copy()
        self._prev_action = executed_action.copy()

        info = self._info()
        info["target_joint_angles"] = targets
        info["raw_action"] = raw_action
        info["executed_action"] = executed_action
        info["raw_action_delta"] = raw_action_delta
        info["executed_action_delta"] = action_delta
        info["reward_terms"] = reward_terms
        info["termination_reason"] = termination_reason
        return obs, reward, terminated, truncated, info

    def _info(self) -> dict:
        return {
            "terrain": self.terrain_type,
            "ground_friction": self.ground_friction,
            "friction_range": self.friction_range,
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
