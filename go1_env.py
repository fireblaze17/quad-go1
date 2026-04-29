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

# At home angles (hip=0, thigh=0.9, calf=-1.8) the leg length trunk→foot is ~0.27 m.
# Spawn trunk at 0.35 m so feet clear the ground by ~0.08 m after assembly.
_SPAWN_HEIGHT = 0.35  # trunk root height; DoAssembly drives legs to home before first step
_TERM_HEIGHT = 0.15
_UPRIGHT_WEIGHT         = 1.0
_MIN_UPRIGHT_ALIGNMENT  = 0.75
_POSE_PENALTY_WEIGHT    = 0.15  # penalise joint deviation from home pose
_CONTROL_PENALTY_WEIGHT = 0.01  # penalise large actions — matches MuJoCo baseline
_ANG_VEL_PENALTY_WEIGHT = 0.05  # penalise trunk angular velocity — matches MuJoCo baseline
_ALIVE_BONUS            = 1.0   # reward per surviving step — matches MuJoCo; terrain-agnostic

# MuJoCo Menagerie unitree_go1/go1.xml:
# <key name="home" qpos="0 0 0.27 1 0 0 0 ..." ctrl="0 0.9 -1.8 ..."/>
# Zero action holds this home control pose.
_HOME_JOINT_ANGLES = np.tile([0.0, 0.9, -1.8], 4).astype(np.float32)
_ACTION_SCALE = 0.25

# Joint limits from go1_chrono.urdf, in _JOINT_NAMES order.
_JOINT_LOW = np.tile([-0.863, -0.686, -2.818], 4).astype(np.float32)
_JOINT_HIGH = np.tile([0.863, 4.501, -0.888], 4).astype(np.float32)

# Revolute joint names. This order is shared by actions, observations, limits,
# and home targets, so keep it synchronized with go1_chrono.urdf.
# Also defines which component of the rotation vector to read for each joint.
# GetRotVec() returns axis*angle in the relative frame.
# Hip joints rotate about URDF X (axis="1 0 0"), which stays Chrono X after
# the -90° spawn rotation → read component 0, sign=+1.
# Thigh/calf rotate about URDF +Y (axis="0 1 0") → Chrono -Z after spawn.
# For rotation θ about -Z: GetRotVec().z = -θ.
# At home (thigh=0.9): GetRotVec().z = -0.9.
# sign=-1 corrects this: reading = -1*(-0.9) = +0.9, matching _HOME_JOINT_ANGLES.
# sign=+1 (tried and wrong): reading = -0.9 → pose_error = (-0.9-0.9)²=3.24 per joint
#   → total pose_penalty ≈ 6.5/step → reward ≈ -6500/episode.
_JOINT_AXES = np.array(
    [0, 2, 2,   # FR: hip=X, thigh=Z, calf=Z
     0, 2, 2,   # FL
     0, 2, 2,   # RR
     0, 2, 2],  # RL
    dtype=np.int32,
)
# Hip: sign=+1 (X component directly correct).
# Thigh/calf: sign=-1 (negates the Chrono -Z rotation sign back to URDF convention).
_JOINT_AXIS_SIGN = np.where(_JOINT_AXES == 0, 1.0, -1.0).astype(np.float32)
_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# External contact shell. We intentionally leave rotors and sensor marker bodies
# non-colliding because they are internal/reference geometry, not terrain-contact
# surfaces. See docs/collision_debug_log.md for the debugging history.
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_hip", "FL_hip", "RR_hip", "RL_hip",
    "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
    "FR_calf", "FL_calf", "RR_calf", "RL_calf",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)


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
    ):
        super().__init__()
        if terrain not in ("flat", "scm"):
            raise ValueError("terrain must be 'flat' or 'scm'")
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
        self.ground_friction = None

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
        ground_mat = chrono.ChContactMaterialSMC()
        ground_mat.SetFriction(self.ground_friction)
        ground_mat.SetRollingFriction(0.0001)
        ground_mat.SetRestitution(0.0)

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

    def _cache_robot_handles(self, system, terrain, parser) -> None:
        self._system = system
        self._terrain = terrain
        self._trunk = parser.GetChBody("trunk")
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
            function = chrono.ChFunctionConst(float(_HOME_JOINT_ANGLES[i]))
            motor.SetMotorFunction(function)
            self._motor_funcs.append(function)

        # ChLinkMotor's angle accessors are not exposed in this PyChrono build,
        # so observations compute joint motion from the linked body frames.
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

    def _joint_angle(self, b1, b2, axis_idx: int, sign: float) -> float:
        """Revolute joint angle from relative body rotation.

        Reads the component of the rotation vector along the joint's actual
        rotation axis in Chrono world space:
          axis_idx=0 (X) for hip abduction joints (URDF axis="1 0 0")
          axis_idx=2 (Z) for thigh/calf joints    (URDF axis="0 1 0" → Chrono -Z)
        sign corrects for URDF Y mapping to Chrono -Z.
        """
        q1 = b1.GetRot()
        q2 = b2.GetRot()
        q_rel = q1.GetInverse() * q2
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

        if self._joint_body_pairs:
            joint_pos = np.array(
                [
                    self._joint_angle(b1, b2, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                    for i, (b1, b2) in enumerate(self._joint_body_pairs)
                ],
                dtype=np.float32,
            )
            joint_vel = np.array(
                [
                    self._joint_vel(b1, b2, int(_JOINT_AXES[i]), float(_JOINT_AXIS_SIGN[i]))
                    for i, (b1, b2) in enumerate(self._joint_body_pairs)
                ],
                dtype=np.float32,
            )
        else:
            joint_pos = np.zeros(12, dtype=np.float32)
            joint_vel = np.zeros(12, dtype=np.float32)

        return np.concatenate([
            [pos.x, pos.y, pos.z],
            [rot.e0, rot.e1, rot.e2, rot.e3],
            [lin_vel.x, lin_vel.y, lin_vel.z],
            [ang_vel.x, ang_vel.y, ang_vel.z],
            joint_pos,
            joint_vel,
        ]).astype(np.float32)

    def _trunk_up_alignment(self) -> float:
        """Return trunk local-up alignment with Chrono world Y-up."""
        trunk_up = self._trunk.GetRot().Rotate(chrono.ChVector3d(0, 0, 1))
        return float(np.clip(trunk_up.y, -1.0, 1.0))

    def _trunk_axis_alignments(self) -> dict[str, float]:
        """Return each trunk local axis alignment with Chrono world Y-up."""
        rot = self._trunk.GetRot()
        return {
            "trunk_x_up": float(np.clip(rot.Rotate(chrono.ChVector3d(1, 0, 0)).y, -1.0, 1.0)),
            "trunk_y_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 1, 0)).y, -1.0, 1.0)),
            "trunk_z_up": float(np.clip(rot.Rotate(chrono.ChVector3d(0, 0, 1)).y, -1.0, 1.0)),
        }

    def _standing_reward(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, dict]:
        """Standing reward built one observed failure mode at a time."""
        if not np.all(np.isfinite(obs)):
            return -10.0, {"invalid_obs": 1.0}

        trunk_y = float(obs[1])
        # Alive bonus: constant +1 per surviving step. Terrain-agnostic —
        # works on flat and SCM deformable ground. Matches MuJoCo baseline.
        alive_bonus = _ALIVE_BONUS

        # Upright: keep trunk vertical.
        upright_score = max(0.0, self._trunk_up_alignment())

        # Stage 3 reward: penalise joint deviation from home pose.
        # Prevents the legs collapsing while trunk stays level (sinking loophole).
        joint_pos = obs[13:25]  # indices in _get_obs: pos(3)+rot(4)+linvel(3)+angvel(3) = 13
        pose_error = float(np.sum(np.square(joint_pos - _HOME_JOINT_ANGLES)))
        pose_penalty = _POSE_PENALTY_WEIGHT * pose_error

        # Stage 4 reward: penalise large actions — matches MuJoCo control_penalty.
        # Discourages saturation (max_abs_action=1.0) and unnecessary joint motion.
        control_penalty = _CONTROL_PENALTY_WEIGHT * float(np.sum(np.square(action)))

        # Stage 5 reward: penalise trunk angular velocity — matches MuJoCo angular_velocity_penalty.
        # obs[10:13] = trunk angular velocity in world frame.
        ang_vel_penalty = _ANG_VEL_PENALTY_WEIGHT * float(np.sum(np.square(obs[10:13])))

        reward = float(alive_bonus + _UPRIGHT_WEIGHT * upright_score - pose_penalty - control_penalty - ang_vel_penalty)

        terms = {
            "alive_bonus": alive_bonus,
            "upright_score": float(upright_score),
            "pose_error": pose_error,
            "pose_penalty": pose_penalty,
            "control_penalty": control_penalty,
            "ang_vel_penalty": ang_vel_penalty,
            "trunk_y": trunk_y,
        }
        terms.update(self._trunk_axis_alignments())
        return reward, terms

    def _termination_reason(self, obs: np.ndarray, reward_terms: dict) -> str | None:
        if not np.all(np.isfinite(obs)):
            return "invalid_obs"
        if float(obs[1]) < _TERM_HEIGHT:
            return "height"
        # Stage 3 termination: falling forward/sideways is a failure, not just
        # a low-reward pose.
        if reward_terms.get("upright_score", 1.0) < _MIN_UPRIGHT_ALIGNMENT:
            return "tip"
        return None

    # ---------------------------------------------------------------------- #
    # Gymnasium interface
    # ---------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_sim()
        self.step_count = 0
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray):
        self.step_count += 1

        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.enable_motors:
            desired_targets = _HOME_JOINT_ANGLES + _ACTION_SCALE * clipped_action
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
        reward, reward_terms = self._standing_reward(obs, clipped_action)
        termination_reason = self._termination_reason(obs, reward_terms)
        terminated = termination_reason is not None
        if terminated:
            reward -= 5.0

        info = self._info()
        info["target_joint_angles"] = targets
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
