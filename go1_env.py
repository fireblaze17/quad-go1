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

# Chrono's URDF parser initializes only the root pose, not a full home keyframe.
# This height keeps the zero-pose feet out of the ground before motors ramp in.
_SPAWN_HEIGHT = 0.48
_TERM_HEIGHT = 0.15

# MuJoCo Menagerie unitree_go1/go1.xml:
# <key name="home" qpos="0 0 0.27 1 0 0 0 ..." ctrl="0 0.9 -1.8 ..."/>
# Zero action holds this home control pose.
_HOME_JOINT_ANGLES = np.tile([0.0, 0.9, -1.8], 4).astype(np.float32)
_ACTION_SCALE = 0.25
_TARGET_RAMP_TIME = 1.0
_TARGET_RAMP_STEPS = max(1, int(_TARGET_RAMP_TIME / _TIME_STEP))

# Joint limits from go1_chrono.urdf, in _JOINT_NAMES order.
_JOINT_LOW = np.tile([-0.863, -0.686, -2.818], 4).astype(np.float32)
_JOINT_HIGH = np.tile([0.863, 4.501, -0.888], 4).astype(np.float32)

# Revolute joint names. This order is shared by actions, observations, limits,
# and home targets, so keep it synchronized with go1_chrono.urdf.
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
    ):
        super().__init__()
        if terrain not in ("flat", "scm"):
            raise ValueError("terrain must be 'flat' or 'scm'")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terrain_type = terrain
        self.enable_motors = enable_motors

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
            self._add_flat_ground(system)

        parser = self._create_robot_parser()
        parser.PopulateSystem(system)
        self._configure_imported_bodies(system, parser)
        self._cache_robot_handles(system, terrain, parser)

        if self.render_mode == "human":
            self._create_visualizer(system)

    def _create_scm_terrain(self, system):
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
        # Ground and foot friction are both set to 0.8 so Chrono's contact
        # combination gives the same effective sliding friction used by the Go1
        # reference while keeping flat terrain fast for debugging/training.
        ground_mat = chrono.ChContactMaterialSMC()
        ground_mat.SetFriction(0.8)
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
        self._motor_funcs = []
        for motor in self._motors:
            function = chrono.ChFunctionConst(0.0)
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

    def _joint_angle(self, b1, b2) -> float:
        """Approximate revolute joint angle from relative body rotation.

        Chrono revolute joints rotate about the Z axis of the joint frame.
        We compute the relative quaternion q_rel = q1_inv * q2 and return
        the Z component of its rotation vector (= axis * angle), which equals
        the rotation angle when the axis is close to Z.
        """
        q1 = b1.GetRot()
        q2 = b2.GetRot()
        q_rel = q1.GetInverse() * q2
        return float(q_rel.GetRotVec().z)

    def _joint_vel(self, b1, b2) -> float:
        """Relative angular velocity around Z in body1's frame."""
        w1 = b1.GetAngVelParent()
        w2 = b2.GetAngVelParent()
        dw_world = chrono.ChVector3d(w2.x - w1.x, w2.y - w1.y, w2.z - w1.z)
        dw_local = b1.GetRot().GetInverse().Rotate(dw_world)
        return float(dw_local.z)

    def _get_obs(self) -> np.ndarray:
        pos = self._trunk.GetPos()
        rot = self._trunk.GetRot()  # Chrono stores w, x, y, z as e0..e3.
        lin_vel = self._trunk.GetPosDt()
        ang_vel = self._trunk.GetAngVelParent()

        if self._joint_body_pairs:
            joint_pos = np.array(
                [self._joint_angle(b1, b2) for b1, b2 in self._joint_body_pairs],
                dtype=np.float32,
            )
            joint_vel = np.array(
                [self._joint_vel(b1, b2) for b1, b2 in self._joint_body_pairs],
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

    # ---------------------------------------------------------------------- #
    # Gymnasium interface
    # ---------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_sim()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1

        if self.enable_motors:
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            desired_targets = _HOME_JOINT_ANGLES + _ACTION_SCALE * action
            desired_targets = np.clip(desired_targets, _JOINT_LOW, _JOINT_HIGH)

            # Ramp target commands during startup so the position motors do not
            # snap from the parser's zero-pose into the Menagerie home pose.
            ramp = min(1.0, self.step_count / _TARGET_RAMP_STEPS)
            targets = ramp * desired_targets
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
        terminated = bool(obs[1] < _TERM_HEIGHT)
        reward = 0.0  # Placeholder until standing/walking rewards are added.

        return obs, reward, terminated, truncated, {"target_joint_angles": targets}

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
