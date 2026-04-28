"""
go1_env.py — Chrono Gymnasium wrapper for the Unitree Go1 quadruped.

Observation (37-dim float32):
    trunk position       (3)   x y z
    trunk quaternion     (4)   w x y z
    trunk linear vel     (3)
    trunk angular vel    (3)
    joint angles         (12)
    joint velocities     (12)

Action (12-dim float32, in [-1, 1]):
    Normalised torques scaled by per-joint limits.
    Order: FR_hip, FR_thigh, FR_calf,
           FL_hip, FL_thigh, FL_calf,
           RR_hip, RR_thigh, RR_calf,
           RL_hip, RL_thigh, RL_calf
"""

import math
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.parsers as parsers
import pychrono.irrlicht as irr

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_URDF = Path(__file__).parent / "models/go1/go1_chrono.urdf"

_TIME_STEP      = 0.002
_TERRAIN_LENGTH = 6.0
_TERRAIN_WIDTH  = 4.0
_TERRAIN_DELTA  = 0.04
_SPAWN_HEIGHT   = 0.45   # trunk y at episode start (m)
_TERM_HEIGHT    = 0.15   # trunk y below this → terminated

# Go1 torque limits (N·m): hip=23.7, thigh=23.7, calf=35.55
_TORQUE_LIMITS = np.tile([23.7, 23.7, 35.55], 4).astype(np.float32)

# Revolute joint names, must match go1_chrono.urdf, in action / obs order
_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

class Go1Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 1000, render_mode: str = None, terrain: str = "flat"):
        super().__init__()
        if terrain not in ("flat", "scm"):
            raise ValueError("terrain must be 'flat' or 'scm'")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terrain_type = terrain

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._system  = None
        self._terrain = None
        self._trunk   = None
        self._motors  = []
        self._vis     = None
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
            # Deformable SCM soil
            terrain = veh.SCMTerrain(system)
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
                3e4,    # damping (Pa·s/m)
            )
            terrain.Initialize(_TERRAIN_LENGTH, _TERRAIN_WIDTH, _TERRAIN_DELTA)
        else:
            # Flat rigid ground
            # Ground friction matches MuJoCo Menagerie floor defaults (scene.xml).
            # Floor has no explicit friction → MuJoCo defaults: sliding=1.0, rolling=0.0001.
            # Effective contact = min(floor, foot=0.8) → sliding=0.8 dominates from foot.
            # In Chrono (averaging composite rule): set ground high so foot material drives result.
            ground_mat = chrono.ChContactMaterialSMC()
            ground_mat.SetFriction(0.8)           # avg(0.8, foot=0.8) = 0.8, matches MuJoCo effective
            ground_mat.SetRollingFriction(0.0001) # MuJoCo floor default rolling friction
            ground_mat.SetRestitution(0.0)
            ground = chrono.ChBodyEasyBox(10, 0.2, 10, 1000, True, True, ground_mat)
            ground.SetFixed(True)
            ground.SetPos(chrono.ChVector3d(0, -0.1, 0))
            # Grey road colour
            if ground.GetVisualModel() is not None:
                for i in range(ground.GetVisualModel().GetNumShapes()):
                    ground.GetVisualModel().GetShape(i).SetColor(
                        chrono.ChColor(0.05, 0.05, 0.05)
                    )
            system.AddBody(ground)
            terrain = None

        # Load robot
        parser = parsers.ChParserURDF(str(_URDF))
        parser.EnableCollisionVisualization()
        parser.SetRootInitPose(
            chrono.ChFramed(
                chrono.ChVector3d(0, _SPAWN_HEIGHT, 0),
                chrono.ChQuaterniond(1, 0, 0, 0),
            )
        )
        parser.SetAllBodiesMeshCollisionType(
            parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
        )
        parser.SetAllJointsActuationType(
            parsers.ChParserURDF.ActuationType_FORCE
        )

        mat = chrono.ChContactMaterialData()
        mat.mu = 0.6
        mat.cr = 0.0
        parser.SetDefaultContactMaterial(mat)

        foot_mat = chrono.ChContactMaterialData()
        foot_mat.mu = 0.8
        foot_mat.cr = 0.0
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            parser.SetBodyContactMaterial(name, foot_mat)

        parser.PopulateSystem(system)

        for body in system.GetBodies():
            if not body.IsFixed():
                body.EnableCollision(True)
                # Blue robot colour
                if body.GetVisualModel() is not None:
                    for i in range(body.GetVisualModel().GetNumShapes()):
                        body.GetVisualModel().GetShape(i).SetColor(
                            chrono.ChColor(0.2, 0.45, 0.85)
                        )

        # Cache references
        self._system  = system
        self._terrain = terrain
        self._trunk   = parser.GetChBody("trunk")
        self._motors  = [parser.GetChMotor(name) for name in _JOINT_NAMES]

        # Pre-allocate one ChFunctionConst per motor so we can update
        # torques in-place each step (SetConstant) without creating new objects.
        # Must use SetMotorFunction — GetChMotor returns ChLinkMotor (base class)
        # which only exposes SetMotorFunction, not SetTorqueFunction.
        self._torque_funcs = []
        for motor in self._motors:
            f = chrono.ChFunctionConst(0.0)
            motor.SetMotorFunction(f)
            self._torque_funcs.append(f)

        # Cache body pairs for joint angle reading.
        # GetMotorAngle/GetMotorAngleDt are not accessible via ChLinkMotor;
        # we compute relative rotation from body frames instead.
        self._joint_body_pairs = [
            (motor.GetBody1(), motor.GetBody2()) for motor in self._motors
        ]

        # Attach (or reattach) the Irrlicht window if rendering is enabled
        if self.render_mode == "human":
            # Always create a fresh visualizer — AttachSystem on an already-
            # initialised window crashes the Irrlicht device.
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
        rv = q_rel.GetRotVec()
        return float(rv.z)

    def _joint_vel(self, b1, b2) -> float:
        """Relative angular velocity around Z in body1's frame."""
        w1 = b1.GetAngVelParent()
        w2 = b2.GetAngVelParent()
        dw_world = chrono.ChVector3d(w2.x - w1.x, w2.y - w1.y, w2.z - w1.z)
        # Express in body1's local frame
        q1 = b1.GetRot()
        dw_local = q1.GetInverse().Rotate(dw_world)
        return float(dw_local.z)

    def _get_obs(self) -> np.ndarray:
        pos     = self._trunk.GetPos()
        rot     = self._trunk.GetRot()        # ChQuaterniond: e0=w, e1=x, e2=y, e3=z
        lin_vel = self._trunk.GetPosDt()
        ang_vel = self._trunk.GetAngVelParent()

        joint_pos = np.array(
            [self._joint_angle(b1, b2) for b1, b2 in self._joint_body_pairs],
            dtype=np.float32,
        )
        joint_vel = np.array(
            [self._joint_vel(b1, b2) for b1, b2 in self._joint_body_pairs],
            dtype=np.float32,
        )

        return np.concatenate([
            [pos.x,     pos.y,     pos.z],
            [rot.e0,    rot.e1,    rot.e2,    rot.e3],
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
        self._build_sim()  # full rebuild clears SCM terrain state
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Scale normalised action → torques and apply in-place
        torques = np.clip(action, -1.0, 1.0) * _TORQUE_LIMITS
        for func, torque in zip(self._torque_funcs, torques):
            func.SetConstant(float(torque))

        # Advance simulation by one time step
        if self._terrain is not None:
            self._terrain.Synchronize(self._system.GetChTime())
        self._system.DoStepDynamics(_TIME_STEP)
        if self._terrain is not None:
            self._terrain.Advance(_TIME_STEP)

        obs = self._get_obs()
        truncated  = self.step_count >= self.max_steps
        terminated = False
        reward     = 0.0  # placeholder — add reward shaping before training

        return obs, reward, terminated, truncated, {}

    def render(self) -> bool:
        """Render one frame. Returns False when the window has been closed."""
        if self._vis is None or not self._vis.Run():
            return False
        self._vis.BeginScene()
        self._vis.Render()
        self._vis.EndScene()
        return True

    def close(self):
        self._vis     = None
        self._system  = None
        self._terrain = None
        self._trunk   = None
        self._motors  = []
