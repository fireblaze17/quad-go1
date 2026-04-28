from pathlib import Path

import pychrono as chrono
import pychrono.irrlicht as irr
import pychrono.parsers as parsers
import pychrono.vehicle as veh
import math


MODEL_PATH = Path("models/go1/go1_chrono.urdf")
TIME_STEP = 0.002
TERRAIN_LENGTH = 6.0
TERRAIN_WIDTH = 4.0
TERRAIN_DELTA = 0.04


def create_system():
    system = chrono.ChSystemSMC()
    system.SetGravityY()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    return system


def create_soil(system):
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
        1.1,    # Bekker n exponent
        0,      # Mohr cohesive limit (Pa)
        30,     # Mohr friction limit (degrees)
        0.01,   # Janosi shear coefficient (m)
        4e7,    # Elastic stiffness (Pa/m)
        3e4,    # Damping (Pa s/m)
    )
    terrain.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, 0, 0.1)
    terrain.Initialize(TERRAIN_LENGTH, TERRAIN_WIDTH, TERRAIN_DELTA)

    return terrain


def load_go1(system):
    parser = parsers.ChParserURDF(str(MODEL_PATH))
    parser.EnableCollisionVisualization()
    parser.SetRootInitPose(
        chrono.ChFramed(
            chrono.ChVector3d(0, 0.45, 0),
            chrono.ChQuaterniond(1, 0, 0, 0),
        )
    )

    # Use convex hulls for all mesh collision shapes.
    # Faster and more stable for contact solving than triangle meshes.
    # Must be called before PopulateSystem.
    parser.SetAllBodiesMeshCollisionType(
        parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
    )

    # Set joints to force/torque actuation to match the real Go1 hardware.
    # The real robot uses EffortJointInterface (torque control).
    # Must be called before PopulateSystem.
    parser.SetAllJointsActuationType(
        parsers.ChParserURDF.ActuationType_FORCE
    )

    # Set a default contact material for all bodies.
    # Values from MuJoCo Menagerie unitree_go1/go1.xml:
    #   default geom friction: 0.6
    #   foot geom friction:    0.8 (first component = sliding friction)
    # MuJoCo uses 3 friction components (sliding, torsional, rolling);
    # Chrono mu maps to the first (sliding) value.
    mat = chrono.ChContactMaterialData()
    mat.mu = 0.6   # sliding friction — default body surfaces
    mat.cr = 0.0   # restitution (inelastic, matches MuJoCo default)
    parser.SetDefaultContactMaterial(mat)

    parser.PopulateSystem(system)

    # Override foot friction to 0.8 to match MuJoCo Menagerie foot geom value.
    foot_mat = chrono.ChContactMaterialData()
    foot_mat.mu = 0.8
    foot_mat.cr = 0.0
    for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
        parser.SetBodyContactMaterial(name, foot_mat)

    # The URDF parser builds collision shapes but leaves collision disabled.
    # Activate it on every non-fixed body so the robot can contact the terrain.
    for body in system.GetBodies():
        if not body.IsFixed():
            body.EnableCollision(True)
    color_trunk_mesh(parser)
    return parser


def color_trunk_mesh(parser):
    pass


def create_visualizer(system):
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1280, 720)
    vis.SetWindowTitle("Chrono Go1 URDF Viewer")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(2.5, 1.5, 2.5), chrono.ChVector3d(0, 0.4, 0))
    vis.AddTypicalLights()
    return vis


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find URDF: {MODEL_PATH}")

    system = create_system()
    terrain = create_soil(system)
    parser = load_go1(system)
    vis = create_visualizer(system)

    print("Loaded Go1 URDF:", MODEL_PATH)
    print("Bodies:", len(system.GetBodies()))
    print("Links:", len(system.GetLinks()))

    while vis.Run():
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        terrain.Synchronize(system.GetChTime())
        system.DoStepDynamics(TIME_STEP)
        terrain.Advance(TIME_STEP)


if __name__ == "__main__":
    main()
