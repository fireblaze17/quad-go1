"""Small Chrono SCM terrain smoke test.

This script keeps the deformable-soil setup separate from the Go1 Gymnasium env.
It is useful when checking that PyChrono, Irrlicht, and SCMTerrain are installed
correctly before debugging robot-specific behavior.
"""

import math

import pychrono as chrono
import pychrono.irrlicht as irr
import pychrono.vehicle as veh


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

    # SCMTerrain defaults to Z-up. Rotate it so this smoke test matches the
    # Y-up convention used by go1_env.py.
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


def add_test_box(system):
    material = chrono.ChContactMaterialSMC()
    material.SetFriction(0.8)
    material.SetRestitution(0.0)

    box = chrono.ChBodyEasyBox(0.4, 0.4, 0.4, 800, True, True, material)
    box.SetPos(chrono.ChVector3d(0, 1.0, 0))
    box.GetVisualShape(0).SetColor(chrono.ChColor(0.1, 0.4, 1.0))
    system.AddBody(box)
    return box


def create_visualizer(system):
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1280, 720)
    vis.SetWindowTitle("Chrono Go1 Soil Demo")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(2.5, 2.0, 3.0), chrono.ChVector3d(0, 0, 0))
    vis.AddTypicalLights()
    return vis


def main():
    system = create_system()
    terrain = create_soil(system)
    box = add_test_box(system)
    vis = create_visualizer(system)
    step = 0

    while vis.Run():
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        terrain.Synchronize(system.GetChTime())
        system.DoStepDynamics(TIME_STEP)
        terrain.Advance(TIME_STEP)

        if step % 100 == 0:
            print(
                "time:",
                round(system.GetChTime(), 3),
                "box y:",
                round(box.GetPos().y, 4),
            )

        step += 1


if __name__ == "__main__":
    main()
