import pychrono as chrono
import pychrono.irrlicht as irr


TIME_STEP = 0.001
SOLVER_ITERATIONS = 50


def create_system():
    """Create the Chrono physics world."""
    system = chrono.ChSystemNSC()

    # Chrono's Irrlicht demos use Y as the up direction.
    system.SetGravityY()

    # Bullet handles collision detection between the box and ground.
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    # More solver iterations make contact handling more stable.
    system.SetSolverType(chrono.ChSolver.Type_PSOR)
    system.GetSolver().AsIterative().SetMaxIterations(SOLVER_ITERATIONS)

    return system


def create_contact_material():
    """Create a simple material shared by the ground and box."""
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(0.5)
    material.SetRestitution(0.1)
    return material


def add_ground(system, material):
    """Add a fixed ground box to the Chrono system."""
    ground = chrono.ChBodyEasyBox(
        5.0,   # x size
        0.2,   # y thickness
        5.0,   # z size
        1000,  # density
        True,  # visualize
        True,  # collide
        material,
    )
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(0, 0, 0))
    system.AddBody(ground)
    return ground


def add_falling_box(system, material):
    """Add a dynamic box above the ground."""
    box = chrono.ChBodyEasyBox(
        0.3,
        0.3,
        0.3,
        1000,
        True,
        True,
        material,
    )
    box.SetPos(chrono.ChVector3d(0, 1.0, 0))
    system.AddBody(box)
    return box


def create_visualizer(system):
    """Create the Irrlicht window for live visualization."""
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle("Chrono Falling Box")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(2, 2, 2))
    vis.AddTypicalLights()
    return vis


def main():
    system = create_system()
    material = create_contact_material()

    add_ground(system, material)
    box = add_falling_box(system, material)
    vis = create_visualizer(system)

    step = 0

    while vis.Run():
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        system.DoStepDynamics(TIME_STEP)

        if step % 100 == 0:
            print(
                "time:",
                round(system.GetChTime(), 3),
                "box y:",
                round(box.GetPos().y, 4),
                "contacts:",
                system.GetNumContacts(),
            )

        step += 1


if __name__ == "__main__":
    main()

