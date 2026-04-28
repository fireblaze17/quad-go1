from view_go1_urdf import TIME_STEP, create_soil, create_system, load_go1


BODY_NAMES = [
    "trunk",
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
    "FR_calf",
    "FL_calf",
    "RR_calf",
    "RL_calf",
]


def print_body_info(parser):
    for name in BODY_NAMES:
        body = parser.GetChBody(name)
        if body is None:
            print(name, "missing")
            continue

        print(
            name,
            "collision_enabled=",
            body.IsCollisionEnabled(),
            "mass=",
            round(body.GetMass(), 6),
            "pos_y=",
            round(body.GetPos().y, 4),
            "visual_shapes=",
            body.GetVisualModel().GetNumShapes(),
        )


def main():
    system = create_system()
    terrain = create_soil(system)
    parser = load_go1(system)
    trunk = parser.GetChBody("trunk")

    print("Initial body info")
    print("-----------------")
    print_body_info(parser)

    print()
    print("Fall/contact trace")
    print("------------------")
    for step in range(800):
        terrain.Synchronize(system.GetChTime())
        system.DoStepDynamics(TIME_STEP)
        terrain.Advance(TIME_STEP)

        if step % 100 == 0:
            print(
                "time=",
                round(system.GetChTime(), 3),
                "trunk_y=",
                round(trunk.GetPos().y, 4),
                "trunk_vy=",
                round(trunk.GetPosDt().y, 4),
                "contacts=",
                system.GetNumContacts(),
            )


if __name__ == "__main__":
    main()

