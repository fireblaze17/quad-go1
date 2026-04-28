from view_go1_urdf import TIME_STEP, create_soil, create_system, load_go1


def main():
    system = create_system()
    terrain = create_soil(system)
    parser = load_go1(system)

    trunk = parser.GetChBody("trunk")
    if trunk is None:
        raise RuntimeError("Could not find trunk body.")

    print("trunk fixed:", trunk.IsFixed())
    print("trunk mass:", trunk.GetMass())
    print("initial trunk y:", trunk.GetPos().y)

    for step in range(1000):
        terrain.Synchronize(system.GetChTime())
        system.DoStepDynamics(TIME_STEP)
        terrain.Advance(TIME_STEP)

        if step % 100 == 0:
            print(
                "time:",
                round(system.GetChTime(), 3),
                "trunk y:",
                round(trunk.GetPos().y, 4),
                "trunk vy:",
                round(trunk.GetPosDt().y, 4),
                "contacts:",
                system.GetNumContacts(),
            )


if __name__ == "__main__":
    main()

