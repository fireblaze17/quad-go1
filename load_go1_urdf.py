from pathlib import Path

import pychrono as chrono
import pychrono.parsers as parsers


MODEL_PATH = Path("models/go1/go1_chrono.urdf")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find URDF: {MODEL_PATH}")

    system = chrono.ChSystemSMC()
    system.SetGravityY()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    parser = parsers.ChParserURDF(str(MODEL_PATH))

    parser.SetRootInitPose(
        chrono.ChFramed(
            chrono.ChVector3d(0, 1.0, 0),
            chrono.ChQuaterniond(1, 0, 0, 0),
        )
    )

    print("Loaded parser for:", MODEL_PATH)

    print("\nURDF model bodies:")
    parser.PrintModelBodies()

    print("\nURDF model joints:")
    parser.PrintModelJoints()

    parser.PopulateSystem(system)

    print("\nChrono bodies after PopulateSystem:")
    parser.PrintChronoBodies()

    print("\nChrono joints after PopulateSystem:")
    parser.PrintChronoJoints()

    print("\nSystem summary:")
    print("number of bodies:", len(system.GetBodies()))
    print("number of links:", len(system.GetLinks()))


if __name__ == "__main__":
    main()
