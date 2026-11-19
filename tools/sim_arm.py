"""Move the simulated six-axis arm's tool to a point and report what happened.

Solves inverse kinematics for the target, then drives the arm there in MuJoCo with gravity on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robosim.arm import Arm, PhysicsArm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, nargs=3, default=[0.45, 0.2, 0.3], metavar=("X", "Y", "Z"),
                        help="tool position in metres")
    parser.add_argument("--seconds", type=float, default=2.0, help="simulated time allowed for the move")
    parser.add_argument("--no-gravity-compensation", action="store_true")
    args = parser.parse_args()

    target = np.array(args.target)
    arm = Arm()
    angles, miss = arm.inverse(target)
    print(f"target (m):               {target.round(3).tolist()}")
    print(f"joint angles (deg):       {np.degrees(angles).round(1).tolist()}")
    if miss > 1e-3:
        print(f"out of reach: the closest the tool gets is {miss * 100:.1f} cm away")
        raise SystemExit(1)

    physics = PhysicsArm(gravity_compensation=not args.no_gravity_compensation)
    reached = physics.move_to(angles, args.seconds)
    print(f"tool after {args.seconds:g} s (m):       {reached.round(3).tolist()}")
    print(f"distance from target:     {np.linalg.norm(reached - target) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
