import numpy as np
import pytest

from robosim.arm import HOME, Arm


def test_forward_kinematics_of_known_poses():
    arm = Arm()
    # Straight up: the link lengths add along Z.
    assert np.allclose(arm.forward(np.zeros(6)), [0, 0, 0.10 + 0.12 + 0.40 + 0.35 + 0.10 + 0.08 + 0.08])
    # Shoulder at 90 degrees: everything above the shoulder joint lies along X.
    bent = arm.forward(np.array([0, np.pi / 2, 0, 0, 0, 0]))
    assert np.allclose(bent, [1.01, 0, 0.22], atol=1e-9)
    # Turning the base swings that reach from X to Y.
    assert np.allclose(arm.forward(np.array([np.pi / 2, np.pi / 2, 0, 0, 0, 0])), [0, 1.01, 0.22], atol=1e-9)


def test_inverse_kinematics_reaches_reachable_targets():
    arm = Arm()
    rng = np.random.default_rng(1)
    for _ in range(20):
        target = np.array([rng.uniform(0.25, 0.6), rng.uniform(-0.4, 0.4), rng.uniform(0.1, 0.6)])
        angles, miss = arm.inverse(target)
        assert miss < 1e-3
        assert np.linalg.norm(arm.forward(angles) - target) < 1e-3
        assert (angles >= arm.lower).all() and (angles <= arm.upper).all()


def test_inverse_kinematics_reports_unreachable_targets():
    _, miss = Arm().inverse(np.array([2.0, 0.0, 0.5]))
    assert miss > 0.5


def test_physics_agrees_with_kinematics_and_reaches_the_target():
    pytest.importorskip("mujoco")
    from robosim.arm import PhysicsArm

    arm, physics = Arm(), PhysicsArm()
    assert np.allclose(physics.tool, arm.forward(HOME), atol=1e-6)
    target = np.array([0.45, 0.2, 0.3])
    angles, _ = arm.inverse(target)
    assert np.linalg.norm(physics.move_to(angles, 2.0) - target) < 0.002


def test_gravity_pulls_the_arm_low_without_compensation():
    pytest.importorskip("mujoco")
    from robosim.arm import PhysicsArm

    arm = Arm()
    target = np.array([0.45, 0.2, 0.3])
    angles, _ = arm.inverse(target)
    reached = PhysicsArm(gravity_compensation=False).move_to(angles, 2.0)
    assert reached[2] < target[2] - 0.005
