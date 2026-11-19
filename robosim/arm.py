from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "public" / "robots" / "arm6.urdf"
HOME = np.array([0.0, 0.6, 1.0, 0.5, 0.0, 0.0])


@dataclass(frozen=True)
class Joint:
    name: str
    origin: np.ndarray  # position of the joint in its parent link's frame
    axis: np.ndarray  # rotation axis, zero for a fixed joint
    lower: float
    upper: float


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a turn of ``angle`` radians about ``axis`` (Rodrigues' formula)."""
    x, y, z = axis / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class Arm:
    """Kinematics of a serial arm read from a URDF. No physics; see ``PhysicsArm`` for that."""

    def __init__(self, urdf: str | Path = DEFAULT_URDF) -> None:
        self.urdf = Path(urdf)
        chain = []
        for node in ET.parse(self.urdf).getroot().findall("joint"):
            origin = np.array([float(v) for v in node.find("origin").get("xyz").split()])
            fixed = node.get("type") == "fixed"
            axis = np.zeros(3) if fixed else np.array([float(v) for v in node.find("axis").get("xyz").split()])
            limit = node.find("limit")
            lower, upper = (0.0, 0.0) if fixed else (float(limit.get("lower")), float(limit.get("upper")))
            chain.append(Joint(node.get("name"), origin, axis, lower, upper))
        self.chain = chain
        self.joints = [j for j in chain if j.axis.any()]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])

    def forward(self, angles: np.ndarray) -> np.ndarray:
        """Tool position in metres for the given joint angles in radians."""
        rotation, position, i = np.eye(3), np.zeros(3), 0
        for joint in self.chain:
            position = position + rotation @ joint.origin
            if joint.axis.any():
                rotation = rotation @ _rotation(joint.axis, angles[i])
                i += 1
        return position

    def jacobian(self, angles: np.ndarray, step: float = 1e-6) -> np.ndarray:
        """How the tool position changes with each joint angle (3 x number of joints)."""
        base = self.forward(angles)
        columns = []
        for i in range(len(angles)):
            nudged = np.array(angles, dtype=float)
            nudged[i] += step
            columns.append((self.forward(nudged) - base) / step)
        return np.stack(columns, axis=1)

    def inverse(self, target: np.ndarray, start: np.ndarray = HOME, iterations: int = 200,
                damping: float = 0.05, tolerance: float = 1e-4) -> tuple[np.ndarray, float]:
        """Joint angles that put the tool at ``target``, by damped least squares.

        Returns the angles and the remaining distance in metres. A distance well above the
        tolerance means the target is out of reach.
        """
        angles = np.array(start, dtype=float)
        error = target - self.forward(angles)
        for _ in range(iterations):
            if np.linalg.norm(error) < tolerance:
                break
            jac = self.jacobian(angles)
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
            angles = np.clip(angles + np.clip(step, -0.2, 0.2), self.lower, self.upper)
            error = target - self.forward(angles)
        return angles, float(np.linalg.norm(error))


class PhysicsArm:
    """The same URDF simulated with MuJoCo: gravity, inertia and a position servo on every joint."""

    def __init__(self, urdf: str | Path = DEFAULT_URDF, stiffness: float = 400.0, damping: float = 40.0,
                 gravity_compensation: bool = True) -> None:
        import mujoco

        self.gravity_compensation = gravity_compensation
        self._mj = mujoco
        self.kinematics = Arm(urdf)
        spec = mujoco.MjSpec.from_file(str(urdf))
        # Stiff servos on light wrist links need an implicit integrator and some rotor inertia
        # to stay numerically stable at a 2 ms time step.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for joint in spec.joints:
            joint.armature = 0.05
        for joint in self.kinematics.joints:
            scale = 0.25 if joint.name.startswith("wrist") else 1.0
            actuator = spec.add_actuator()
            actuator.name = f"{joint.name}_servo"
            actuator.target = joint.name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = stiffness * scale
            actuator.biasprm[:3] = [0.0, -stiffness * scale, -damping * scale]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._tool_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3")
        self._tool_offset = self.kinematics.chain[-1].origin
        self.reset()

    def reset(self, angles: np.ndarray = HOME) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = angles
        self.data.ctrl[:] = angles
        self._mj.mj_forward(self.model, self.data)

    @property
    def angles(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def tool(self) -> np.ndarray:
        rotation = self.data.xmat[self._tool_body].reshape(3, 3)
        return self.data.xpos[self._tool_body] + rotation @ self._tool_offset

    def move_to(self, angles: np.ndarray, seconds: float = 2.0) -> np.ndarray:
        """Command joint targets and step the physics. Returns the tool position afterwards."""
        self.data.ctrl[:] = np.clip(angles, self.kinematics.lower, self.kinematics.upper)
        for _ in range(int(seconds / self.model.opt.timestep)):
            if self.gravity_compensation:
                # Feed forward the torque that holds the arm against gravity, as real arm
                # controllers do. Without it the servos settle a few centimetres low.
                self.data.qfrc_applied[:] = self.data.qfrc_bias
            self._mj.mj_step(self.model, self.data)
        return self.tool
