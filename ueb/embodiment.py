from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOINT_FEATURES = 12
ROOT_FEATURES = 16


@dataclass(frozen=True)
class EmbodimentSpec:
    """A robot body described as data, so the policy can take the body as an input.

    One row of ``joint_features`` per actuated joint: joint type, axis, range, actuator gear,
    depth in the kinematic tree, and the mass and position of the link it moves.
    """

    name: str
    joint_names: tuple[str, ...]
    joint_features: np.ndarray  # [J, JOINT_FEATURES]
    joint_lower: np.ndarray  # [J] radians or metres
    joint_upper: np.ndarray  # [J]
    control_hz: float

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


def spec_from_mujoco(name: str, model, control_hz: float) -> EmbodimentSpec:
    """Build the body descriptor straight from a MuJoCo model (works for any morphology)."""
    import mujoco

    names, rows, lower, upper = [], [], [], []
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        body = int(model.jnt_bodyid[j])
        depth, b = 0, body
        while b > 0:
            b, depth = int(model.body_parentid[b]), depth + 1
        limited = bool(model.jnt_limited[j])
        lo, hi = (model.jnt_range[j] if limited else (-np.pi, np.pi))
        jtype = int(model.jnt_type[j])
        rows.append(
            [
                float(jtype == mujoco.mjtJoint.mjJNT_HINGE),
                float(jtype == mujoco.mjtJoint.mjJNT_SLIDE),
                *model.jnt_axis[j],
                lo / np.pi,
                hi / np.pi,
                np.log10(max(float(model.actuator_gear[a, 0]), 1.0)) / 3.0,
                depth / 10.0,
                np.log10(max(float(model.body_mass[body]), 1e-3)) / 2.0,
                float(np.linalg.norm(model.body_pos[body])),
                float(limited),
            ]
        )
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}")
        lower.append(lo)
        upper.append(hi)
    features = np.asarray(rows, dtype=np.float32)
    assert features.shape[1] == JOINT_FEATURES
    return EmbodimentSpec(
        name=name,
        joint_names=tuple(names),
        joint_features=features,
        joint_lower=np.asarray(lower, dtype=np.float32),
        joint_upper=np.asarray(upper, dtype=np.float32),
        control_hz=control_hz,
    )
