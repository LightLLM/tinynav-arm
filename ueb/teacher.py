from __future__ import annotations

import numpy as np

from .env import BodyEnv


class PoseHoldTeacher:
    """A hand-written PD controller that tries to hold the reset pose.

    It exists only to produce consistent demonstrations for checking the training pipeline end to
    end. It is not a competent humanoid controller and stands in for teleoperated data.
    """

    def __init__(self, env: BodyEnv, kp: float = 1.0, kd: float = 0.1) -> None:
        self.env, self.kp, self.kd = env, kp, kd
        self.target = np.zeros(env.spec.num_joints, dtype=np.float32)

    def reset(self, joints: np.ndarray) -> None:
        self.target = joints[:, 0].copy()

    def act(self, joints: np.ndarray) -> np.ndarray:
        command = self.kp * (self.target - joints[:, 0]) - self.kd * joints[:, 1]
        return np.clip(command, self.env.ctrl_low, self.env.ctrl_high).astype(np.float32)
