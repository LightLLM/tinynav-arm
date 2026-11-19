from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embodiment import ROOT_FEATURES, EmbodimentSpec, spec_from_mujoco

# Gymnasium MuJoCo bodies usable with the same wrapper. The humanoid is the primary target.
BODIES = {"humanoid": "Humanoid-v5", "ant": "Ant-v5", "walker": "Walker2d-v5"}


@dataclass
class Observation:
    joints: np.ndarray  # [J, 2] position and velocity of each actuated joint
    root: np.ndarray  # [ROOT_FEATURES] floating-base state (height, orientation, velocities)
    height: float  # world z of the root body, used by the safety reflex
    upright: float  # cosine between the torso's up axis and world up (1 = upright)


class BodyEnv:
    """A MuJoCo robot exposed as per-joint tokens instead of one body-specific vector."""

    def __init__(self, body: str = "humanoid", seed: int = 0) -> None:
        import gymnasium as gym

        self.body = body
        self.env = gym.make(BODIES[body])
        self.sim = self.env.unwrapped
        self.model, self.data = self.sim.model, self.sim.data
        self.spec: EmbodimentSpec = spec_from_mujoco(body, self.model, 1.0 / self.sim.dt)
        joint_ids = self.model.actuator_trnid[:, 0]
        self._qpos_idx = self.model.jnt_qposadr[joint_ids]
        self._qvel_idx = self.model.jnt_dofadr[joint_ids]
        self._root_qpos = np.setdiff1d(np.arange(self.model.nq), self._qpos_idx)
        self._root_qvel = np.setdiff1d(np.arange(self.model.nv), self._qvel_idx)
        self._root_body = int(self.model.jnt_bodyid[0])
        self._seed = seed
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)

    def observe(self) -> Observation:
        joints = np.stack([self.data.qpos[self._qpos_idx], self.data.qvel[self._qvel_idx]], axis=1)
        # Drop the first two root coordinates (global x, y) so the policy is translation invariant.
        raw = np.concatenate([self.data.qpos[self._root_qpos][2:], self.data.qvel[self._root_qvel]])
        root = np.zeros(ROOT_FEATURES, dtype=np.float32)
        root[: min(len(raw), ROOT_FEATURES)] = raw[:ROOT_FEATURES]
        rot = self.data.xmat[self._root_body].reshape(3, 3)
        return Observation(
            joints=joints.astype(np.float32),
            root=np.clip(root, -10, 10),
            height=float(self.data.xpos[self._root_body][2]),
            upright=float(rot[2, 2]),
        )

    def reset(self, seed: int | None = None) -> Observation:
        self.env.reset(seed=self._seed if seed is None else seed)
        return self.observe()

    def step(self, action: np.ndarray) -> tuple[Observation, bool]:
        """Apply one normalised motor command per joint. Returns the observation and whether the body fell."""
        action = np.clip(action, self.ctrl_low, self.ctrl_high)
        _, _, terminated, _, _ = self.env.step(action)
        return self.observe(), bool(terminated)

    def close(self) -> None:
        self.env.close()
