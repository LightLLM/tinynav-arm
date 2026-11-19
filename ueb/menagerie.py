"""Real robot models from MuJoCo Menagerie behind the same per-joint interface as ``ueb.env``.

Get the models with:
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
or point ``UEB_MENAGERIE`` at an existing checkout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embodiment import ROOT_FEATURES, EmbodimentSpec, spec_from_mujoco
from .env import Observation

CONTROL_HZ = 50.0
POSITION_STEP_RAD = 0.25  # a position_delta command of 1.0 moves the target this far from the joint


@dataclass(frozen=True)
class BodyInfo:
    scene: str
    keyframe: str
    command_mode: str
    robot_class: str
    episode_steps: int
    reset_noise: float  # uniform noise added to actuated joint positions (or drone pose) at reset


MENAGERIE_BODIES = {
    "g1": BodyInfo("unitree_g1/scene.xml", "stand", "position_delta", "humanoid", 500, 0.03),
    "ur5e": BodyInfo("universal_robots_ur5e/scene.xml", "home", "position_delta", "industrial arm", 150, 0.8),
    "x2": BodyInfo("skydio_x2/scene.xml", "hover", "effort", "drone", 500, 0.15),
    "go2": BodyInfo("unitree_go2/scene.xml", "home", "effort", "quadruped", 500, 0.05),
}


def menagerie_root() -> Path:
    root = Path(os.environ.get("UEB_MENAGERIE", Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"))
    if not (root / "unitree_g1").is_dir():
        raise FileNotFoundError(f"MuJoCo Menagerie not found at {root}. See the docstring of ueb.menagerie.")
    return root


class MenagerieEnv:
    """One Menagerie robot with a fixed task: hold a reference pose (stand, hover, or return home)."""

    def __init__(self, body: str) -> None:
        import mujoco

        self._mj = mujoco
        self.body, self.info = body, MENAGERIE_BODIES[body]
        self.model = mujoco.MjModel.from_xml_path(str(menagerie_root() / self.info.scene))
        self.data = mujoco.MjData(self.model)
        self.key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.info.keyframe)
        self.decimation = max(1, round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.spec: EmbodimentSpec = spec_from_mujoco(body, self.model, CONTROL_HZ, self.info.command_mode)

        m = self.model
        self._is_joint = m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
        joint_ids = np.where(self._is_joint, m.actuator_trnid[:, 0], 0)
        self._qpos_idx = m.jnt_qposadr[joint_ids]
        self._qvel_idx = m.jnt_dofadr[joint_ids]
        self.floating = bool(m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
        self._root_body = int(m.jnt_bodyid[0])
        self._ctrl_mid = m.actuator_ctrlrange.mean(axis=1)
        self._ctrl_half = (m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]) / 2
        self.ctrl_low = -np.ones(m.nu, dtype=np.float32)
        self.ctrl_high = np.ones(m.nu, dtype=np.float32)
        self.reference_qpos = m.key_qpos[self.key].copy()
        self.reference_joints = self.reference_qpos[self._qpos_idx].astype(np.float32)
        self.steps = 0

    # -- observation ---------------------------------------------------------------------------
    def observe(self) -> Observation:
        d = self.data
        joints = np.stack([d.qpos[self._qpos_idx], d.qvel[self._qvel_idx]], axis=1) * self._is_joint[:, None]
        root = np.zeros(ROOT_FEATURES, dtype=np.float32)
        if self.floating:
            # Position relative to the reference, orientation, and body velocities.
            raw = np.concatenate([d.qpos[:3] - self.reference_qpos[:3], d.qpos[3:7], d.qvel[:6]])
            root[: len(raw)] = raw
        rot = d.xmat[self._root_body].reshape(3, 3)
        return Observation(joints.astype(np.float32), np.clip(root, -10, 10),
                           float(d.xpos[self._root_body][2]), float(rot[2, 2]))

    # -- control -------------------------------------------------------------------------------
    def reset(self, seed: int = 0) -> Observation:
        mj, m, d = self._mj, self.model, self.data
        rng = np.random.default_rng(seed)
        mj.mj_resetDataKeyframe(m, d, self.key)
        noise = self.info.reset_noise
        if self.info.robot_class == "drone":
            d.qpos[:3] += rng.uniform(-noise, noise, 3)
            d.qvel[:3] = rng.uniform(-noise, noise, 3)
        else:
            idx = self._qpos_idx[self._is_joint]
            d.qpos[idx] += rng.uniform(-noise, noise, len(idx))
        mj.mj_forward(m, d)
        self.steps = 0
        return self.observe()

    def step(self, command: np.ndarray) -> tuple[Observation, bool]:
        """``command`` is one value in [-1, 1] per actuator. Returns the observation and whether the episode ended in failure."""
        mj, m, d = self._mj, self.model, self.data
        command = np.clip(np.nan_to_num(command), -1.0, 1.0)
        if self.spec.command_mode == "position_delta":
            target = d.qpos[self._qpos_idx] + command * POSITION_STEP_RAD
            d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        else:
            d.ctrl[:] = self._ctrl_mid + command * self._ctrl_half
        for _ in range(self.decimation):
            mj.mj_step(m, d)
        self.steps += 1
        return self.observe(), self.failed()

    def failed(self) -> bool:
        d, cls = self.data, self.info.robot_class
        upright = d.xmat[self._root_body].reshape(3, 3)[2, 2]
        if cls == "humanoid":
            return bool(d.qpos[2] < 0.5 or upright < 0.5)
        if cls == "quadruped":
            return bool(d.qpos[2] < 0.15 or upright < 0.5)
        if cls == "drone":
            return bool(np.linalg.norm(d.qpos[:3] - self.reference_qpos[:3]) > 1.0 or upright < 0.5 or d.qpos[2] < 0.05)
        return False

    def success(self) -> bool:
        """Task outcome at the end of an episode."""
        if self.info.robot_class == "industrial arm":
            return bool(np.abs(self.data.qpos[self._qpos_idx] - self.reference_joints).max() < 0.05)
        return not self.failed() and self.steps >= self.info.episode_steps

    def close(self) -> None:
        pass


class ReferenceTeacher:
    """Scripted controllers that hold each body's reference pose. They stand in for teleoperated data.

    Position-controlled bodies step their joint targets toward the reference; the quadruped uses a
    joint PD loop; the drone uses a cascaded position and attitude PD loop with motor mixing.
    """

    def __init__(self, env: MenagerieEnv) -> None:
        self.env = env
        m = env.model
        if env.info.robot_class == "drone":
            sites = m.actuator_trnid[:, 0]
            pos = m.site_pos[sites]
            mix = np.stack([np.ones(m.nu), pos[:, 1], -pos[:, 0], m.actuator_gear[:, 5]])
            self._unmix = np.linalg.pinv(mix)
            self._mass = float(m.body_subtreemass[env._root_body])

    def reset(self, joints: np.ndarray) -> None:
        pass

    def act(self, joints: np.ndarray) -> np.ndarray:
        env, d = self.env, self.env.data
        cls = env.info.robot_class
        if env.spec.command_mode == "position_delta":
            return np.clip((env.reference_joints - joints[:, 0]) / POSITION_STEP_RAD, -1, 1).astype(np.float32)
        if cls == "quadruped":
            torque = 60.0 * (env.reference_joints - joints[:, 0]) - 3.0 * joints[:, 1]
            return np.clip((torque - env._ctrl_mid) / env._ctrl_half, -1, 1).astype(np.float32)
        # Drone: world-frame position error -> desired tilt and thrust -> body torques -> rotor thrusts.
        err = env.reference_qpos[:3] - d.qpos[:3]
        vel = d.qvel[:3]
        rot = d.xmat[env._root_body].reshape(3, 3)
        want = 4.0 * err - 3.0 * vel  # desired world acceleration
        thrust = self._mass * (9.81 + want[2]) / max(rot[2, 2], 0.3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        ax = np.cos(yaw) * want[0] + np.sin(yaw) * want[1]
        ay = -np.sin(yaw) * want[0] + np.cos(yaw) * want[1]
        want_pitch, want_roll = np.clip(ax / 9.81, -0.4, 0.4), np.clip(-ay / 9.81, -0.4, 0.4)
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = -np.arcsin(np.clip(rot[2, 0], -1, 1))
        omega = d.qvel[3:6]
        torque = np.array([3.0 * (want_roll - roll) - 0.5 * omega[0],
                           3.0 * (want_pitch - pitch) - 0.5 * omega[1],
                           -0.5 * yaw - 0.2 * omega[2]])
        rotors = self._unmix @ np.concatenate([[thrust], torque])
        return np.clip((rotors - env._ctrl_mid) / env._ctrl_half, -1, 1).astype(np.float32)
