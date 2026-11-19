from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env import BODIES, BodyEnv
from .menagerie import MenagerieEnv, ReferenceTeacher
from .model import BrainConfig, UnifiedBrain
from .safety import PRESETS, ReflexLayer
from .teacher import PoseHoldTeacher


def make_env(body: str):
    """Gymnasium toy bodies and Menagerie robot models share one interface."""
    return BodyEnv(body) if body in BODIES else MenagerieEnv(body)


def make_teacher(env):
    return PoseHoldTeacher(env) if isinstance(env, BodyEnv) else ReferenceTeacher(env)


def episode_limit(env, default: int = 300) -> int:
    return env.info.episode_steps if isinstance(env, MenagerieEnv) else default


VELOCITY_SCALE = 0.1  # joint velocities are an order of magnitude larger than positions


def joint_tensor(joints: np.ndarray) -> np.ndarray:
    scaled = joints.copy()
    scaled[..., 1] *= VELOCITY_SCALE
    return np.clip(scaled, -5, 5)


@dataclass
class BodyData:
    env: BodyEnv
    embodiment_id: int
    joint_features: torch.Tensor
    joints: torch.Tensor  # [N, J, 2]
    root: torch.Tensor  # [N, R]
    actions: torch.Tensor  # [N, J, H]
    future_joints: torch.Tensor  # [N, J, 2]

    def batch(self, size: int, generator: torch.Generator) -> dict:
        idx = torch.randint(0, self.joints.shape[0], (size,), generator=generator)
        return {"joints": self.joints[idx], "root": self.root[idx], "actions": self.actions[idx],
                "future_joints": self.future_joints[idx]}


def collect(body: str, embodiment_id: int, episodes: int, horizon: int, max_steps: int | None = None,
            noise: float = 0.05, seed: int = 0) -> BodyData:
    """Roll out the teacher with small exploration noise and record clean teacher actions as labels."""
    env = make_env(body)
    teacher = make_teacher(env)
    max_steps = max_steps or episode_limit(env)
    rng = np.random.default_rng(seed)
    joints, roots, chunks, futures = [], [], [], []
    for episode in range(episodes):
        obs = env.reset(seed=seed * 10_000 + episode)
        teacher.reset(obs.joints)
        ep_joints, ep_root, ep_actions = [], [], []
        for _ in range(max_steps):
            action = teacher.act(obs.joints)
            ep_joints.append(joint_tensor(obs.joints))
            ep_root.append(obs.root)
            ep_actions.append(action)
            obs, fell = env.step(action + rng.normal(0, noise, action.shape).astype(np.float32))
            if fell:
                break
        ep_joints.append(joint_tensor(obs.joints))
        for t in range(len(ep_actions) - horizon + 1):
            joints.append(ep_joints[t])
            roots.append(ep_root[t])
            chunks.append(np.stack(ep_actions[t : t + horizon], axis=1))
            futures.append(ep_joints[t + horizon])
    as_tensor = lambda rows: torch.from_numpy(np.asarray(rows, dtype=np.float32))
    return BodyData(env, embodiment_id, torch.from_numpy(env.spec.joint_features), as_tensor(joints),
                    as_tensor(roots), as_tensor(chunks), as_tensor(futures))


def train(datasets: list[BodyData], steps: int, batch_size: int = 128, lr: float = 3e-4, seed: int = 0,
          config: BrainConfig | None = None, log_every: int = 0) -> UnifiedBrain:
    """Train one set of weights on every body in ``datasets``, one body per gradient step in rotation."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = UnifiedBrain(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        data = datasets[step % len(datasets)]
        loss, parts = model.loss(data.batch(batch_size, generator), data.joint_features, data.embodiment_id,
                                 data.env.spec.control_hz)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  {data.env.body:9s} action={parts['action']:.4f} predict={parts['predict']:.4f}")
    return model.eval()


@dataclass
class RolloutStats:
    mean_steps: float
    steps: list[int]
    reflex_modified_rate: float
    reflex_stops: int
    success_rate: float | None = None


def rollout(env, policy, episodes: int = 10, max_steps: int | None = None, use_reflex: bool = True,
            seed_offset: int = 50_000) -> RolloutStats:
    """Run ``policy(obs) -> command`` for several seeds. Episode length is steps before falling."""
    reflex = ReflexLayer(env.spec, PRESETS[env.body])
    max_steps = max_steps or episode_limit(env)
    lengths, modified, total, stops, successes = [], 0, 0, 0, []
    for episode in range(episodes):
        obs = env.reset(seed=seed_offset + episode)
        reflex.reset()
        if hasattr(policy, "reset"):
            policy.reset(obs)
        count = 0
        for count in range(1, max_steps + 1):
            command = policy(obs)
            if use_reflex:
                decision = reflex.filter(command, obs.joints[:, 0], obs.height, obs.upright)
                command = decision.command
                modified += int(decision.modified)
                stops += int(decision.action.value == "STOP")
            total += 1
            obs, fell = env.step(command)
            if fell:
                break
        lengths.append(count)
        if hasattr(env, "success"):
            successes.append(env.success())
    return RolloutStats(float(np.mean(lengths)), lengths, modified / max(total, 1), stops,
                        float(np.mean(successes)) if successes else None)


class BrainPolicy:
    """Receding-horizon execution: sample an action chunk, run the first few steps, then replan."""

    def __init__(self, model: UnifiedBrain, env, embodiment_id: int, replan_every: int = 4,
                 flow_steps: int = 8, seed: int = 0) -> None:
        self.model, self.env, self.embodiment_id = model, env, embodiment_id
        self.replan_every, self.flow_steps = replan_every, flow_steps
        self.features = torch.from_numpy(env.spec.joint_features)
        self.generator = torch.Generator().manual_seed(seed)
        self.chunk, self.index = None, 0

    def reset(self, obs) -> None:
        self.chunk, self.index = None, 0

    def __call__(self, obs) -> np.ndarray:
        if self.chunk is None or self.index >= self.replan_every:
            joints = torch.from_numpy(joint_tensor(obs.joints))[None]
            root = torch.from_numpy(obs.root)[None]
            self.chunk = self.model.sample(self.features, joints, root, self.embodiment_id,
                                           self.env.spec.control_hz, self.flow_steps, self.generator)[0].numpy()
            self.index = 0
        command = self.chunk[:, self.index]
        self.index += 1
        return command.astype(np.float32)


class TeacherPolicy:
    def __init__(self, env) -> None:
        self.teacher = make_teacher(env)

    def reset(self, obs) -> None:
        self.teacher.reset(obs.joints)

    def __call__(self, obs) -> np.ndarray:
        return self.teacher.act(obs.joints)


class ZeroPolicy:
    def __init__(self, env) -> None:
        self.size = env.spec.num_joints

    def __call__(self, obs) -> np.ndarray:
        return np.zeros(self.size, dtype=np.float32)
