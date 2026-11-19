from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .embodiment import JOINT_FEATURES, ROOT_FEATURES


@dataclass(frozen=True)
class BrainConfig:
    width: int = 128
    layers: int = 4
    heads: int = 4
    horizon: int = 8  # action chunk length, in control steps
    max_embodiments: int = 16
    num_tasks: int = 16


def _mlp(inputs: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))


class UnifiedBrain(nn.Module):
    """One transformer that controls any body whose joints are described to it.

    Every actuated joint contributes two tokens: its current state and its (noisy) action chunk.
    Both carry an embedding of that joint's physical description, so the number of joints is just
    the sequence length and no layer has a body-specific shape. The only per-body parameters are
    one learned soft-prompt vector per embodiment.

    Heads: a flow-matching action head (velocity field over the action chunk) and a prediction head
    that forecasts each joint's state at the end of the chunk.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or BrainConfig()
        self.joint_descriptor = _mlp(JOINT_FEATURES, c.width)
        self.joint_state = _mlp(2, c.width)
        self.root_state = _mlp(ROOT_FEATURES, c.width)
        self.action_in = _mlp(c.horizon, c.width)
        self.time_in = _mlp(c.width, c.width)
        self.rate_in = _mlp(1, c.width)
        self.soft_prompt = nn.Embedding(c.max_embodiments, c.width)
        self.task = nn.Embedding(c.num_tasks, c.width)
        self.kind = nn.Embedding(4, c.width)  # 0 context, 1 root, 2 joint state, 3 action
        layer = nn.TransformerEncoderLayer(c.width, c.heads, c.width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.width)
        self.action_out = nn.Linear(c.width, c.horizon)
        self.predict_out = nn.Linear(c.width, 2)

    def _time_features(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.config.width // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
        angles = tau[:, None] * 1000.0 * freqs[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(
        self,
        joint_features: torch.Tensor,  # [J, JOINT_FEATURES], the body description
        joints: torch.Tensor,  # [B, J, 2]
        root: torch.Tensor,  # [B, ROOT_FEATURES]
        noisy_actions: torch.Tensor,  # [B, J, H]
        tau: torch.Tensor,  # [B] flow time in [0, 1]
        embodiment_id: int,
        control_hz: float,
        task_id: torch.Tensor | None = None,  # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_joints, _ = joints.shape
        device = joints.device
        body = self.joint_descriptor(joint_features)[None]  # [1, J, W]
        if task_id is None:
            task_id = torch.zeros(batch, dtype=torch.long, device=device)
        emb_id = torch.full((batch,), embodiment_id, dtype=torch.long, device=device)
        rate = torch.full((batch, 1), math.log10(control_hz) / 3.0, device=device)
        context = self.soft_prompt(emb_id) + self.task(task_id) + self.rate_in(rate) + self.kind.weight[0]
        root_token = self.root_state(root) + self.kind.weight[1]
        state_tokens = self.joint_state(joints) + body + self.kind.weight[2]
        time_token = self.time_in(self._time_features(tau))[:, None]
        action_tokens = self.action_in(noisy_actions) + body + time_token + self.kind.weight[3]
        tokens = torch.cat([context[:, None], root_token[:, None], state_tokens, action_tokens], dim=1)
        hidden = self.norm(self.trunk(tokens))
        predicted_state = self.predict_out(hidden[:, 2 : 2 + num_joints])
        velocity = self.action_out(hidden[:, 2 + num_joints :])
        return velocity, predicted_state

    def loss(self, batch: dict, joint_features: torch.Tensor, embodiment_id: int, control_hz: float,
             predict_weight: float = 0.1) -> tuple[torch.Tensor, dict[str, float]]:
        """Conditional flow matching on the action chunk plus future-state prediction."""
        actions = batch["actions"]  # [B, J, H]
        noise = torch.randn_like(actions)
        tau = torch.rand(actions.shape[0], device=actions.device)
        mix = tau[:, None, None]
        noisy = mix * actions + (1 - mix) * noise
        velocity, predicted = self(joint_features, batch["joints"], batch["root"], noisy, tau,
                                   embodiment_id, control_hz, batch.get("task_id"))
        act = ((velocity - (actions - noise)) ** 2).mean()
        pred = (predicted - batch["future_joints"]).abs().mean()
        return act + predict_weight * pred, {"action": float(act.detach()), "predict": float(pred.detach())}

    @torch.no_grad()
    def sample(self, joint_features: torch.Tensor, joints: torch.Tensor, root: torch.Tensor,
               embodiment_id: int, control_hz: float, steps: int = 8,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Integrate the learned flow from noise to an action chunk. Returns [B, J, H]."""
        batch, num_joints, _ = joints.shape
        actions = torch.randn(batch, num_joints, self.config.horizon, generator=generator, device=joints.device)
        for i in range(steps):
            tau = torch.full((batch,), i / steps, device=joints.device)
            velocity, _ = self(joint_features, joints, root, actions, tau, embodiment_id, control_hz)
            actions = actions + velocity / steps
        return actions
