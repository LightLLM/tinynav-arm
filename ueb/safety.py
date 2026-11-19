from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinynav.types import Action

from .embodiment import EmbodimentSpec


@dataclass(frozen=True)
class ReflexConfig:
    max_command: float = 0.4  # absolute limit on any normalised motor command
    max_delta: float = 0.2  # largest allowed change in a command between control steps
    limit_margin_rad: float = 0.05  # start pushing back this close to a joint limit
    min_upright: float = 0.5  # torso tilt (cosine to vertical) below which the robot is falling
    min_height_m: float = 0.8  # root height below which the robot is falling


@dataclass(frozen=True)
class ReflexDecision:
    action: Action
    command: np.ndarray
    reason: str
    modified: bool


# Per-body limits. Heights are the root-body height below which the body counts as fallen.
PRESETS = {
    "humanoid": ReflexConfig(max_command=0.4, max_delta=0.2, min_upright=0.5, min_height_m=0.8),
    "walker": ReflexConfig(max_command=1.0, max_delta=0.5, min_upright=0.5, min_height_m=0.7),
    "ant": ReflexConfig(max_command=1.0, max_delta=0.5, min_upright=0.2, min_height_m=0.2),
}


class ReflexLayer:
    """Deterministic filter between the learned brain and the motors.

    The brain proposes a command; this layer can clamp it or replace it with a stop, and the
    brain has no way to override that. It is a research demonstration, not certified safety software.
    """

    def __init__(self, spec: EmbodimentSpec, config: ReflexConfig | None = None) -> None:
        self.spec = spec
        self.config = config or ReflexConfig()
        self._previous = np.zeros(spec.num_joints, dtype=np.float32)
        self._latched = False

    def reset(self) -> None:
        self._previous = np.zeros(self.spec.num_joints, dtype=np.float32)
        self._latched = False

    def filter(self, proposed: np.ndarray, joint_pos: np.ndarray, height: float, upright: float) -> ReflexDecision:
        c = self.config
        zeros = np.zeros(self.spec.num_joints, dtype=np.float32)
        if not np.all(np.isfinite(proposed)):
            self._latched = True
            return ReflexDecision(Action.STOP, zeros, "non-finite command from the brain", True)
        if self._latched or upright < c.min_upright or height < c.min_height_m:
            self._latched = True  # stay stopped until an operator resets the layer
            return ReflexDecision(Action.STOP, zeros, "fall detected, motors relaxed", True)

        command = np.clip(proposed, -c.max_command, c.max_command)
        command = np.clip(command, self._previous - c.max_delta, self._previous + c.max_delta)
        # Never push further into a joint limit.
        at_upper = joint_pos >= self.spec.joint_upper - c.limit_margin_rad
        at_lower = joint_pos <= self.spec.joint_lower + c.limit_margin_rad
        command = np.where(at_upper, np.minimum(command, 0.0), command)
        command = np.where(at_lower, np.maximum(command, 0.0), command)
        command = command.astype(np.float32)

        modified = not np.allclose(command, proposed, atol=1e-3)
        self._previous = command
        if modified:
            return ReflexDecision(Action.SLOW, command, "command limited by the reflex layer", True)
        return ReflexDecision(Action.GO, command, "command passed unchanged", False)
