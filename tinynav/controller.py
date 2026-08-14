from __future__ import annotations

from dataclasses import dataclass

from .types import Action, ControlDecision, Detection


@dataclass(frozen=True)
class SafetyConfig:
    stop_distance_m: float = 1.0
    slow_distance_m: float = 2.5
    confidence_threshold: float = 0.45
    cruise_speed_mps: float = 1.2
    slow_speed_mps: float = 0.4
    image_width: int = 640


class SafetyController:
    """Converts perception output into deterministic robot control decisions.

    This is a demonstration safety layer, not certified control software.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def decide(self, detections: list[Detection]) -> ControlDecision:
        valid = [d for d in detections if d.confidence >= self.config.confidence_threshold]
        with_distance = [d for d in valid if d.distance_m is not None]
        nearest = min(with_distance, key=lambda d: d.distance_m or 9999, default=None)

        if nearest is None:
            return ControlDecision(Action.GO, self.config.cruise_speed_mps, 5, "path clear", None)

        distance = float(nearest.distance_m or 0)
        if distance <= self.config.stop_distance_m:
            return ControlDecision(Action.STOP, 0.0, 95, f"{nearest.label} inside stop zone", distance)
        if distance <= self.config.slow_distance_m:
            return ControlDecision(Action.SLOW, self.config.slow_speed_mps, 65, f"{nearest.label} in caution zone", distance)

        x1, _, x2, _ = nearest.bbox
        center = (x1 + x2) / 2
        if center < self.config.image_width * 0.33:
            return ControlDecision(Action.TURN_RIGHT, 0.7, 28, f"passing {nearest.label} on right", distance)
        if center > self.config.image_width * 0.67:
            return ControlDecision(Action.TURN_LEFT, 0.7, 28, f"passing {nearest.label} on left", distance)
        return ControlDecision(Action.GO, self.config.cruise_speed_mps, 12, "safe forward clearance", distance)
