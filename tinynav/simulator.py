from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .types import Detection


SCENARIOS = ("clear", "forklift", "blocked")


@dataclass
class SimulatedFrame:
    image: np.ndarray
    detections: list[Detection]
    timestamp: float
    scenario: str


class WarehouseSimulator:
    """Creates deterministic synthetic sensor frames for zero-hardware demos."""

    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height
        self.frame_id = 0

    def next_frame(self, scenario: str = "forklift") -> SimulatedFrame:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.frame_id += 1
        image = self._background()
        phase = math.sin(self.frame_id / 10) * 8
        detections: list[Detection] = []

        if scenario == "clear":
            detections.append(Detection("pallet", 0.91, (280, 175, 360, 230), 4.8))
        elif scenario == "forklift":
            x = int(315 + phase)
            detections.extend([
                Detection("forklift", 0.96, (x, 135, x + 150, 305), 2.1),
                Detection("worker", 0.89, (205, 145, 260, 310), 2.7),
            ])
        else:
            detections.extend([
                Detection("crate", 0.98, (245, 205, 445, 345), 0.7),
                Detection("worker", 0.92, (160, 105, 235, 315), 1.3),
            ])
        return SimulatedFrame(image, detections, time.time(), scenario)

    def _background(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:140] = (35, 43, 37)
        frame[140:] = (26, 31, 26)
        for y in range(150, self.height, 40):
            frame[y:y + 1] = (60, 66, 58)
        return frame
