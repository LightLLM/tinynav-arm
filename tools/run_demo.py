#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinynav.controller import SafetyController
from tinynav.simulator import SCENARIOS, WarehouseSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyNav's deterministic warehouse demo")
    parser.add_argument("--scenario", choices=SCENARIOS, default="forklift")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.15)
    args = parser.parse_args()
    simulator, controller = WarehouseSimulator(), SafetyController()
    for index in range(args.frames):
        frame = simulator.next_frame(args.scenario)
        decision = controller.decide(frame.detections)
        print(json.dumps({"frame": index + 1, "objects": [d.label for d in frame.detections], **decision.to_dict()}))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
