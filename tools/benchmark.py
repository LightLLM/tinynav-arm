#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinynav.benchmarking import benchmark, synthetic_workload
from tinynav.inference import OnnxDetector
from tinynav.simulator import WarehouseSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an honest TinyNav benchmark artifact")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if args.model:
        detector = OnnxDetector(args.model)
        image = WarehouseSimulator().next_frame("forklift").image
        fn, label = lambda: detector.infer(image), args.model.name
    else:
        print("No ONNX model supplied; running a measured synthetic CPU workload.")
        fn, label = synthetic_workload, "synthetic-matmul"
    result = benchmark(fn, label, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write(args.output)
    print(result)
    print(f"Wrote measured results to {args.output}")


if __name__ == "__main__":
    main()
