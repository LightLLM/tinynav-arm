from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class BenchmarkResult:
    label: str
    iterations: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    architecture: str
    processor: str
    python: str
    measured: bool = True

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def benchmark(fn: Callable[[], object], label: str, iterations: int = 200, warmup: int = 20) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(samples)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(len(ordered) * p))]
    median = statistics.median(samples)
    return BenchmarkResult(label, iterations, round(median, 3), round(percentile(.95), 3), round(percentile(.99), 3), round(1000 / median, 2), platform.machine(), platform.processor() or "unknown", platform.python_version())


def synthetic_workload(size: int = 256) -> float:
    matrix = np.ones((size, size), dtype=np.float32)
    return float(np.mean(matrix @ matrix))
