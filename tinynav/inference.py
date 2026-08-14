from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class OnnxDetector:
    """Small ONNX Runtime wrapper with Arm-friendly CPU session settings.

    Post-processing varies by model, so this class returns raw output tensors.
    Adapt ``decode`` for the chosen detector before claiming model accuracy.
    """

    def __init__(self, model_path: str | Path, threads: int = 4) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the 'onnx' dependency group") from exc
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]

    def infer(self, image: np.ndarray) -> tuple[list[np.ndarray], float]:
        tensor = self.preprocess(image)
        start = time.perf_counter_ns()
        outputs = self.session.run(None, {self.input.name: tensor})
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return outputs, elapsed_ms

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        shape = self.input.shape
        height = int(shape[2]) if isinstance(shape[2], int) else 320
        width = int(shape[3]) if isinstance(shape[3], int) else 320
        try:
            import cv2
            resized = cv2.resize(image, (width, height))
        except ImportError:
            resized = np.resize(image, (height, width, 3))
        return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
