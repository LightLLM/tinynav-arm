#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize an ONNX detector to INT8")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model does not exist: {args.model}")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit("Install with: pip install -e '.[onnx]'") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(args.model), str(args.output), weight_type=QuantType.QInt8)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")
    print("Validate model quality and benchmark on real Arm64 hardware before publishing claims.")


if __name__ == "__main__":
    main()
