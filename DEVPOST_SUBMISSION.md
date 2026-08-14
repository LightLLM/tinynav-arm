# TinyNav-Arm

## Tagline

Faster reflexes for Physical AI on Arm.

## Track

Physical AI

## Inspiration

Affordable robots often operate with strict limits on CPU performance, memory,
power, and connectivity. A perception model that looks impressive on a
workstation can react too slowly on an edge device. TinyNav-Arm explores a
practical developer workflow for turning perception output into fast,
deterministic robot-control decisions on Arm64-oriented deployments.

## What it does

TinyNav-Arm is an open-source Physical AI reference application with five parts:

1. A deterministic warehouse sensor simulator with clear-aisle, forklift, and
   blocked-path scenarios.
2. A safety controller that converts detected objects, confidence, bounding
   boxes, and estimated distance into GO, SLOW, TURN_LEFT, TURN_RIGHT, or STOP.
3. An optional ONNX Runtime adapter configured for CPU inference, graph
   optimization, controlled threading, memory arenas, and memory-pattern reuse.
4. Command-line tools for ONNX INT8 weight quantization and repeatable latency
   benchmarking with warm-up, median, p95, p99, FPS, and platform metadata.
5. An interactive dashboard that explains the robot-perception loop and lets a
   user explore scenarios, decisions, FP32/INT8 modes, and deployment artifacts.

The included Arm64 GitHub Actions workflow runs the controller tests and creates
a measured synthetic CPU benchmark artifact on an Arm64 runner. Developers can
then provide their own permissively licensed ONNX detector and run the same
benchmark command on a Raspberry Pi, Arm laptop, or other AArch64 device.

## How we built it

- Python 3.10+
- FastAPI and Uvicorn
- NumPy
- ONNX Runtime CPU execution provider
- ONNX Runtime quantization tools
- React 19, TypeScript, Next.js/Vinext, and Tailwind CSS
- Docker and Docker Compose
- Pytest
- GitHub Actions on `ubuntu-24.04-arm`

The controller is intentionally deterministic and separate from model-specific
post-processing. This makes every action explainable and allows developers to
swap detectors without rewriting the safety policy.

## Arm-specific work

TinyNav-Arm makes Arm64 a reproducible target rather than treating it as a label:

- An Arm64 CI job validates the Python control path on `ubuntu-24.04-arm`.
- ONNX Runtime is configured for CPU-only inference with graph optimizations and
  explicit thread settings suitable for edge tuning.
- The benchmark artifact records the actual machine architecture and refuses to
  disguise dashboard examples as measured results.
- The quantization tool produces an INT8 ONNX artifact that can be compared with
  an FP32 baseline using the same runner.
- Setup documentation covers Raspberry Pi 4/5, Arm Linux, Apple Silicon, and
  Arm64 cloud runners.

## Challenges we ran into

The biggest challenge was separating an engaging visualization from honest
performance evidence. The dashboard needs representative data to explain the
concept, but competition claims must come from a named target device. We solved
this by labeling dashboard numbers as prototype values and generating measured
results into a machine-readable JSON artifact that includes architecture and
runtime metadata.

Another challenge was model portability. Different detectors expose different
output tensors and post-processing requirements. TinyNav therefore provides a
generic ONNX inference wrapper and leaves the decode adapter explicit instead of
silently assuming one model family.

## Accomplishments that we're proud of

- A complete sense-decide-act loop that runs without specialized hardware.
- Explainable control decisions with confidence and distance thresholds.
- A reusable Arm64 validation workflow.
- Clear separation between simulated examples and measured evidence.
- A polished dashboard that communicates the project in under three minutes.
- A small, permissively licensed codebase that other learners can extend.

## What we learned

Physical AI optimization is not only model compression. End-to-end response
time also depends on preprocessing, runtime graph optimization, thread count,
memory behavior, post-processing, and the controller that consumes inference
output. We also learned that p95 and p99 latency can be more important to a
robot than an attractive average.

## What's next for TinyNav-Arm

- Add a permissively licensed detector with model checksum and attribution.
- Add static INT8 quantization using representative calibration frames.
- Measure task accuracy before and after quantization.
- Capture power and peak-memory measurements on Raspberry Pi 5.
- Add a ROS 2 publisher/subscriber adapter.
- Connect to a low-risk educational rover in a controlled environment.

## Try it

Interactive dashboard: https://tinynav-arm.telbises.chatgpt.site

## Testing instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python tools/run_demo.py --scenario forklift
uvicorn tinynav.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`, call `/api/frame/forklift`, and confirm the
returned action is `SLOW`. Call `/api/frame/blocked` and confirm the returned
action is `STOP`.

To generate a measured benchmark artifact:

```bash
python tools/benchmark.py --iterations 200
```

To benchmark a real ONNX model:

```bash
pip install -e '.[onnx]'
python tools/benchmark.py --model models/detector.onnx --iterations 500
```

## Important disclosure

TinyNav-Arm is a learning and hackathon reference implementation, not certified
robot-safety software. The dashboard's comparison values are representative
visualization data until replaced by results from a named Arm64 target.
