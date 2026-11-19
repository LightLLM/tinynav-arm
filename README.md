# TinyNav-Arm

**Faster reflexes for Physical AI on Arm.**

TinyNav-Arm is an open-source reference project that turns camera detections
into deterministic robot navigation decisions and provides a reproducible path
for benchmarking ONNX inference on Arm64. It includes a zero-hardware warehouse
simulation, REST API, safety controller, optional ONNX Runtime adapter,
quantization and benchmarking tools, tests, Arm64 CI, and an interactive web
dashboard.

## Important scope

This is a learning and hackathon reference implementation. It is **not certified
robot safety software** and must not directly control machinery around people.
Test inside simulation or on a low-risk educational platform under appropriate
adult/institutional supervision.

The dashboard contains representative values for visualization. Only JSON
produced by `tools/benchmark.py` on your named target should be presented as a
measured result.

## Architecture

```text
camera or simulator → detections → risk estimator → GO/SLOW/TURN/STOP
                           ↓
                   Arm64 benchmark report
```

## Quick start: simulation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python tools/run_demo.py --scenario forklift
uvicorn tinynav.api:app --reload
```

Open `http://localhost:8000/docs` for the interactive API.

## Dashboard

```bash
cd dashboard
npm install
npm run dev
```

## Test

```bash
pytest -q
```

## Run a measured benchmark

Synthetic CPU validation:

```bash
python tools/benchmark.py --iterations 200
```

Real ONNX inference:

```bash
pip install -e '.[onnx]'
python tools/benchmark.py --model models/detector.onnx --iterations 500
```

The generated `results/benchmark.json` records median, p95, p99, FPS,
architecture, processor, and Python version.

## Quantize a model

```bash
python tools/quantize.py \
  --model models/detector-fp32.onnx \
  --output models/detector-int8.onnx
```

Dynamic quantization is provided as a portable starting point. Vision models
often benefit from static quantization using representative calibration frames;
implement and validate that path for the detector selected for the final entry.

## Arm64 validation

Use Raspberry Pi 4/5 with a 64-bit OS, an Arm Linux laptop, Apple Silicon, or an
Arm64 cloud runner. Record the exact device, CPU, OS, model checksum, ONNX
Runtime version, thread configuration, warmup count, run count, model size,
latency percentiles, memory, throughput, and accuracy delta.

## Cloud deployment

The repository deploys to Vercel as one project: `tinynav.api:app` runs as the API (set in
`pyproject.toml` under `[tool.vercel]`) and `public/` is served as the website, which calls the
API for its live demo. Training code in `ueb/` is not part of the deployment.

## Unified Embodied Brain prototype (humanoid first)

`ueb/` is a research prototype of one policy that controls different robot bodies with the same
weights, tested first on a simulated humanoid. The body is described to the model joint by joint,
so nothing in the network depends on how many joints a robot has. A deterministic reflex layer
(`ueb/safety.py`) sits between the learned model and the motors, following the same idea as
`tinynav/controller.py`.

```bash
pip install -e '.[dev,ueb]'
python tools/ueb_humanoid.py
```

The script trains a humanoid-only specialist and a shared model, compares them in MuJoCo, and
writes `results/ueb_humanoid.json`. See [docs/TECH_STACK.md](docs/TECH_STACK.md) for the full
stack, the limits of the prototype, and the path to a real humanoid.

## Repository map

```text
tinynav/       Python simulation, controller, API and ONNX adapter
tools/         Demo, quantization and benchmarking CLIs
tests/         Controller and API tests
dashboard/     Complete interactive visualization website
config/        Safety thresholds
results/       Generated benchmark evidence
.github/       Arm64 continuous integration
```

## Devpost checklist

- [ ] Replace representative UI numbers with measured Arm64 results
- [ ] Select and document a permissively licensed detector
- [ ] Add model attribution and checksum
- [ ] Add real calibration frames with documented provenance
- [ ] Record accuracy before and after quantization
- [ ] Capture the app running on the target Arm64 device
- [ ] Record a demo under three minutes
- [ ] Put the MIT license in the GitHub About section

## License

MIT
