# Humanoid-first tech stack for the Unified Embodied Brain

This is the stack for testing the "one brain, many bodies" idea on a humanoid first. It is split
into what runs in this repository today and what to add at each later stage.

## Layers

| Layer | In this repo now | Add next | Add for a real robot |
| --- | --- | --- | --- |
| Robot body | Gymnasium `Humanoid-v5` (17 joints, MuJoCo) | Unitree G1 model from MuJoCo Menagerie | Unitree G1 (a developer/EDU unit with SDK access) |
| Physics | MuJoCo on CPU | GPU-parallel simulation (MuJoCo MJX / Playground or Isaac Lab) | - |
| Brain | `ueb.model.UnifiedBrain`: small transformer, per-joint tokens, flow-matching action head, future-state prediction head | Cameras and language through a pretrained VLA (SmolVLA, then GR00T N1.7 or pi0.5) via LeRobot | Same model, small tier on board |
| Body description | `ueb.embodiment`: joint type, axis, limits, gear, tree depth, link mass, read from the MuJoCo model | Same code on the G1 model | Same |
| Balance and locomotion | None; the brain drives joint torques directly | Learned whole-body controller trained with RL in simulation; the brain commands it | Same controller, validated on hardware |
| Demonstrations | `ueb.teacher.PoseHoldTeacher`, a scripted PD controller | Simulated teleoperation; open humanoid datasets | VR teleoperation, stored in LeRobot dataset format |
| Safety | `ueb.safety.ReflexLayer`: command, rate and joint-limit clamps, latched stop on fall or bad output | Self-collision and workspace limits | Hardware e-stop, overhead gantry, and the vendor's low-level protections |
| Middleware | Direct Python calls | ROS 2 topics for observations and commands | ROS 2 plus the robot vendor's SDK |
| Compute | CPU (4 cores is enough for this prototype) | One 24 GB GPU for training | Jetson-class computer on board, GPU workstation off board |
| Evaluation | `tools/ueb_humanoid.py`: specialist vs shared brain, JSON report | Task success on reach, pick and walk tasks | Trials on hardware with counted successes |
| Tooling | Python 3.10+, PyTorch, pytest, ruff | Experiment tracking (TensorBoard or W&B) | Signed model releases, staged rollout |

## What the prototype does

Every actuated joint becomes two tokens, its state and its action chunk, each tagged with an
embedding of that joint's physical description. The transformer therefore has no layer whose
shape depends on the robot: a 17-joint humanoid, a 6-joint walker and an 8-joint quadruped run
through the same weights. The only per-body parameters are one soft-prompt vector each.

`tools/ueb_humanoid.py` runs the smallest version of the paper's first hypothesis (H1):

1. Collect demonstrations from the scripted teacher on each body.
2. Train a **specialist** on the humanoid only.
3. Train a **shared brain** on all bodies, with the same number of gradient steps on the humanoid.
4. Compare both on the humanoid: mean control steps before it falls.

```bash
pip install -e '.[dev,ueb]'
python tools/ueb_humanoid.py            # full run, about 15 minutes on 4 CPU cores
pytest -q                               # includes tests/test_ueb.py
```

## Measured result

One run on 4 CPU cores (seed 0, 60 teacher demonstrations per body, 2,000 gradient steps on the
humanoid for each model, 20 evaluation episodes capped at 300 steps). Full numbers are in
`results/ueb_humanoid.json`.

| Policy on the humanoid | Mean steps before falling |
| --- | --- |
| No motor commands | 40.2 |
| Scripted teacher | 75.0 |
| Humanoid-only specialist | 51.5 |
| Shared brain (humanoid + walker + ant) | 58.7 |
| Shared brain with the reflex layer | 49.8 |

The shared brain did not do worse than the specialist on the humanoid, which is the direction H1
predicts, but this is one seed and 20 episodes, so the gap is not established. Both learned
policies fall short of the teacher. The same shared weights also ran the walker (184.1 steps
against the teacher's 295.1) and the ant (300 of 300).

The reflex layer changed every command and shortened the episode. The simulator's reset pose puts
at least one joint just outside its stated range, so the layer blocks commands that push it
further. That is the layer doing what it was written to do, and it shows the limits need tuning to
the body before it is useful.

## Limits of this prototype

- **The teacher is weak.** A PD controller holding the reset pose keeps the humanoid up for well
  under two seconds. The run checks that the pipeline learns and that weights are shared; it does
  not produce a humanoid that stands or walks.
- **No vision, audio, touch or language yet.** Only joint and floating-base state are tokenized.
- **Direct torque control.** A real humanoid needs a balance controller underneath the brain.
- **The reflex layer is a demonstration**, not certified safety software.
- **The other bodies are simulated toys** (walker, ant), not a drone or an industrial arm.

## Order of work from here

1. Replace `Humanoid-v5` with the Unitree G1 model and joint position targets instead of torques.
2. Train a standing and walking controller with RL, and use it as the teacher and as the low-level layer.
3. Add a camera and a language instruction by putting a pretrained VLA in the trunk.
4. Add a fourth and fifth body (an arm and a quadrotor) and rerun the specialist-vs-shared comparison.
5. Only then move to hardware, on a gantry, with the reflex layer and an e-stop in the loop.
