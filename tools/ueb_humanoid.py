"""Humanoid-first test of the Unified Embodied Brain prototype.

Trains two models with the same architecture and the same number of humanoid gradient steps:
a specialist that sees only the humanoid, and a shared brain that also trains on other bodies.
Then compares them on the humanoid in simulation and writes the numbers to a JSON report.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ueb.pipeline import BrainPolicy, TeacherPolicy, ZeroPolicy, collect, rollout, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", nargs="+", default=["humanoid", "walker", "ant"], help="first one is the target body")
    parser.add_argument("--episodes", type=int, default=60, help="teacher demonstrations per body")
    parser.add_argument("--steps", type=int, default=2000, help="gradient steps on the target body")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/ueb_humanoid.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/ueb_shared.pt"))
    args = parser.parse_args()

    started = time.time()
    datasets = [collect(body, i, args.episodes, horizon=8, seed=args.seed) for i, body in enumerate(args.bodies)]
    for data in datasets:
        print(f"collected {data.env.body}: {data.joints.shape[0]} samples, {data.env.spec.num_joints} joints")

    target = datasets[0]
    print("training specialist (target body only)")
    specialist = train([target], args.steps, seed=args.seed, log_every=1000)
    print("training shared brain (all bodies)")
    shared = train(datasets, args.steps * len(datasets), seed=args.seed, log_every=1000)

    report = {
        "target_body": target.env.body,
        "bodies": args.bodies,
        "metric": "mean control steps before falling (higher is better)",
        "eval_episodes": args.eval_episodes,
        "max_steps": 300,
        "train_steps_on_target": args.steps,
        "demonstrations_per_body": args.episodes,
        "parameters": sum(p.numel() for p in shared.parameters()),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "results": {},
    }

    def record(name: str, env, policy, use_reflex: bool) -> None:
        stats = rollout(env, policy, args.eval_episodes, use_reflex=use_reflex)
        report["results"][name] = {"mean_steps": round(stats.mean_steps, 1), "min": min(stats.steps), "max": max(stats.steps),
                                   "reflex_modified_rate": round(stats.reflex_modified_rate, 3), "reflex_stops": stats.reflex_stops}
        print(f"{name:42s} {stats.mean_steps:6.1f} steps  (reflex changed {stats.reflex_modified_rate:.1%} of commands)")

    env = target.env
    record("zero_torque", env, ZeroPolicy(env), False)
    record("teacher", env, TeacherPolicy(env), False)
    record("specialist", env, BrainPolicy(specialist, env, 0, seed=args.seed), False)
    record("shared_brain", env, BrainPolicy(shared, env, 0, seed=args.seed), False)
    record("shared_brain_with_reflex", env, BrainPolicy(shared, env, 0, seed=args.seed), True)
    for data in datasets[1:]:
        record(f"shared_brain_on_{data.env.body}", data.env, BrainPolicy(shared, data.env, data.embodiment_id, seed=args.seed), False)
        record(f"teacher_on_{data.env.body}", data.env, TeacherPolicy(data.env), False)

    report["wall_seconds"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"config": shared.config.__dict__, "state_dict": shared.state_dict(), "bodies": args.bodies}, args.checkpoint)
    print(f"wrote {args.output} and {args.checkpoint}")


if __name__ == "__main__":
    main()
