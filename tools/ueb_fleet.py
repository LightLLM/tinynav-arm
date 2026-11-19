"""One brain, four real robot models: a humanoid, an industrial arm, a drone and a quadruped.

Uses MuJoCo Menagerie models (see ueb/menagerie.py for how to get them). For each seed it trains
a humanoid-only specialist and a shared brain on all bodies, with the same number of gradient
steps on the humanoid, then reports task success for every body.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ueb.menagerie import MENAGERIE_BODIES
from ueb.pipeline import BrainPolicy, TeacherPolicy, ZeroPolicy, collect, rollout, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", nargs="+", default=["g1", "ur5e", "x2", "go2"], help="first one is the target body")
    parser.add_argument("--episodes", type=int, default=20, help="teacher demonstrations per body")
    parser.add_argument("--steps", type=int, default=2000, help="gradient steps on the target body")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/ueb_fleet.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/ueb_fleet.pt"))
    args = parser.parse_args()

    started = time.time()
    runs: dict[str, list[dict]] = {}

    def record(name: str, env, policy, use_reflex: bool = False) -> None:
        stats = rollout(env, policy, args.eval_episodes, use_reflex=use_reflex)
        runs.setdefault(name, []).append({"success_rate": stats.success_rate, "mean_steps": stats.mean_steps,
                                          "reflex_modified_rate": stats.reflex_modified_rate, "reflex_stops": stats.reflex_stops})
        print(f"  {name:28s} success {stats.success_rate:5.0%}  steps {stats.mean_steps:6.1f}", flush=True)

    shared = None
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        datasets = [collect(body, i, args.episodes, horizon=8, seed=seed) for i, body in enumerate(args.bodies)]
        target = datasets[0]
        specialist = train([target], args.steps, seed=seed)
        shared = train(datasets, args.steps * len(datasets), seed=seed)
        tenv, name = target.env, target.env.body
        if seed == args.seeds[0]:
            record(f"{name}/no_command", tenv, ZeroPolicy(tenv))
            for data in datasets:
                record(f"{data.env.body}/teacher", data.env, TeacherPolicy(data.env))
        record(f"{name}/specialist", tenv, BrainPolicy(specialist, tenv, 0, seed=seed))
        for data in datasets:
            record(f"{data.env.body}/shared", data.env, BrainPolicy(shared, data.env, data.embodiment_id, seed=seed))
        record(f"{name}/shared_with_reflex", tenv, BrainPolicy(shared, tenv, 0, seed=seed), use_reflex=True)

    summary = {}
    for name, rows in runs.items():
        rates = [r["success_rate"] for r in rows]
        summary[name] = {"success_rate_mean": round(float(np.mean(rates)), 3), "success_rate_per_seed": rates,
                         "mean_steps": round(float(np.mean([r["mean_steps"] for r in rows])), 1),
                         "reflex_modified_rate": round(float(np.mean([r["reflex_modified_rate"] for r in rows])), 3),
                         "reflex_stops": int(sum(r["reflex_stops"] for r in rows))}
    report = {
        "bodies": {b: {"class": MENAGERIE_BODIES[b].robot_class, "episode_steps": MENAGERIE_BODIES[b].episode_steps} for b in args.bodies},
        "task": "hold the reference pose: stand (humanoid, quadruped), hover (drone), return to home (arm)",
        "seeds": args.seeds, "eval_episodes_per_seed": args.eval_episodes, "demonstrations_per_body": args.episodes,
        "train_steps_on_target": args.steps, "parameters": sum(p.numel() for p in shared.parameters()),
        "torch": torch.__version__, "machine": platform.machine(), "wall_seconds": round(time.time() - started, 1),
        "results": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"config": shared.config.__dict__, "state_dict": shared.state_dict(), "bodies": args.bodies}, args.checkpoint)
    print(f"wrote {args.output} and {args.checkpoint}")


if __name__ == "__main__":
    main()
