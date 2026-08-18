"""Phase 2C — multi-episode D1 Presence evaluation on live MineRL.

Runs the **same** D1 Presence task (Lava / Water / Obsidian) for
``--num-episodes`` consecutive live episodes against a
:class:`obsidianlink.env.controlled_scene_env.ControlledSceneEnv`.
The only intentional variable is the model path
(``--model-path``) — this is the model-scale control experiment
the user asked for.

Failure-mode contract
---------------------

The D1 Presence Evaluator distinguishes two failure modes:

* ``output_protocol_error`` — the model response was not parseable
  as the required ``{"visible": bool}`` schema.
* ``perception_error`` — the response was well-formed but the
  boolean disagrees with the hidden ground truth.

The aggregate reports both rates separately; the
``output_protocol_error`` rate is the model's *formatting*
capability, and the ``perception_error`` rate is its *visual
perception* capability. We do NOT collapse them — the user
explicitly asked for this split.

Output
------

Per-episode: full :class:`Result` evidence bag (success, reason,
report_visible, ground_truth_visible, raw model response).

Aggregate: success_rate, perception_error_rate,
output_protocol_error_rate, reason breakdown.

Saved to ``experiments/runs/d1_presence_<target>_<model>_<N>ep_<TS>.json``.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/multi_episode_d1_presence.py \\
        --target lava \\
        --model-path /Users/joey/Documents/Projects/ObsidianLink/models/Qwen3-VL-2B-Instruct \\
        --num-episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from obsidianlink.agents.qwen_vl_client import QwenVLModelClient
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.tasks.diagnostic import (
    D1_PRESENCE_TASKS,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


_RUNS_DIR = "experiments/runs"


def _resolve_env_id(target: str) -> str:
    """Map a target name to the corresponding herobraine env id."""
    return f"MineRLControlled{target.capitalize()}-v0"


def _run_one_episode(
    model: QwenVLModelClient,
    target: str,
    episode_idx: int,
    total_episodes: int,
) -> tuple[Result, float]:
    """Run a single D1 Presence episode. Returns ``(Result, wall_seconds)``."""
    task = D1_PRESENCE_TASKS[f"d1_{target}_presence"]
    env_id = _resolve_env_id(target)
    print(f"\n=== Episode {episode_idx + 1}/{total_episodes} (target={target}, env={env_id}) ===")
    sys.stdout.flush()
    # Fresh env per episode. The controlled-scene envs are all
    # deterministic (fixed spawn + fixed drawn block), so the
    # "world variation" we see across episodes is from MineRL's
    # internal handling (chunk loading, slight timing variance),
    # not from random world generation.
    env = ControlledSceneEnv(env_id=env_id)
    agent = D1PresenceAgent(model=model, target_name=target)
    evaluator = D1PresenceEvaluator()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=task,
        env=env,
        agent=agent,
        evaluator=evaluator,
    )
    elapsed = time.perf_counter() - t0

    print(f"  success              : {result.success}")
    print(f"  reason               : {result.evidence.get('reason')!r}")
    print(f"  report_visible       : {result.evidence.get('report_visible')!r}")
    print(f"  ground_truth_visible : {result.evidence.get('ground_truth_visible')!r}")
    print(f"  model_calls          : {result.model_calls}")
    print(f"  elapsed              : {elapsed:.2f}s")
    sys.stdout.flush()
    return result, elapsed


def _episode_record(episode_idx: int, result: Result, elapsed: float) -> dict[str, Any]:
    return {
        "episode": episode_idx + 1,
        "success": result.success,
        "reason": result.evidence.get("reason"),
        "report_visible": result.evidence.get("report_visible"),
        "ground_truth_visible": result.evidence.get("ground_truth_visible"),
        "steps": result.steps,
        "model_calls": result.model_calls,
        "invalid_actions": result.invalid_actions,
        "elapsed_time": result.elapsed_time,
        "wall_time": elapsed,
        "evidence": dict(result.evidence),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_episodes": 0}
    n_success = sum(1 for r in records if r["success"])
    n_perception_err = sum(1 for r in records if r["reason"] == "perception_error")
    n_protocol_err = sum(1 for r in records if r["reason"] == "output_protocol_error")
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "perception_error_rate": n_perception_err / n,
        "output_protocol_error_rate": n_protocol_err / n,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        choices=("lava", "water", "obsidian"),
        help="D1 presence target. Phase 2C ships only the lava env live.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-dir",
        default=_RUNS_DIR,
        help="Directory to write per-run JSON records into.",
    )
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    target = args.target
    env_id = _resolve_env_id(target)
    task_id = f"d1_{target}_presence"

    print("ObsidianLink — Phase 2C D1 Presence multi-episode evaluation")
    print(f"  target      : {target}")
    print(f"  task_id     : {task_id}")
    print(f"  env_id      : {env_id}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N episodes  : {args.num_episodes}")
    print(f"  device      : {args.device}")
    print(f"  prompt      : D1PresenceAgent(target={target!r}) (target-specific, unchanged)")
    print(f"  evaluator   : D1PresenceEvaluator (unchanged)")
    print(
        f"  ground truth: hidden via Task.ground_truth (NOT in observation/prompt)"
    )
    print()
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for i in range(args.num_episodes):
        try:
            result, ep_elapsed = _run_one_episode(
                model, target, i, args.num_episodes
            )
        except Exception as exc:
            print(f"  episode raised: {type(exc).__name__}: {exc}")
            records.append({
                "episode": i + 1,
                "success": False,
                "reason": f"exception:{type(exc).__name__}",
                "report_visible": None,
                "ground_truth_visible": None,
                "steps": 0,
                "model_calls": 0,
                "invalid_actions": 0,
                "elapsed_time": 0.0,
                "wall_time": 0.0,
                "evidence": {"exception": f"{type(exc).__name__}: {exc}"},
            })
            sys.stdout.flush()
            continue
        records.append(_episode_record(i, result, ep_elapsed))
    total_elapsed = time.perf_counter() - total_t0

    agg = _aggregate(records)
    n = agg["n_episodes"]
    print(f"\n=== Aggregate ({model_name}, {target}, {n} episodes) ===")
    print(
        f"  success_rate                : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  perception_error_rate       : {agg['perception_error_rate']:.1%}  "
        f"({int(agg['perception_error_rate'] * n)}/{n})"
    )
    print(
        f"  output_protocol_error_rate  : {agg['output_protocol_error_rate']:.1%}  "
        f"({int(agg['output_protocol_error_rate'] * n)}/{n})"
    )
    print("  reason breakdown:")
    for reason, count in sorted(agg["reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"  total wall time    : {total_elapsed:.1f}s")
    print(f"  per-episode (mean) : {total_elapsed / max(n, 1):.1f}s")
    sys.stdout.flush()

    # Persist for failure-taxonomy re-analysis.
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_name = f"d1_presence_{target}_{model_name}_{n}ep_{timestamp}.json"
    out_path = os.path.join(args.save_dir, out_name)
    payload = {
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "target": target,
        "task_id": task_id,
        "env_id": env_id,
        "num_episodes": n,
        "total_wall_time": total_elapsed,
        "aggregate": agg,
        "episodes": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
