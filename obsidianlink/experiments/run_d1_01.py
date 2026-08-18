"""D1-01 Lava Presence — live 2B / 4B evaluation.

Runs the redesigned D1-01 task (640×360 obsidian sky-platform,
binary lava presence, ``max_steps=1``) against a local Qwen3-VL
checkpoint. Prompt, evaluator, and model weights are not tuned
here: this is one real capability pass after the scene redesign.

Each episode launches a fresh MineRL env. Positive and negative
conditions share the Agent prompt; only the hidden
``Task.ground_truth`` and the floor patch differ.

Saved to ``obsidianlink/experiments/runs/d1_01_lava_<model>_<N>ep_<TS>.json``.
Per-episode VLM input frames go under a sibling ``frames/`` folder
via ``BenchmarkRunner.debug_save_dir``.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/run_d1_01.py \\
        --model-path /Users/joey/Documents/Projects/ObsidianLink/models/Qwen3-VL-2B-Instruct \\
        --num-episodes 1
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
from obsidianlink.env.d1_v2_lava_scene import D1_V2_RESOLUTION
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_ENV_IDS,
    D1_01_LAVA_TASKS,
    D1_01_WARMUP_STEPS,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


_RUNS_DIR = os.path.join("obsidianlink", "experiments", "runs")
_CONDITIONS = ("positive", "negative")


def _run_one_episode(
    model: QwenVLModelClient,
    condition: str,
    episode_idx: int,
    total_episodes: int,
    debug_save_dir: str | None,
) -> tuple[Result, float]:
    task = D1_01_LAVA_TASKS[condition]
    env_id = D1_01_LAVA_ENV_IDS[condition]
    print(
        f"\n=== {condition} episode {episode_idx + 1}/{total_episodes} "
        f"(env={env_id}, gt={task.ground_truth}) ==="
    )
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D1_01_WARMUP_STEPS)
    agent = D1PresenceAgent(model=model, target_name="lava")
    evaluator = D1PresenceEvaluator()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=task,
        env=env,
        agent=agent,
        evaluator=evaluator,
        debug_save_dir=debug_save_dir,
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


def _episode_record(
    condition: str,
    episode_idx: int,
    result: Result,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "condition": condition,
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
    n_protocol_err = sum(
        1 for r in records if r["reason"] == "output_protocol_error"
    )
    reasons: dict[str, int] = {}
    by_condition: dict[str, dict[str, Any]] = {}
    for r in records:
        reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
        cond = r.get("condition") or "unknown"
        bucket = by_condition.setdefault(
            cond,
            {"n": 0, "n_success": 0, "n_perception_error": 0, "n_protocol_error": 0},
        )
        bucket["n"] += 1
        if r["success"]:
            bucket["n_success"] += 1
        if r["reason"] == "perception_error":
            bucket["n_perception_error"] += 1
        if r["reason"] == "output_protocol_error":
            bucket["n_protocol_error"] += 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "perception_error_rate": n_perception_err / n,
        "output_protocol_error_rate": n_protocol_err / n,
        "reasons": reasons,
        "by_condition": by_condition,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--condition",
        choices=("both",) + _CONDITIONS,
        default="both",
        help="Which D1-01 scene(s) to run. Default: positive then negative.",
    )
    parser.add_argument(
        "--save-dir",
        default=_RUNS_DIR,
        help="Directory to write per-run JSON records into.",
    )
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    conditions = _CONDITIONS if args.condition == "both" else (args.condition,)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_stem = f"d1_01_lava_{model_name}_{args.num_episodes}ep_{timestamp}"
    frames_root = os.path.join(args.save_dir, run_stem + "_frames")

    print("ObsidianLink — D1-01 Lava Presence live evaluation")
    print(f"  task_id     : d1_01_lava_presence")
    print(f"  resolution  : {D1_V2_RESOLUTION[0]}x{D1_V2_RESOLUTION[1]}")
    print(f"  conditions  : {', '.join(conditions)}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N / cond    : {args.num_episodes}")
    print(f"  warmup      : {D1_01_WARMUP_STEPS}")
    print(f"  device      : {args.device}")
    print("  prompt      : D1PresenceAgent(target='lava') (unchanged)")
    print("  evaluator   : D1PresenceEvaluator (unchanged)")
    print("  ground truth: hidden via Task.ground_truth")
    print()
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for condition in conditions:
        for i in range(args.num_episodes):
            debug_dir = os.path.join(
                frames_root, f"{condition}_ep{i + 1}"
            )
            try:
                result, ep_elapsed = _run_one_episode(
                    model,
                    condition,
                    i,
                    args.num_episodes,
                    debug_dir,
                )
            except Exception as exc:
                print(f"  episode raised: {type(exc).__name__}: {exc}")
                records.append({
                    "condition": condition,
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
            records.append(_episode_record(condition, i, result, ep_elapsed))
    total_elapsed = time.perf_counter() - total_t0

    agg = _aggregate(records)
    n = agg["n_episodes"]
    print(f"\n=== Aggregate ({model_name}, D1-01, {n} episodes) ===")
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
    print("  by condition:")
    for cond, bucket in agg.get("by_condition", {}).items():
        print(
            f"    {cond}: {bucket['n_success']}/{bucket['n']} ok, "
            f"perception_error={bucket['n_perception_error']}, "
            f"protocol_error={bucket['n_protocol_error']}"
        )
    print("  reason breakdown:")
    for reason, count in sorted(agg["reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"  total wall time    : {total_elapsed:.1f}s")
    print(f"  per-episode (mean) : {total_elapsed / max(n, 1):.1f}s")
    sys.stdout.flush()

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, run_stem + ".json")
    payload = {
        "task_id": "d1_01_lava_presence",
        "resolution": list(D1_V2_RESOLUTION),
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "conditions": list(conditions),
        "env_ids": {c: D1_01_LAVA_ENV_IDS[c] for c in conditions},
        "warmup_steps": D1_01_WARMUP_STEPS,
        "num_episodes_per_condition": args.num_episodes,
        "num_episodes": n,
        "total_wall_time": total_elapsed,
        "frames_dir": frames_root,
        "aggregate": agg,
        "episodes": records,
        "pilot": False,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
