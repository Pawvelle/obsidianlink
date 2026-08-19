"""D2-02 Spatial Region Grounding — live 2B / 4B evaluation.

Controlled 640×360 lava courtyard. Nine spawn-pose conditions
(3×3 screen regions). The Agent classifies which region contains
the lava from one RGB frame and emits WAIT. ``max_steps=1``.
No camera or movement.

Prompt, evaluator, and model weights are not tuned here.
Not a statistical capability claim.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/run_d2_02.py \\
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
from obsidianlink.env.d2_02_scene import (
    D2_02_ENV_IDS,
    D2_02_REGIONS,
    D2_02_RESOLUTION,
    D2_02_SPAWN_POSES,
    D2_02_TARGET_NAME,
)
from obsidianlink.tasks.diagnostic import (
    D2_02_MAX_STEPS,
    D2_02_TASKS,
    D2_02_WARMUP_STEPS,
    D2SpatialRegionGroundingAgent,
    D2SpatialRegionGroundingEvaluator,
)


_RUNS_DIR = os.path.join("obsidianlink", "experiments", "runs")


def _run_one_episode(
    model: QwenVLModelClient,
    condition: str,
    episode_idx: int,
    total_episodes: int,
    debug_save_dir: str | None,
) -> tuple[Result, float]:
    task = D2_02_TASKS[condition]
    env_id = D2_02_ENV_IDS[condition]
    yaw, pitch = D2_02_SPAWN_POSES[condition]
    print(
        f"\n=== {condition} episode {episode_idx + 1}/{total_episodes} "
        f"(env={env_id}, gt={task.ground_truth}, "
        f"spawn_yaw={yaw}, spawn_pitch={pitch}) ==="
    )
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D2_02_WARMUP_STEPS)
    agent = D2SpatialRegionGroundingAgent(
        model=model, target_name=D2_02_TARGET_NAME
    )
    evaluator = D2SpatialRegionGroundingEvaluator()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=task,
        env=env,
        agent=agent,
        evaluator=evaluator,
        debug_save_dir=debug_save_dir,
    )
    elapsed = time.perf_counter() - t0

    print(f"  success                 : {result.success}")
    print(f"  reason                  : {result.evidence.get('reason')!r}")
    print(f"  report_target           : {result.evidence.get('report_target')!r}")
    print(f"  report_region           : {result.evidence.get('report_region')!r}")
    print(f"  ground_truth_region     : {result.evidence.get('ground_truth_region')!r}")
    print(f"  model_calls             : {result.model_calls}")
    print(f"  elapsed                 : {elapsed:.2f}s")
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
        "report_target": result.evidence.get("report_target"),
        "report_region": result.evidence.get("report_region"),
        "ground_truth_region": result.evidence.get("ground_truth_region"),
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
    n_grounding = sum(1 for r in records if r["reason"] == "grounding_error")
    n_protocol = sum(1 for r in records if r["reason"] == "output_protocol_error")
    reasons: dict[str, int] = {}
    by_condition: dict[str, dict[str, Any]] = {}
    for r in records:
        reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
        cond = r.get("condition") or "unknown"
        bucket = by_condition.setdefault(
            cond,
            {
                "n": 0,
                "n_success": 0,
                "n_grounding_error": 0,
                "n_protocol_error": 0,
            },
        )
        bucket["n"] += 1
        if r["success"]:
            bucket["n_success"] += 1
        if r["reason"] == "grounding_error":
            bucket["n_grounding_error"] += 1
        if r["reason"] == "output_protocol_error":
            bucket["n_protocol_error"] += 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "grounding_error_rate": n_grounding / n,
        "output_protocol_error_rate": n_protocol / n,
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
        choices=("all",) + D2_02_REGIONS,
        default="all",
        help="Which 3×3 scene(s) to run. Default: all nine.",
    )
    parser.add_argument("--save-dir", default=_RUNS_DIR)
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    conditions = D2_02_REGIONS if args.condition == "all" else (args.condition,)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_stem = f"d2_02_region_{model_name}_{args.num_episodes}ep_{timestamp}"
    frames_root = os.path.join(args.save_dir, run_stem + "_frames")

    print("ObsidianLink — D2-02 Spatial Region Grounding live evaluation")
    print("  task_id     : d2_02_spatial_region_grounding")
    print(f"  resolution  : {D2_02_RESOLUTION[0]}x{D2_02_RESOLUTION[1]}")
    print(f"  max_steps   : {D2_02_MAX_STEPS}")
    print(f"  conditions  : {', '.join(conditions)}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N / cond    : {args.num_episodes}")
    print(f"  warmup      : {D2_02_WARMUP_STEPS}")
    print(f"  device      : {args.device}")
    print("  prompt      : D2SpatialRegionGroundingAgent (untuned)")
    print("  evaluator   : D2SpatialRegionGroundingEvaluator (hidden region GT)")
    print("  motor       : none (WAIT only)")
    print()
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for condition in conditions:
        for i in range(args.num_episodes):
            debug_dir = os.path.join(frames_root, f"{condition}_ep{i + 1}")
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
                    "report_target": None,
                    "report_region": None,
                    "ground_truth_region": None,
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
    print(f"\n=== Aggregate ({model_name}, D2-02, {n} episodes) ===")
    print(
        f"  success_rate                : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  grounding_error_rate        : {agg['grounding_error_rate']:.1%}  "
        f"({int(agg['grounding_error_rate'] * n)}/{n})"
    )
    print(
        f"  output_protocol_error_rate  : {agg['output_protocol_error_rate']:.1%}  "
        f"({int(agg['output_protocol_error_rate'] * n)}/{n})"
    )
    print("  by condition:")
    for cond, bucket in agg.get("by_condition", {}).items():
        print(
            f"    {cond}: {bucket['n_success']}/{bucket['n']} ok, "
            f"grounding_error={bucket['n_grounding_error']}, "
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
        "task_id": "d2_02_spatial_region_grounding",
        "design": "spatial_region_grounding",
        "motor_execution": False,
        "resolution": list(D2_02_RESOLUTION),
        "max_steps": D2_02_MAX_STEPS,
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "conditions": list(conditions),
        "env_ids": {c: D2_02_ENV_IDS[c] for c in conditions},
        "spawn_poses": {
            c: {"yaw": D2_02_SPAWN_POSES[c][0], "pitch": D2_02_SPAWN_POSES[c][1]}
            for c in conditions
        },
        "target": D2_02_TARGET_NAME,
        "warmup_steps": D2_02_WARMUP_STEPS,
        "num_episodes_per_condition": args.num_episodes,
        "num_episodes": n,
        "total_wall_time": total_elapsed,
        "frames_dir": frames_root,
        "aggregate": agg,
        "episodes": records,
        "pilot": True,
        "historical_exploratory": False,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
