"""D3-01 Camera Alignment — live 2B / 4B evaluation.

Controlled 640×360 lava courtyard. Three spawn-yaw conditions
(left / center / right). The Agent issues camera yaw (or wait)
from RGB until the lava is centered. ``max_steps=8``. Movement
is not executed. Success is the final hidden Minecraft yaw,
not a model text claim.

Prompt, evaluator, and model weights are not tuned here.
Not a statistical capability claim.

D3-02 Target Approach is not implemented and is not run.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/run_d3_01.py \\
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
from obsidianlink.env.d3_01_scene import (
    D3_01_CENTER_YAW_TOLERANCE,
    D3_01_CONDITIONS,
    D3_01_ENV_IDS,
    D3_01_RESOLUTION,
    D3_01_SPAWN_YAWS,
    D3_01_TARGET_NAME,
    D3_01_TARGET_YAW,
)
from obsidianlink.tasks.diagnostic import (
    D3_01_MAX_STEPS,
    D3_01_TASKS,
    D3_01_WARMUP_STEPS,
    D3CameraAlignmentAgent,
    D3CameraAlignmentEvaluator,
)


_RUNS_DIR = os.path.join("obsidianlink", "experiments", "runs")


def _run_one_episode(
    model: QwenVLModelClient,
    condition: str,
    episode_idx: int,
    total_episodes: int,
    debug_save_dir: str | None,
) -> tuple[Result, float]:
    task = D3_01_TASKS[condition]
    env_id = D3_01_ENV_IDS[condition]
    print(
        f"\n=== {condition} episode {episode_idx + 1}/{total_episodes} "
        f"(env={env_id}, initial={task.ground_truth}, "
        f"spawn_yaw={D3_01_SPAWN_YAWS[condition]}) ==="
    )
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D3_01_WARMUP_STEPS)
    agent = D3CameraAlignmentAgent(model=model, target_name=D3_01_TARGET_NAME)
    evaluator = D3CameraAlignmentEvaluator()

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
    print(f"  last_action             : {result.evidence.get('last_action')!r}")
    print(f"  final_yaw               : {result.evidence.get('final_yaw')!r}")
    print(f"  yaw_error               : {result.evidence.get('yaw_error')!r}")
    print(f"  yaw_tolerance           : {result.evidence.get('yaw_tolerance')!r}")
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
        "last_action": result.evidence.get("last_action"),
        "final_yaw": result.evidence.get("final_yaw"),
        "yaw_error": result.evidence.get("yaw_error"),
        "initial_direction": result.evidence.get("initial_direction"),
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
    n_orientation = sum(1 for r in records if r["reason"] == "orientation_error")
    n_protocol = sum(1 for r in records if r["reason"] == "output_protocol_error")
    n_missing = sum(1 for r in records if r["reason"] == "missing_world_truth")
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
                "n_orientation_error": 0,
                "n_protocol_error": 0,
                "n_missing_world_truth": 0,
            },
        )
        bucket["n"] += 1
        if r["success"]:
            bucket["n_success"] += 1
        if r["reason"] == "orientation_error":
            bucket["n_orientation_error"] += 1
        if r["reason"] == "output_protocol_error":
            bucket["n_protocol_error"] += 1
        if r["reason"] == "missing_world_truth":
            bucket["n_missing_world_truth"] += 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "orientation_error_rate": n_orientation / n,
        "output_protocol_error_rate": n_protocol / n,
        "missing_world_truth_rate": n_missing / n,
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
        choices=("all",) + D3_01_CONDITIONS,
        default="all",
        help="Which spawn-yaw scene(s) to run. Default: left, center, right.",
    )
    parser.add_argument("--save-dir", default=_RUNS_DIR)
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    conditions = D3_01_CONDITIONS if args.condition == "all" else (args.condition,)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_stem = f"d3_01_camera_{model_name}_{args.num_episodes}ep_{timestamp}"
    frames_root = os.path.join(args.save_dir, run_stem + "_frames")

    print("ObsidianLink — D3-01 Camera Alignment live evaluation")
    print("  task_id     : d3_01_camera_alignment")
    print(f"  resolution  : {D3_01_RESOLUTION[0]}x{D3_01_RESOLUTION[1]}")
    print(f"  max_steps   : {D3_01_MAX_STEPS}")
    print(f"  conditions  : {', '.join(conditions)}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N / cond    : {args.num_episodes}")
    print(f"  warmup      : {D3_01_WARMUP_STEPS}")
    print(f"  device      : {args.device}")
    print(f"  target_yaw  : {D3_01_TARGET_YAW}")
    print(f"  yaw_tol     : ±{D3_01_CENTER_YAW_TOLERANCE}")
    print("  prompt      : D3CameraAlignmentAgent (untuned)")
    print("  evaluator   : D3CameraAlignmentEvaluator (final hidden yaw)")
    print("  motor       : camera / wait only")
    print("  D3-02       : not implemented / not run")
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
                    "last_action": None,
                    "final_yaw": None,
                    "yaw_error": None,
                    "initial_direction": condition,
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
    print(f"\n=== Aggregate ({model_name}, D3-01, {n} episodes) ===")
    print(
        f"  success_rate                : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  orientation_error_rate      : {agg['orientation_error_rate']:.1%}  "
        f"({int(agg['orientation_error_rate'] * n)}/{n})"
    )
    print(
        f"  output_protocol_error_rate  : {agg['output_protocol_error_rate']:.1%}  "
        f"({int(agg['output_protocol_error_rate'] * n)}/{n})"
    )
    print(
        f"  missing_world_truth_rate    : {agg['missing_world_truth_rate']:.1%}  "
        f"({int(agg['missing_world_truth_rate'] * n)}/{n})"
    )
    print("  by condition:")
    for cond, bucket in agg.get("by_condition", {}).items():
        print(
            f"    {cond}: {bucket['n_success']}/{bucket['n']} ok, "
            f"orientation_error={bucket['n_orientation_error']}, "
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
        "task_id": "d3_01_camera_alignment",
        "design": "camera_alignment",
        "motor_execution": True,
        "allowed_actions": ["camera", "wait"],
        "resolution": list(D3_01_RESOLUTION),
        "max_steps": D3_01_MAX_STEPS,
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "conditions": list(conditions),
        "env_ids": {c: D3_01_ENV_IDS[c] for c in conditions},
        "spawn_yaws": {c: D3_01_SPAWN_YAWS[c] for c in conditions},
        "target": D3_01_TARGET_NAME,
        "target_yaw": D3_01_TARGET_YAW,
        "yaw_tolerance": D3_01_CENTER_YAW_TOLERANCE,
        "warmup_steps": D3_01_WARMUP_STEPS,
        "num_episodes_per_condition": args.num_episodes,
        "num_episodes": n,
        "total_wall_time": total_elapsed,
        "frames_dir": frames_root,
        "aggregate": agg,
        "episodes": records,
        "d3_02": False,
        "pilot": True,
        "historical_exploratory": False,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
