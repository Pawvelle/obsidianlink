"""D3-02 Target Approach — live 2B / 4B evaluation.

Controlled 640×360 lava courtyard. Lava is already visible and
centered. The Agent issues move (forward) or wait from RGB until
it is at an interaction distance. ``max_steps=20``. Camera is not
executed. Success is the final hidden distance to the lava AABB,
not a model text claim.

Prompt, evaluator, and model weights are not tuned here.
Not a statistical capability claim.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/run_d3_02.py \\
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
from obsidianlink.env.d3_02_scene import (
    D3_02_ENV_ID,
    D3_02_GOAL_DISTANCE,
    D3_02_MIN_DISTANCE,
    D3_02_PLAYER_Z,
    D3_02_RESOLUTION,
    D3_02_TARGET_NAME,
)
from obsidianlink.tasks.diagnostic import (
    D3_02_APPROACH,
    D3_02_MAX_STEPS,
    D3_02_WARMUP_STEPS,
    D3TargetApproachAgent,
    D3TargetApproachEvaluator,
)


_RUNS_DIR = os.path.join("obsidianlink", "experiments", "runs")


def _run_one_episode(
    model: QwenVLModelClient,
    episode_idx: int,
    total_episodes: int,
    debug_save_dir: str | None,
) -> tuple[Result, float]:
    print(
        f"\n=== episode {episode_idx + 1}/{total_episodes} "
        f"(env={D3_02_ENV_ID}, spawn_z={D3_02_PLAYER_Z}) ==="
    )
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=D3_02_ENV_ID, warmup_steps=D3_02_WARMUP_STEPS)
    agent = D3TargetApproachAgent(model=model, target_name=D3_02_TARGET_NAME)
    evaluator = D3TargetApproachEvaluator()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=D3_02_APPROACH,
        env=env,
        agent=agent,
        evaluator=evaluator,
        debug_save_dir=debug_save_dir,
    )
    elapsed = time.perf_counter() - t0

    print(f"  success                 : {result.success}")
    print(f"  reason                  : {result.evidence.get('reason')!r}")
    print(f"  last_action             : {result.evidence.get('last_action')!r}")
    print(f"  final_distance          : {result.evidence.get('final_distance')!r}")
    print(f"  xpos / zpos             : {result.evidence.get('xpos')!r} / {result.evidence.get('zpos')!r}")
    print(f"  band                    : {D3_02_MIN_DISTANCE}–{D3_02_GOAL_DISTANCE}")
    print(f"  model_calls             : {result.model_calls}")
    print(f"  elapsed                 : {elapsed:.2f}s")
    sys.stdout.flush()
    return result, elapsed


def _episode_record(
    episode_idx: int,
    result: Result,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "episode": episode_idx + 1,
        "success": result.success,
        "reason": result.evidence.get("reason"),
        "last_action": result.evidence.get("last_action"),
        "final_distance": result.evidence.get("final_distance"),
        "xpos": result.evidence.get("xpos"),
        "zpos": result.evidence.get("zpos"),
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
    n_approach = sum(1 for r in records if r["reason"] == "approach_error")
    n_overshoot = sum(1 for r in records if r["reason"] == "overshoot_error")
    n_protocol = sum(1 for r in records if r["reason"] == "output_protocol_error")
    n_missing = sum(1 for r in records if r["reason"] == "missing_world_truth")
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "approach_error_rate": n_approach / n,
        "overshoot_error_rate": n_overshoot / n,
        "output_protocol_error_rate": n_protocol / n,
        "missing_world_truth_rate": n_missing / n,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default=_RUNS_DIR)
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_stem = f"d3_02_approach_{model_name}_{args.num_episodes}ep_{timestamp}"
    frames_root = os.path.join(args.save_dir, run_stem + "_frames")

    print("ObsidianLink — D3-02 Target Approach live evaluation")
    print("  task_id     : d3_02_target_approach")
    print(f"  resolution  : {D3_02_RESOLUTION[0]}x{D3_02_RESOLUTION[1]}")
    print(f"  max_steps   : {D3_02_MAX_STEPS}")
    print(f"  env_id      : {D3_02_ENV_ID}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N           : {args.num_episodes}")
    print(f"  warmup      : {D3_02_WARMUP_STEPS}")
    print(f"  device      : {args.device}")
    print(f"  spawn_z     : {D3_02_PLAYER_Z}")
    print(f"  band        : {D3_02_MIN_DISTANCE}–{D3_02_GOAL_DISTANCE}")
    print("  prompt      : D3TargetApproachAgent (untuned)")
    print("  evaluator   : D3TargetApproachEvaluator (final hidden distance)")
    print("  motor       : move forward / wait only")
    print()
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for i in range(args.num_episodes):
        debug_dir = os.path.join(frames_root, f"ep{i + 1}")
        try:
            result, ep_elapsed = _run_one_episode(
                model, i, args.num_episodes, debug_dir
            )
        except Exception as exc:
            print(f"  episode raised: {type(exc).__name__}: {exc}")
            records.append({
                "episode": i + 1,
                "success": False,
                "reason": f"exception:{type(exc).__name__}",
                "last_action": None,
                "final_distance": None,
                "xpos": None,
                "zpos": None,
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
    print(f"\n=== Aggregate ({model_name}, D3-02, {n} episodes) ===")
    print(
        f"  success_rate                : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  approach_error_rate         : {agg['approach_error_rate']:.1%}  "
        f"({int(agg['approach_error_rate'] * n)}/{n})"
    )
    print(
        f"  overshoot_error_rate        : {agg['overshoot_error_rate']:.1%}  "
        f"({int(agg['overshoot_error_rate'] * n)}/{n})"
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

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, run_stem + ".json")
    payload = {
        "task_id": "d3_02_target_approach",
        "design": "target_approach",
        "motor_execution": True,
        "allowed_actions": ["move", "wait"],
        "resolution": list(D3_02_RESOLUTION),
        "max_steps": D3_02_MAX_STEPS,
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "env_id": D3_02_ENV_ID,
        "spawn_z": D3_02_PLAYER_Z,
        "target": D3_02_TARGET_NAME,
        "goal_distance": D3_02_GOAL_DISTANCE,
        "min_distance": D3_02_MIN_DISTANCE,
        "warmup_steps": D3_02_WARMUP_STEPS,
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
