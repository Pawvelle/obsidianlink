"""L1 Controlled Construction — Reactive Agent live pilot.

Wires a real vision-capable MLLM (Qwen3-VL) into the L1
Reactive Agent and runs the L1 task on the controlled scene.

This is a **pilot** run. The goal is to observe the failure
modes of a non-planning, non-reflection agent on the first
end-to-end portal construction task — not to chase success.
A failure is the expected outcome at this stage; the
project's Research-First plan does NOT immediately escalate
to a planner / reflection agent on a pilot failure. We
record the failure and stop.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/run_l1_reactive.py \\
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
from obsidianlink.env.l1_scene import L1_ENV_ID, L1_MAX_STEPS, L1_WARMUP_STEPS
from obsidianlink.tasks.portal import (
    L1Evaluator,
    L1ReactiveAgent,
    L1_TASK,
)


_RUNS_DIR = os.path.join("obsidianlink", "experiments", "runs")


def _run_one_episode(
    model: QwenVLModelClient,
    episode_idx: int,
    total_episodes: int,
    debug_save_dir: str | None,
) -> tuple[Result, float]:
    print(
        f"\n=== L1 reactive episode {episode_idx + 1}/{total_episodes} "
        f"(env={L1_ENV_ID}) ==="
    )
    sys.stdout.flush()
    env = ControlledSceneEnv(
        env_id=L1_ENV_ID, warmup_steps=L1_WARMUP_STEPS,
    )
    agent = L1ReactiveAgent(model=model)
    evaluator = L1Evaluator()

    print(f"  calling env.reset() ... (cold start ~30-60s)")
    sys.stdout.flush()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=L1_TASK,
        env=env,
        agent=agent,
        evaluator=evaluator,
        debug_save_dir=debug_save_dir,
    )
    elapsed = time.perf_counter() - t0

    print(f"  success              : {result.success}")
    print(f"  reason               : {result.evidence.get('reason')!r}")
    print(f"  frame_complete       : {result.evidence.get('frame_complete')}")
    print(
        f"  frame_obsidian_count : "
        f"{result.evidence.get('frame_obsidian_count')}/"
        f"{result.evidence.get('frame_total')}"
    )
    print(f"  portal_ignited       : {result.evidence.get('portal_ignited')}")
    print(f"  entered_nether       : {result.evidence.get('entered_nether')}")
    print(f"  last_action          : {result.evidence.get('last_action')!r}")
    print(f"  model_calls          : {result.model_calls}")
    print(f"  invalid_actions      : {result.invalid_actions}")
    print(f"  vision_completions   : {getattr(model, 'vision_completions', 'n/a')}")
    print(f"  text_completions     : {getattr(model, 'completions', 'n/a')}")
    print(f"  elapsed              : {elapsed:.2f}s")
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
        "frame_complete": result.evidence.get("frame_complete"),
        "frame_obsidian_count": result.evidence.get("frame_obsidian_count"),
        "frame_total": result.evidence.get("frame_total"),
        "portal_ignited": result.evidence.get("portal_ignited"),
        "portal_cell_count": result.evidence.get("portal_cell_count"),
        "interior_total": result.evidence.get("interior_total"),
        "entered_nether": result.evidence.get("entered_nether"),
        "final_xpos": result.evidence.get("final_xpos"),
        "final_ypos": result.evidence.get("final_ypos"),
        "final_zpos": result.evidence.get("final_zpos"),
        "last_action": result.evidence.get("last_action"),
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
    n_frame = sum(1 for r in records if r.get("reason") == "portal_frame_incomplete")
    n_ignited = sum(1 for r in records if r.get("reason") == "portal_not_ignited")
    n_max = sum(1 for r in records if r.get("reason") == "max_steps_reached")
    n_missing = sum(1 for r in records if r.get("reason") == "missing_world_truth")
    reasons: dict[str, int] = {}
    for r in records:
        reason = r.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "portal_frame_incomplete_rate": n_frame / n,
        "portal_not_ignited_rate": n_ignited / n,
        "max_steps_reached_rate": n_max / n,
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
    run_stem = f"l1_reactive_{model_name}_{args.num_episodes}ep_{timestamp}"
    frames_root = os.path.join(args.save_dir, run_stem + "_frames")

    print("ObsidianLink — L1 Controlled Construction Reactive Agent")
    print(f"  task_id     : {L1_TASK.task_id}")
    print(f"  env_id      : {L1_ENV_ID}")
    print(f"  max_steps   : {L1_MAX_STEPS}")
    print(f"  model       : {args.model_path}")
    print(f"  model_name  : {model_name}")
    print(f"  N           : {args.num_episodes}")
    print(f"  warmup      : {L1_WARMUP_STEPS}")
    print(f"  device      : {args.device}")
    print("  agent       : L1ReactiveAgent (vision-capable MLLM)")
    print("  evaluator   : L1Evaluator (real Minecraft grid + ypos)")
    print("  pilot       : True (single-episode observation)")
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for i in range(args.num_episodes):
        debug_dir = os.path.join(frames_root, f"ep{i + 1}")
        try:
            result, ep_elapsed = _run_one_episode(
                model, i, args.num_episodes, debug_dir,
            )
        except Exception as exc:
            print(f"  episode raised: {type(exc).__name__}: {exc}")
            records.append({
                "episode": i + 1,
                "success": False,
                "reason": f"exception:{type(exc).__name__}",
                "frame_complete": None,
                "frame_obsidian_count": None,
                "frame_total": None,
                "portal_ignited": None,
                "portal_cell_count": None,
                "interior_total": None,
                "entered_nether": None,
                "final_xpos": None,
                "final_ypos": None,
                "final_zpos": None,
                "last_action": None,
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
    print(f"\n=== Aggregate (L1 reactive, {model_name}, {n} episodes) ===")
    print(
        f"  success_rate                 : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  portal_frame_incomplete_rate : "
        f"{agg['portal_frame_incomplete_rate']:.1%}  "
        f"({int(agg['portal_frame_incomplete_rate'] * n)}/{n})"
    )
    print(
        f"  portal_not_ignited_rate      : "
        f"{agg['portal_not_ignited_rate']:.1%}  "
        f"({int(agg['portal_not_ignited_rate'] * n)}/{n})"
    )
    print(
        f"  max_steps_reached_rate       : "
        f"{agg['max_steps_reached_rate']:.1%}  "
        f"({int(agg['max_steps_reached_rate'] * n)}/{n})"
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
        "task_id": L1_TASK.task_id,
        "design": "l1_controlled_construction",
        "level": "L1",
        "motor_execution": True,
        "allowed_actions": [
            "move", "camera", "place", "use", "equip", "wait",
        ],
        "max_steps": L1_MAX_STEPS,
        "env_id": L1_ENV_ID,
        "warmup_steps": L1_WARMUP_STEPS,
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "num_episodes": n,
        "total_wall_time": total_elapsed,
        "frames_dir": frames_root,
        "agent": "L1ReactiveAgent",
        "evaluator": "L1Evaluator",
        "aggregate": agg,
        "episodes": records,
        "pilot": True,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
