"""Phase 2B — multi-episode D1 evaluation on live MineRL.

Runs the **same** D1 task (D1_INVENTORY_PERCEPTION), with the
**same** prompt (defined on :class:`D1InventoryPerceptionAgent`),
against the **same** environment (``MineRLTreechop-v0``), and
graded by the **same** :class:`D1InventoryPerceptionEvaluator`,
for ``--num-episodes`` consecutive live episodes. The only
intentional variable between two runs of this script is the
``--model-path`` (e.g. Qwen3-VL-2B-Instruct vs 4B-Instruct) — the
model-scale control experiment the user asked for.

What this script is NOT
-----------------------

* It does NOT modify the prompt, evaluator, action set, env, or
  runner between calls. Success rate is what the model produces,
  not what we engineered it to produce.
* It does NOT add retry / recovery / fallback logic. A bad
  episode stays a bad episode.
* It does NOT cache or share the MineRL env across episodes; each
  episode launches a fresh MineRL instance so world variation
  between episodes is real (MineRL picks a new random seed on each
  ``gym.make``). This is what the Master Plan calls
  "multi-episode, multi-seed" and is the right thing for
  Benchmark MVP statistics.

Output
------

Per-episode: full :class:`Result` evidence bag (success, reason,
inventory_match, selected_match, parsed report, ground truth,
raw model response).

Aggregate: per-model success rate, inventory_match rate,
selected_match rate, reason breakdown.

Saved to ``experiments/runs/d1_<model>_<N>ep_<timestamp>.json``
for later failure-taxonomy re-analysis (per Master Plan §正式统计要求).

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/multi_episode_d1.py \\
        --model-path /Users/joey/Documents/Projects/ObsidianLink/models/Qwen3-VL-2B-Instruct \\
        --num-episodes 5

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/multi_episode_d1.py \\
        --model-path /Users/joey/Documents/Projects/ObsidianLink/models/Qwen3-VL-4B-Instruct \\
        --num-episodes 5
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
from obsidianlink.env.minerl import MineRLEnvironment
from obsidianlink.tasks.diagnostic import (
    D1_INVENTORY_PERCEPTION,
    D1InventoryPerceptionAgent,
    D1InventoryPerceptionEvaluator,
)


_RUNS_DIR = "experiments/runs"


def _run_one_episode(
    model: QwenVLModelClient,
    episode_idx: int,
    total_episodes: int,
) -> tuple[Result, float]:
    """Run a single D1 episode. Returns ``(Result, wall_seconds)``."""
    print(f"\n=== Episode {episode_idx + 1}/{total_episodes} ===")
    sys.stdout.flush()
    # Fresh env per episode -> real MineRL cold start per episode
    # (each gym.make() gets a new random world seed). This is the
    # right semantics for the multi-episode control experiment.
    env = MineRLEnvironment()
    # Fresh agent per episode so model_calls and last_report
    # are isolated. The model is shared (loaded once) — that is
    # the expensive thing; agents are cheap.
    agent = D1InventoryPerceptionAgent(model=model)
    evaluator = D1InventoryPerceptionEvaluator()

    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=evaluator,
    )
    elapsed = time.perf_counter() - t0

    print(f"  success       : {result.success}")
    print(f"  reason        : {result.evidence.get('reason')!r}")
    inv_m = result.evidence.get("inventory_match")
    sel_m = result.evidence.get("selected_match")
    print(f"  inventory_m   : {inv_m}")
    print(f"  selected_m    : {sel_m}")
    print(f"  steps         : {result.steps}")
    print(f"  model_calls   : {result.model_calls}")
    print(f"  elapsed       : {elapsed:.2f}s")
    sys.stdout.flush()
    return result, elapsed


def _episode_record(episode_idx: int, result: Result, elapsed: float) -> dict[str, Any]:
    """Per-episode dict that goes into the saved JSON."""
    return {
        "episode": episode_idx + 1,
        "success": result.success,
        "reason": result.evidence.get("reason"),
        "inventory_match": result.evidence.get("inventory_match"),
        "selected_match": result.evidence.get("selected_match"),
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
    n_inv_match = sum(1 for r in records if r["inventory_match"])
    n_sel_match = sum(1 for r in records if r["selected_match"])
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "inventory_match_rate": n_inv_match / n,
        "selected_match_rate": n_sel_match / n,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-dir",
        default=_RUNS_DIR,
        help="Directory to write per-run JSON records into.",
    )
    args = parser.parse_args(argv)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    print("ObsidianLink — Phase 2B multi-episode D1 evaluation")
    print(f"  model      : {args.model_path}")
    print(f"  model_name : {model_name}")
    print(f"  N episodes : {args.num_episodes}")
    print(f"  device     : {args.device}")
    print(f"  task       : {D1_INVENTORY_PERCEPTION.task_id}")
    print(f"  env        : MineRLTreechop-v0 (fresh per episode)")
    print(
        f"  prompt     : D1InventoryPerceptionAgent.D1_PROMPT (unchanged)"
    )
    print(
        f"  evaluator  : D1InventoryPerceptionEvaluator (unchanged)"
    )
    print()
    sys.stdout.flush()

    model = QwenVLModelClient(model_path=args.model_path, device=args.device)

    records: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for i in range(args.num_episodes):
        try:
            result, elapsed = _run_one_episode(model, i, args.num_episodes)
        except Exception as exc:
            # A broken episode is itself a research datum. Record
            # the failure and continue with the next episode.
            print(f"  episode raised: {type(exc).__name__}: {exc}")
            records.append({
                "episode": i + 1,
                "success": False,
                "reason": f"exception:{type(exc).__name__}",
                "inventory_match": None,
                "selected_match": None,
                "steps": 0,
                "model_calls": 0,
                "invalid_actions": 0,
                "elapsed_time": 0.0,
                "wall_time": 0.0,
                "evidence": {"exception": f"{type(exc).__name__}: {exc}"},
            })
            sys.stdout.flush()
            continue
        records.append(_episode_record(i, result, elapsed))
    total_elapsed = time.perf_counter() - total_t0

    agg = _aggregate(records)
    n = agg["n_episodes"]
    print(f"\n=== Aggregate ({model_name}, {n} episodes) ===")
    print(
        f"  success_rate       : {agg['success_rate']:.1%}  "
        f"({int(agg['success_rate'] * n)}/{n})"
    )
    print(
        f"  inventory_match    : {agg['inventory_match_rate']:.1%}  "
        f"({int(agg['inventory_match_rate'] * n)}/{n})"
    )
    print(
        f"  selected_match     : {agg['selected_match_rate']:.1%}  "
        f"({int(agg['selected_match_rate'] * n)}/{n})"
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
    out_name = f"d1_{model_name}_{n}ep_{timestamp}.json"
    out_path = os.path.join(args.save_dir, out_name)
    payload = {
        "model": args.model_path,
        "model_name": model_name,
        "device": args.device,
        "task_id": D1_INVENTORY_PERCEPTION.task_id,
        "env_id": "MineRLTreechop-v0",
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
