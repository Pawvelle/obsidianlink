"""D1 lava presence live run.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_d1.py --condition positive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from obsidianlink.agents.qwen_vl import QwenVLModelClient
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.scene import ControlledSceneEnv
from obsidianlink.tasks.diagnostic import (
    D1_ENV_IDS,
    D1_TASKS,
    D1LavaEvaluator,
    d1_prompt,
)

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "Qwen3-VL-2B-Instruct"
        ),
    )
    parser.add_argument("--condition", choices=("positive", "negative"), default="positive")
    args = parser.parse_args(argv)
    model_path = os.path.abspath(args.model_path)

    task = D1_TASKS[args.condition]
    env = ControlledSceneEnv(env_id=D1_ENV_IDS[args.condition])
    model = QwenVLModelClient(model_path)
    agent = ReactiveAgent(model=model, prompt_builder=d1_prompt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    debug_dir = os.path.join(
        _RUNS_DIR, f"d1_lava_{os.path.basename(model_path)}_{args.condition}_{stamp}_frames"
    )
    print(f"D1 {args.condition} env={env.env_id} model={model_path}")
    sys.stdout.flush()
    t0 = time.perf_counter()
    result = BenchmarkRunner().run(
        task=task,
        env=env,
        agent=agent,
        evaluator=D1LavaEvaluator(),
        debug_save_dir=debug_dir,
    )
    payload = {
        "task_id": result.task_id,
        "success": result.success,
        "steps": result.steps,
        "model_calls": result.model_calls,
        "invalid_actions": result.invalid_actions,
        "elapsed_time": result.elapsed_time,
        "vision_completions": model.vision_completions,
        "text_completions": model.completions,
        "historical": False,
        "valid_for_l1_conclusion": False,
        "evidence": dict(result.evidence),
        "wall_time": time.perf_counter() - t0,
    }
    os.makedirs(_RUNS_DIR, exist_ok=True)
    out_path = os.path.join(
        _RUNS_DIR, f"d1_lava_{os.path.basename(model_path)}_{args.condition}_{stamp}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    if model.vision_completions < 1:
        print("FAIL: vision_completions == 0 (text-only fallback)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
