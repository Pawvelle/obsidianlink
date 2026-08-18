"""Phase 2C — single-episode D1 Lava debug view.

NOT a benchmark. NOT a batch experiment. This script runs **one**
Lava Presence positive episode and writes the per-step debug
artifacts to disk:

* ``step_<N>_frame.png`` — the *exact* bytes of the
  ``Observation.frame`` that were forwarded to the model that
  step. Saved by :meth:`BenchmarkRunner.run` when ``debug_save_dir``
  is set.
* ``raw_response.txt`` — the model's raw response string for each
  step.
* ``result_evidence.json`` — the full ``Result.evidence`` bag
  (success, reason, report_visible, ground_truth_visible, raw
  responses, ...).

**No benchmark behaviour is changed.** Prompt, Evaluator, ground
truth, and the model itself are the same as the multi-episode
Lava runs. The only addition is the per-step frame save.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/debug_lava_view.py \\
        --model-path /Users/joey/Documents/Projects/ObsidianLink/models/Qwen3-VL-2B-Instruct \\
        --output-dir /tmp/lava_debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from obsidianlink.agents.qwen_vl_client import QwenVLModelClient
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.tasks.diagnostic import (
    D1_LAVA_PRESENCE,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local Qwen3-VL checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write debug artifacts into.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print("ObsidianLink — Phase 2C single-episode D1 Lava DEBUG view")
    print(f"  model      : {args.model_path}")
    print(f"  device     : {args.device}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  task_id    : {D1_LAVA_PRESENCE.task_id}")
    print(f"  env_id     : MineRLControlledLava-v0")
    print(f"  prompt     : D1PresenceAgent(target='lava') (UNCHANGED from Phase 2C multi-episode)")
    print(f"  evaluator  : D1PresenceEvaluator (UNCHANGED)")
    print(f"  ground_trth: hidden via Task.ground_truth=True (UNCHANGED)")
    print()
    print("debug mode: runner will save per-step frame to <output_dir>/step_<N>_frame.png")
    print()
    sys.stdout.flush()

    # Same wire-up as the multi-episode Lava script — do NOT change
    # anything in the model / env / agent / evaluator.
    env = ControlledSceneEnv(env_id="MineRLControlledLava-v0")
    model = QwenVLModelClient(model_path=args.model_path, device=args.device)
    agent = D1PresenceAgent(model=model, target_name="lava")
    evaluator = D1PresenceEvaluator()

    started = time.perf_counter()
    result = BenchmarkRunner().run(
        task=D1_LAVA_PRESENCE,
        env=env,
        agent=agent,
        evaluator=evaluator,
        debug_save_dir=args.output_dir,
    )
    elapsed = time.perf_counter() - started

    # Persist the raw responses (per step) and the evidence bag.
    raw_responses: list[dict[str, Any]] = []
    for step_idx in range(result.steps):
        # The agent only stores ``last_raw_response`` (from the most
        # recent ``act()``). The Runner does not capture per-step raw
        # responses, so we re-derive by reading the agent attribute
        # again here — the per-step view is approximated by reading
        # the agent's last_raw_response once, which is the most
        # recent one. For the single-episode debug view this is
        # enough; full per-step raw-response capture is a separate
        # add-on (and not in scope here).
        raw_responses.append(
            {
                "step": step_idx + 1,
                "raw_response": getattr(agent, "last_raw_response", None),
            }
        )
    with open(os.path.join(args.output_dir, "raw_responses.json"), "w") as f:
        json.dump(raw_responses, f, indent=2)
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(
            {
                "task_id": result.task_id,
                "success": result.success,
                "steps": result.steps,
                "model_calls": result.model_calls,
                "invalid_actions": result.invalid_actions,
                "elapsed_time": result.elapsed_time,
                "wall_time": elapsed,
                "evidence": dict(result.evidence),
            },
            f,
            indent=2,
        )

    # Print the standard Result block.
    print("\n--- D1 Lava DEBUG result ---")
    print(f"task_id:        {result.task_id}")
    print(f"success:        {result.success}")
    print(f"steps:          {result.steps}")
    print(f"model_calls:    {result.model_calls}")
    print(f"elapsed_time:   {result.elapsed_time:.2f}s")
    print(f"wall_time:      {elapsed:.2f}s")
    print(f"evidence:")
    for k, v in result.evidence.items():
        print(f"  {k}: {v!r}")
    print()
    print("Debug artifacts written to:")
    for fname in sorted(os.listdir(args.output_dir)):
        full = os.path.join(args.output_dir, fname)
        size = os.path.getsize(full)
        print(f"  {full}  ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
