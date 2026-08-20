"""Run repeated formal L1 Portal evaluations for the deterministic Oracle.

This is deliberately a thin experiment wrapper over the production
``BenchmarkRunner`` and ``L1Evaluator``.  It does not reinterpret success:
only evaluator-confirmed Nether entry counts.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_oracle_eval.py --episodes 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from obsidianlink.agents.portal_agent import OraclePortalAgent
from obsidianlink.benchmark.l1_evaluator import L1Evaluator
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.result import Result
from obsidianlink.env.l1_scene import L1ControlledEnv
from obsidianlink.tasks.portal import L1_PORTAL_TASK


def failure_reason(result: Result) -> str:
    """One stable label for an unsuccessful episode."""
    if result.success:
        return "ok"
    reason = result.evidence.get("reason")
    return str(reason) if reason else "unknown_failure"


def summarize(results: Iterable[Result]) -> dict[str, Any]:
    """Serialize exactly the aggregate metrics used by the benchmark report."""
    episodes = list(results)
    if not episodes:
        raise ValueError("at least one episode result is required")
    failures = Counter(failure_reason(result) for result in episodes if not result.success)
    return {
        "agent_name": "OraclePortalAgent",
        "episodes": len(episodes),
        "success_rate": sum(result.success for result in episodes) / len(episodes),
        "average_steps": sum(result.steps for result in episodes) / len(episodes),
        "average_duration": sum(result.elapsed_time for result in episodes) / len(episodes),
        "failure_reason_distribution": dict(sorted(failures.items())),
        "episode_results": [asdict(result) for result in episodes],
    }


def run(episodes: int, max_steps: int) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    task = replace(L1_PORTAL_TASK, max_steps=max_steps)
    results: list[Result] = []
    for _ in range(episodes):
        # A fresh server instance is required: completed/failed MineRL worlds
        # are not safe to reuse and would contaminate latency measurements.
        results.append(
            BenchmarkRunner().run(
                task,
                L1ControlledEnv(),
                OraclePortalAgent(),
                L1Evaluator(),
            )
        )
    return summarize(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OraclePortalAgent on formal L1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=L1_PORTAL_TASK.max_steps)
    parser.add_argument("--output", default="results/oracle_eval.json")
    args = parser.parse_args()
    report = run(max(1, args.episodes), max(1, args.max_steps))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
