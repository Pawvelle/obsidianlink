"""Run one bounded, asynchronous local-VLM Phase 3 A0 episode.

The MineRL owner always advances with a safe one-tick wait while inference is
pending.  A returned decision is accepted only for the same episode/agent and
within the explicit age bound; all other model output is logged then dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VENDORED_MINERL = ROOT / "vendor/minerl"
PINNED_JAVA_HOME = Path("/opt/anaconda3/envs/mc-agent")
TASK_PATH = ROOT / "benchmark/instances/route_a_a0_phase3.json"
MODEL_PATH = ROOT / "models/Qwen3-VL-2B-Instruct"


def _configure_pinned_java_home() -> None:
    java = PINNED_JAVA_HOME / "bin/java"
    if not java.is_file():
        raise RuntimeError(f"missing pinned Java runtime: {java}")
    os.environ["JAVA_HOME"] = str(PINNED_JAVA_HOME)
    os.environ["PATH"] = f"{PINNED_JAVA_HOME / 'bin'}:{os.environ.get('PATH', '')}"


_configure_pinned_java_home()
for import_root in (ROOT, VENDORED_MINERL):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from obsidianlink.agents import (
    AsyncA0PolicyWorker,
    DirectA0Policy,
    LocalQwenResponder,
    MiniMaxM3Responder,
    WorkflowA0Policy,
)
from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.drivers.scripted_a0 import _step_deadline
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation import PortalEvaluator


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_task() -> TaskInstance:
    return TaskInstance.from_dict(json.loads(TASK_PATH.read_text(encoding="utf-8")))


def _experiment_path(mode: str, planner_backend: str) -> Path:
    if planner_backend == "minimax_m3":
        return ROOT / f"configs/experiments/phase3_minimax_m3_{mode}_a0.json"
    return ROOT / f"configs/experiments/phase3_single_{mode}_a0.json"


def _write_snapshot(run_dir: Path, args: argparse.Namespace, experiment: Path) -> None:
    shutil.copyfile(TASK_PATH, run_dir / "task_instance.json")
    shutil.copyfile(experiment, run_dir / "experiment_config.json")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
        capture_output=True, text=True,
    )
    (run_dir / "code_version.json").write_text(
        json.dumps(
            {
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "runner_arguments": _json_ready(vars(args)),
                "task_instance": str(TASK_PATH.relative_to(ROOT)),
                "experiment_config": str(experiment.relative_to(ROOT)),
                "model_path": str(MODEL_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _evaluation_summary(backend: MineRLEnvironmentBackend) -> dict[str, Any]:
    state = backend.get_evaluation_state()
    result = PortalEvaluator().evaluate(state)
    return {
        "success": result.success,
        "step_id": result.step_id,
        "milestones": list(result.milestones),
        "blocking_conditions": list(result.blocking_conditions),
        "failure_type": result.failure_type,
        "failure_step": result.failure_step,
        "last_successful_milestone": result.last_successful_milestone,
        "episode_terminated": result.episode_terminated,
        "terminated_step": result.terminated_step,
        "terminated_reason": result.terminated_reason,
        "entered_via_episode_portal": result.entered_via_episode_portal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("workflow", "direct"), default="workflow")
    parser.add_argument(
        "--planner-backend", choices=("local_qwen", "minimax_m3"),
        default="local_qwen",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/phase3-vlm-a0")
    parser.add_argument("--step-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-decision-age-steps", type=int, default=160)
    parser.add_argument("--min-step-interval-seconds", type=float, default=0.25)
    parser.add_argument(
        "--preload-model", action=argparse.BooleanOptionalAction, default=True,
        help="load the local model before opening MineRL; never during owner stepping",
    )
    args = parser.parse_args()
    if args.step_timeout_seconds <= 0:
        parser.error("--step-timeout-seconds must be positive")
    if args.max_decision_age_steps < 0:
        parser.error("--max-decision-age-steps must be non-negative")
    if args.min_step_interval_seconds < 0:
        parser.error("--min-step-interval-seconds must be non-negative")

    experiment = _experiment_path(args.mode, args.planner_backend)
    experiment_config = json.loads(experiment.read_text(encoding="utf-8"))
    task = _load_task()
    if experiment_config["max_model_calls"] != task.limits["max_model_calls"]:
        raise RuntimeError("experiment and task model-call budgets disagree")
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_snapshot(run_dir, args, experiment)

    responder = (
        LocalQwenResponder(MODEL_PATH)
        if args.planner_backend == "local_qwen"
        else MiniMaxM3Responder()
    )
    policy = WorkflowA0Policy(responder) if args.mode == "workflow" else DirectA0Policy(responder)
    worker = AsyncA0PolicyWorker(policy)
    backend = MineRLEnvironmentBackend()
    if args.preload_model and args.planner_backend == "local_qwen":
        responder.prepare()
    backend.open()
    worker.start()
    try:
        observations = backend.reset(task)
        observation = observations["agent_1"]
        Image.fromarray(observation.frame).save(run_dir / "initial.png")
        calls_submitted = 0
        decisions_applied = 0
        decisions_dropped_stale = 0
        decisions_rejected = 0
        event_rows: list[dict[str, Any]] = []
        while observation.step_id < task.limits["max_environment_steps"]:
            iteration_started = time.monotonic()
            if worker.failure is not None:
                raise RuntimeError(
                    "local VLM worker failed: " + str(worker.failure)
                ) from worker.failure
            if calls_submitted < task.limits["max_model_calls"]:
                if worker.submit(task, observation):
                    calls_submitted += 1
            action = MacroAction.wait()
            action_source_step: int | None = None
            pending = worker.poll(
                episode_id=observation.episode_id, agent_id=observation.agent_id,
            )
            if pending is not None:
                age = observation.step_id - pending.step_id
                if age < 0 or age > args.max_decision_age_steps:
                    decisions_dropped_stale += 1
                elif not pending.decision.accepted:
                    decisions_rejected += 1
                else:
                    action = pending.decision.action
                    action_source_step = pending.step_id
                    decisions_applied += 1
            with _step_deadline(args.step_timeout_seconds):
                step = backend.step({"agent_1": action})
            event_rows.append(
                {
                    "episode_id": step.episode_id,
                    "agent_id": "agent_1",
                    "step_id": step.step_id,
                    "event_type": "environment_action",
                    "timestamp": time.time(),
                    "payload": {
                        "action": {
                            "action_type": action.action_type,
                            "target": action.target,
                            "duration_ticks": action.duration_ticks,
                            "parameters": dict(action.parameters),
                        },
                        "action_source_step": action_source_step,
                        "model_calls_submitted": calls_submitted,
                    },
                }
            )
            observation = step.observations["agent_1"]
            if step.terminated:
                break
            remaining = args.min_step_interval_seconds - (
                time.monotonic() - iteration_started
            )
            if remaining > 0:
                time.sleep(remaining)
        backend.mark_terminated(reason="vlm_a0_budget_complete")
        Image.fromarray(observation.frame).save(run_dir / "final.png")
        evaluation = _evaluation_summary(backend)
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            for event in event_rows:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        evaluator_state = backend.get_evaluation_state()
        with (run_dir / "evaluator_events.jsonl").open("w", encoding="utf-8") as handle:
            for event in evaluator_state.milestone_events():
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        status = "passed" if evaluation["success"] else "blocked"
        summary = {
            "status": status,
            "mode": args.mode,
            "planner_backend": args.planner_backend,
            "model_device": responder.device if args.planner_backend == "local_qwen" else None,
            "remote_request": (
                asdict(responder.last_request)
                if args.planner_backend == "minimax_m3"
                and responder.last_request is not None
                else None
            ),
            "steps_completed": observation.step_id,
            "model_calls_submitted": calls_submitted,
            "decisions_applied": decisions_applied,
            "decisions_dropped_stale": decisions_dropped_stale,
            "decisions_rejected": decisions_rejected,
            "formal_evaluation": evaluation,
            "run_dir": str(run_dir),
            "artifacts": {
                "task_instance": "task_instance.json", "experiment_config": "experiment_config.json",
                "code_version": "code_version.json", "initial_frame": "initial.png",
                "final_frame": "final.png", "action_events": "events.jsonl",
                "evaluator_events": "evaluator_events.jsonl",
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if status == "passed" else 2
    except Exception as error:
        summary = {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        worker.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
