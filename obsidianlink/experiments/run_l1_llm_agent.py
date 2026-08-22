"""First formal L1 Portal Benchmark episode with LLMAgent.

Uses the existing L1 controlled environment, L1_PORTAL_TASK, LLMAgent,
and L1Evaluator. Does not add a planner, memory, RAG, multi-agent loop,
or hard-coded solver. The agent only sees Observation.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_llm_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.agents.llm_agent import LLMAgent
from obsidianlink.benchmark.l1_evaluator import L1Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.l1_scene import L1_ENV_ID
from obsidianlink.models.minimax_client import MiniMaxClient, redact
from obsidianlink.tasks.portal import L1_PORTAL_TASK

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
DEFAULT_MAX_STEPS = 500
EXPERIMENT_NAME = "L1 LLMAgent Prompt Baseline v2"
VISION_EXPERIMENT_NAME = "L1 LLMAgent Vision Baseline v3"
PROMPT_VARIANT = "baseline_v2"
VISION_PROMPT_VARIANT = "baseline_v3_vision"
_DISTRIBUTION_VERBS: tuple[str, ...] = (
    ActionType.MOVE.value,
    ActionType.CAMERA.value,
    ActionType.USE.value,
    ActionType.ATTACK.value,
    ActionType.EQUIP.value,
    ActionType.WAIT.value,
)


def action_distribution(verbs: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in _DISTRIBUTION_VERBS}
    for verb in verbs:
        if verb in counts:
            counts[verb] += 1
    return counts


def new_experiment_id(stamp: str | None = None) -> str:
    tag = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"l1_llm_{tag}"


def new_run_dir(experiment_id: str) -> str:
    path = os.path.join(_RUNS_DIR, experiment_id)
    os.makedirs(path, exist_ok=True)
    return path


def episode_payload(
    result: Result,
    *,
    experiment_id: str,
    model: str | None,
    api_url: str | None,
    agent_name: str,
    task: Task,
    verbs: list[str],
    parsed_ok_count: int,
    experiment_name: str = EXPERIMENT_NAME,
    prompt_variant: str = PROMPT_VARIANT,
    use_vision: bool = False,
    vision_calls: int = 0,
    last_used_vision: bool = False,
    last_fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Compact L1 LLM episode record. Never includes the API key."""
    evidence = dict(result.evidence or {})
    milestones = evidence.get("milestones") if isinstance(evidence.get("milestones"), dict) else {}
    nether_entered = bool(milestones.get("nether_entered", False))
    portal_activated = bool(milestones.get("portal_activated", False))
    failure_reason = evidence.get("reason")
    if result.success:
        failure_reason = None
    error = evidence.get("error")
    failure_detail = evidence.get("failure_detail") or error
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "prompt_variant": prompt_variant,
        "kind": "l1_llm_portal",
        "nether_portal_attempt": True,
        "valid_for_l1_capability_conclusion": False,
        "task_id": result.task_id,
        "env_id": L1_ENV_ID,
        "agent": agent_name,
        "model": model,
        "api_url": api_url,
        "api_key_env": "MINIMAX_API_KEY",
        "success": bool(result.success),
        "nether_entered": nether_entered,
        "portal_activated": portal_activated,
        "portal_constructed": milestones.get("portal_constructed", "unknown"),
        "steps": int(result.steps),
        "elapsed_time": float(result.elapsed_time),
        "model_calls": int(result.model_calls),
        "invalid_actions": int(result.invalid_actions),
        "failure_reason": failure_reason,
        "failure_detail": failure_detail,
        "failure_class": evidence.get("failure_class"),
        "error": error,
        "task_max_steps": int(L1_PORTAL_TASK.max_steps),
        "episode_max_steps": int(task.max_steps),
        "action_distribution": action_distribution(verbs),
        "verbs": list(verbs),
        "parsed_ok_count": int(parsed_ok_count),
        "use_vision": bool(use_vision),
        "vision_calls": int(vision_calls),
        "last_used_vision": bool(last_used_vision),
        "last_fallback_reason": last_fallback_reason,
        "episode_result": "success" if result.success else "failure",
    }


def _write_json(path: str, payload: Any) -> None:
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if key:
        text = redact(text, key)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


class _TraceAgent(BaseAgent):
    """Experiment-only wrapper. Does not change LLMAgent."""

    def __init__(self, inner: BaseAgent) -> None:
        self._inner = inner
        self.verbs: list[str] = []
        self.parsed_ok_count = 0

    def reset(self) -> None:
        reset = getattr(self._inner, "reset", None)
        if callable(reset):
            reset()
        self.verbs = []
        self.parsed_ok_count = 0

    def act(self, observation: Observation) -> Action:
        action = self._inner.act(observation)
        self.verbs.append(action.type.value)
        if bool(getattr(self._inner, "last_parsed_ok", True)):
            self.parsed_ok_count += 1
        return action

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def run_l1_llm_episode(
    agent: BaseAgent,
    env: Environment,
    *,
    task: Task | None = None,
    evaluator: L1Evaluator | None = None,
    experiment_id: str | None = None,
    run_dir: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Official L1 loop: Observation → LLMAgent → Action → env.step → Evaluator."""
    base_task = task if task is not None else L1_PORTAL_TASK
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        episode_task = replace(base_task, max_steps=int(max_steps))
    else:
        episode_task = base_task
    exp_id = experiment_id or new_experiment_id()
    dest = run_dir or new_run_dir(exp_id)
    os.makedirs(dest, exist_ok=True)
    traced = agent if isinstance(agent, _TraceAgent) else _TraceAgent(agent)
    client = getattr(traced, "_client", None)
    runner = BenchmarkRunner()
    result = runner.run(
        episode_task,
        env,
        traced,
        evaluator if evaluator is not None else L1Evaluator(),
    )
    use_vision = bool(getattr(traced, "use_vision", False))
    payload = episode_payload(
        result,
        experiment_id=exp_id,
        model=getattr(client, "model", None),
        api_url=getattr(client, "url", None),
        agent_name=type(getattr(traced, "_inner", traced)).__name__,
        task=episode_task,
        verbs=list(traced.verbs),
        parsed_ok_count=int(traced.parsed_ok_count),
        experiment_name=VISION_EXPERIMENT_NAME if use_vision else EXPERIMENT_NAME,
        prompt_variant=VISION_PROMPT_VARIANT if use_vision else PROMPT_VARIANT,
        use_vision=use_vision,
        vision_calls=int(getattr(traced, "vision_calls", 0) or 0),
        last_used_vision=bool(getattr(traced, "last_used_vision", False)),
        last_fallback_reason=getattr(traced, "last_fallback_reason", None),
    )
    payload["run_dir"] = dest
    _write_json(os.path.join(dest, "result.json"), payload)
    return payload


def _make_agent(*, use_vision: bool) -> LLMAgent:
    timeout_s = 90.0 if use_vision else 60.0
    return LLMAgent(MiniMaxClient(timeout_s=timeout_s), use_vision=use_vision)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLMAgent on the formal L1 Nether Portal benchmark"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Episode budget (default: {DEFAULT_MAX_STEPS}; official task is {L1_PORTAL_TASK.max_steps})",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Attach Observation.frame RGB to MiniMax-M3 (Vision Baseline v3)",
    )
    args = parser.parse_args()

    from obsidianlink.env.l1_scene import L1ControlledEnv

    exp_id = new_experiment_id()
    dest = new_run_dir(exp_id)
    name = VISION_EXPERIMENT_NAME if args.vision else EXPERIMENT_NAME
    print(
        f"[l1-llm] experiment={name} experiment_id={exp_id} "
        f"max_steps={max(1, int(args.max_steps))} vision={bool(args.vision)} "
        f"run_dir={dest}",
        flush=True,
    )
    env = L1ControlledEnv()
    agent = _make_agent(use_vision=bool(args.vision))
    report = run_l1_llm_episode(
        agent,
        env,
        experiment_id=exp_id,
        run_dir=dest,
        max_steps=max(1, int(args.max_steps)),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
