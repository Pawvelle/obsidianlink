"""Agent ↔ Environment loop for the unified Agent API.

Does not modify the Oracle, Task, or L1Evaluator. The agent only sees
``Observation``. Reward / done are read from evaluator-only
``hidden_state`` by this runner for logging, never passed into
``agent.act``.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_agent.py --agent random
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_agent.py --agent reactive
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_agent.py --agent llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.agents.random_agent import RandomAgent
from obsidianlink.agents.reactive_agent import ReactiveAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

_FORBIDDEN = frozenset({ActionType.EQUIP, ActionType.PLACE})


def _hidden(env: Environment) -> dict[str, Any]:
    raw = getattr(env, "hidden_state", None)
    return dict(raw) if isinstance(raw, dict) else {}


def run_episode(
    agent: BaseAgent,
    env: Environment,
    *,
    max_steps: int = 128,
) -> dict[str, Any]:
    """reset → act(observation) → step until done / finished / max_steps."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    agent.reset()
    t0 = time.perf_counter()
    observation: Observation = env.reset()
    steps = 0
    last_reward: float | None = None
    done = False
    error: str | None = None
    try:
        while steps < max_steps:
            hidden = _hidden(env)
            if bool(hidden.get("done")):
                done = True
                last_reward = hidden.get("reward")
                break
            if getattr(agent, "finished", False):
                break
            action = agent.act(observation)
            if not isinstance(action, Action):
                raise TypeError(f"agent.act must return Action, got {type(action)!r}")
            if action.type in _FORBIDDEN:
                action = Action(type=ActionType.WAIT)
            observation = env.step(action)
            steps += 1
            hidden = _hidden(env)
            last_reward = hidden.get("reward")
            if bool(hidden.get("done")):
                done = True
                break
            if getattr(agent, "finished", False):
                break
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    # Pipeline success: the loop connected agent ↔ env without crashing.
    # Task success is not claimed here (RandomAgent is interface smoke).
    report = {
        "success": error is None,
        "reward": last_reward,
        "steps": steps,
        "time": elapsed,
        "done": done,
        "agent_finished": bool(getattr(agent, "finished", False)),
        "error": error,
        "agent": type(agent).__name__,
    }
    return report


def _make_agent(name: str) -> BaseAgent:
    key = name.strip().lower()
    if key == "random":
        return RandomAgent()
    if key == "reactive":
        return ReactiveAgent()
    if key == "llm":
        from obsidianlink.agents.llm_agent import LLMAgent
        from obsidianlink.models.minimax_client import MiniMaxClient

        return LLMAgent(MiniMaxClient())
    raise ValueError("unknown agent; use 'random', 'reactive', or 'llm'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RandomAgent, ReactiveAgent, or LLMAgent on L1")
    parser.add_argument("--agent", choices=("random", "reactive", "llm"), default="random")
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    from obsidianlink.env.l1_scene import L1ControlledEnv

    agent = _make_agent(args.agent)
    env = L1ControlledEnv()
    try:
        report = run_episode(agent, env, max_steps=max(1, int(args.max_steps)))
    finally:
        try:
            env.close()
        except Exception:
            pass
    print(json.dumps(report, indent=2))
    sys.stdout.flush()
    return 0 if report.get("success") and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
