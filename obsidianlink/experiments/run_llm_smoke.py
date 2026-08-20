"""First live LLM embodied-control smoke.

Not a Nether Portal attempt. The prompt only asks the model to look,
move, select a hotbar slot, and optionally USE or WAIT.

Does not modify Environment / Evaluator / Oracle / Task.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_llm_smoke.py --agent llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.agents.llm_agent import LLMAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.models.minimax_client import MiniMaxClient, redact

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
_FORBIDDEN = frozenset({ActionType.EQUIP, ActionType.PLACE})

SMOKE_GOAL = (
    "This is an interface smoke test, NOT a Nether Portal task. "
    "Do not build a portal and do not try to enter the Nether. "
    "Over a few steps: (1) look around with camera, (2) move a little, "
    "(3) select a hotbar slot from 1 to 5, (4) optionally USE once or WAIT. "
    "Prefer simple legal JSON. One action per response."
)


def observation_summary(observation: Observation) -> dict[str, Any]:
    frame = observation.frame
    frame_info: dict[str, Any]
    if frame is None:
        frame_info = {"present": False}
    else:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        mean = None
        try:
            mean = float(getattr(frame, "mean")())
        except Exception:
            mean = None
        frame_info = {
            "present": True,
            "shape": list(shape) if shape is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "mean": mean,
        }
    return {
        "inventory": dict(observation.inventory or {}),
        "selected_item": observation.selected_item,
        "frame": frame_info,
    }


def action_record(action: Action) -> dict[str, Any]:
    return {
        "type": action.type.value,
        "dx": action.dx,
        "dz": action.dz,
        "yaw": action.yaw,
        "pitch": action.pitch,
        "target": action.target,
        "sneak": action.sneak,
    }


def _hidden(env: Environment) -> dict[str, Any]:
    raw = getattr(env, "hidden_state", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _secret() -> str:
    return os.environ.get("MINIMAX_API_KEY", "").strip()


def _safe(value: Any) -> Any:
    key = _secret()
    if isinstance(value, str):
        return redact(value, key) if key else value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value


def new_run_dir(stamp: str | None = None) -> str:
    tag = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    path = os.path.join(_RUNS_DIR, f"llm_smoke_{tag}")
    os.makedirs(path, exist_ok=True)
    return path


def write_run_files(
    run_dir: str,
    *,
    prompts: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    mapping = {
        "prompts.json": prompts,
        "responses.json": responses,
        "actions.json": actions,
        "result.json": result,
    }
    for name, payload in mapping.items():
        path = os.path.join(run_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_safe(payload), fh, indent=2)


def run_llm_smoke_episode(
    agent: BaseAgent,
    env: Environment,
    *,
    max_steps: int = 8,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """reset → LLM act → env.step, writing traces after every step."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    dest = run_dir or new_run_dir()
    prompts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    agent.reset()
    t0 = time.perf_counter()
    steps = 0
    done = False
    error: str | None = None
    last_reward: float | None = None
    api_calls = 0
    api_ok = 0
    verbs: list[str] = []
    executed = 0
    observation = env.reset()
    client = getattr(agent, "_client", None)
    result: dict[str, Any] = {
        "kind": "llm_embodied_smoke",
        "valid_for_l1_agent_conclusion": False,
        "nether_portal_attempt": False,
        "run_dir": dest,
        "agent": type(agent).__name__,
        "goal": SMOKE_GOAL,
        "model": getattr(client, "model", None),
        "api_url": getattr(client, "url", None),
        "api_key_env": "MINIMAX_API_KEY",
    }

    def flush() -> None:
        result.update(
            {
                "success": error is None and executed >= 1,
                "reward": last_reward,
                "steps": steps,
                "time": time.perf_counter() - t0,
                "done": done,
                "error": error,
                "api_calls": api_calls,
                "api_successes": api_ok,
                "minecraft_steps": executed,
                "verbs": verbs,
                "parsed_ok_count": sum(1 for row in actions if row.get("parsed_ok")),
                "invalid_actions": getattr(agent, "invalid_actions", 0),
            }
        )
        write_run_files(
            dest,
            prompts=prompts,
            responses=responses,
            actions=actions,
            result=result,
        )

    try:
        while steps < max_steps:
            hidden = _hidden(env)
            if bool(hidden.get("done")):
                done = True
                last_reward = hidden.get("reward")
                break
            stamp = datetime.now(timezone.utc).isoformat()
            obs_before = observation_summary(observation)
            try:
                action = agent.act(observation)
                api_calls += 1
                api_ok += 1
            except Exception as exc:
                api_calls += 1
                error = redact(f"{type(exc).__name__}: {exc}", _secret())
                prompts.append(
                    {
                        "step": steps,
                        "timestamp": stamp,
                        "prompt": getattr(agent, "last_prompt", None),
                    }
                )
                responses.append(
                    {
                        "step": steps,
                        "timestamp": stamp,
                        "text": getattr(agent, "last_raw_response", None),
                        "raw": getattr(getattr(agent, "_client", None), "last_raw_response", None),
                        "error": error,
                    }
                )
                break
            if not isinstance(action, Action):
                error = f"agent.act must return Action, got {type(action)!r}"
                break
            if action.type in _FORBIDDEN:
                action = Action(type=ActionType.WAIT)
            parsed_ok = bool(getattr(agent, "last_parsed_ok", True))
            client = getattr(agent, "_client", None)
            prompts.append(
                {
                    "step": steps,
                    "timestamp": stamp,
                    "prompt": getattr(agent, "last_prompt", None),
                    "observation": obs_before,
                }
            )
            responses.append(
                {
                    "step": steps,
                    "timestamp": stamp,
                    "text": getattr(agent, "last_raw_response", None),
                    "raw": getattr(client, "last_raw_response", None),
                    "error": getattr(client, "last_error", None),
                }
            )
            observation = env.step(action)
            executed += 1
            steps += 1
            verbs.append(action.type.value)
            hidden = _hidden(env)
            last_reward = hidden.get("reward")
            actions.append(
                {
                    "step": steps,
                    "timestamp": stamp,
                    "observation": obs_before,
                    "parsed_ok": parsed_ok,
                    "action": action_record(action),
                    "minecraft_executed": True,
                }
            )
            flush()
            if bool(hidden.get("done")):
                done = True
                break
    except Exception as exc:  # noqa: BLE001
        error = redact(f"{type(exc).__name__}: {exc}", _secret())
    flush()
    result["run_dir"] = dest
    return result


def _make_agent(name: str) -> BaseAgent:
    key = name.strip().lower()
    if key != "llm":
        raise ValueError("run_llm_smoke.py only supports --agent llm")
    client = MiniMaxClient()
    return LLMAgent(client, goal=SMOKE_GOAL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live MiniMax LLM smoke on L1 env")
    parser.add_argument("--agent", choices=("llm",), default="llm")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    from obsidianlink.env.l1_scene import L1ControlledEnv

    dest = new_run_dir()
    print(f"[llm-smoke] run_dir={dest}")
    sys.stdout.flush()
    agent = _make_agent(args.agent)
    env = L1ControlledEnv()
    try:
        report = run_llm_smoke_episode(
            agent,
            env,
            max_steps=max(1, int(args.max_steps)),
            run_dir=dest,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
    print(json.dumps(_safe(report), indent=2))
    sys.stdout.flush()
    return 0 if report.get("success") and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
