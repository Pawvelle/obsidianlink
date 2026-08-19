"""Minimal benchmark loop.

reset → observe → agent.act → clamp → env.step → loop → evaluator → Result → close

Exceptions become a structured :class:`Result`; they do not abort the
caller without evidence.
"""

from __future__ import annotations

import os
import time
from typing import Any

from obsidianlink.agents.base import Agent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import (
    AGENT_FAILURE,
    ENVIRONMENT_FAILURE,
    EVALUATOR_FAILURE,
    Result,
)
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation


def clamp_to_allowed(
    action: Action, allowed_actions: tuple[str, ...]
) -> tuple[Action, str | None]:
    """Replace a disallowed action with WAIT.

    An empty ``allowed_actions`` means unrestricted. Returns
    ``(action, None)`` when the action is allowed, otherwise
    ``(WAIT, original_verb)``.
    """
    if not allowed_actions:
        return action, None
    allowed = {name.strip().lower() for name in allowed_actions if name}
    if action.type.value in allowed:
        return action, None
    return Action(type=ActionType.WAIT), action.type.value


def _save_frame_png(frame: Any, path: str) -> bool:
    if frame is None:
        return False
    try:
        from PIL import Image

        if isinstance(frame, Image.Image):
            frame.save(path)
            return True
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3:
            return False
        Image.fromarray(frame).save(path)
        return True
    except Exception:
        return False


def _attach_disallowed(result: Result, disallowed: str | None) -> Result:
    evidence = dict(result.evidence)
    if disallowed:
        evidence["disallowed_action"] = disallowed
        evidence.setdefault("reason", "disallowed_action")
    return Result(
        task_id=result.task_id,
        success=result.success,
        steps=result.steps,
        model_calls=result.model_calls,
        invalid_actions=result.invalid_actions,
        elapsed_time=result.elapsed_time,
        evidence=evidence,
    )


def _agent_tool_evidence(agent: Agent) -> dict[str, Any]:
    """Expose optional agent-local tool metrics without changing Agent API."""
    evidence: dict[str, Any] = {}
    wiki_calls = getattr(agent, "wiki_calls", None)
    if wiki_calls is not None:
        evidence["wiki_calls"] = wiki_calls
    wiki_queries = getattr(agent, "wiki_queries", None)
    if wiki_queries is not None:
        evidence["wiki_queries"] = list(wiki_queries)
    tool_trace = getattr(agent, "last_tool_trace", None)
    if tool_trace:
        evidence["tool_trace_summary"] = list(tool_trace)
    return evidence


def _attach_agent_evidence(result: Result, agent: Agent) -> Result:
    agent_evidence = _agent_tool_evidence(agent)
    if not agent_evidence:
        return result
    evidence = dict(result.evidence)
    evidence.update(agent_evidence)
    return Result(
        task_id=result.task_id,
        success=result.success,
        steps=result.steps,
        model_calls=result.model_calls,
        invalid_actions=result.invalid_actions,
        elapsed_time=result.elapsed_time,
        evidence=evidence,
    )


class BenchmarkRunner:
    def run(
        self,
        task: Task,
        env: Environment,
        agent: Agent,
        evaluator: Evaluator,
        *,
        debug_save_dir: str | None = None,
    ) -> Result:
        started = time.perf_counter()
        last_input_observation: Observation | None = None
        last_hidden_state: Any = None
        steps = 0
        extra_invalid = 0
        last_disallowed: str | None = None
        aborted: Result | None = None
        debug_dir: str | None = None
        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)
            debug_dir = debug_save_dir

        def _abort(failure_class: str, reason: str, exc: Exception) -> Result:
            evidence: dict[str, Any] = {
                "failure_class": failure_class,
                "reason": reason,
                "error": f"{type(exc).__name__}: {exc}",
                "used_vision": getattr(agent, "last_used_vision", None),
                "fallback_reason": getattr(agent, "last_fallback_reason", None),
                "vision_calls": getattr(agent, "vision_calls", 0),
                **_agent_tool_evidence(agent),
            }
            if last_disallowed:
                evidence["disallowed_action"] = last_disallowed
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=getattr(agent, "model_calls", 0),
                invalid_actions=getattr(agent, "invalid_actions", 0) + extra_invalid,
                elapsed_time=time.perf_counter() - started,
                evidence=evidence,
            )

        try:
            try:
                env.reset()
                observation = env.observe()
                last_hidden_state = getattr(env, "hidden_state", None)
            except Exception as exc:
                aborted = _abort(ENVIRONMENT_FAILURE, "environment_exception", exc)
            else:
                for step_idx in range(task.max_steps):
                    last_input_observation = observation
                    try:
                        action = agent.act(observation)
                    except Exception as exc:
                        aborted = _abort(AGENT_FAILURE, "agent_exception", exc)
                        break
                    action, disallowed = clamp_to_allowed(
                        action, task.allowed_actions
                    )
                    if disallowed is not None:
                        extra_invalid += 1
                        last_disallowed = disallowed
                    try:
                        env.step(action)
                        observation = env.observe()
                        last_hidden_state = getattr(env, "hidden_state", None)
                    except Exception as exc:
                        aborted = _abort(
                            ENVIRONMENT_FAILURE, "environment_exception", exc
                        )
                        break
                    steps += 1
                    if debug_dir is not None:
                        _save_frame_png(
                            getattr(last_input_observation, "frame", None),
                            os.path.join(
                                debug_dir, f"step_{step_idx + 1}_frame.png"
                            ),
                        )
        finally:
            try:
                env.close()
            except Exception:
                pass

        if aborted is not None:
            return aborted

        invalid_actions = getattr(agent, "invalid_actions", 0) + extra_invalid
        try:
            result = evaluator.evaluate(
                task,
                steps=steps,
                model_calls=getattr(agent, "model_calls", 0),
                invalid_actions=invalid_actions,
                elapsed_time=time.perf_counter() - started,
                observation=last_input_observation,
                raw_response=getattr(agent, "last_raw_response", None),
                ground_truth=task.ground_truth,
                hidden_state=last_hidden_state,
                used_vision=getattr(agent, "last_used_vision", None),
                fallback_reason=getattr(agent, "last_fallback_reason", None),
                vision_calls=getattr(agent, "vision_calls", 0),
            )
        except Exception as exc:
            return _abort(EVALUATOR_FAILURE, "evaluator_exception", exc)
        return _attach_disallowed(_attach_agent_evidence(result, agent), last_disallowed)
