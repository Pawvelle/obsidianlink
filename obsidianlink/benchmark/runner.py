"""Minimal benchmark loop.

reset → observe → agent.act → env.step → loop → evaluator → Result → close
"""

from __future__ import annotations

import os
import time
from typing import Any

from obsidianlink.agents.base import Agent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import (
    ENVIRONMENT_FAILURE,
    Result,
)
from obsidianlink.benchmark.task import Task
from obsidianlink.env.environment import Environment, Observation


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
        debug_dir: str | None = None
        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)
            debug_dir = debug_save_dir
        try:
            try:
                env.reset()
                observation = env.observe()
                last_hidden_state = getattr(env, "hidden_state", None)
                for step_idx in range(task.max_steps):
                    last_input_observation = observation
                    action = agent.act(observation)
                    env.step(action)
                    observation = env.observe()
                    last_hidden_state = getattr(env, "hidden_state", None)
                    steps += 1
                    if debug_dir is not None:
                        _save_frame_png(
                            getattr(last_input_observation, "frame", None),
                            os.path.join(debug_dir, f"step_{step_idx + 1}_frame.png"),
                        )
            except Exception as exc:
                return Result(
                    task_id=task.task_id,
                    success=False,
                    steps=steps,
                    model_calls=getattr(agent, "model_calls", 0),
                    invalid_actions=getattr(agent, "invalid_actions", 0),
                    elapsed_time=time.perf_counter() - started,
                    evidence={
                        "failure_class": ENVIRONMENT_FAILURE,
                        "reason": "environment_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "used_vision": getattr(agent, "last_used_vision", None),
                        "fallback_reason": getattr(
                            agent, "last_fallback_reason", None
                        ),
                        "vision_calls": getattr(agent, "vision_calls", 0),
                    },
                )
        finally:
            env.close()

        return evaluator.evaluate(
            task,
            steps=steps,
            model_calls=getattr(agent, "model_calls", 0),
            invalid_actions=getattr(agent, "invalid_actions", 0),
            elapsed_time=time.perf_counter() - started,
            observation=last_input_observation,
            raw_response=getattr(agent, "last_raw_response", None),
            ground_truth=task.ground_truth,
            hidden_state=last_hidden_state,
            used_vision=getattr(agent, "last_used_vision", None),
            fallback_reason=getattr(agent, "last_fallback_reason", None),
            vision_calls=getattr(agent, "vision_calls", 0),
        )
