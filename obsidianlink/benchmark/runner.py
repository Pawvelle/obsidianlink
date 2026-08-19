"""Minimal benchmark loop. Does not start Minecraft.

The runner owns the env lifecycle (``reset -> step loop -> close``) and
forwards a fixed minimum of evidence to the :class:`Evaluator`:

* the primary metric set from the Development Plan
  (``steps`` / ``model_calls`` / ``invalid_actions`` / ``elapsed_time``);
* the latest side-channel payload the Agent emitted (e.g. a
  :class:`PerceptionReport` for D1) via ``report=``;
* the *agent-visible* observation the Agent most recently acted on via
  ``observation=``. This is also the evaluator-only ground truth for
  Phase 2A; the runner captures it **before** calling ``env.step()``
  so the Agent's view and the Evaluator's view are aligned.

``invalid_actions`` stays at 0 for Phase 2A: no D / L task yet
implements action validation. Action validation lands when the bounded
action set has tasks that care about action legality (D3 / L1+).

D2-01 additionally receives ``hidden_state`` (evaluator-only pose
from MineRL monitors) and ``final_observation`` (post-last-step).
D1 evaluators ignore both.

Debug mode
----------

``run()`` accepts an optional ``debug_save_dir`` keyword argument.
When set, the runner writes the agent-visible ``Observation.frame``
for every step into that directory as ``step_<N>_frame.png`` (the
exact bytes that were forwarded to the model). This is purely
additive: when ``debug_save_dir`` is ``None`` (default), the runner
behaves identically to the non-debug path. The flag is intended for
human sanity-checks of the observation pipeline (D1 v2 debug)
debug) and does NOT change model input, prompt, evaluator, or
ground truth in any way.
"""

from __future__ import annotations

import os
import time
from typing import Any

from obsidianlink.agents.base import Agent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task
from obsidianlink.env.environment import Environment, Observation


def _save_frame_png(frame: Any, path: str) -> bool:
    """Write a frame (PIL Image or ndarray) to ``path`` as PNG.

    Returns True on success, False if the frame is ``None`` or not a
    recognised image type. Permissive on purpose: a failed debug
    save must never crash the benchmark.
    """
    if frame is None:
        return False
    try:
        from PIL import Image  # local import

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
        observation: Observation = env.reset()
        steps = 0
        # Track the last agent-visible observation (i.e. what the Agent
        # saw when it produced the most recent action / side-channel
        # payload), the latest side-channel payload it emitted, and the
        # raw model response for the last act() call. The runner
        # forwards all three to the Evaluator so the Evaluator has the
        # agent-visible ground truth for Diagnostic tasks without ever
        # touching the Agent's prompts or memory.
        last_input_observation: Observation = observation
        last_report: Any = None
        last_raw_response: Any = None
        last_hidden_state: Any = getattr(env, "hidden_state", None)
        debug_dir: str | None = None
        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)
            debug_dir = debug_save_dir
        try:
            for step_idx in range(task.max_steps):
                last_input_observation = observation
                action = agent.act(observation)
                observation = env.step(action)
                last_report = getattr(agent, "last_report", None)
                last_raw_response = getattr(agent, "last_raw_response", None)
                last_hidden_state = getattr(env, "hidden_state", None)
                steps += 1
                # Debug-only: persist the exact frame the model saw
                # this step. We save *after* agent.act() so the saved
                # bytes are identical to what ``call_model`` received
                # (the Agent does not mutate the observation).
                if debug_dir is not None:
                    frame = getattr(last_input_observation, "frame", None)
                    _save_frame_png(
                        frame,
                        os.path.join(debug_dir, f"step_{step_idx + 1}_frame.png"),
                    )
            if debug_dir is not None:
                _save_frame_png(
                    getattr(observation, "frame", None),
                    os.path.join(debug_dir, "final_frame.png"),
                )
        finally:
            env.close()

        model_calls = getattr(agent, "model_calls", 0)
        return evaluator.evaluate(
            task,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=0,
            elapsed_time=time.perf_counter() - started,
            report=last_report,
            observation=last_input_observation,
            raw_response=last_raw_response,
            ground_truth=task.ground_truth,
            final_observation=observation,
            hidden_state=last_hidden_state,
        )
