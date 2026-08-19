"""ObsidianLink entry points.

Modes:

* ``OBSIDIANLINK_OFFLINE=1`` — no Java / MineRL
* ``OBSIDIANLINK_PHASE=1`` (default) — Validation A: live Treechop loop
* ``OBSIDIANLINK_PHASE=2`` — Validation B/C/D: D1 lava presence on live MineRL
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

NUM_LIVE_STEPS = 16


def main() -> int:
    if os.environ.get("OBSIDIANLINK_OFFLINE") == "1":
        return _run_offline()
    phase = os.environ.get("OBSIDIANLINK_PHASE", "1")
    if phase == "1":
        return _run_phase1()
    if phase == "2":
        return _run_phase2()
    print(f"unknown OBSIDIANLINK_PHASE={phase!r}; expected 1 or 2", file=sys.stderr)
    return 2


def _run_offline() -> int:
    class _StubEnvironment(Environment):
        def __init__(self) -> None:
            self._obs = Observation()
            self.closed = False

        def reset(self) -> Observation:
            self._obs = Observation(frame="offline-stub-frame", inventory={}, selected_item=None)
            return self._obs

        def observe(self) -> Observation:
            return self._obs

        def step(self, action: Action) -> Observation:
            del action
            self._obs = Observation(frame="offline-stub-frame-1", inventory={}, selected_item=None)
            return self._obs

        def close(self) -> None:
            self.closed = True

    print("ObsidianLink offline stub")
    env = _StubEnvironment()
    env.reset()
    env.observe()
    env.step(Action(type=ActionType.WAIT))
    env.close()
    print("offline: reset/observe/step/close wired")
    return 0


def _frame_mean(frame: Any) -> float | None:
    try:
        import numpy as np

        if frame is None:
            return None
        return float(np.mean(frame))
    except Exception:
        return None


def _run_phase1() -> int:
    from obsidianlink.agents.heuristic import HeuristicModelClient
    from obsidianlink.agents.reactive import ReactiveAgent
    from obsidianlink.env.minerl import MineRLEnvironment

    print("ObsidianLink Phase 1 — live MineRL agent loop")
    env = MineRLEnvironment()
    agent = ReactiveAgent(model=HeuristicModelClient())
    print(f"env_id: {env.env_id}")
    sys.stdout.flush()
    started = time.perf_counter()
    try:
        env.reset()
        observation = env.observe()
        first_mean = _frame_mean(observation.frame)
        print(
            "reset ok: "
            f"frame={_frame_shape(observation.frame)} "
            f"mean={first_mean} "
            f"inventory={observation.inventory} "
            f"hidden={env.hidden_state}"
        )
        sys.stdout.flush()
        last_mean = first_mean
        for i in range(NUM_LIVE_STEPS):
            action = agent.act(observation)
            env.step(action)
            observation = env.observe()
            last_mean = _frame_mean(observation.frame)
            print(
                f"step {i + 1}/{NUM_LIVE_STEPS}: {action.type.value} "
                f"frame_mean={last_mean} used_vision={agent.last_used_vision}"
            )
            sys.stdout.flush()
        elapsed = time.perf_counter() - started
        print(
            f"Phase 1 done in {elapsed:.1f}s; "
            f"frame_mean {first_mean} -> {last_mean}; "
            f"vision_calls={agent.vision_calls} text_calls={agent.text_calls}"
        )
        if first_mean is None or last_mean is None:
            print("VALIDATION A FAIL: missing RGB frame")
            return 1
        print("VALIDATION A: reset → RGB → action → next observation OK")
        return 0
    finally:
        env.close()


def _run_phase2() -> int:
    from obsidianlink.agents.qwen_vl import QwenVLModelClient
    from obsidianlink.agents.reactive import ReactiveAgent
    from obsidianlink.benchmark.runner import BenchmarkRunner
    from obsidianlink.env.environment import observation_field_names
    from obsidianlink.env.scene import ControlledSceneEnv
    from obsidianlink.tasks.diagnostic import (
        D1_ENV_IDS,
        D1_TASKS,
        D1LavaEvaluator,
        d1_prompt,
    )

    model_path = os.environ.get(
        "OBSIDIANLINK_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-VL-2B-Instruct"),
    )
    model_path = os.path.abspath(model_path)
    condition = os.environ.get("OBSIDIANLINK_D1_CONDITION", "positive")
    print("ObsidianLink Phase 2 — D1 lava presence")
    print(f"model: {model_path}")
    print(f"condition: {condition}")
    sys.stdout.flush()

    task = D1_TASKS[condition]
    env = ControlledSceneEnv(env_id=D1_ENV_IDS[condition])
    model = QwenVLModelClient(model_path)
    agent = ReactiveAgent(model=model, prompt_builder=d1_prompt)
    result = BenchmarkRunner().run(task=task, env=env, agent=agent, evaluator=D1LavaEvaluator())
    evidence = dict(result.evidence)
    print(f"success={result.success} reason={evidence.get('reason')}")
    print(f"failure_class={evidence.get('failure_class')}")
    print(f"used_vision={evidence.get('used_vision')} vision_calls={evidence.get('vision_calls')}")
    print(f"vision_completions={model.vision_completions} text_completions={model.completions}")
    print(f"observation_fields={evidence.get('observation_fields')}")
    print(f"hidden_state_keys={evidence.get('hidden_state_keys')}")
    print(f"raw_response={evidence.get('raw_response')!r}")
    print(f"report_visible={evidence.get('report_visible')} gt={evidence.get('ground_truth_visible')}")

    ok = True
    if model.vision_completions < 1:
        print("VALIDATION B FAIL: vision_completions == 0")
        ok = False
    else:
        print("VALIDATION B: Observation.frame reached complete_with_vision")
    print("VALIDATION C: Task → Runner → Agent → Minecraft → Evaluator → Result")
    fields = set(evidence.get("observation_fields") or [])
    if fields != observation_field_names():
        print(f"VALIDATION D FAIL: observation fields {fields}")
        ok = False
    elif "target_truths" in (evidence.get("hidden_state_keys") or []):
        print("VALIDATION D: agent/evaluator boundary OK (GT in hidden_state only)")
    else:
        print("VALIDATION D: observation fields OK; hidden_state keys logged above")
    # Perception success is not a pipeline requirement.
    return 0 if ok else 1


def _frame_shape(frame: Any) -> str:
    shape = getattr(frame, "shape", None)
    return str(shape) if shape is not None else type(frame).__name__


if __name__ == "__main__":
    raise SystemExit(main())
