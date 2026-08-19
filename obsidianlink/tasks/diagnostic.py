"""D1 Lava Presence — the Phase 2 representative diagnostic.

D1 asks: is lava visible in the current RGB frame?

* Task does not contain a solver.
* Hidden ground truth is scene-defined (we drew lava or we did not).
  It is not read from ObservationFromGrid.
* The Agent is a ReactiveAgent with a D1 prompt. The RGB frame is
  passed through call_model; text-only fallback is evaluation_error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from obsidianlink.agents.reactive import extract_json_object
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import (
    AGENT_FAILURE,
    EVALUATOR_FAILURE,
    Result,
)
from obsidianlink.benchmark.task import Task
from obsidianlink.env.environment import Observation, observation_field_names
from obsidianlink.env.scene import NEGATIVE_ENV_ID, POSITIVE_ENV_ID

_GOAL = (
    "Look at the Minecraft frame and report whether LAVA is "
    "visible anywhere in it. Respond with the JSON object "
    '{"visible": true} or {"visible": false}.'
)

D1_LAVA_POSITIVE = Task(
    task_id="d1_01_lava_presence",
    goal=_GOAL,
    max_steps=1,
    initial_condition="Controlled obsidian courtyard; 3x3 lava patch in view.",
    allowed_actions=("wait",),
    evaluation_condition="parsed visible boolean matches hidden scene label",
    ground_truth=True,
)

D1_LAVA_NEGATIVE = Task(
    task_id="d1_01_lava_presence",
    goal=_GOAL,
    max_steps=1,
    initial_condition="Same courtyard; floor patch is obsidian, no lava.",
    allowed_actions=("wait",),
    evaluation_condition="parsed visible boolean matches hidden scene label",
    ground_truth=False,
)

D1_ENV_IDS = {
    "positive": POSITIVE_ENV_ID,
    "negative": NEGATIVE_ENV_ID,
}

D1_TASKS = {
    "positive": D1_LAVA_POSITIVE,
    "negative": D1_LAVA_NEGATIVE,
}


def d1_prompt(_observation: Observation) -> str:
    return (
        "You are looking at a first-person Minecraft image.\n"
        "Question: is LAVA visible anywhere in the image?\n"
        "Respond with a JSON object and nothing else:\n"
        '{"visible": true} or {"visible": false}.\n'
        "Do not include any other keys. Do not use markdown fences."
    )


@dataclass(frozen=True)
class PresenceReport:
    visible: bool | None = None

    def is_well_formed(self) -> bool:
        return isinstance(self.visible, bool)


def parse_presence_report(response: str) -> PresenceReport | None:
    data = extract_json_object(response)
    if data is None:
        return None
    raw = data.get("visible")
    if raw is None and isinstance(data.get("report"), dict):
        raw = data["report"].get("visible")
    if isinstance(raw, bool):
        return PresenceReport(visible=raw)
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        return PresenceReport(visible=raw.strip().lower() == "true")
    return PresenceReport(visible=None)


class D1LavaEvaluator(Evaluator):
    """Grade D1 from scene-defined GT, never from model self-report of success."""

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        hidden_state: Any = None,
        used_vision: bool | None = None,
        fallback_reason: str | None = None,
        vision_calls: int = 0,
    ) -> Result:
        report = parse_presence_report(raw_response) if raw_response else None
        leaked = _leaked_evaluator_fields(observation)
        evidence: dict[str, Any] = {
            "report_visible": None if report is None else report.visible,
            "ground_truth_visible": ground_truth,
            "raw_response": raw_response,
            "used_vision": used_vision,
            "fallback_reason": fallback_reason,
            "vision_calls": vision_calls,
            "observation_fields": sorted(observation_field_names()),
            "hidden_state_keys": (
                sorted(hidden_state.keys()) if isinstance(hidden_state, dict) else []
            ),
        }

        def _result(success: bool, failure_class: str | None, reason: str) -> Result:
            evidence["reason"] = reason
            if failure_class is not None:
                evidence["failure_class"] = failure_class
            return Result(
                task_id=task.task_id,
                success=success,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if leaked:
            evidence["leaked_fields"] = leaked
            return _result(False, EVALUATOR_FAILURE, "evaluation_error")

        if used_vision is not True:
            evidence["vision_fallback"] = True
            return _result(False, EVALUATOR_FAILURE, "evaluation_error")

        if ground_truth is None:
            return _result(False, EVALUATOR_FAILURE, "evaluation_error")

        if report is None or not report.is_well_formed():
            return _result(False, AGENT_FAILURE, "output_protocol_error")

        if report.visible == bool(ground_truth):
            return _result(True, None, "ok")
        return _result(False, AGENT_FAILURE, "perception_error")


def _leaked_evaluator_fields(observation: Any) -> list[str]:
    if observation is None:
        return []
    banned = (
        "ground_truth",
        "target_truths",
        "hidden_state",
        "success",
        "l1_grid",
        "xpos",
        "ypos",
        "zpos",
        "yaw",
        "pitch",
    )
    leaked = [name for name in banned if hasattr(observation, name)]
    extra = [
        name
        for name in getattr(observation, "__dict__", {})
        if name not in observation_field_names()
    ]
    return sorted(set(leaked + extra))


__all__ = [
    "D1_ENV_IDS",
    "D1_LAVA_NEGATIVE",
    "D1_LAVA_POSITIVE",
    "D1_TASKS",
    "D1LavaEvaluator",
    "PresenceReport",
    "d1_prompt",
    "parse_presence_report",
]
