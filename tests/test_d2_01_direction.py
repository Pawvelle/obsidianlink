"""Offline tests for D2-01 Direction Grounding.

No Minecraft, no VLM. Covers:

* DirectionGroundingReport parsing (valid / invalid / malformed)
* grounding success, mismatch, protocol failure
* hidden GT never enters the agent prompt
* Runner + evaluator integration (WAIT only, max_steps=1)
* D1 presence / inventory pilots are unchanged
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.benchmark.perception import (
    DirectionGroundingReport,
    parse_direction_grounding_report,
)
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.d2_01_scene import (
    D2_01_CENTER_ENV_ID,
    D2_01_ENV_IDS,
    D2_01_LEFT_ENV_ID,
    D2_01_RESOLUTION,
    D2_01_RIGHT_ENV_ID,
    D2_01_SPAWN_YAWS,
    d2_01_scene_xml,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_PRESENCE_POSITIVE,
    D1_INVENTORY_PERCEPTION,
    D2_01_CENTER,
    D2_01_ENV_IDS_BY_CONDITION,
    D2_01_LEFT,
    D2_01_MAX_STEPS,
    D2_01_RIGHT,
    D2_01_TASKS,
    D2_01_WARMUP_STEPS,
    D2DirectionGroundingAgent,
    D2DirectionGroundingEvaluator,
)


# ---------------------------------------------------------------------------
# Tasks / scene
# ---------------------------------------------------------------------------


def test_d2_01_tasks_share_id_and_differ_only_in_gt() -> None:
    assert D2_01_LEFT.task_id == D2_01_CENTER.task_id == D2_01_RIGHT.task_id
    assert D2_01_LEFT.task_id == "d2_01_direction_grounding"
    assert D2_01_LEFT.ground_truth == "left"
    assert D2_01_CENTER.ground_truth == "center"
    assert D2_01_RIGHT.ground_truth == "right"
    assert D2_01_LEFT.goal == D2_01_CENTER.goal == D2_01_RIGHT.goal
    assert D2_01_LEFT.max_steps == D2_01_MAX_STEPS == 1


def test_d2_01_env_ids_are_distinct_from_d1() -> None:
    assert D2_01_ENV_IDS_BY_CONDITION == D2_01_ENV_IDS
    assert D2_01_ENV_IDS["left"] == D2_01_LEFT_ENV_ID
    assert D2_01_ENV_IDS["center"] == D2_01_CENTER_ENV_ID
    assert D2_01_ENV_IDS["right"] == D2_01_RIGHT_ENV_ID
    assert "MineRLD1LavaPositive-v0" not in D2_01_ENV_IDS.values()


def test_d2_01_warmup_and_resolution_match_d1_v2() -> None:
    assert D2_01_WARMUP_STEPS > 0
    assert D2_01_RESOLUTION == (640, 360)


def test_d2_01_spawn_yaws_put_target_left_center_right() -> None:
    """Positive yaw looks right, so the +Z lava appears on the left."""
    assert D2_01_SPAWN_YAWS["left"] > 0
    assert D2_01_SPAWN_YAWS["center"] == 0.0
    assert D2_01_SPAWN_YAWS["right"] < 0
    assert D2_01_SPAWN_YAWS["left"] == -D2_01_SPAWN_YAWS["right"]


def test_d2_01_xml_is_lava_positive_courtyard() -> None:
    xml = d2_01_scene_xml()
    assert "DrawCuboid" not in xml
    assert xml.count('type="lava"') == 9
    assert "water" not in xml
    assert 'type="obsidian"' in xml


def test_d1_and_phase1_tasks_unchanged_by_d2_01() -> None:
    assert D1_01_LAVA_PRESENCE_POSITIVE.max_steps == 1
    assert D1_INVENTORY_PERCEPTION.max_steps == 2
    assert D1_INVENTORY_PERCEPTION.ground_truth is None
    assert set(D2_01_TASKS) == {"left", "center", "right"}
    assert D2_01_TASKS["left"] is D2_01_LEFT


# ---------------------------------------------------------------------------
# DirectionGroundingReport
# ---------------------------------------------------------------------------


def test_direction_grounding_report_valid_labels() -> None:
    for label in ("left", "center", "right"):
        report = DirectionGroundingReport(target="lava", direction=label)
        assert report.is_well_formed()


def test_parse_direction_grounding_report_valid() -> None:
    assert parse_direction_grounding_report(
        '{"target": "lava", "direction": "left"}'
    ) == DirectionGroundingReport(target="lava", direction="left")
    assert parse_direction_grounding_report(
        '{"target": "lava", "direction": "CENTER"}'
    ) == DirectionGroundingReport(target="lava", direction="center")
    nested = parse_direction_grounding_report(
        '{"report": {"target": "lava", "direction": "Right"}}'
    )
    assert nested == DirectionGroundingReport(target="lava", direction="right")


def test_parse_direction_grounding_report_invalid_direction() -> None:
    bad = parse_direction_grounding_report(
        '{"target": "lava", "direction": "up"}'
    )
    assert bad == DirectionGroundingReport(target="lava", direction=None)
    assert bad is not None and not bad.is_well_formed()


def test_parse_direction_grounding_report_malformed_json() -> None:
    assert parse_direction_grounding_report("not json") is None
    assert parse_direction_grounding_report("") is None
    assert parse_direction_grounding_report("[1]") is None
    missing = parse_direction_grounding_report('{"target": "lava"}')
    assert missing == DirectionGroundingReport(target="lava", direction=None)
    assert missing is not None and not missing.is_well_formed()


# ---------------------------------------------------------------------------
# Agent — WAIT only, GT not in prompt
# ---------------------------------------------------------------------------


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
        return self.response


def test_d2_01_prompt_does_not_leak_ground_truth() -> None:
    model = _StaticModel('{"target": "lava", "direction": "left"}')
    agent = D2DirectionGroundingAgent(model=model)
    agent.act(Observation(frame=None, inventory={}, selected_item=None))
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "lava" in lowered
    assert "left" in lowered and "center" in lowered and "right" in lowered
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "spawn" not in lowered
    assert "35" not in prompt
    assert "MineRLD201" not in prompt
    assert "yaw" not in lowered
    assert "approach" not in lowered
    # Same prompt for every condition; episode GT is not injected.
    model_center = _StaticModel('{"target": "lava", "direction": "center"}')
    D2DirectionGroundingAgent(model=model_center).act(Observation())
    assert model_center.calls[0] == prompt


def test_d2_01_agent_always_waits_and_parses_report() -> None:
    model = _StaticModel(
        '{"target": "lava", "direction": "left", "action": "camera", "yaw": -15}'
    )
    agent = D2DirectionGroundingAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert action.yaw == 0.0
    assert agent.last_report == DirectionGroundingReport(
        target="lava", direction="left"
    )
    assert agent.last_raw_response == model.response
    assert agent.model_calls == 1


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _eval(
    *,
    report: Any,
    ground_truth: Any = "left",
    raw_response: str | None = None,
) -> Result:
    return D2DirectionGroundingEvaluator().evaluate(
        D2_01_LEFT,
        steps=1,
        model_calls=1,
        invalid_actions=0,
        elapsed_time=0.1,
        report=report,
        observation=None,
        raw_response=raw_response,
        ground_truth=ground_truth,
        hidden_state={"yaw": 35.0},
    )


def test_evaluator_grounding_success() -> None:
    result = _eval(
        report=DirectionGroundingReport(target="lava", direction="left"),
        ground_truth="left",
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["report_direction"] == "left"
    assert result.evidence["ground_truth_direction"] == "left"
    assert "orientation_error" not in result.evidence
    assert "yaw_error" not in result.evidence


def test_evaluator_grounding_mismatch() -> None:
    result = _eval(
        report=DirectionGroundingReport(target="lava", direction="right"),
        ground_truth="left",
    )
    assert result.success is False
    assert result.evidence["reason"] == "grounding_error"


def test_evaluator_protocol_failure() -> None:
    none_result = _eval(report=None, raw_response="not json")
    assert none_result.success is False
    assert none_result.evidence["reason"] == "output_protocol_error"
    assert none_result.evidence["raw_response"] == "not json"

    bad = _eval(report=DirectionGroundingReport(target="lava", direction=None))
    assert bad.success is False
    assert bad.evidence["reason"] == "output_protocol_error"


# ---------------------------------------------------------------------------
# Runner + stub env
# ---------------------------------------------------------------------------


class _StubEnv(Environment):
    def __init__(self) -> None:
        self.reset_called = 0
        self.close_called = 0
        self.steps = 0
        self.actions: list[Action] = []

    def reset(self) -> Observation:
        self.reset_called += 1
        self.steps = 0
        return Observation(frame=None, inventory={}, selected_item=None)

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        self.steps += 1
        return Observation(frame=None, inventory={}, selected_item=None)

    def close(self) -> None:
        self.close_called += 1


def test_runner_evaluator_integration_success_and_wait_only() -> None:
    env = _StubEnv()
    result = BenchmarkRunner().run(
        task=D2_01_LEFT,
        env=env,
        agent=D2DirectionGroundingAgent(
            model=_StaticModel('{"target": "lava", "direction": "left"}')
        ),
        evaluator=D2DirectionGroundingEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["ground_truth_direction"] == "left"
    assert env.close_called == 1
    assert env.steps == D2_01_MAX_STEPS == 1
    assert env.actions[0].type is ActionType.WAIT


def test_runner_grounding_error_when_direction_mismatches() -> None:
    env = _StubEnv()
    result = BenchmarkRunner().run(
        task=D2_01_RIGHT,
        env=env,
        agent=D2DirectionGroundingAgent(
            model=_StaticModel('{"target": "lava", "direction": "left"}')
        ),
        evaluator=D2DirectionGroundingEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "grounding_error"
    assert result.evidence["report_direction"] == "left"
    assert result.evidence["ground_truth_direction"] == "right"
