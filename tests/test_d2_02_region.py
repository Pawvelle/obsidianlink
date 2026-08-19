"""Offline tests for D2-02 Spatial Region Grounding.

No Minecraft, no VLM. Covers:

* SpatialRegionGroundingReport parsing (valid / invalid / malformed)
* grounding success, mismatch, protocol failure
* hidden GT never enters the agent prompt
* Runner + evaluator integration (WAIT only, max_steps=1)
* D2-01 / D1 existing behaviour is unchanged
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.benchmark.perception import (
    SpatialRegionGroundingReport,
    parse_spatial_region_grounding_report,
)
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.d2_02_scene import (
    D2_02_ENV_IDS,
    D2_02_PITCHES,
    D2_02_REGIONS,
    D2_02_RESOLUTION,
    D2_02_SPAWN_POSES,
    D2_02_YAWS,
    d2_02_region_from_norm,
    d2_02_scene_xml,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_PRESENCE_POSITIVE,
    D2_01_LEFT,
    D2_01_MAX_STEPS,
    D2_02_MAX_STEPS,
    D2_02_TASKS,
    D2_02_WARMUP_STEPS,
    D2SpatialRegionGroundingAgent,
    D2SpatialRegionGroundingEvaluator,
)


def test_d2_02_nine_tasks_share_id_and_differ_only_in_gt() -> None:
    assert len(D2_02_REGIONS) == 9
    assert set(D2_02_TASKS) == set(D2_02_REGIONS)
    ids = {task.task_id for task in D2_02_TASKS.values()}
    assert ids == {"d2_02_spatial_region_grounding"}
    for region, task in D2_02_TASKS.items():
        assert task.ground_truth == region
        assert task.max_steps == D2_02_MAX_STEPS == 1
        assert task.goal == D2_02_TASKS["center"].goal


def test_d2_02_env_ids_and_poses_are_distinct() -> None:
    assert len(set(D2_02_ENV_IDS.values())) == 9
    assert "MineRLD201Left-v0" not in D2_02_ENV_IDS.values()
    assert D2_02_ENV_IDS["center"] == "MineRLD202Center-v0"
    assert D2_02_ENV_IDS["upper_left"] == "MineRLD202UpperLeft-v0"
    assert D2_02_SPAWN_POSES["center"] == (0.0, D2_02_PITCHES["center"])
    assert D2_02_SPAWN_POSES["upper_left"][0] == D2_02_YAWS["left"]
    assert D2_02_SPAWN_POSES["upper_left"][1] == D2_02_PITCHES["upper"]
    assert D2_02_SPAWN_POSES["lower_right"][0] == D2_02_YAWS["right"]
    assert D2_02_PITCHES["upper"] > D2_02_PITCHES["center"] > D2_02_PITCHES["lower"]


def test_d2_02_warmup_resolution_and_xml_match_d1() -> None:
    assert D2_02_WARMUP_STEPS > 0
    assert D2_02_RESOLUTION == (640, 360)
    xml = d2_02_scene_xml()
    assert xml.count('type="lava"') == 9
    assert 'type="obsidian"' in xml


def test_d2_02_centroid_bin_helper() -> None:
    assert d2_02_region_from_norm(0.2, 0.2) == "upper_left"
    assert d2_02_region_from_norm(0.5, 0.5) == "center"
    assert d2_02_region_from_norm(0.8, 0.8) == "lower_right"


def test_d2_01_and_d1_unchanged_by_d2_02() -> None:
    assert D2_01_LEFT.task_id == "d2_01_direction_grounding"
    assert D2_01_LEFT.max_steps == D2_01_MAX_STEPS == 1
    assert D1_01_LAVA_PRESENCE_POSITIVE.max_steps == 1


def test_spatial_region_report_valid_labels() -> None:
    for region in D2_02_REGIONS:
        report = SpatialRegionGroundingReport(target="lava", region=region)
        assert report.is_well_formed()


def test_parse_spatial_region_report_valid() -> None:
    assert parse_spatial_region_grounding_report(
        '{"target": "lava", "region": "upper_left"}'
    ) == SpatialRegionGroundingReport(target="lava", region="upper_left")
    assert parse_spatial_region_grounding_report(
        '{"target": "lava", "region": "CENTER"}'
    ) == SpatialRegionGroundingReport(target="lava", region="center")
    nested = parse_spatial_region_grounding_report(
        '{"report": {"target": "lava", "region": "Center_Right"}}'
    )
    assert nested == SpatialRegionGroundingReport(
        target="lava", region="center_right"
    )


def test_parse_spatial_region_report_invalid_region() -> None:
    bad = parse_spatial_region_grounding_report(
        '{"target": "lava", "region": "center_center"}'
    )
    assert bad == SpatialRegionGroundingReport(target="lava", region=None)
    assert bad is not None and not bad.is_well_formed()
    also = parse_spatial_region_grounding_report(
        '{"target": "lava", "region": "middle"}'
    )
    assert also is not None and not also.is_well_formed()


def test_parse_spatial_region_report_malformed_json() -> None:
    assert parse_spatial_region_grounding_report("not json") is None
    assert parse_spatial_region_grounding_report("") is None
    assert parse_spatial_region_grounding_report("[1]") is None
    missing = parse_spatial_region_grounding_report('{"target": "lava"}')
    assert missing == SpatialRegionGroundingReport(target="lava", region=None)
    assert missing is not None and not missing.is_well_formed()


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
        return self.response


def test_d2_02_prompt_does_not_leak_ground_truth() -> None:
    model = _StaticModel('{"target": "lava", "region": "upper_left"}')
    agent = D2SpatialRegionGroundingAgent(model=model)
    agent.act(Observation(frame=None, inventory={}, selected_item=None))
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "lava" in lowered
    for region in D2_02_REGIONS:
        assert region in lowered
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "spawn" not in lowered
    assert "45" not in prompt
    assert "35" not in prompt
    assert "MineRLD202" not in prompt
    assert "yaw" not in lowered
    assert "pitch" not in lowered
    other = _StaticModel('{"target": "lava", "region": "lower_right"}')
    D2SpatialRegionGroundingAgent(model=other).act(Observation())
    assert other.calls[0] == prompt


def test_d2_02_agent_always_waits_and_parses_report() -> None:
    model = _StaticModel(
        '{"target": "lava", "region": "center", "action": "camera", "yaw": -15}'
    )
    agent = D2SpatialRegionGroundingAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert action.yaw == 0.0
    assert agent.last_report == SpatialRegionGroundingReport(
        target="lava", region="center"
    )
    assert agent.last_raw_response == model.response
    assert agent.model_calls == 1


def _eval(
    *,
    report: Any,
    ground_truth: Any = "upper_left",
    raw_response: str | None = None,
) -> Result:
    return D2SpatialRegionGroundingEvaluator().evaluate(
        D2_02_TASKS["upper_left"],
        steps=1,
        model_calls=1,
        invalid_actions=0,
        elapsed_time=0.1,
        report=report,
        observation=None,
        raw_response=raw_response,
        ground_truth=ground_truth,
        hidden_state={"yaw": 35.0, "pitch": 45.0},
    )


def test_evaluator_region_success() -> None:
    result = _eval(
        report=SpatialRegionGroundingReport(target="lava", region="upper_left"),
        ground_truth="upper_left",
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["report_region"] == "upper_left"
    assert result.evidence["ground_truth_region"] == "upper_left"
    assert "orientation_error" not in result.evidence


def test_evaluator_region_mismatch() -> None:
    result = _eval(
        report=SpatialRegionGroundingReport(target="lava", region="lower_right"),
        ground_truth="upper_left",
    )
    assert result.success is False
    assert result.evidence["reason"] == "grounding_error"


def test_evaluator_region_protocol_failure() -> None:
    none_result = _eval(report=None, raw_response="not json")
    assert none_result.success is False
    assert none_result.evidence["reason"] == "output_protocol_error"
    assert none_result.evidence["raw_response"] == "not json"
    bad = _eval(
        report=SpatialRegionGroundingReport(target="lava", region=None)
    )
    assert bad.success is False
    assert bad.evidence["reason"] == "output_protocol_error"


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


def test_runner_region_integration_success_and_wait_only() -> None:
    env = _StubEnv()
    result = BenchmarkRunner().run(
        task=D2_02_TASKS["center_right"],
        env=env,
        agent=D2SpatialRegionGroundingAgent(
            model=_StaticModel(
                '{"target": "lava", "region": "center_right"}'
            )
        ),
        evaluator=D2SpatialRegionGroundingEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["ground_truth_region"] == "center_right"
    assert env.close_called == 1
    assert env.steps == D2_02_MAX_STEPS == 1
    assert env.actions[0].type is ActionType.WAIT


def test_runner_region_grounding_error_on_mismatch() -> None:
    env = _StubEnv()
    result = BenchmarkRunner().run(
        task=D2_02_TASKS["lower_left"],
        env=env,
        agent=D2SpatialRegionGroundingAgent(
            model=_StaticModel('{"target": "lava", "region": "upper_right"}')
        ),
        evaluator=D2SpatialRegionGroundingEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "grounding_error"
    assert result.evidence["report_region"] == "upper_right"
    assert result.evidence["ground_truth_region"] == "lower_left"
