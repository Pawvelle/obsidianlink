"""Offline tests for D1-02 Water Presence (640×360, live-verified).

No Minecraft, no VLM. Mirrors the D1-01 lava checks: single-step
binary tasks, water XML contract, prompt does not leak truth.
"""

from __future__ import annotations

from typing import List

from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action
from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_WATER_NEGATIVE_ENV_ID,
    D1_V2_WATER_POSITIVE_ENV_ID,
    d1_v2_water_scene_xml,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_02_WATER_ENV_IDS,
    D1_02_WATER_PRESENCE_NEGATIVE,
    D1_02_WATER_PRESENCE_POSITIVE,
    D1_02_WARMUP_STEPS,
    D1_WATER_PRESENCE,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


def test_d1_02_tasks_are_single_step_binary() -> None:
    assert D1_02_WATER_PRESENCE_POSITIVE.max_steps == 1
    assert D1_02_WATER_PRESENCE_NEGATIVE.max_steps == 1
    assert D1_02_WATER_PRESENCE_POSITIVE.ground_truth is True
    assert D1_02_WATER_PRESENCE_NEGATIVE.ground_truth is False
    assert (
        D1_02_WATER_PRESENCE_POSITIVE.task_id
        == D1_02_WATER_PRESENCE_NEGATIVE.task_id
        == "d1_02_water_presence"
    )
    assert D1_02_WATER_PRESENCE_POSITIVE.goal == D1_02_WATER_PRESENCE_NEGATIVE.goal


def test_phase_2c_water_pilot_task_is_unchanged() -> None:
    assert D1_WATER_PRESENCE.task_id == "d1_water_presence"
    assert D1_WATER_PRESENCE.max_steps == 2
    assert D1_WATER_PRESENCE.ground_truth is True


def test_d1_02_env_ids_are_distinct() -> None:
    assert D1_02_WATER_ENV_IDS["positive"] == D1_V2_WATER_POSITIVE_ENV_ID
    assert D1_02_WATER_ENV_IDS["negative"] == D1_V2_WATER_NEGATIVE_ENV_ID
    assert "MineRLControlledWater-v0" not in D1_02_WATER_ENV_IDS.values()


def test_d1_02_warmup_matches_lava() -> None:
    assert D1_02_WARMUP_STEPS > 0


def test_d1_02_draw_xml_has_no_water() -> None:
    xml = d1_v2_water_scene_xml()
    assert "DrawCuboid" not in xml
    assert "water" not in xml
    assert 'type="obsidian"' in xml
    assert 'type="lava"' not in xml


def test_d1_02_setup_actions_only_on_positive() -> None:
    from obsidianlink.tasks.diagnostic import d1_02_setup_actions

    assert d1_02_setup_actions("negative") == ()
    pos = d1_02_setup_actions("positive")
    assert len(pos) > 0
    assert any(a.type.value == "use" for a in pos)


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
        return self.response


def test_d1_02_prompt_does_not_leak_ground_truth() -> None:
    model = _StaticModel('{"visible": false}')
    agent = D1PresenceAgent(model=model, target_name="water")
    agent.act(Observation(frame=None, inventory={}, selected_item=None))
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "target_present" not in lowered
    assert "water" in lowered
    assert "positive" not in lowered
    assert "negative" not in lowered


class _StubEnv(Environment):
    def __init__(self) -> None:
        self.reset_called = 0
        self.close_called = 0
        self.steps = 0

    def reset(self) -> Observation:
        self.reset_called += 1
        self.steps = 0
        return Observation(frame=None, inventory={}, selected_item=None)

    def step(self, action: Action) -> Observation:
        del action
        self.steps += 1
        return Observation(frame=None, inventory={}, selected_item=None)

    def close(self) -> None:
        self.close_called += 1


class _ReportingModel:
    completions = 0

    def __init__(self, visible: bool) -> None:
        self._visible = visible

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        self.__class__.completions += 1
        return '{"visible": ' + ("true" if self._visible else "false") + "}"


def test_runner_d1_02_positive_success() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(True), target_name="water")
    result = BenchmarkRunner().run(
        task=D1_02_WATER_PRESENCE_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert env.steps == 1
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_runner_d1_02_negative_success() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(False), target_name="water")
    result = BenchmarkRunner().run(
        task=D1_02_WATER_PRESENCE_NEGATIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is True
    assert result.evidence["ground_truth_visible"] is False


def test_runner_d1_02_perception_error() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(True), target_name="water")
    result = BenchmarkRunner().run(
        task=D1_02_WATER_PRESENCE_NEGATIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "perception_error"
