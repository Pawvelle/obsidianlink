"""Offline tests for D1 v2 / D1-01 Lava Presence (640×360, live-verified).

No Minecraft, no VLM. Covers:

* D1-01 tasks are single-step with hidden bool ground truth
* the Phase 2C lava task is unchanged (historical pilot)
* scene XML: positive draws lava, negative does not
* hidden truth is not in the Agent prompt
* Runner takes exactly one step
* ControlledSceneEnv warmup is env-side only
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_NEGATIVE_ENV_ID,
    D1_V2_POSITIVE_ENV_ID,
    D1_V2_RESOLUTION,
    d1_v2_lava_scene_xml,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_ENV_IDS,
    D1_01_LAVA_PRESENCE_NEGATIVE,
    D1_01_LAVA_PRESENCE_POSITIVE,
    D1_01_WARMUP_STEPS,
    D1_INVENTORY_PERCEPTION,
    D1_LAVA_PRESENCE,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_d1_01_tasks_are_single_step_binary() -> None:
    assert D1_01_LAVA_PRESENCE_POSITIVE.max_steps == 1
    assert D1_01_LAVA_PRESENCE_NEGATIVE.max_steps == 1
    assert D1_01_LAVA_PRESENCE_POSITIVE.ground_truth is True
    assert D1_01_LAVA_PRESENCE_NEGATIVE.ground_truth is False
    assert (
        D1_01_LAVA_PRESENCE_POSITIVE.task_id
        == D1_01_LAVA_PRESENCE_NEGATIVE.task_id
        == "d1_01_lava_presence"
    )
    assert D1_01_LAVA_PRESENCE_POSITIVE.goal == D1_01_LAVA_PRESENCE_NEGATIVE.goal


def test_phase_2c_lava_pilot_task_is_unchanged() -> None:
    """Old lava presence stays a 2-step pilot; do not silently migrate it."""
    assert D1_LAVA_PRESENCE.task_id == "d1_lava_presence"
    assert D1_LAVA_PRESENCE.max_steps == 2
    assert D1_LAVA_PRESENCE.ground_truth is True


def test_inventory_pilot_task_is_unchanged() -> None:
    assert D1_INVENTORY_PERCEPTION.task_id == "d1_inventory_perception"
    assert D1_INVENTORY_PERCEPTION.max_steps == 2
    assert D1_INVENTORY_PERCEPTION.ground_truth is None


def test_d1_01_env_ids_are_distinct_from_pilot() -> None:
    assert D1_01_LAVA_ENV_IDS["positive"] == D1_V2_POSITIVE_ENV_ID
    assert D1_01_LAVA_ENV_IDS["negative"] == D1_V2_NEGATIVE_ENV_ID
    assert "MineRLControlledLava-v0" not in D1_01_LAVA_ENV_IDS.values()


def test_d1_01_warmup_is_positive() -> None:
    assert D1_01_WARMUP_STEPS > 0


def test_d1_01_resolution_is_not_treechop_64() -> None:
    """64×64 made the HUD look like lava. D1-01 uses MineRL 640×360."""
    assert D1_V2_RESOLUTION == (640, 360)


# ---------------------------------------------------------------------------
# Scene XML (no Minecraft)
# ---------------------------------------------------------------------------


def test_d1_v2_positive_xml_draws_lava_blocks() -> None:
    xml = d1_v2_lava_scene_xml(lava_present=True)
    assert "DrawCuboid" not in xml
    assert "DrawBlock" in xml
    assert 'type="lava"' in xml
    assert xml.count('type="lava"') == 9
    assert 'type="obsidian"' in xml
    assert 'type="stone"' not in xml


def test_d1_v2_negative_xml_has_no_lava() -> None:
    xml = d1_v2_lava_scene_xml(lava_present=False)
    assert "lava" not in xml
    assert "DrawCuboid" not in xml
    assert "DrawBlock" in xml
    assert 'type="obsidian"' in xml
    assert 'type="stone"' not in xml


def test_d1_v2_pos_and_neg_xml_differ_only_in_patch_type() -> None:
    pos = d1_v2_lava_scene_xml(lava_present=True)
    neg = d1_v2_lava_scene_xml(lava_present=False)
    assert pos.replace('type="lava"', 'type="obsidian"') == neg


# ---------------------------------------------------------------------------
# Prompt must not contain hidden truth
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


def test_d1_01_prompt_does_not_leak_ground_truth() -> None:
    model = _StaticModel('{"visible": false}')
    agent = D1PresenceAgent(model=model, target_name="lava")
    agent.act(Observation(frame=None, inventory={}, selected_item=None))
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "target_present" not in lowered
    assert "true" in lowered and "false" in lowered
    assert "positive" not in lowered
    assert "negative" not in lowered


# ---------------------------------------------------------------------------
# Runner + evaluator, max_steps=1
# ---------------------------------------------------------------------------


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


def test_runner_d1_01_takes_exactly_one_step() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(True), target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_01_LAVA_PRESENCE_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert env.steps == 1
    assert result.steps == 1
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert env.close_called == 1


def test_runner_d1_01_negative_success() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(False), target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_01_LAVA_PRESENCE_NEGATIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["ground_truth_visible"] is False


def test_runner_d1_01_perception_error_on_negative() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_ReportingModel(True), target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_01_LAVA_PRESENCE_NEGATIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "perception_error"


def test_runner_d1_01_output_protocol_error() -> None:
    env = _StubEnv()
    agent = D1PresenceAgent(model=_StaticModel("not json"), target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_01_LAVA_PRESENCE_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "output_protocol_error"


# ---------------------------------------------------------------------------
# Warmup is env-side, default 0 preserves the pilot
# ---------------------------------------------------------------------------


class _FakeMineRL:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.n_reset = 0
        self.n_step = 0
        self.closed = False

    def reset(self) -> Observation:
        self.n_reset += 1
        return Observation(frame="reset-frame", inventory={}, selected_item=None)

    def step(self, action: Action) -> Observation:
        del action
        self.n_step += 1
        return Observation(
            frame=f"warmup-{self.n_step}", inventory={}, selected_item=None
        )

    def close(self) -> None:
        self.closed = True


def test_controlled_scene_env_default_warmup_is_zero(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env._ensure_specs_registered",
        lambda: None,
    )
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env.MineRLEnvironment",
        _FakeMineRL,
    )
    env = ControlledSceneEnv(env_id="MineRLControlledLava-v0")
    assert env.warmup_steps == 0
    obs = env.reset()
    assert env._env.n_reset == 1
    assert env._env.n_step == 0
    assert obs.frame == "reset-frame"


def test_controlled_scene_env_warmup_skips_wait_ticks(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env._ensure_specs_registered",
        lambda: None,
    )
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env.MineRLEnvironment",
        _FakeMineRL,
    )
    env = ControlledSceneEnv(
        env_id="MineRLD1LavaPositive-v0",
        warmup_steps=5,
    )
    obs = env.reset()
    assert env._env.n_reset == 1
    assert env._env.n_step == 5
    assert obs.frame == "warmup-5"


def test_controlled_scene_env_hidden_truth_not_on_observation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env._ensure_specs_registered",
        lambda: None,
    )
    monkeypatch.setattr(
        "obsidianlink.env.controlled_scene_env.MineRLEnvironment",
        _FakeMineRL,
    )
    env = ControlledSceneEnv(env_id="MineRLD1LavaNegative-v0")
    obs = env.reset()
    assert env.target_truths == {"lava": False}
    assert not hasattr(obs, "target_truths")
    assert getattr(obs, "ground_truth", None) is None
