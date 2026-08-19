"""Tests for the historical Phase 2C D1 Presence lava pilot.

That scene is **not** D1 v2. Formal D1 v2 tests are
``test_d1_v2_lava.py`` / ``test_d1_v2_water.py``.

These tests cover the offline components of the D1 Presence
pipeline:

* :class:`PresenceReport` construction + JSON parsing edge cases
  (visible=true / visible=false / missing / wrong type /
  under ``report`` key);
* :class:`D1PresenceAgent` builds a target-specific prompt and
  parses the model's response;
* :class:`D1PresenceEvaluator` distinguishes
  ``output_protocol_error`` (bad schema) from
  ``perception_error`` (well-formed but wrong boolean) and
  reads the hidden ground truth from ``Task.ground_truth``;
* the Runner forwards ``ground_truth=task.ground_truth`` to the
  Evaluator (so the controlled-scene env can carry the hidden
  truth on the ``Task`` and the Agent never sees it);
* the existing inventory pilot is unchanged by Phase 2C.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from obsidianlink.agents.heuristic_model import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.perception import (
    PresenceReport,
    parse_presence_report,
)
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1InventoryPerceptionAgent,
    D1InventoryPerceptionEvaluator,
    D1_LAVA_PRESENCE,
    D1_OBSIDIAN_PRESENCE,
    D1_WATER_PRESENCE,
    D1PresenceAgent,
    D1PresenceEvaluator,
)


# ---------------------------------------------------------------------------
# PresenceReport
# ---------------------------------------------------------------------------


def test_presence_report_default_visible_is_none() -> None:
    assert PresenceReport().visible is None


def test_presence_report_well_formed_with_true() -> None:
    assert PresenceReport(visible=True).is_well_formed()


def test_presence_report_well_formed_with_false() -> None:
    assert PresenceReport(visible=False).is_well_formed()


def test_presence_report_not_well_formed_with_none() -> None:
    assert not PresenceReport(visible=None).is_well_formed()


# ---------------------------------------------------------------------------
# parse_presence_report
# ---------------------------------------------------------------------------


def test_parse_presence_report_extracts_top_level_true() -> None:
    assert parse_presence_report('{"visible": true}') == PresenceReport(visible=True)


def test_parse_presence_report_extracts_top_level_false() -> None:
    assert parse_presence_report('{"visible": false}') == PresenceReport(visible=False)


def test_parse_presence_report_tolerates_action_key() -> None:
    parsed = parse_presence_report('{"action": "WAIT", "visible": true}')
    assert parsed == PresenceReport(visible=True)


def test_parse_presence_report_tolerates_nested_under_report() -> None:
    parsed = parse_presence_report(
        '{"action": "WAIT", "report": {"visible": true}}'
    )
    assert parsed == PresenceReport(visible=True)


def test_parse_presence_report_returns_none_for_unparseable_json() -> None:
    assert parse_presence_report("not json at all") is None
    assert parse_presence_report("") is None
    assert parse_presence_report("[1, 2, 3]") is None


def test_parse_presence_report_returns_none_for_non_dict() -> None:
    assert parse_presence_report('"hello"') is None
    assert parse_presence_report("42") is None


def test_parse_presence_report_returns_well_formed_false_when_visible_missing() -> None:
    """A dict with no ``visible`` key returns PresenceReport(visible=None)
    (which is *not* well-formed). This is what the Evaluator maps to
    ``output_protocol_error``."""
    parsed = parse_presence_report('{"action": "WAIT"}')
    assert parsed == PresenceReport(visible=None)
    assert parsed is not None
    assert not parsed.is_well_formed()


def test_parse_presence_report_returns_well_formed_false_when_visible_wrong_type() -> None:
    """``"true"`` (string) or ``1`` (int) are NOT valid booleans and
    must be treated as protocol errors, not silently coerced."""
    assert parse_presence_report('{"visible": "true"}') == PresenceReport(visible=None)
    assert parse_presence_report('{"visible": 1}') == PresenceReport(visible=None)
    assert parse_presence_report('{"visible": []}') == PresenceReport(visible=None)


# ---------------------------------------------------------------------------
# D1PresenceAgent
# ---------------------------------------------------------------------------


class _StaticModel:
    """Model that returns a fixed response for every prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
        return self.response


def test_d1_presence_agent_prompt_mentions_target() -> None:
    """The agent's prompt MUST mention the target so the model knows
    what to look for. Different targets -> different prompts -> the
    cross-target comparison stays clean."""
    for target, task in (
        ("lava", D1_LAVA_PRESENCE),
        ("water", D1_WATER_PRESENCE),
        ("obsidian", D1_OBSIDIAN_PRESENCE),
    ):
        model = _StaticModel('{"visible": true}')
        agent = D1PresenceAgent(model=model, target_name=target)
        agent.act(Observation(frame=None, inventory={}, selected_item=None))
        prompt = model.calls[0]
        assert target.upper() in prompt
        assert "visible" in prompt


def test_d1_presence_agent_parses_model_visible_true() -> None:
    model = _StaticModel('{"visible": true}')
    agent = D1PresenceAgent(model=model, target_name="lava")
    action = agent.act(Observation(frame=None, inventory={}, selected_item=None))
    assert action.type is ActionType.WAIT
    assert agent.last_report is not None
    assert agent.last_report.visible is True


def test_d1_presence_agent_parses_model_visible_false() -> None:
    model = _StaticModel('{"visible": false}')
    agent = D1PresenceAgent(model=model, target_name="lava")
    agent.act(Observation())
    assert agent.last_report is not None
    assert agent.last_report.visible is False


def test_d1_presence_agent_stores_raw_response() -> None:
    model = _StaticModel('{"visible": true}')
    agent = D1PresenceAgent(model=model, target_name="lava")
    agent.act(Observation())
    assert agent.last_raw_response == '{"visible": true}'


def test_d1_presence_agent_handles_missing_visible_as_protocol_error_signal() -> None:
    """A model that emits valid JSON without ``visible`` should
    surface as ``last_report.visible = None``, which the Evaluator
    maps to ``output_protocol_error``."""
    model = _StaticModel('{"action": "WAIT"}')
    agent = D1PresenceAgent(model=model, target_name="lava")
    agent.act(Observation())
    assert agent.last_report is not None
    assert agent.last_report.visible is None


# ---------------------------------------------------------------------------
# D1PresenceEvaluator
# ---------------------------------------------------------------------------


_TASK = D1_LAVA_PRESENCE


def _eval(
    *,
    report: Any,
    ground_truth: Any = True,
    raw_response: str | None = None,
    steps: int = 2,
    model_calls: int = 2,
    elapsed_time: float = 0.1,
) -> Result:
    return D1PresenceEvaluator().evaluate(
        _TASK,
        steps=steps,
        model_calls=model_calls,
        invalid_actions=0,
        elapsed_time=elapsed_time,
        report=report,
        observation=None,
        raw_response=raw_response,
        ground_truth=ground_truth,
    )


def test_evaluator_success_on_matching_visible_true() -> None:
    result = _eval(report=PresenceReport(visible=True), ground_truth=True)
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["report_visible"] is True
    assert result.evidence["ground_truth_visible"] is True


def test_evaluator_success_on_matching_visible_false() -> None:
    result = _eval(report=PresenceReport(visible=False), ground_truth=False)
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_evaluator_perception_error_when_visible_disagrees() -> None:
    """The well-formed schema with a wrong boolean is a PERCEPTION
    error, not a protocol error. This is the Phase 2C contract."""
    result = _eval(report=PresenceReport(visible=False), ground_truth=True)
    assert result.success is False
    assert result.evidence["reason"] == "perception_error"


def test_evaluator_perception_error_on_inverse_mismatch() -> None:
    result = _eval(report=PresenceReport(visible=True), ground_truth=False)
    assert result.success is False
    assert result.evidence["reason"] == "perception_error"


def test_evaluator_output_protocol_error_when_report_is_none() -> None:
    """No report at all is an ``output_protocol_error``, NOT a
    perception error. The user explicitly asked for this split so
    that JSON / truncation / wrong-schema failures do not pollute
    the perception signal."""
    result = _eval(report=None, ground_truth=True, raw_response="not json")
    assert result.success is False
    assert result.evidence["reason"] == "output_protocol_error"
    assert result.evidence["raw_response"] == "not json"


def test_evaluator_output_protocol_error_when_visible_is_none() -> None:
    """A well-formed PresenceReport with ``visible=None`` is
    also a protocol error (the schema is broken)."""
    report = PresenceReport(visible=None)
    result = _eval(report=report, ground_truth=True, raw_response='{"action":"WAIT"}')
    assert result.success is False
    assert result.evidence["reason"] == "output_protocol_error"


def test_evaluator_output_protocol_error_when_wrong_type() -> None:
    """Passing a PerceptionReport (the wrong type) is a protocol
    error; the Evaluator only accepts PresenceReport for presence
    tasks."""
    from obsidianlink.benchmark.perception import PerceptionReport

    wrong_type = PerceptionReport(inventory={}, selected_item=None)
    result = _eval(report=wrong_type, ground_truth=True)
    assert result.success is False
    assert result.evidence["reason"] == "output_protocol_error"


def test_evaluator_fails_loudly_on_missing_ground_truth() -> None:
    """A presence task without a ground truth is a wiring bug.
    The Evaluator must NOT silently pass — it must mark the run
    as ``missing_ground_truth`` so the experiment scripts flag it."""
    result = _eval(
        report=PresenceReport(visible=True), ground_truth=None
    )
    assert result.success is False
    assert result.evidence["reason"] == "missing_ground_truth"


# ---------------------------------------------------------------------------
# Runner forwards Task.ground_truth to Evaluator
# ---------------------------------------------------------------------------


class _CapturingPresenceEvaluator(Evaluator):
    """Evaluator that records what ``ground_truth`` it received."""

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        report: Any = None,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        final_observation: Any = None,
        hidden_state: Any = None,
    ) -> Result:
        # Stash the ground truth for the test to inspect.
        self.received_truth: Any = ground_truth
        return Result(
            task_id=task.task_id,
            success=True,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
        )


class _ReportingModel:
    completions = 0

    def __init__(self, visible: bool) -> None:
        self._visible = visible

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        self.__class__.completions += 1
        return '{"visible": ' + ("true" if self._visible else "false") + "}"


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


def test_runner_forwards_task_ground_truth_to_evaluator() -> None:
    """The Runner MUST forward ``Task.ground_truth`` to the Evaluator,
    so the Evaluator can grade against the controlled-scene truth
    without it ever entering the agent-visible observation."""
    env = _StubEnv()
    model = _ReportingModel(visible=True)
    agent = D1PresenceAgent(model=model, target_name="lava")
    evaluator = _CapturingPresenceEvaluator()
    BenchmarkRunner().run(
        task=D1_LAVA_PRESENCE,  # ground_truth=True
        env=env,
        agent=agent,
        evaluator=evaluator,
    )
    assert evaluator.received_truth is True


def test_runner_d1_presence_success_path_on_stub_env() -> None:
    env = _StubEnv()
    model = _ReportingModel(visible=True)
    agent = D1PresenceAgent(model=model, target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_LAVA_PRESENCE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_runner_d1_presence_perception_error_path() -> None:
    """Model says ``visible=false`` but ground truth is ``True``
    (lava is in the world). The Evaluator must mark this as a
    *perception* error, NOT a protocol error."""
    env = _StubEnv()
    model = _ReportingModel(visible=False)  # wrong answer
    agent = D1PresenceAgent(model=model, target_name="lava")
    result = BenchmarkRunner().run(
        task=D1_LAVA_PRESENCE,
        env=env,
        agent=agent,
        evaluator=D1PresenceEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "perception_error"


# ---------------------------------------------------------------------------
# Inventory pilot is unchanged by Phase 2C
# ---------------------------------------------------------------------------


class _InventoryReportingModel:
    completions = 0

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        self.__class__.completions += 1
        return (
            '{"action": "WAIT", '
            '"report": {"inventory": {}, "selected_item": null}}'
        )


def test_inventory_pilot_still_works_after_phase_2c() -> None:
    """The Phase 2A / 2B inventory pilot must keep working. The
    only change the Runner / Evaluator saw was a new keyword
    argument ``ground_truth``, which defaults to ``None`` for the
    pilot (the pilot derives truth from the observation)."""
    from obsidianlink.tasks.diagnostic import D1_INVENTORY_PERCEPTION

    env = _StubEnv()
    model = _InventoryReportingModel()
    agent = D1InventoryPerceptionAgent(model=model)
    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=D1InventoryPerceptionEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_inventory_pilot_runner_forwards_none_ground_truth() -> None:
    """The pilot Task has ``ground_truth=None``; the runner must
    forward that to the Evaluator without trying to invent a truth."""
    from obsidianlink.tasks.diagnostic import D1_INVENTORY_PERCEPTION

    assert D1_INVENTORY_PERCEPTION.ground_truth is None

    captured: dict[str, Any] = {}

    class _PilotCapturingEvaluator(Evaluator):
        def evaluate(self, task, *, steps, model_calls, invalid_actions,
                     elapsed_time, report=None, observation=None,
                     raw_response=None, ground_truth=None,
                     final_observation=None, hidden_state=None) -> Result:
            captured["ground_truth"] = ground_truth
            return Result(
                task_id=task.task_id,
                success=True,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
            )

    env = _StubEnv()
    model = _InventoryReportingModel()
    agent = D1InventoryPerceptionAgent(model=model)
    BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=_PilotCapturingEvaluator(),
    )
    assert captured["ground_truth"] is None
