"""Tests for Phase 2A — D1 Inventory & Selected-Item Perception.

These tests exercise the full Benchmark vertical slice offline (no
MineRL, no Java):

* :class:`PerceptionReport` construction + JSON parsing edge cases.
* :class:`ReactiveAgent` extracts the optional ``report`` field from
  the model response into ``last_report``.
* :class:`D1InventoryPerceptionEvaluator` grades a report against an
  agent-visible observation (match, mismatch, missing report, malformed
  report).
* :class:`D1InventoryPerceptionAgent` derives a report from the
  observation and emits a WAIT action.
* :class:`BenchmarkRunner` drives the full chain end-to-end on a stub
  environment — both the success and failure paths.

A live MineRL smoke for D1 lives in ``obsidianlink.main`` behind
``OBSIDIANLINK_PHASE=2a``.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from obsidianlink.agents.heuristic_model import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.perception import (
    PerceptionReport,
    parse_perception_report,
)
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_INVENTORY_PERCEPTION,
    D1InventoryPerceptionAgent,
    D1InventoryPerceptionEvaluator,
    D1InventoryPerceptionModel,
)


# ---------------------------------------------------------------------------
# PerceptionReport
# ---------------------------------------------------------------------------


def test_perception_report_defaults() -> None:
    report = PerceptionReport()
    assert report.inventory is None
    assert report.selected_item is None


def test_perception_report_is_well_formed_with_empty_dict() -> None:
    assert PerceptionReport(inventory={}, selected_item=None).is_well_formed()


def test_perception_report_is_well_formed_with_real_dict() -> None:
    report = PerceptionReport(
        inventory={"dirt": 4, "oak_log": 2},
        selected_item="dirt",
    )
    assert report.is_well_formed()


def test_perception_report_is_not_well_formed_without_dict() -> None:
    assert not PerceptionReport(inventory=None).is_well_formed()
    assert not PerceptionReport(inventory=["dirt", 4]).is_well_formed()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_perception_report
# ---------------------------------------------------------------------------


def test_parse_perception_report_extracts_full_payload() -> None:
    response = (
        '{"action": "WAIT", '
        '"report": {"inventory": {"dirt": 4, "oak_log": 2}, '
        '"selected_item": "dirt"}}'
    )
    parsed = parse_perception_report(response)
    assert parsed is not None
    assert parsed.inventory == {"dirt": 4, "oak_log": 2}
    assert parsed.selected_item == "dirt"


def test_parse_perception_report_handles_empty_inventory() -> None:
    parsed = parse_perception_report(
        '{"action": "WAIT", "report": {"inventory": {}, "selected_item": null}}'
    )
    assert parsed is not None
    assert parsed.inventory == {}
    assert parsed.selected_item is None


def test_parse_perception_report_missing_report_key() -> None:
    assert parse_perception_report('{"action": "WAIT"}') is None


def test_parse_perception_report_report_not_dict() -> None:
    assert parse_perception_report('{"action": "WAIT", "report": []}') is None
    assert parse_perception_report('{"action": "WAIT", "report": "dirt"}') is None


def test_parse_perception_report_invalid_json() -> None:
    assert parse_perception_report("not json at all") is None
    assert parse_perception_report("") is None
    assert parse_perception_report("[1, 2, 3]") is None


def test_parse_perception_report_coerces_quantity_types() -> None:
    """Quantities that come in as strings / floats should be coerced to int.

    Garbage quantities are dropped (matches the rest of the project:
    "garbage in -> noise out" for diagnostic signals).
    """
    parsed = parse_perception_report(
        '{"report": {"inventory": {"dirt": "4", "oak_log": 2.7, "bad": "x"}, '
        '"selected_item": "dirt"}}'
    )
    assert parsed is not None
    assert parsed.inventory == {"dirt": 4, "oak_log": 2}


# ---------------------------------------------------------------------------
# ReactiveAgent.last_report
# ---------------------------------------------------------------------------


class _RecordingModel:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_reactive_agent_extracts_report_from_model_response() -> None:
    model = _RecordingModel(
        [r'{"action": "WAIT", "report": {"inventory": {"dirt": 4}, "selected_item": "dirt"}}']
    )
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation(inventory={"dirt": 4}, selected_item="dirt"))
    assert action.type is ActionType.WAIT
    assert agent.last_report is not None
    assert agent.last_report.inventory == {"dirt": 4}
    assert agent.last_report.selected_item == "dirt"


def test_reactive_agent_last_report_is_none_when_missing() -> None:
    model = _RecordingModel([r'{"action": "WAIT"}'])
    agent = ReactiveAgent(model=model)
    agent.act(Observation())
    assert agent.last_report is None


def test_reactive_agent_last_report_is_none_on_garbage() -> None:
    model = _RecordingModel(["not json at all"])
    agent = ReactiveAgent(model=model)
    agent.act(Observation())
    assert agent.last_report is None


def test_reactive_agent_default_starts_with_no_last_report() -> None:
    agent = ReactiveAgent(HeuristicModelClient())
    assert agent.last_report is None


# ---------------------------------------------------------------------------
# D1InventoryPerceptionEvaluator
# ---------------------------------------------------------------------------


_TASK = D1_INVENTORY_PERCEPTION


def _eval(
    *,
    report: Any,
    observation: Any,
    steps: int = 2,
    model_calls: int = 2,
    invalid_actions: int = 0,
    elapsed_time: float = 0.1,
) -> Result:
    return D1InventoryPerceptionEvaluator().evaluate(
        _TASK,
        steps=steps,
        model_calls=model_calls,
        invalid_actions=invalid_actions,
        elapsed_time=elapsed_time,
        report=report,
        observation=observation,
    )


def test_evaluator_success_when_report_matches_observation() -> None:
    obs = Observation(inventory={"dirt": 4, "oak_log": 2}, selected_item="dirt")
    report = PerceptionReport(inventory={"dirt": 4, "oak_log": 2}, selected_item="dirt")
    result = _eval(report=report, observation=obs)
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["inventory_match"] is True
    assert result.evidence["selected_match"] is True


def test_evaluator_success_on_empty_inventory_and_no_selected() -> None:
    obs = Observation(inventory={}, selected_item=None)
    report = PerceptionReport(inventory={}, selected_item=None)
    result = _eval(report=report, observation=obs)
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_evaluator_fails_when_inventory_mismatches() -> None:
    obs = Observation(inventory={"dirt": 4}, selected_item="dirt")
    report = PerceptionReport(inventory={"dirt": 3}, selected_item="dirt")
    result = _eval(report=report, observation=obs)
    assert result.success is False
    assert result.evidence["reason"] == "inventory_mismatch"
    assert result.evidence["inventory_match"] is False
    assert result.evidence["selected_match"] is True


def test_evaluator_fails_when_selected_item_mismatches() -> None:
    obs = Observation(inventory={"dirt": 4}, selected_item="dirt")
    report = PerceptionReport(inventory={"dirt": 4}, selected_item="oak_log")
    result = _eval(report=report, observation=obs)
    assert result.success is False
    assert result.evidence["reason"] == "selected_item_mismatch"


def test_evaluator_fails_when_no_report_emitted() -> None:
    obs = Observation(inventory={"dirt": 4}, selected_item="dirt")
    result = _eval(report=None, observation=obs)
    assert result.success is False
    assert result.evidence["reason"] == "no_report_emitted"
    assert result.evidence["report"] is None


def test_evaluator_fails_when_report_malformed() -> None:
    """A PerceptionReport whose ``inventory`` isn't a dict is malformed."""
    obs = Observation(inventory={"dirt": 4}, selected_item="dirt")
    report = PerceptionReport(inventory=None, selected_item="dirt")  # type: ignore[arg-type]
    result = _eval(report=report, observation=obs)
    assert result.success is False
    assert result.evidence["reason"] == "report_malformed"


def test_evaluator_treats_missing_observation_as_empty_truth() -> None:
    """When the observation is missing the D1 ground truth is the empty
    state. A report of ``inventory={}, selected_item=None`` therefore
    matches it. (A non-empty report would still fail, see
    ``test_evaluator_fails_when_inventory_mismatches``.)
    """
    report = PerceptionReport(inventory={}, selected_item=None)
    result = _eval(report=report, observation=None)
    assert result.evidence["ground_truth_inv"] == {}
    assert result.evidence["ground_truth_sel"] is None
    assert result.success is True
    assert result.evidence["reason"] == "ok"


def test_evaluator_fails_when_agent_says_none_but_truth_is_set() -> None:
    obs = Observation(inventory={"dirt": 4}, selected_item="dirt")
    report = PerceptionReport(inventory={"dirt": 4}, selected_item=None)
    result = _eval(report=report, observation=obs)
    assert result.success is False
    assert result.evidence["reason"] == "selected_item_mismatch"


def test_evaluator_preserves_primary_metric_set() -> None:
    """The Evaluator must still emit the dev-plan-fixed metric set."""
    obs = Observation(inventory={}, selected_item=None)
    report = PerceptionReport(inventory={}, selected_item=None)
    result = _eval(
        report=report,
        observation=obs,
        steps=7,
        model_calls=11,
        invalid_actions=3,
        elapsed_time=1.25,
    )
    assert result.steps == 7
    assert result.model_calls == 11
    assert result.invalid_actions == 3
    assert result.elapsed_time == 1.25
    assert result.task_id == D1_INVENTORY_PERCEPTION.task_id


# ---------------------------------------------------------------------------
# D1InventoryPerceptionAgent
# ---------------------------------------------------------------------------


def test_d1_agent_emits_wait_and_reads_report_from_model_response() -> None:
    """The D1 agent must read ``last_report`` from the model response,
    NOT auto-derive it from the observation. The whole point of D1 is
    whether the model can perceive the observation; the agent must
    not leak the ground truth into the report.
    """

    class _ReportingModel:
        completions = 0

        def complete(self, prompt: str) -> str:
            del prompt
            self.__class__.completions += 1
            return (
                '{"action": "WAIT", '
                '"report": {"inventory": {"dirt": 4, "oak_log": 2}, '
                '"selected_item": "dirt"}}'
            )

    agent = D1InventoryPerceptionAgent(model=_ReportingModel())
    obs = Observation(inventory={"dirt": 4, "oak_log": 2}, selected_item="dirt")
    action = agent.act(obs)
    assert action.type is ActionType.WAIT
    assert agent.model_calls == 1
    assert agent.last_report is not None
    assert agent.last_report.inventory == {"dirt": 4, "oak_log": 2}
    assert agent.last_report.selected_item == "dirt"


def test_d1_agent_handles_empty_report_from_model() -> None:
    """If the model response has no ``report`` key, the agent's
    ``last_report`` must be ``None`` (NOT auto-derived)."""

    class _NoReportModel:
        completions = 0

        def complete(self, prompt: str) -> str:
            del prompt
            self.__class__.completions += 1
            return '{"action": "WAIT"}'

    agent = D1InventoryPerceptionAgent(model=_NoReportModel())
    action = agent.act(Observation(inventory={"dirt": 4}, selected_item="dirt"))
    assert action.type is ActionType.WAIT
    assert agent.last_report is None


def test_d1_agent_handles_empty_inventory_from_model() -> None:
    """A model response of ``inventory: {}, selected_item: null`` is
    a valid empty D1 report."""

    class _EmptyModel:
        completions = 0

        def complete(self, prompt: str) -> str:
            del prompt
            self.__class__.completions += 1
            return (
                '{"action": "WAIT", '
                '"report": {"inventory": {}, "selected_item": null}}'
            )

    agent = D1InventoryPerceptionAgent(model=_EmptyModel())
    action = agent.act(Observation(inventory={}, selected_item=None))
    assert action.type is ActionType.WAIT
    assert agent.last_report is not None
    assert agent.last_report.inventory == {}
    assert agent.last_report.selected_item is None


# ---------------------------------------------------------------------------
# BenchmarkRunner vertical slice (offline)
# ---------------------------------------------------------------------------


class _StubEnv(Environment):
    """A controllable stub env for offline vertical-slice tests.

    Each ``step()`` call leaves the observation unchanged so the
    D1 ground truth stays stable across the whole episode.
    """

    def __init__(self, inventory: dict[str, int], selected_item: str | None) -> None:
        self._inventory = inventory
        self._selected_item = selected_item
        self.reset_called = 0
        self.close_called = 0
        self.steps = 0

    def reset(self) -> Observation:
        self.reset_called += 1
        self.steps = 0
        return Observation(
            frame=None,
            inventory=dict(self._inventory),
            selected_item=self._selected_item,
        )

    def step(self, action: Action) -> Observation:
        del action
        self.steps += 1
        return Observation(
            frame=None,
            inventory=dict(self._inventory),
            selected_item=self._selected_item,
        )

    def close(self) -> None:
        self.close_called += 1


class _WrongReportAgent:
    """Agent that emits a deliberately-wrong PerceptionReport."""

    model_calls = 0

    def __init__(self) -> None:
        self.last_report: PerceptionReport | None = None

    def act(self, observation: Observation) -> Action:  # noqa: ARG002
        type(self).model_calls += 1
        self.last_report = PerceptionReport(
            inventory={"diamond": 99},  # wrong on purpose
            selected_item="diamond",
        )
        return Action(type=ActionType.WAIT)


def test_runner_d1_success_path_on_stub_env() -> None:
    """End-to-end success: D1InventoryPerceptionModel returns
    ``inventory={}, selected_item=null`` and the stub env also has
    empty inventory, so the evaluator marks the run as a match.

    This mirrors the live ``MineRLTreechop-v0`` run (Treechop spawns
    with an empty inventory).
    """
    env = _StubEnv(inventory={}, selected_item=None)
    model = D1InventoryPerceptionModel()
    agent = D1InventoryPerceptionAgent(model=model)
    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=D1InventoryPerceptionEvaluator(),
    )
    assert env.close_called == 1
    assert env.steps == D1_INVENTORY_PERCEPTION.max_steps
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.steps == D1_INVENTORY_PERCEPTION.max_steps
    assert result.model_calls == D1_INVENTORY_PERCEPTION.max_steps
    assert result.invalid_actions == 0
    assert result.elapsed_time >= 0.0


def test_runner_d1_failure_path_on_stub_env() -> None:
    """A wrong report from the Agent must produce a structured failure."""
    env = _StubEnv(inventory={"dirt": 4}, selected_item="dirt")
    _WrongReportAgent.model_calls = 0
    agent = _WrongReportAgent()
    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=D1InventoryPerceptionEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "inventory_mismatch"
    assert result.task_id == D1_INVENTORY_PERCEPTION.task_id
    assert result.steps == D1_INVENTORY_PERCEPTION.max_steps
    assert result.invalid_actions == 0


def test_runner_closes_env_even_on_agent_failure() -> None:
    """The runner must call env.close() in a finally block."""
    class _RaisingAgent:
        model_calls = 0
        last_report = None

        def act(self, observation: Observation) -> Action:  # noqa: ARG002
            raise RuntimeError("synthetic agent failure")

    env = _StubEnv(inventory={}, selected_item=None)
    with pytest.raises(RuntimeError):
        BenchmarkRunner().run(
            task=D1_INVENTORY_PERCEPTION,
            env=env,
            agent=_RaisingAgent(),
            evaluator=D1InventoryPerceptionEvaluator(),
        )
    assert env.close_called == 1


def test_runner_forwards_agent_visible_observation_to_evaluator() -> None:
    """The runner must hand the Evaluator the *agent-visible* observation,
    not the post-step one. This is the D1 ground-truth contract.
    """

    captured: dict[str, Any] = {}

    class _CapturingEvaluator(Evaluator):
        def evaluate(self, task, *, steps, model_calls, invalid_actions,
                     elapsed_time, report=None, observation=None,
                     raw_response=None, ground_truth=None) -> Result:
            captured["report"] = report
            captured["observation"] = observation
            captured["raw_response"] = raw_response
            captured["ground_truth"] = ground_truth
            return Result(
                task_id=task.task_id,
                success=True,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
            )

    # Use a stub model that returns a report matching the stub env's
    # inventory, so the runner's "report + observation" forwarding
    # can be inspected on a non-mismatched example.
    class _MatchingModel:
        completions = 0

        def complete(self, prompt: str) -> str:  # noqa: ARG002
            self.__class__.completions += 1
            return (
                '{"action": "WAIT", '
                '"report": {"inventory": {"dirt": 4}, '
                '"selected_item": "dirt"}}'
            )

    env = _StubEnv(inventory={"dirt": 4}, selected_item="dirt")
    agent = D1InventoryPerceptionAgent(model=_MatchingModel())
    BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=_CapturingEvaluator(),
    )
    assert captured["observation"] is not None
    assert captured["observation"].inventory == {"dirt": 4}
    assert captured["observation"].selected_item == "dirt"
    assert captured["report"] is not None
    assert captured["report"].inventory == {"dirt": 4}
    assert captured["report"].selected_item == "dirt"
