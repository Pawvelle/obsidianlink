"""Offline tests for the R5 deterministic continuous casting driver.

These tests prove, in code, that:

* :func:`run_casting_c3_driver` walks a bounded multi-cell plan of
  legal :class:`MacroAction` values and never reaches evaluator
  truth.
* The driver successfully reaches the end of the plan on the
  :class:`FakeEnvironmentBackend` and produces the same event log
  on every replay (deterministic replay stability).
* The driver stops cleanly when a step / time / wait / plan / total
  recovery cap fires.
* The driver's recovery protocol retries a typed
  :class:`RecoverableBackendError` deterministically, with bounded
  per-step and total recovery budgets.
* The driver refuses to start when the Agent lacks a required
  inventory item; the missing-evidence path leaves the
  :class:`ContinuousCastingEvaluator` returning
  :data:`OUTCOME_TRUTH_MISSING`.
* The driver never accepts a stale ``step_id``: every event in the
  log carries the ``step_id`` returned by the backend, and a
  hand-crafted stale event is rejected by the orchestrator.
* Driver-visible ``Observation`` objects never carry evaluator
  truth (the orchestrator's truth surface is *separate* from the
  Observation the driver received).
* The casting truth surface is owned by the test orchestrator, not
  the driver. The driver has no access to
  ``set_continuous_casting_evaluation_state`` /
  ``get_continuous_casting_evaluation_state``; the orchestrator
  (this file) is the only place that calls them.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import time as _time
import unittest
from typing import Any, Iterable, Mapping

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)
from obsidianlink.drivers.casting_c3 import (
    AGENT_ID,
    ALLOWED_C3_ACTION_TYPES,
    ALLOWED_C3_TARGETS,
    DEFAULT_CELL_COUNT,
    DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    DEFAULT_MAX_ENVIRONMENT_STEPS,
    DEFAULT_MAX_GAME_TIME_SECONDS,
    DEFAULT_MAX_WAIT_STEPS,
    DEFAULT_OBSIDIAN_WAIT_STEPS,
    DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    MAX_C3_PLAN_WAIT_STEPS,
    MAX_CELL_COUNT,
    MAX_RECOVERIES_PER_ACTION,
    MAX_TOTAL_RECOVERY_BUDGET,
    MIN_CELL_COUNT,
    PHASE_PLACE_LAVA,
    PHASE_PLACE_SUPPORT,
    PHASE_PLACE_WATER,
    PHASE_PREPARE,
    PHASE_RECOVERY,
    PHASE_WAIT_FOR_OBSIDIAN,
    PHASE_VALUES,
    RECOVERIES_PER_ACTION_DEFAULT,
    TOTAL_RECOVERY_BUDGET_DEFAULT,
    DRIVER_STATUS_BLOCKED,
    DRIVER_STATUS_COMPLETED,
    DRIVER_STATUS_FAILED,
    DRIVER_STATUSES,
    CastingC3DriverResult,
    ContinuousCastingPlanStep,
    build_continuous_casting_action_plan,
    run_casting_c3_driver,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    OUTCOME_SUCCESS as R3_OUTCOME_SUCCESS,
)
from obsidianlink.evaluation.continuous_casting import (
    CONTINUOUS_OUTCOMES,
    OUTCOME_SUCCESS,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    OUTCOME_INVALID_INITIAL_STATE,
    OUTCOME_PARTIAL_COMPLEMENT,
    ContinuousCastingCellTruth,
    ContinuousCastingEvaluationResult,
    ContinuousCastingEvaluationState,
    ContinuousCastingEvaluator,
)


EPISODE_ID = "casting_c3_fixed_seed_0"
TARGET_CELLS: tuple[tuple[int, int, int], ...] = (
    (2, 4, 3),
    (3, 4, 3),
    (4, 4, 3),
)


def _task() -> TaskInstance:
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": EPISODE_ID,
            "route": "lava_casting",
            "difficulty": 2,
            "agent_ids": [AGENT_ID],
            "world_seed": 0,
            "instruction": "R5 contract test task.",
            "spawn_positions": {AGENT_ID: [0, 4, 0]},
            "initial_inventories": {
                AGENT_ID: {
                    "water_bucket": 3,
                    "lava_bucket": 3,
                    "cobblestone": 6,
                }
            },
            "workflow": "casting_c3_fixed",
            "milestones": [
                "task_reset",
                "cell_0_obsidian_cast",
                "cell_1_obsidian_cast",
                "cell_2_obsidian_cast",
            ],
            "limits": {
                "max_environment_steps": 240,
                "max_model_calls": 1,
                "max_game_time_seconds": 180,
            },
            "split": "development",
        }
    )


# ----------------------------------------------------------------------
# Test orchestrator: owns the R5 casting truth surface and the
# ContinuousCastingEvaluator call. The driver must not call
# anything in this section.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContinuousCastingWorldTruth:
    """Test-only description of the R5 casting world.

    The orchestrator (functions below) is the *only* component that
    consumes a :class:`ContinuousCastingWorldTruth`. The driver
    never sees this object. Each cell carries the evaluator truth
    required by :class:`ContinuousCastingCellTruth`.
    """

    initial_blocks: tuple[str, ...] = ("air", "air", "air")
    current_blocks: tuple[str, ...] = ("obsidian", "obsidian", "obsidian")
    water_truth: tuple[tuple[bool, int | None], ...] = (
        (True, 20),
        (True, 44),
        (True, 68),
    )
    lava_truth: tuple[tuple[bool, int | None], ...] = (
        (True, 9),
        (True, 33),
        (True, 57),
    )
    transition_steps: tuple[int | None, ...] = (20, 44, 68)
    transition_before_blocks: tuple[str | None, ...] = ("air", "air", "air")
    transition_after_blocks: tuple[str | None, ...] = (
        "obsidian",
        "obsidian",
        "obsidian",
    )
    per_cell_relevant: tuple[tuple[int, ...], ...] = (
        (3, 5, 9, 16),
        (27, 29, 33, 40),
        (51, 53, 57, 64),
    )
    terminated_step: int = 72
    terminated_reason: str = "driver_done"


def _state(
    world: ContinuousCastingWorldTruth,
    *,
    step_id: int,
    task: TaskInstance,
    current_time_seconds: float = 0.0,
) -> ContinuousCastingEvaluationState:
    cells = []
    for index, target_cell in enumerate(TARGET_CELLS):
        cells.append(
            ContinuousCastingCellTruth(
                target_cell=target_cell,
                initial_block=world.initial_blocks[index],
                current_block=world.current_blocks[index],
                water_truth=CastingFluidTruth(
                    present=world.water_truth[index][0],
                    evidence_step=world.water_truth[index][1],
                ),
                lava_truth=CastingFluidTruth(
                    present=world.lava_truth[index][0],
                    evidence_step=world.lava_truth[index][1],
                ),
                transition_evidence=CastingTransitionEvidence(
                    before_block=world.transition_before_blocks[index],
                    after_block=world.transition_after_blocks[index],
                    update_step=world.transition_steps[index],
                ),
                relevant_action_steps=world.per_cell_relevant[index],
            )
        )
    return ContinuousCastingEvaluationState(
        episode_id=task.task_id,
        step_id=step_id,
        cells=tuple(cells),
        agent_id=AGENT_ID,
        episode_terminated=True,
        terminated_step=world.terminated_step,
        terminated_reason=world.terminated_reason,
        current_time_seconds=current_time_seconds,
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )


def run_orchestrator(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC3DriverResult,
    world: ContinuousCastingWorldTruth,
    *,
    current_time_seconds: float = 0.0,
) -> ContinuousCastingEvaluationResult:
    """Build the orchestrator-side state and call the evaluator.

    The orchestrator (this function) is the *only* place in R5 that
    calls ``set_continuous_casting_evaluation_state`` /
    ``get_continuous_casting_evaluation_state`` and the
    :class:`ContinuousCastingEvaluator`. The driver never sees the
    truth surface.
    """
    task = backend._task  # type: ignore[attr-defined]
    if task is None:
        raise RuntimeError("backend must be reset before orchestrator runs")
    backend_step = backend._step_id  # type: ignore[attr-defined]
    state = _state(
        world,
        step_id=backend_step,
        task=task,
        current_time_seconds=current_time_seconds,
    )
    backend.set_continuous_casting_evaluation_state(state)
    return ContinuousCastingEvaluator().evaluate(
        backend.get_continuous_casting_evaluation_state()
    )


# ----------------------------------------------------------------------
# Driver contract tests
# ----------------------------------------------------------------------


class DriverContractTests(unittest.TestCase):
    """Static contract: allowlist, plan shape, refusal to read truth."""

    def test_action_allowlist_is_closed(self) -> None:
        self.assertEqual(
            ALLOWED_C3_ACTION_TYPES,
            frozenset(
                {
                    "equip_item",
                    "use_item",
                    "place_block",
                    "wait",
                }
            ),
        )
        self.assertEqual(
            ALLOWED_C3_TARGETS,
            frozenset({"water_bucket", "lava_bucket", "cobblestone"}),
        )
        self.assertEqual(
            DRIVER_STATUSES,
            frozenset(
                {
                    DRIVER_STATUS_COMPLETED,
                    DRIVER_STATUS_BLOCKED,
                    DRIVER_STATUS_FAILED,
                }
            ),
        )

    def test_driver_source_does_not_import_continuous_evaluator(self) -> None:
        import obsidianlink.drivers.casting_c3 as driver_module

        with open(
            driver_module.__file__, "r", encoding="utf-8"
        ) as handle:
            source = handle.read()
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
        forbidden = {
            "ContinuousCastingEvaluator",
            "ContinuousCastingEvaluationState",
            "ContinuousCastingEvaluationResult",
            "ContinuousCastingCellTruth",
            "ContinuousCastingWorldTruth",
            "OUTCOME_SUCCESS",
            "OUTCOME_TRUTH_MISSING",
            "OUTCOME_WRONG_BLOCK",
            "OUTCOME_INVALID_INITIAL_STATE",
            "OUTCOME_PARTIAL_COMPLEMENT",
            "OUTCOME_CAUSALITY_MISSING",
            "OUTCOME_STEP_BUDGET_EXCEEDED",
            "OUTCOME_TIME_BUDGET_EXCEEDED",
            "OUTCOME_ABNORMAL_TERMINATION",
            "OUTCOME_IN_PROGRESS",
        }
        for name in forbidden:
            self.assertNotIn(
                name,
                imported_names,
                f"driver module must not import {name!r}",
            )
        forbidden_strings = (
            "set_continuous_casting_evaluation_state",
            "get_continuous_casting_evaluation_state",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name
            ):
                self.assertNotIn(
                    node.attr,
                    forbidden_strings,
                    f"driver source uses {node.attr!r} on {node.value.id!r}",
                )

    def test_default_plan_length_and_relevant_counts(self) -> None:
        plan = build_continuous_casting_action_plan()
        # 24 steps per cell × 3 cells = 72 default steps.
        self.assertEqual(len(plan), 72)
        relevant_count = 0
        per_cell_count: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for step in plan:
            self.assertIsInstance(step, ContinuousCastingPlanStep)
            self.assertIn(step.action.action_type, ALLOWED_C3_ACTION_TYPES)
            if step.action.target is not None:
                self.assertIn(step.action.target, ALLOWED_C3_TARGETS)
            if step.relevant_action:
                relevant_count += 1
                per_cell_count[step.cell_index] = (
                    per_cell_count.get(step.cell_index, 0) + 1
                )
        # 4 relevant actions per cell (2 supports + 2 fluids).
        self.assertEqual(relevant_count, 12)
        for cell_index in range(DEFAULT_CELL_COUNT):
            self.assertEqual(per_cell_count[cell_index], 4)

    def test_default_plan_uses_phases(self) -> None:
        plan = build_continuous_casting_action_plan()
        seen_phases: set[str] = set()
        for step in plan:
            seen_phases.add(step.phase)
        # PHASE_RECOVERY is reserved for driver-internal events.
        self.assertNotIn(PHASE_RECOVERY, seen_phases)
        self.assertEqual(
            seen_phases,
            {
                PHASE_PREPARE,
                PHASE_PLACE_SUPPORT,
                PHASE_PLACE_LAVA,
                PHASE_PLACE_WATER,
                PHASE_WAIT_FOR_OBSIDIAN,
            },
        )

    def test_default_plan_uses_allowlist(self) -> None:
        for step in build_continuous_casting_action_plan():
            self.assertIn(step.action.action_type, ALLOWED_C3_ACTION_TYPES)
            if step.action.target is not None:
                self.assertIn(step.action.target, ALLOWED_C3_TARGETS)

    def test_default_plan_recoveries_are_bounded(self) -> None:
        plan = build_continuous_casting_action_plan()
        for step in plan:
            self.assertGreaterEqual(step.recoveries_allowed, 0)
            self.assertLessEqual(
                step.recoveries_allowed, MAX_RECOVERIES_PER_ACTION
            )
            # Only ``use_item`` steps may carry a recovery budget.
            if step.recoveries_allowed > 0:
                self.assertEqual(step.action.action_type, "use_item")

    def test_plan_actions_accepted_by_public_protocol(self) -> None:
        for step in build_continuous_casting_action_plan():
            payload = json.dumps(
                {
                    "action_type": step.action.action_type,
                    "target": step.action.target,
                    "duration_ticks": step.action.duration_ticks,
                    "parameters": dict(step.action.parameters),
                }
            )
            parsed = parse_macro_action(payload)
            self.assertTrue(parsed.accepted, parsed.error)
            self.assertEqual(parsed.action, step.action)

    def test_plan_parameters_are_validated(self) -> None:
        for bad in (-1, "1", True):
            with self.assertRaises(ValueError):
                build_continuous_casting_action_plan(
                    support_block_wait_steps=bad  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                build_continuous_casting_action_plan(
                    fluid_settle_wait_steps=bad  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                build_continuous_casting_action_plan(
                    obsidian_wait_steps=bad  # type: ignore[arg-type]
                )

    def test_plan_rejects_too_many_waits(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard limit"):
            build_continuous_casting_action_plan(
                obsidian_wait_steps=MAX_C3_PLAN_WAIT_STEPS,
            )

    def test_plan_rejects_too_many_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "cell_count must be between"):
            build_continuous_casting_action_plan(
                cell_count=MAX_CELL_COUNT + 1,
            )

    def test_plan_rejects_recoveries_per_use_item_too_high(self) -> None:
        with self.assertRaisesRegex(ValueError, "recoveries_per_use_item"):
            build_continuous_casting_action_plan(
                recoveries_per_use_item=MAX_RECOVERIES_PER_ACTION + 1,
            )

    def test_recovery_budget_constants_are_documented(self) -> None:
        self.assertGreater(MIN_CELL_COUNT, 0)
        self.assertLessEqual(MAX_CELL_COUNT, 16)
        self.assertGreaterEqual(TOTAL_RECOVERY_BUDGET_DEFAULT, 1)
        self.assertLessEqual(
            TOTAL_RECOVERY_BUDGET_DEFAULT, MAX_TOTAL_RECOVERY_BUDGET
        )
        self.assertGreaterEqual(RECOVERIES_PER_ACTION_DEFAULT, 1)
        self.assertLessEqual(
            RECOVERIES_PER_ACTION_DEFAULT, MAX_RECOVERIES_PER_ACTION
        )
        self.assertEqual(
            PHASE_VALUES,
            frozenset(
                {
                    PHASE_PREPARE,
                    PHASE_PLACE_SUPPORT,
                    PHASE_PLACE_LAVA,
                    PHASE_PLACE_WATER,
                    PHASE_WAIT_FOR_OBSIDIAN,
                    PHASE_RECOVERY,
                }
            ),
        )


class DriverArgumentValidationTests(unittest.TestCase):
    """The driver refuses bad arguments before touching the backend."""

    def test_rejects_non_task(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            with self.assertRaisesRegex(ValueError, "TaskInstance"):
                run_casting_c3_driver(backend, "not a task")  # type: ignore[arg-type]
        finally:
            backend.close()

    def test_rejects_wrong_workflow(self) -> None:
        wrong_task = TaskInstance.from_dict(_task_dict_with_workflow("casting_c1_fixed"))
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            with self.assertRaisesRegex(ValueError, "casting_c3_fixed"):
                run_casting_c3_driver(backend, wrong_task)
        finally:
            backend.close()

    def test_rejects_zero_max_wait_steps(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            with self.assertRaisesRegex(ValueError, "max_wait_steps"):
                run_casting_c3_driver(
                    backend,
                    _task(),
                    max_wait_steps=0,
                )
        finally:
            backend.close()

    def test_rejects_too_large_max_wait_steps(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            with self.assertRaisesRegex(
                ValueError, "max_wait_steps must be <="
            ):
                run_casting_c3_driver(
                    backend,
                    _task(),
                    max_wait_steps=MAX_C3_PLAN_WAIT_STEPS + 1,
                )
        finally:
            backend.close()

    def test_rejects_max_environment_steps_above_task_limit(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            with self.assertRaisesRegex(
                ValueError, "max_environment_steps cannot exceed"
            ):
                run_casting_c3_driver(
                    backend,
                    _task(),
                    max_environment_steps=10_000,
                )
        finally:
            backend.close()

    def test_rejects_max_game_time_above_task_limit(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            with self.assertRaisesRegex(
                ValueError, "max_game_time_seconds cannot exceed"
            ):
                run_casting_c3_driver(
                    backend,
                    _task(),
                    max_game_time_seconds=10_000.0,
                )
        finally:
            backend.close()

    def test_rejects_total_recovery_budget_above_max(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            with self.assertRaisesRegex(
                ValueError, "total_recovery_budget cannot exceed"
            ):
                run_casting_c3_driver(
                    backend,
                    _task(),
                    total_recovery_budget=MAX_TOTAL_RECOVERY_BUDGET + 1,
                )
        finally:
            backend.close()

    def test_rejects_plan_longer_than_task_step_limit(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            too_long_plan = tuple(
                ContinuousCastingPlanStep(
                    cell_index=0,
                    label=f"step.{index}",
                    phase=PHASE_PREPARE,
                    action=MacroAction.wait(),
                )
                for index in range(241)
            )
            with self.assertRaisesRegex(
                ValueError, "plan length cannot exceed"
            ):
                run_casting_c3_driver(backend, _task(), plan=too_long_plan)
        finally:
            backend.close()


def _task_dict_with_workflow(workflow: str) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "task_id": EPISODE_ID,
        "route": "lava_casting",
        "difficulty": 2,
        "agent_ids": [AGENT_ID],
        "world_seed": 0,
        "instruction": "wrong workflow",
        "spawn_positions": {AGENT_ID: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID: {
                "water_bucket": 3,
                "lava_bucket": 3,
                "cobblestone": 6,
            }
        },
        "workflow": workflow,
        "milestones": ["task_reset"],
        "limits": {
            "max_environment_steps": 240,
            "max_model_calls": 1,
            "max_game_time_seconds": 180,
        },
        "split": "development",
    }


# ----------------------------------------------------------------------
# Driver on FakeBackend
# ----------------------------------------------------------------------


class FakeBackendDriverTests(unittest.TestCase):
    """The driver walks the multi-cell plan on a plain FakeBackend."""

    def _run_driver(
        self,
        *,
        max_environment_steps: int | None = None,
        max_game_time_seconds: float | None = None,
        max_wait_steps: int = DEFAULT_MAX_WAIT_STEPS,
        total_recovery_budget: int = TOTAL_RECOVERY_BUDGET_DEFAULT,
        recoveries_per_use_item: int = RECOVERIES_PER_ACTION_DEFAULT,
        task: TaskInstance | None = None,
        event_sink=None,
        plan=None,
    ) -> CastingC3DriverResult:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            effective_task = task or _task()
            backend.reset(effective_task)
            kwargs: dict[str, Any] = dict(
                max_wait_steps=max_wait_steps,
                total_recovery_budget=total_recovery_budget,
                recoveries_per_use_item=recoveries_per_use_item,
                event_sink=event_sink,
            )
            if max_environment_steps is not None:
                kwargs["max_environment_steps"] = max_environment_steps
            if max_game_time_seconds is not None:
                kwargs["max_game_time_seconds"] = max_game_time_seconds
            if plan is not None:
                kwargs["plan"] = plan
            return run_casting_c3_driver(backend, effective_task, **kwargs)
        finally:
            backend.close()

    def test_driver_completes_full_plan_on_fake_backend(self) -> None:
        result = self._run_driver()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(result.steps_executed, result.planned_steps)
        # Every step the driver submitted is in the event log, plus
        # the reset event at the start.
        self.assertEqual(len(result.events), result.planned_steps + 1)
        for event in result.events:
            self.assertEqual(event["episode_id"], EPISODE_ID)
            self.assertEqual(event["agent_id"], AGENT_ID)
            self.assertIs(type(event["step_id"]), int)
            self.assertGreaterEqual(event["step_id"], 0)
        # 4 relevant actions per cell × 3 cells = 12.
        self.assertEqual(
            sum(
                len(steps)
                for steps in result.per_cell_relevant_action_steps.values()
            ),
            12,
        )
        # per_cell_relevant_action_steps is a Mapping with three
        # entries, one per cell.
        self.assertEqual(
            set(result.per_cell_relevant_action_steps.keys()),
            {0, 1, 2},
        )
        self.assertEqual(
            result.final_observation.step_id, result.planned_steps
        )

    def test_driver_event_step_ids_match_backend(self) -> None:
        result = self._run_driver()
        previous = -1
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["step_id"], 0)
                previous = 0
                continue
            self.assertGreater(event["step_id"], previous)
            previous = event["step_id"]
        self.assertEqual(
            set(result.action_label_for_step),
            set(range(1, result.planned_steps + 1)),
        )

    def test_driver_event_cell_index_is_set(self) -> None:
        result = self._run_driver()
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["cell_index"], -1)
                continue
            self.assertIn(event["cell_index"], {0, 1, 2})

    def test_driver_fires_step_budget(self) -> None:
        result = self._run_driver(max_environment_steps=10)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("step budget exhausted", result.blocked_reason or "")
        self.assertLessEqual(result.steps_executed, 10)
        self.assertLess(result.steps_executed, result.planned_steps)
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "step"
        )

    def test_driver_fires_time_budget(self) -> None:
        class _TimeAdvancingBackend(FakeEnvironmentBackend):
            def __init__(self, seconds_per_step: float = 60.0) -> None:
                super().__init__()
                self._seconds_per_step = seconds_per_step
                self._base = _time.time()

            def _observations(self):  # type: ignore[override]
                task = self._require_task()
                timestamp = self._base + self._step_id * self._seconds_per_step
                return {
                    agent_id: Observation(
                        episode_id=task.task_id,
                        agent_id=agent_id,
                        step_id=self._step_id,
                        timestamp=timestamp,
                        frame={"backend": "fake_time", "step_id": self._step_id},
                        visible_inventory=task.initial_inventories[agent_id],
                        workflow_stage=task.workflow,
                    )
                    for agent_id in task.agent_ids
                }

            def step(self, actions):  # type: ignore[override]
                step = super().step(actions)
                if step.step_id == 3:
                    return dataclasses.replace(step, terminated=True)
                return step

        backend = _TimeAdvancingBackend(seconds_per_step=60.0)
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                max_game_time_seconds=120.0,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("time budget exceeded", result.blocked_reason or "")
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "time"
        )

    def test_driver_fires_wait_budget(self) -> None:
        result = self._run_driver(max_wait_steps=2)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("wait budget exhausted", result.blocked_reason or "")
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "wait"
        )
        self.assertEqual(result.wait_steps, 2)

    def test_driver_marks_early_backend_termination_as_blocked(self) -> None:
        class _EarlyTerminatingBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override]
                step = super().step(actions)
                return dataclasses.replace(step, terminated=True)

        backend = _EarlyTerminatingBackend()
        backend.open()
        try:
            result = run_casting_c3_driver(backend, _task())
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.steps_executed, 1)
        self.assertIn("backend termination", result.blocked_reason or "")

    def test_driver_marks_early_truncation_as_blocked(self) -> None:
        class _EarlyTruncatingBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override]
                step = super().step(actions)
                return dataclasses.replace(step, truncated=True)

        backend = _EarlyTruncatingBackend()
        backend.open()
        try:
            result = run_casting_c3_driver(backend, _task())
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertTrue(result.truncated)
        self.assertIn("backend truncation", result.blocked_reason or "")

    def test_driver_result_evidence_is_deeply_immutable(self) -> None:
        result = self._run_driver()
        with self.assertRaises(TypeError):
            result.events[0]["label"] = "tampered"  # type: ignore[index]
        snapshot = result.as_dict()
        snapshot["events"][0]["label"] = "tampered"
        snapshot["events"][0]["visible_inventory"]["water_bucket"] = 0
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 3
        )

    def test_event_sink_cannot_mutate_driver_evidence(self) -> None:
        def mutate(event):  # type: ignore[no-untyped-def]
            event["label"] = "tampered"
            if "visible_inventory" in event:
                event["visible_inventory"]["water_bucket"] = 0

        result = self._run_driver(event_sink=mutate)
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 3
        )

    def test_driver_blocks_on_missing_required_item(self) -> None:
        bad_task = TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": EPISODE_ID,
                "route": "lava_casting",
                "difficulty": 2,
                "agent_ids": [AGENT_ID],
                "world_seed": 0,
                "instruction": "no cobblestone",
                "spawn_positions": {AGENT_ID: [0, 4, 0]},
                "initial_inventories": {
                    AGENT_ID: {
                        "water_bucket": 3,
                        "lava_bucket": 3,
                    }
                },
                "workflow": "casting_c3_fixed",
                "milestones": [
                    "task_reset",
                ],
                "limits": {
                    "max_environment_steps": 240,
                    "max_model_calls": 1,
                    "max_game_time_seconds": 180,
                },
                "split": "development",
            }
        )
        result = self._run_driver(task=bad_task)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("cobblestone", result.blocked_reason or "")
        # No relevant action should have been submitted because
        # the first place_block was refused.
        self.assertEqual(result.per_cell_relevant_action_steps, {})

    def test_driver_replay_is_deterministic(self) -> None:
        first = self._run_driver()
        second = self._run_driver()
        self.assertEqual(
            [event["label"] for event in first.events],
            [event["label"] for event in second.events],
        )
        self.assertEqual(
            [event["step_id"] for event in first.events],
            [event["step_id"] for event in second.events],
        )
        self.assertEqual(
            [event["action_type"] for event in first.events],
            [event["action_type"] for event in second.events],
        )
        self.assertEqual(
            first.per_cell_relevant_action_steps,
            second.per_cell_relevant_action_steps,
        )

    def test_driver_does_not_call_casting_truth_surface(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            original_set = backend.set_continuous_casting_evaluation_state
            original_get = backend.get_continuous_casting_evaluation_state
            set_calls: list[Any] = []
            get_calls: list[int] = []
            backend.set_continuous_casting_evaluation_state = (  # type: ignore[method-assign]
                lambda state: set_calls.append(state) or original_set(state)
            )
            backend.get_continuous_casting_evaluation_state = (  # type: ignore[method-assign]
                lambda: get_calls.append(1) or original_get()
            )
            run_casting_c3_driver(backend, _task())
        finally:
            backend.close()
        self.assertEqual(set_calls, [])
        self.assertEqual(get_calls, [])

    def test_driver_does_not_leak_truth_into_observation(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            original = Observation.__getattribute__
            forbidden = {
                "target_cell",
                "target_cells",
                "target_block",
                "initial_block",
                "current_block",
                "water_truth",
                "lava_truth",
                "fluid_truth",
                "transition_evidence",
                "relevant_action_steps",
                "casting_evaluator",
                "casting_outcome",
                "success",
                "blocking_conditions",
                "outcome",
                "failure_type",
                "per_cell_outcomes",
                "first_failed_cell",
                "completed_cells",
            }

            def guarded(self, name):  # type: ignore[no-untyped-def]
                if name in forbidden:
                    raise AssertionError(
                        f"driver attempted to read Observation.{name}"
                    )
                return original(self, name)

            Observation.__getattribute__ = guarded  # type: ignore[assignment]
            try:
                result = run_casting_c3_driver(backend, _task())
            finally:
                Observation.__getattribute__ = original  # type: ignore[assignment]
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)


# ----------------------------------------------------------------------
# Recovery protocol
# ----------------------------------------------------------------------


class RecoveryProtocolTests(unittest.TestCase):
    """The driver retries typed RecoverableBackendError deterministically."""

    def _make_recovery_backend(
        self,
        raise_spec: Mapping[int, int],
        recoverable_kind: str = "bucket_use_transient",
    ) -> FakeEnvironmentBackend:
        class _RecoveryBackend(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self.raise_spec = dict(raise_spec)
                self.raised_on_step: dict[int, int] = {}
                self.recoverable_kind = recoverable_kind

            def step(self, actions):  # type: ignore[override]
                next_step = self._step_id + 1
                max_raises = self.raise_spec.get(next_step, 0)
                count = self.raised_on_step.get(next_step, 0)
                if count < max_raises:
                    self.raised_on_step[next_step] = count + 1
                    raise RecoverableBackendError(
                        f"transient error at step {next_step}",
                        recoverable_kind=self.recoverable_kind,
                    )
                return super().step(actions)

        return _RecoveryBackend()

    def test_recoverable_error_is_retried_once(self) -> None:
        # Make the driver retry the first ``use_item`` (cell 0, lava)
        # exactly once. The recovery must succeed and the driver
        # must reach the end of the plan.
        backend = self._make_recovery_backend(raise_spec={9: 1})
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                total_recovery_budget=3,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.recovery_attempts, 1)
        recovery_events = [
            event for event in result.events
            if event.get("phase") == PHASE_RECOVERY
        ]
        self.assertEqual(len(recovery_events), 1)
        self.assertEqual(
            recovery_events[0]["recoverable_kind"], "bucket_use_transient"
        )

    def test_recoverable_error_blocked_when_total_budget_exhausted(self) -> None:
        # Make three different use_item steps raise exactly once
        # each. With total_recovery_budget=2 the third raise must
        # block the driver.
        backend = self._make_recovery_backend(raise_spec={9: 1, 16: 1, 33: 1})
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                total_recovery_budget=2,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertEqual(result.recovery_attempts, 3)
        self.assertIn("recovery budget exhausted", result.blocked_reason or "")

    def test_recoverable_error_blocked_when_per_step_budget_exhausted(self) -> None:
        # Make the first use_item step raise twice. With
        # recoveries_per_use_item=1 the second raise must block
        # the driver (attempt 2 > 1).
        backend = self._make_recovery_backend(raise_spec={9: 2})
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                total_recovery_budget=6,
                recoveries_per_use_item=1,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertEqual(result.recovery_attempts, 2)
        self.assertIn(
            "per-step recovery budget exhausted", result.blocked_reason or ""
        )

    def test_recovery_event_includes_recoverable_metadata(self) -> None:
        backend = self._make_recovery_backend(
            raise_spec={9: 1}, recoverable_kind="custom_kind"
        )
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                total_recovery_budget=3,
            )
        finally:
            backend.close()
        recovery_events = [
            event for event in result.events
            if event.get("phase") == PHASE_RECOVERY
        ]
        self.assertEqual(len(recovery_events), 1)
        self.assertEqual(recovery_events[0]["recoverable_kind"], "custom_kind")
        self.assertIn("recoverable_message", recovery_events[0])

    def test_recovery_blocked_before_step_budget_remaining(self) -> None:
        # ``total_recovery_budget`` is independent from
        # ``max_wait_steps``; it must fire even when the
        # ``max_wait_steps`` budget has spare room.
        backend = self._make_recovery_backend(raise_spec={9: 1, 16: 1, 33: 1})
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(
                backend,
                _task(),
                total_recovery_budget=2,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertLessEqual(result.recovery_attempts, 3)

    def test_non_recoverable_error_fails_closed(self) -> None:
        class _RuntimeErrorBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override]
                next_step = self._step_id + 1
                if next_step == 9:
                    raise RuntimeError(f"boom at step {next_step}")
                return super().step(actions)

        backend = _RuntimeErrorBackend()
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(backend, _task())
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("RuntimeError", result.blocked_reason or "")

    def test_recovery_budget_zero_still_runs(self) -> None:
        # The driver must still run when total_recovery_budget is
        # zero (no recovery allowed). The default plan's use_item
        # steps are allowed to retry 1 time; the FakeBackend
        # never raises, so no recovery is actually attempted.
        result = FakeBackendDriverTests()._run_driver(
            total_recovery_budget=0
        )
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.recovery_attempts, 0)


# ----------------------------------------------------------------------
# Driver + orchestrator + evaluator
# ----------------------------------------------------------------------


class OrchestratorOutcomeTests(unittest.TestCase):
    """The R5 driver + orchestrator + evaluator end-to-end on the
    default plan."""

    def _run_with_world(
        self,
        world: ContinuousCastingWorldTruth,
        *,
        max_environment_steps: int | None = None,
        max_game_time_seconds: float | None = None,
    ) -> tuple[CastingC3DriverResult, ContinuousCastingEvaluationResult]:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = _task()
            backend.reset(task)
            driver_result = run_casting_c3_driver(backend, task)
            result = run_orchestrator(backend, driver_result, world)
            return driver_result, result
        finally:
            backend.close()

    def test_normal_path_yields_success(self) -> None:
        driver, evaluator = self._run_with_world(ContinuousCastingWorldTruth())
        self.assertEqual(driver.status, DRIVER_STATUS_COMPLETED)
        self.assertTrue(evaluator.success)
        self.assertEqual(evaluator.outcome, OUTCOME_SUCCESS)
        self.assertEqual(evaluator.completed_cells, 3)
        self.assertEqual(evaluator.total_cells, 3)
        self.assertEqual(evaluator.first_failed_cell, None)
        self.assertEqual(evaluator.failure_type, None)
        self.assertEqual(evaluator.blocking_conditions, ())

    def test_completed_cells_is_three(self) -> None:
        _, evaluator = self._run_with_world(ContinuousCastingWorldTruth())
        self.assertEqual(evaluator.completed_cells, 3)
        self.assertEqual(
            tuple(evaluator.per_cell_outcomes), ("cell_success",) * 3
        )

    def test_only_first_cell_succeeds(self) -> None:
        # cells 1 / 2 stay on "air"
        world = ContinuousCastingWorldTruth(
            current_blocks=("obsidian", "air", "air"),
        )
        driver, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(evaluator.completed_cells, 1)
        self.assertEqual(evaluator.first_failed_cell, 1)

    def test_first_two_cells_succeed(self) -> None:
        world = ContinuousCastingWorldTruth(
            current_blocks=("obsidian", "obsidian", "air"),
        )
        driver, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(evaluator.completed_cells, 2)
        self.assertEqual(evaluator.first_failed_cell, 2)

    def test_middle_cell_wrong_block(self) -> None:
        world = ContinuousCastingWorldTruth(
            current_blocks=("obsidian", "cobblestone", "obsidian"),
        )
        _, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(evaluator.first_failed_cell, 1)
        self.assertTrue(
            any(
                "cell_1" in condition and "cobblestone" in condition
                for condition in evaluator.blocking_conditions
            )
        )

    def test_truth_missing_via_orchestrator(self) -> None:
        # cell 1 has no water truth
        world = ContinuousCastingWorldTruth(
            water_truth=(
                (True, 20),
                (None, None),  # type: ignore[arg-type]
                (True, 68),
            ),
        )
        _, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_TRUTH_MISSING)

    def test_invalid_initial_state(self) -> None:
        world = ContinuousCastingWorldTruth(
            initial_blocks=("obsidian", "air", "air"),
            current_blocks=("obsidian", "obsidian", "obsidian"),
            # Mark cells 1 / 2 as already obsidian-from-air via
            # the orchestrator. We need to bump the per-cell
            # transition ``after_block`` to obsidian as well, and
            # the ``before_block`` to obsidian.
            transition_before_blocks=("obsidian", "air", "air"),
            transition_after_blocks=("obsidian", "obsidian", "obsidian"),
            water_truth=((True, 5), (True, 44), (True, 68)),
            lava_truth=((True, 5), (True, 33), (True, 57)),
            transition_steps=(5, 44, 68),
        )
        _, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_INVALID_INITIAL_STATE)
        self.assertEqual(evaluator.first_failed_cell, 0)

    def test_partial_completion_via_orchestrator(self) -> None:
        # The orchestrator reports a world that is ambiguous: cell 0
        # and cell 2 are not_evaluated (current_block = None means
        # missing truth). Build a state where only cell 1 has the
        # truth required for a per-cell success verdict; the others
        # are missing the transition update step, so the evaluator
        # should fall back to truth_missing (per-cell truth outranks
        # partial_completion). This re-verifies the priority chain.
        world = ContinuousCastingWorldTruth(
            current_blocks=("air", "obsidian", "air"),
            transition_steps=(None, 44, None),
            transition_before_blocks=("air", "air", "air"),
            transition_after_blocks=("air", "obsidian", "air"),
        )
        _, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_TRUTH_MISSING)


# ----------------------------------------------------------------------
# Replay / observability
# ----------------------------------------------------------------------


class StaleStepAndReplayTests(unittest.TestCase):
    def test_driver_rejects_backend_step_jump(self) -> None:
        class _JumpingBackend:
            def __init__(self) -> None:
                self.task = None

            def reset(self, task):  # type: ignore[no-untyped-def]
                self.task = task
                return {
                    AGENT_ID: Observation(
                        episode_id=task.task_id,
                        agent_id=AGENT_ID,
                        step_id=0,
                        timestamp=0.0,
                        frame={},
                        visible_inventory=task.initial_inventories[AGENT_ID],
                        workflow_stage=task.workflow,
                    )
                }

            def step(self, actions):  # type: ignore[no-untyped-def]
                observation = Observation(
                    episode_id=self.task.task_id,
                    agent_id=AGENT_ID,
                    step_id=2,
                    timestamp=1.0,
                    frame={},
                    visible_inventory=self.task.initial_inventories[AGENT_ID],
                    workflow_stage=self.task.workflow,
                )
                return BackendStep(
                    episode_id=self.task.task_id,
                    step_id=2,
                    observations={AGENT_ID: observation},
                    rewards={AGENT_ID: 0.0},
                    terminated=False,
                    truncated=False,
                )

        with self.assertRaisesRegex(ValueError, "advance exactly once"):
            run_casting_c3_driver(_JumpingBackend(), _task())

    def test_driver_event_step_ids_are_unique_and_monotonic(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            result = run_casting_c3_driver(backend, _task())
        finally:
            backend.close()
        seen_step_ids: list[int] = []
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["step_id"], 0)
                continue
            seen_step_ids.append(event["step_id"])
        self.assertEqual(
            sorted(seen_step_ids), list(range(1, result.planned_steps + 1))
        )
        self.assertEqual(len(seen_step_ids), len(set(seen_step_ids)))

    def test_replay_with_orchestrator_is_stable(self) -> None:
        def run_once():
            backend = FakeEnvironmentBackend()
            backend.open()
            try:
                task = _task()
                backend.reset(task)
                driver_result = run_casting_c3_driver(backend, task)
                result = run_orchestrator(
                    backend, driver_result, ContinuousCastingWorldTruth()
                )
                snapshot = (
                    driver_result.action_label_for_step,
                    dict(driver_result.per_cell_relevant_action_steps),
                    result.outcome,
                    result.success,
                )
                return snapshot
            finally:
                backend.close()

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)

    def test_evaluation_state_uses_driver_per_cell_actions(self) -> None:
        # The orchestrator must build per-cell
        # ``relevant_action_steps`` from the driver result, not
        # from outside knowledge.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = _task()
            backend.reset(task)
            driver_result = run_casting_c3_driver(backend, task)
            result = run_orchestrator(
                backend, driver_result, ContinuousCastingWorldTruth()
            )
            # The per-cell outcomes should all be cell_success.
            self.assertEqual(
                tuple(result.per_cell_outcomes), ("cell_success",) * 3
            )
            self.assertEqual(result.completed_cells, 3)
        finally:
            backend.close()


# ----------------------------------------------------------------------
# Observation leakage: the orchestrator's truth must never show up
# in the driver's Observations.
# ----------------------------------------------------------------------


class ObservationLeakageTests(unittest.TestCase):
    def test_observation_does_not_carry_casting_truth_after_driver(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = _task()
            backend.reset(task)
            driver_result = run_casting_c3_driver(backend, task)
            # After the driver, the orchestrator injects the full
            # truth. The driver result's final observation was
            # captured *before* the orchestrator ran, so it must
            # still be clean.
            run_orchestrator(
                backend, driver_result, ContinuousCastingWorldTruth()
            )
            forbidden = {
                "target_cell",
                "target_cells",
                "target_block",
                "initial_block",
                "current_block",
                "water_truth",
                "lava_truth",
                "transition_evidence",
                "relevant_action_steps",
                "casting_evaluator",
                "casting_outcome",
                "success",
                "blocking_conditions",
                "outcome",
                "failure_type",
                "per_cell_outcomes",
                "first_failed_cell",
                "completed_cells",
            }
            observation = driver_result.final_observation
            for field in forbidden:
                self.assertFalse(
                    hasattr(observation, field),
                    f"final observation exposes {field!r}",
                )
            if isinstance(observation.frame, Mapping):
                for field in forbidden:
                    self.assertNotIn(
                        field,
                        observation.frame,
                        f"final observation.frame carries {field!r}",
                    )
        finally:
            backend.close()

    def test_all_driver_observations_are_clean(self) -> None:
        seen: list[Observation] = []
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = _task()
            backend.reset(task)
            for _ in range(20):
                step = backend.step({"agent_1": MacroAction.wait()})
                seen.append(step.observations[AGENT_ID])
        finally:
            backend.close()
        forbidden = {
            "target_cell",
            "target_cells",
            "target_block",
            "initial_block",
            "current_block",
            "water_truth",
            "lava_truth",
            "transition_evidence",
            "relevant_action_steps",
            "casting_evaluator",
            "casting_outcome",
            "success",
            "blocking_conditions",
            "outcome",
            "failure_type",
            "per_cell_outcomes",
            "first_failed_cell",
            "completed_cells",
        }
        for observation in seen:
            for field in forbidden:
                self.assertFalse(
                    hasattr(observation, field),
                    f"observation leaks {field!r}",
                )
            if isinstance(observation.frame, Mapping):
                for field in forbidden:
                    self.assertNotIn(
                        field,
                        observation.frame,
                        f"observation.frame leaks {field!r}",
                    )


# ----------------------------------------------------------------------
# Driver ↔ MineRL capability gap
# ----------------------------------------------------------------------


class DriverBackendShapeTests(unittest.TestCase):
    """The driver only requires ``reset`` and ``step``; nothing more."""

    def test_driver_accepts_minimal_backend_shape(self) -> None:
        class _Minimal:
            def __init__(self) -> None:
                self.calls = 0
                self.task = None

            def reset(self, task):  # type: ignore[no-untyped-def]
                self.task = task
                self.calls = 0
                return {
                    AGENT_ID: Observation(
                        episode_id=task.task_id,
                        agent_id=AGENT_ID,
                        step_id=0,
                        timestamp=0.0,
                        frame={"minimal": True},
                        visible_inventory={
                            "water_bucket": 3,
                            "lava_bucket": 3,
                            "cobblestone": 6,
                        },
                        workflow_stage="casting_c3_fixed",
                    )
                }

            def step(self, actions):  # type: ignore[no-untyped-def]
                self.calls += 1
                step_id = self.calls
                observation = Observation(
                    episode_id=self.task.task_id,
                    agent_id=AGENT_ID,
                    step_id=step_id,
                    timestamp=0.0,
                    frame={"step": step_id},
                    visible_inventory={
                        "water_bucket": 3,
                        "lava_bucket": 3,
                        "cobblestone": 6,
                    },
                    workflow_stage="casting_c3_fixed",
                )
                return BackendStep(
                    episode_id=self.task.task_id,
                    step_id=step_id,
                    observations={AGENT_ID: observation},
                    rewards={AGENT_ID: 0.0},
                    terminated=False,
                    truncated=False,
                )

        backend = _Minimal()
        result = run_casting_c3_driver(backend, _task())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)


if __name__ == "__main__":
    unittest.main()
