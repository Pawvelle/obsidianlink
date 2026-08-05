"""Offline tests for the R5 continuous casting evaluator.

These tests prove, in code, that:

* :class:`ContinuousCastingCellTruth`,
  :class:`ContinuousCastingEvaluationState`, and
  :class:`ContinuousCastingEvaluationResult` are frozen, type-strict,
  and JSON-serializable, with a recursive frozen evidence tree.
* :class:`ContinuousCastingEvaluator` returns the same result for the
  same state on repeated calls, never reads Agent text / images /
  Planner input, and never imports the driver / planner / model
  surface.
* The closed outcome set and the locked priority order are honoured
  for every required scenario (success, partial completion, wrong
  block, missing truth per cell, causality issues, abnormal
  termination, budget exceeded, invalid initial state, in progress).
* Per-cell evidence cannot be lifted from one cell into another.
* The :class:`FakeEnvironmentBackend` exposes a
  ``set_continuous_casting_evaluation_state`` /
  ``get_continuous_casting_evaluation_state`` surface that is
  identity-guarded, identity-checked on every step, and that does
  not leak into :class:`Observation`.

The tests never start Minecraft, MineRL, or Gradle.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from typing import Any, Mapping

from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
)
from obsidianlink.evaluation.continuous_casting import (
    CONTINUOUS_OUTCOMES,
    DEFAULT_CELL_COUNT,
    MAX_CELL_COUNT,
    MIN_CELL_COUNT,
    OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_CAUSALITY_MISSING,
    OUTCOME_IN_PROGRESS,
    OUTCOME_INVALID_INITIAL_STATE,
    OUTCOME_PARTIAL_COMPLEMENT,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    PER_CELL_CAUSALITY_MISSING,
    PER_CELL_NOT_EVALUATED,
    PER_CELL_SUCCESS,
    PER_CELL_TRUTH_MISSING,
    PER_CELL_WRONG_BLOCK,
    ContinuousCastingCellTruth,
    ContinuousCastingEvaluationResult,
    ContinuousCastingEvaluationState,
    ContinuousCastingEvaluator,
)
from obsidianlink.core.types import TaskInstance


EPISODE_ID = "casting_c3_fixed_seed_0"
AGENT_ID = "agent_1"
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


def _success_cell(
    target_cell: tuple[int, int, int],
    *,
    last_action_step: int,
    relevant_action_steps: tuple[int, ...],
    initial_block: str = "air",
    current_block: str = "obsidian",
    water_step: int | None = None,
    lava_step: int | None = None,
) -> ContinuousCastingCellTruth:
    water_step = last_action_step if water_step is None else water_step
    lava_step = (
        min(relevant_action_steps) if lava_step is None else lava_step
    )
    return ContinuousCastingCellTruth(
        target_cell=target_cell,
        initial_block=initial_block,
        current_block=current_block,
        water_truth=CastingFluidTruth(present=True, evidence_step=water_step),
        lava_truth=CastingFluidTruth(present=True, evidence_step=lava_step),
        transition_evidence=CastingTransitionEvidence(
            before_block=initial_block,
            after_block=current_block,
            update_step=last_action_step,
        ),
        relevant_action_steps=relevant_action_steps,
    )


def _state(
    *,
    cells: tuple[ContinuousCastingCellTruth, ...],
    step_id: int,
    terminated_step: int | None = None,
    terminated_reason: str | None = "driver_done",
    episode_terminated: bool = True,
    current_time_seconds: float = 0.0,
    max_environment_steps: int = 240,
    max_game_time_seconds: float = 180.0,
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW_STEPS,
) -> ContinuousCastingEvaluationState:
    if not episode_terminated:
        terminated_step = None
        terminated_reason = None
    elif terminated_step is None:
        terminated_step = step_id
    # Most focused evaluator tests supply only the cell under test.
    # The production state is fixed to exactly three ordered targets,
    # so fill the remaining contract cells with independent success
    # evidence. Step-zero backend lifecycle tests use empty truth
    # because they never evaluate the injected state.
    by_target = {cell.target_cell: cell for cell in cells}
    if set(by_target).issubset(set(TARGET_CELLS)) and len(by_target) < 3:
        used_steps = {
            action_step
            for cell in by_target.values()
            for action_step in cell.relevant_action_steps
        }
        available_steps = [
            candidate
            for candidate in range(step_id + 1)
            if candidate not in used_steps
        ]
        for target_cell in TARGET_CELLS:
            if target_cell in by_target:
                continue
            if available_steps:
                action_step = available_steps.pop(0)
                by_target[target_cell] = _success_cell(
                    target_cell,
                    last_action_step=action_step,
                    relevant_action_steps=(action_step,),
                )
            else:
                by_target[target_cell] = ContinuousCastingCellTruth(
                    target_cell=target_cell,
                    initial_block="air",
                    current_block=None,
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
        cells = tuple(by_target[target_cell] for target_cell in TARGET_CELLS)
    return ContinuousCastingEvaluationState(
        episode_id=EPISODE_ID,
        step_id=step_id,
        cells=cells,
        agent_id=AGENT_ID,
        causality_window_steps=causality_window_steps,
        episode_terminated=episode_terminated,
        terminated_step=terminated_step,
        terminated_reason=terminated_reason,
        current_time_seconds=current_time_seconds,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
    )


# ----------------------------------------------------------------------
# Outcome constant contract
# ----------------------------------------------------------------------


class OutcomeContractTests(unittest.TestCase):
    def test_outcome_constants_are_unique(self) -> None:
        self.assertEqual(
            len(CONTINUOUS_OUTCOMES),
            len(set(CONTINUOUS_OUTCOMES)),
        )

    def test_outcome_constants_cover_required_set(self) -> None:
        required = {
            OUTCOME_SUCCESS,
            OUTCOME_IN_PROGRESS,
            OUTCOME_PARTIAL_COMPLEMENT,
            OUTCOME_WRONG_BLOCK,
            OUTCOME_TRUTH_MISSING,
            OUTCOME_STEP_BUDGET_EXCEEDED,
            OUTCOME_TIME_BUDGET_EXCEEDED,
            OUTCOME_INVALID_INITIAL_STATE,
            OUTCOME_CAUSALITY_MISSING,
            OUTCOME_ABNORMAL_TERMINATION,
        }
        self.assertTrue(required.issubset(CONTINUOUS_OUTCOMES))

    def test_partial_completion_id_is_stable(self) -> None:
        self.assertEqual(OUTCOME_PARTIAL_COMPLEMENT, "partial_completion")


# ----------------------------------------------------------------------
# State and result immutability
# ----------------------------------------------------------------------


class StateImmutabilityTests(unittest.TestCase):
    def test_cell_truth_is_frozen(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cell.current_block = "air"  # type: ignore[misc]

    def test_state_is_frozen(self) -> None:
        cells = (
            _success_cell(
                TARGET_CELLS[0],
                last_action_step=24,
                relevant_action_steps=(3, 5, 9, 16),
            ),
        )
        state = _state(cells=cells, step_id=24, terminated_step=24)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.step_id = 25  # type: ignore[misc]

    def test_state_rejects_empty_cells(self) -> None:
        from dataclasses import dataclass as _dc

        @_dc(frozen=True)
        class _Holder:
            value: Any

        with self.assertRaisesRegex(ValueError, "non-empty tuple"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
                cells=(),
            )

    def test_state_rejects_duplicate_target_cells(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "duplicate target_cell"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=(cell, cell, cell),
            )

    def test_state_rejects_fewer_than_three_cells(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "requires exactly 3 cells"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=(cell,),
            )

    def test_state_rejects_wrong_target_order(self) -> None:
        cells = (
            _success_cell(TARGET_CELLS[1], last_action_step=2, relevant_action_steps=(2,)),
            _success_cell(TARGET_CELLS[0], last_action_step=3, relevant_action_steps=(3,)),
            _success_cell(TARGET_CELLS[2], last_action_step=4, relevant_action_steps=(4,)),
        )
        with self.assertRaisesRegex(ValueError, "frozen casting_c3_fixed target order"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=cells,
            )

    def test_state_rejects_future_relevant_step(self) -> None:
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=4
            ),
            relevant_action_steps=(10,),
        )
        with self.assertRaisesRegex(ValueError, "future step"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=5,
                cells=(cell,),
            )

    def test_state_rejects_non_cell_truth(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ContinuousCastingCellTruth"
        ):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
                cells=("not a cell",),  # type: ignore[arg-type]
            )

    def test_state_rejects_too_many_cells(self) -> None:
        # Build MAX_CELL_COUNT + 1 fake cells (with different target
        # coordinates) to trip the upper bound.
        from obsidianlink.evaluation.continuous_casting import MAX_CELL_COUNT

        cells = tuple(
            ContinuousCastingCellTruth(
                target_cell=(0, 0, index),
                initial_block="air",
                current_block="air",
                water_truth=None,
                lava_truth=None,
                transition_evidence=None,
                relevant_action_steps=(),
            )
            for index in range(MAX_CELL_COUNT + 1)
        )
        with self.assertRaisesRegex(ValueError, "requires exactly 3 cells"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
                cells=cells,
            )

    def test_state_rejects_zero_causality_window(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "causality_window_steps"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=(cell,),
                causality_window_steps=0,
            )

    def test_state_rejects_terminated_step_in_future(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=5,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "terminated_step"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=5,
                cells=(cell,),
                episode_terminated=True,
                terminated_step=10,
                terminated_reason="driver_done",
            )

    def test_state_rejects_terminated_reason_without_terminated(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=5,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "episode_terminated=True"):
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=5,
                cells=(cell,),
                episode_terminated=False,
                terminated_reason="driver_done",
            )

    def test_state_rejects_nan_time(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=5,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            _state(
                cells=(cell,),
                step_id=5,
                current_time_seconds=float("nan"),
            )

    def test_state_rejects_non_positive_budgets(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=5,
            relevant_action_steps=(3,),
        )
        with self.assertRaisesRegex(ValueError, "max_environment_steps"):
            _state(
                cells=(cell,),
                step_id=5,
                max_environment_steps=0,
                episode_terminated=False,
            )
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            _state(
                cells=(cell,),
                step_id=5,
                max_game_time_seconds=0.0,
                episode_terminated=False,
            )

    def test_result_is_frozen(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        result = ContinuousCastingEvaluator().evaluate(state)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_result_as_dict_is_json_serializable(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        result = ContinuousCastingEvaluator().evaluate(state)
        snapshot = result.as_dict()
        encoded = json.dumps(snapshot, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["outcome"], result.outcome)

    def test_result_as_dict_is_detached(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        result = ContinuousCastingEvaluator().evaluate(state)
        snapshot = result.as_dict()
        snapshot["evidence"]["mutated"] = True
        snapshot["per_cell_outcomes"].append("tampered")
        # The result's evidence tree is read-only and was not
        # mutated by the snapshot manipulation.
        self.assertFalse(result.evidence.get("mutated", False))


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_evaluate_is_deterministic(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3,),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        first = ContinuousCastingEvaluator().evaluate(state)
        second = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(first, second)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_priority_is_stable_for_same_input(self) -> None:
        # A single mixed-failure state (one truth_missing, one
        # wrong_block) must always return truth_missing regardless
        # of how the cells are iterated.
        truth_missing_cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=None,  # type: ignore[arg-type]
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air",
                after_block="obsidian",
                update_step=5,
            ),
            relevant_action_steps=(4,),
        )
        wrong_block_cell = _success_cell(
            TARGET_CELLS[1],
            last_action_step=8,
            relevant_action_steps=(6,),
            current_block="cobblestone",
        )
        state = _state(
            cells=(truth_missing_cell, wrong_block_cell),
            step_id=10,
            terminated_step=10,
        )
        first = ContinuousCastingEvaluator().evaluate(state)
        second = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(first.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(second.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(first, second)


# ----------------------------------------------------------------------
# Success path
# ----------------------------------------------------------------------


class SuccessPathTests(unittest.TestCase):
    def test_three_cell_success(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertTrue(result.success)
        self.assertEqual(result.completed_cells, 3)
        self.assertEqual(result.total_cells, 3)
        self.assertEqual(result.first_failed_cell, None)
        self.assertEqual(result.per_cell_outcomes, (PER_CELL_SUCCESS,) * 3)
        self.assertEqual(result.failure_type, None)
        self.assertEqual(result.blocking_conditions, ())

    def test_causality_delta_zero_is_success(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        # The per-cell evidence has causality_delta_steps = 0.
        self.assertEqual(
            result.evidence["per_cell_evidence"][0]["causality_delta_steps"], 0
        )


# ----------------------------------------------------------------------
# Partial completion / mixed outcomes
# ----------------------------------------------------------------------


class PartialCompletionTests(unittest.TestCase):
    def test_only_first_cell_succeeds(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        # cell 1 ended on cobblestone (truth present, all in budget)
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
            current_block="cobblestone",
        )
        # The successful cells form the ordered prefix [0], so this
        # is a genuine partial completion rather than a hole.
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
            current_block="air",
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 1)
        self.assertEqual(result.first_failed_cell, 1)
        self.assertEqual(
            result.per_cell_outcomes,
            (PER_CELL_SUCCESS, PER_CELL_WRONG_BLOCK, PER_CELL_WRONG_BLOCK),
        )

    def test_only_first_cell_succeeds_others_truth_missing(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=30),
            # lava truth missing on purpose
            lava_truth=None,  # type: ignore[arg-type]
            transition_evidence=CastingTransitionEvidence(
                before_block="air",
                after_block="obsidian",
                update_step=32,
            ),
            relevant_action_steps=(27, 29, 33),
        )
        cell2 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[2],
            initial_block="air",
            current_block="obsidian",
            # no transition evidence
            water_truth=CastingFluidTruth(present=True, evidence_step=60),
            lava_truth=CastingFluidTruth(present=True, evidence_step=51),
            transition_evidence=None,
            relevant_action_steps=(51, 53, 57),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(result.completed_cells, 1)
        self.assertEqual(result.first_failed_cell, 1)
        self.assertEqual(
            result.per_cell_outcomes,
            (
                PER_CELL_SUCCESS,
                PER_CELL_TRUTH_MISSING,
                PER_CELL_TRUTH_MISSING,
            ),
        )

    def test_partial_completion_only_when_no_cell_issue(self) -> None:
        # cell 0 succeeds; cells 1 and 2 are not_evaluated because
        # they were never marked obsidian (the orchestrator did not
        # report any transition / current_block truth for them).
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        # not_evaluated cells: current_block missing => not_evaluated
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block=None,  # type: ignore[arg-type]
            water_truth=CastingFluidTruth(present=True, evidence_step=30),
            lava_truth=CastingFluidTruth(present=True, evidence_step=27),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=33
            ),
            relevant_action_steps=(27, 29, 33),
        )
        cell2 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[2],
            initial_block="air",
            current_block=None,  # type: ignore[arg-type]
            water_truth=CastingFluidTruth(present=True, evidence_step=60),
            lava_truth=CastingFluidTruth(present=True, evidence_step=51),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=57
            ),
            relevant_action_steps=(51, 53, 57),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        # cell 1 / 2 are missing truth, so truth_missing outranks
        # partial_completion.
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_partial_completion_with_truth_complete_remaining_cells(self) -> None:
        # cell 0 success, cell 1 wrong_block, cell 2 wrong_block
        # => the successful prefix is complete and the remaining
        # cells have truth-complete wrong-block verdicts.
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
            current_block="cobblestone",
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
            current_block="stone",
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 1)
        self.assertEqual(
            result.per_cell_outcomes,
            (PER_CELL_SUCCESS, PER_CELL_WRONG_BLOCK, PER_CELL_WRONG_BLOCK),
        )

    def test_partial_completion_is_reported_when_truth_intact(self) -> None:
        # All three cells are per-cell SUCCESS, so the overall
        # outcome is success. This re-verifies the contract.
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)


# ----------------------------------------------------------------------
# Wrong block / causality / truth missing per cell
# ----------------------------------------------------------------------


class WrongBlockTests(unittest.TestCase):
    def test_middle_cell_wrong_block(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
            current_block="cobblestone",
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.first_failed_cell, 1)
        for condition in result.blocking_conditions:
            if "wrong_block" in condition:
                self.assertIn("cell_1", condition)
                self.assertIn("cobblestone", condition)


class TruthMissingTests(unittest.TestCase):
    def test_one_cell_missing_water(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block="obsidian",
            water_truth=None,  # missing
            lava_truth=CastingFluidTruth(present=True, evidence_step=30),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=33
            ),
            relevant_action_steps=(27, 29, 33),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(result.first_failed_cell, 1)
        self.assertTrue(
            any(
                "cell_1" in condition and "water_truth" in condition
                for condition in result.blocking_conditions
            )
        )

    def test_one_cell_missing_lava(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=30),
            lava_truth=None,  # missing
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=33
            ),
            relevant_action_steps=(27, 29, 33),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "cell_1" in condition and "lava_truth" in condition
                for condition in result.blocking_conditions
            )
        )

    def test_one_cell_missing_transition(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=30),
            lava_truth=CastingFluidTruth(present=True, evidence_step=27),
            transition_evidence=None,  # missing
            relevant_action_steps=(27, 29, 33),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            terminated_step=72,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "cell_1" in condition and "transition_evidence" in condition
                for condition in result.blocking_conditions
            )
        )

    def test_one_cell_uses_other_cell_relevant_actions(self) -> None:
        # The orchestrator / driver must never lift another cell's
        # relevant actions into this cell. Build cell 1 with
        # relevant_action_steps = (3, 5, 9, 16) (cell 0's steps)
        # and a transition that requires cell 0's actions.
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[1],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=48),
            lava_truth=CastingFluidTruth(present=True, evidence_step=27),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=48
            ),
            # "borrowed" from cell 0 — this is what the contract
            # forbids. The evaluator must still be able to compute
            # causality for cell 1 from its own steps alone.
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        with self.assertRaisesRegex(ValueError, "disjoint across cells"):
            _state(
                cells=(cell0, cell1, cell2),
                step_id=72,
                terminated_step=72,
            )


class CausalityTests(unittest.TestCase):
    def test_transition_before_action(self) -> None:
        # transition happened at step 5, but the relevant action
        # steps are all after step 5.
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=8),
            lava_truth=CastingFluidTruth(present=True, evidence_step=8),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(7, 8, 9, 10),
        )
        state = _state(cells=(cell,), step_id=10, terminated_step=10)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)

    def test_transition_outside_window(self) -> None:
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=20
            ),
            relevant_action_steps=(3, 5, 9, 10),
        )
        state = _state(cells=(cell,), step_id=20, terminated_step=20)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("outside_window" in c for c in result.blocking_conditions)
        )

    def test_water_not_present(self) -> None:
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=False, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(3, 4, 5, 6),
        )
        state = _state(cells=(cell,), step_id=6, terminated_step=6)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any(
                "water_not_present" in c
                for c in result.blocking_conditions
            )
        )

    def test_lava_not_present(self) -> None:
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=False, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(3, 4, 5, 6),
        )
        state = _state(cells=(cell,), step_id=6, terminated_step=6)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("lava_not_present" in c for c in result.blocking_conditions)
        )

    def test_transition_after_block_not_obsidian(self) -> None:
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="cobblestone", update_step=5
            ),
            relevant_action_steps=(3, 4, 5, 6),
        )
        state = _state(cells=(cell,), step_id=6, terminated_step=6)
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)


# ----------------------------------------------------------------------
# Invalid initial state / in progress / abnormal termination
# ----------------------------------------------------------------------


class ResetStateTests(unittest.TestCase):
    def test_cell_starts_as_obsidian(self) -> None:
        cell0 = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="obsidian",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=10),
            lava_truth=CastingFluidTruth(present=True, evidence_step=5),
            transition_evidence=CastingTransitionEvidence(
                before_block="obsidian", after_block="obsidian", update_step=10
            ),
            relevant_action_steps=(3, 5, 9, 10),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=20,
            relevant_action_steps=(15, 17, 19, 20),
        )
        state = _state(
            cells=(cell0, cell1),
            step_id=20,
            terminated_step=20,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INVALID_INITIAL_STATE)
        self.assertEqual(result.first_failed_cell, 0)


class InProgressTests(unittest.TestCase):
    def test_episode_not_terminated(self) -> None:
        cell0 = _success_cell(
            TARGET_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            TARGET_CELLS[1],
            last_action_step=44,
            relevant_action_steps=(27, 29, 33, 40),
        )
        cell2 = _success_cell(
            TARGET_CELLS[2],
            last_action_step=68,
            relevant_action_steps=(51, 53, 57, 64),
        )
        state = _state(
            cells=(cell0, cell1, cell2),
            step_id=72,
            episode_terminated=False,
            terminated_step=None,
            terminated_reason=None,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)
        self.assertFalse(result.success)
        self.assertEqual(result.per_cell_outcomes, (PER_CELL_NOT_EVALUATED,) * 3)


class AbnormalTerminationTests(unittest.TestCase):
    def test_abnormal_termination_reason(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        state = _state(
            cells=(cell,),
            step_id=10,
            terminated_step=10,
            terminated_reason="crashed",
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_ABNORMAL_TERMINATION)
        self.assertEqual(result.failure_type, OUTCOME_ABNORMAL_TERMINATION)
        self.assertEqual(
            result.blocking_conditions, ("abnormal_termination",)
        )

    def test_normal_termination_reason_routes_to_per_cell(self) -> None:
        # "driver_done" is in NORMAL_TERMINATION_REASONS so the
        # evaluator falls through to the per-cell checks.
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        state = _state(
            cells=(cell,),
            step_id=10,
            terminated_step=10,
            terminated_reason="driver_done",
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)


# ----------------------------------------------------------------------
# Budget outcomes
# ----------------------------------------------------------------------


class BudgetOutcomeTests(unittest.TestCase):
    def test_step_budget_exceeded(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        # step_id (11) > max_environment_steps (10) => budget
        # exceeded.
        state = _state(
            cells=(cell,),
            step_id=11,
            terminated_step=11,
            max_environment_steps=10,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)
        self.assertIn("step_budget_exceeded", result.blocking_conditions)

    def test_time_budget_exceeded(self) -> None:
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        state = _state(
            cells=(cell,),
            step_id=10,
            terminated_step=10,
            current_time_seconds=200.0,
            max_game_time_seconds=180.0,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TIME_BUDGET_EXCEEDED)
        self.assertIn("time_budget_exceeded", result.blocking_conditions)


# ----------------------------------------------------------------------
# FakeBackend surface
# ----------------------------------------------------------------------


class FakeBackendContinuousCastingStateTests(unittest.TestCase):
    """The new surface mirrors the R3 single-cell surface."""

    def _reset(self) -> tuple[FakeEnvironmentBackend, TaskInstance]:
        backend = FakeEnvironmentBackend()
        backend.open()
        task = _task()
        backend.reset(task)
        return backend, task

    def test_set_then_get(self) -> None:
        backend, task = self._reset()
        # Advance the backend to step 5 so we can inject a state
        # with step_id=5.
        from obsidianlink.core.types import MacroAction

        for _ in range(5):
            backend.step({"agent_1": MacroAction.wait()})
        cell = _success_cell(
            TARGET_CELLS[0],
            last_action_step=5,
            relevant_action_steps=(3, 4, 5),
        )
        state = _state(cells=(cell,), step_id=5, terminated_step=5)
        backend.set_continuous_casting_evaluation_state(state)
        out = backend.get_continuous_casting_evaluation_state()
        self.assertEqual(out, state)

    def test_get_without_injection_raises(self) -> None:
        backend, _ = self._reset()
        with self.assertRaisesRegex(
            RuntimeError, "continuous casting evaluation state is unavailable"
        ):
            backend.get_continuous_casting_evaluation_state()

    def test_wrong_episode_id_rejected(self) -> None:
        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        bad = dataclasses.replace(
            _state(cells=(cell,), step_id=0, terminated_step=0),
            episode_id="other_episode",
        )
        with self.assertRaisesRegex(ValueError, "episode_id must match"):
            backend.set_continuous_casting_evaluation_state(bad)

    def test_wrong_step_id_rejected(self) -> None:
        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        bad = _state(cells=(cell,), step_id=2, terminated_step=2)
        with self.assertRaisesRegex(ValueError, "step_id must match"):
            backend.set_continuous_casting_evaluation_state(bad)

    def test_non_state_type_rejected(self) -> None:
        backend, _ = self._reset()
        with self.assertRaisesRegex(TypeError, "must be a ContinuousCastingEvaluationState"):
            backend.set_continuous_casting_evaluation_state("not a state")  # type: ignore[arg-type]

    def test_step_clears_state(self) -> None:
        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        state = _state(cells=(cell,), step_id=0, terminated_step=0)
        backend.set_continuous_casting_evaluation_state(state)
        from obsidianlink.core.types import MacroAction
        backend.step({"agent_1": MacroAction.wait()})
        with self.assertRaisesRegex(
            RuntimeError, "continuous casting evaluation state is unavailable"
        ):
            backend.get_continuous_casting_evaluation_state()

    def test_close_clears_state(self) -> None:
        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        state = _state(cells=(cell,), step_id=0, terminated_step=0)
        backend.set_continuous_casting_evaluation_state(state)
        backend.close()
        # Re-open and try a new reset. The new reset must not see
        # the previously injected state.
        backend2 = FakeEnvironmentBackend()
        backend2.open()
        try:
            backend2.reset(_task())
            with self.assertRaisesRegex(
                RuntimeError, "continuous casting evaluation state is unavailable"
            ):
                backend2.get_continuous_casting_evaluation_state()
        finally:
            backend2.close()

    def test_observation_does_not_leak_continuous_casting_truth(self) -> None:
        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        state = _state(cells=(cell,), step_id=0, terminated_step=0)
        backend.set_continuous_casting_evaluation_state(state)
        observation = backend.reset(_task())[AGENT_ID]
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
        for field in forbidden:
            self.assertFalse(
                hasattr(observation, field),
                f"observation exposes {field!r}",
            )
        if isinstance(observation.frame, Mapping):
            for field in forbidden:
                self.assertNotIn(
                    field,
                    observation.frame,
                    f"observation.frame carries {field!r}",
                )

    def test_step_observation_does_not_leak_continuous_casting_truth(
        self,
    ) -> None:
        from obsidianlink.core.types import MacroAction

        backend, _ = self._reset()
        cell = ContinuousCastingCellTruth(
            target_cell=TARGET_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        state = _state(cells=(cell,), step_id=0, terminated_step=0)
        backend.set_continuous_casting_evaluation_state(state)
        step = backend.step({"agent_1": MacroAction.wait()})
        observation = step.observations[AGENT_ID]
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
        for field in forbidden:
            self.assertFalse(
                hasattr(observation, field),
                f"step observation exposes {field!r}",
            )
        if isinstance(observation.frame, Mapping):
            for field in forbidden:
                self.assertNotIn(
                    field,
                    observation.frame,
                    f"step observation.frame carries {field!r}",
                )


# ----------------------------------------------------------------------
# Isolation: the evaluator must not import driver / planner / model
# surfaces
# ----------------------------------------------------------------------


class EvaluatorIsolationTests(unittest.TestCase):
    def test_evaluator_source_does_not_import_drivers_or_workflows(self) -> None:
        import obsidianlink.evaluation.continuous_casting as module

        with open(module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        forbidden_substrings = [
            "obsidianlink.agents",
            "obsidianlink.workflows",
            "obsidianlink.drivers",
            "MacroAction",
            "VLM",
            "vlm",
            "Qwen",
        ]
        for forbidden in forbidden_substrings:
            self.assertNotIn(
                forbidden,
                source,
                f"evaluator source must not reference {forbidden!r}",
            )

    def test_evaluator_signature_only_accepts_state(self) -> None:
        import typing

        hints = typing.get_type_hints(ContinuousCastingEvaluator.evaluate)
        self.assertEqual(
            hints,
            {
                "state": ContinuousCastingEvaluationState,
                "return": ContinuousCastingEvaluationResult,
            },
        )


if __name__ == "__main__":
    unittest.main()
