"""Offline tests for the R6 Casting-S-C3 frozen-frame evaluator.

These tests prove, in code, that:

* :class:`FrozenFrameCellTruth`,
  :class:`FrozenFrameInteriorCellTruth`,
  :class:`FrozenFrameEvaluationState`, and
  :class:`FrozenFrameEvaluationResult` are frozen, type-strict, and
  JSON-serializable, with a recursive frozen evidence tree.
* :class:`FrozenFrameEvaluator` returns the same result for the same
  state on repeated calls, never reads Agent text / images / Planner
  input, and never imports the driver / planner / workflow / Agent
  / model surface.
* The closed outcome set and the locked priority order are honoured
  for every required scenario: 14-cell success, vanilla 10-cell
  frame with missing corners, 1-13 cell partial completion, baseline
  pre-existing obsidian, wrong block, interior blocker, missing
  water / lava / transition truth, transition outside the causality
  window, action attribution to the wrong Agent, episode_id /
  step_id inconsistency, step / time budget, abnormal termination,
  deterministic replay, state / result immutability, JSON
  serialization, FakeBackend set / get / clear, observation
  non-leakage, and task-origin / grid-origin coordinate conversion
  (success, out-of-bounds, missing origin, type error).
* The :class:`FakeEnvironmentBackend` exposes a
  ``set_frame_evaluation_state`` / ``get_frame_evaluation_state``
  / ``clear_frame_evaluation_state`` surface that is identity-guarded
  and that does not leak into :class:`Observation`.
* The C1 / C2 / portal evaluator surfaces keep working.

The tests never start Minecraft, MineRL, or Gradle.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from pathlib import Path
from typing import Any, Mapping

from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation import (
    CASTING_S_C3_CORNER_CELL_COUNT,
    CASTING_S_C3_CORNER_CELLS,
    CASTING_S_C3_FRAME_CELLS,
    CASTING_S_C3_INTERIOR_CELL_COUNT,
    CASTING_S_C3_INTERIOR_CELLS,
    CASTING_S_C3_REQUIRED_CELL_COUNT,
    CASTING_S_C3_REQUIRED_CELLS,
    CASTING_S_C3_TARGET_CELL_COUNT,
    FRAME_OUTCOMES,
    FrozenFrameActionEvidence,
    FrozenFrameCellTruth,
    FrozenFrameEvaluationResult,
    FrozenFrameEvaluationState,
    FrozenFrameEvaluator,
    FrozenFrameInteriorCellTruth,
    FrozenFrameOriginAnchor,
    OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_CAUSALITY_MISSING,
    OUTCOME_IN_PROGRESS,
    OUTCOME_INTERIOR_BLOCKED,
    OUTCOME_INVALID_INITIAL_STATE,
    OUTCOME_PARTIAL_COMPLEMENT,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    PER_CELL_CAUSALITY_MISSING,
    PER_CELL_INCOMPLETE,
    PER_CELL_NOT_EVALUATED,
    PER_CELL_SUCCESS,
    PER_CELL_TRUTH_MISSING,
    PER_CELL_WRONG_BLOCK,
    PER_INTERIOR_CELL_ALLOWED,
    PER_INTERIOR_CELL_BLOCKED,
    PER_INTERIOR_CELL_NOT_EVALUATED,
    PER_INTERIOR_CELL_TRUTH_MISSING,
    default_c3_anchor,
)
from obsidianlink.evaluation.casting import (
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
)
from obsidianlink.evaluation.continuous_casting import (
    ContinuousCastingEvaluationState,
    ContinuousCastingEvaluator,
)
from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator


EPISODE_ID = "casting_s_c3_fixed_seed_0"
AGENT_ID = "agent_1"
WRONG_AGENT_ID = "agent_2"

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE = (
    ROOT / "obsidianlink/evaluation/casting_frame_evaluator.py"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _task() -> TaskInstance:
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": EPISODE_ID,
            "route": "lava_casting",
            "difficulty": 3,
            "agent_ids": [AGENT_ID],
            "world_seed": 0,
            "instruction": (
                "Casting-S-C3 frozen-frame evaluator unit-test task."
            ),
            "spawn_positions": {AGENT_ID: [0, 4, 0]},
            "initial_inventories": {
                AGENT_ID: {
                    "water_bucket": 14,
                    "lava_bucket": 14,
                    "cobblestone": 28,
                }
            },
            "workflow": "casting_s_c3_fixed",
            "milestones": [
                "task_reset",
                "first_obsidian_cast",
                "build_site_selected",
                "valid_portal_frame",
            ],
            "limits": {
                "max_environment_steps": 640,
                "max_model_calls": 1,
                "max_game_time_seconds": 600,
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
) -> FrozenFrameCellTruth:
    water_step = last_action_step if water_step is None else water_step
    if lava_step is None:
        if relevant_action_steps:
            lava_step = min(relevant_action_steps)
        else:
            lava_step = last_action_step
    return FrozenFrameCellTruth(
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
        action_evidence=tuple(
            FrozenFrameActionEvidence(
                episode_id=EPISODE_ID,
                step_id=step,
                agent_id=AGENT_ID,
                action_type="use_item",
                item=("water_bucket" if index % 2 == 0 else "lava_bucket"),
                target_cell=target_cell,
            )
            for index, step in enumerate(relevant_action_steps)
        ),
        transition_action_step=(
            last_action_step if relevant_action_steps else None
        ),
    )


def _actions(
    target_cell: tuple[int, int, int],
    steps: tuple[int, ...],
    *,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
) -> tuple[FrozenFrameActionEvidence, ...]:
    return tuple(
        FrozenFrameActionEvidence(
            episode_id=episode_id,
            step_id=step,
            agent_id=agent_id,
            action_type="use_item",
            item="water_bucket" if index % 2 == 0 else "lava_bucket",
            target_cell=target_cell,
        )
        for index, step in enumerate(steps)
    )


def _all_success_cells(
    *,
    step_budget: int = 640,
    start_step: int = 4,
) -> tuple[FrozenFrameCellTruth, ...]:
    """Build 14 success cells in the canonical order, disjoint steps.

    Each cell consumes 4 disjoint steps (per-cell relevant actions).
    The 14th cell's last step is ``start_step + 14*4 - 1``.
    """
    step_cursor = start_step
    cells: list[FrozenFrameCellTruth] = []
    for index, target_cell in enumerate(CASTING_S_C3_FRAME_CELLS):
        last_action = step_cursor + 3
        steps = (step_cursor, step_cursor + 1, step_cursor + 2, last_action)
        cells.append(
            _success_cell(
                target_cell,
                last_action_step=last_action,
                relevant_action_steps=steps,
            )
        )
        step_cursor = last_action + 1
    return tuple(cells)


def _all_allowed_interior(
    *, block: str | None = "air"
) -> tuple[FrozenFrameInteriorCellTruth, ...]:
    return tuple(
        FrozenFrameInteriorCellTruth(
            target_cell=cell, current_block=block
        )
        for cell in CASTING_S_C3_INTERIOR_CELLS
    )


_AUTO = object()


def _state(
    *,
    cells: tuple[FrozenFrameCellTruth, ...] | None = None,
    interior_cells: tuple[FrozenFrameInteriorCellTruth, ...] | None = None,
    step_id: int = 100,
    terminated_step: int | None | object = _AUTO,
    terminated_reason: str | None = "driver_done",
    episode_terminated: bool = True,
    current_time_seconds: float = 0.0,
    max_environment_steps: int = 640,
    max_game_time_seconds: float = 600.0,
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW_STEPS,
    agent_id: str | None = AGENT_ID,
) -> FrozenFrameEvaluationState:
    if cells is None:
        cells = _all_success_cells(step_budget=max_environment_steps)
    if interior_cells is None:
        interior_cells = _all_allowed_interior()
    if not episode_terminated:
        terminated_step = None
        terminated_reason = None
    else:
        if terminated_step is _AUTO or terminated_step is None:
            terminated_step = step_id
    return FrozenFrameEvaluationState(
        episode_id=EPISODE_ID,
        step_id=step_id,
        cells=cells,
        interior_cells=interior_cells,
        agent_id=agent_id,
        causality_window_steps=causality_window_steps,
        episode_terminated=episode_terminated,
        terminated_step=terminated_step,
        terminated_reason=terminated_reason,
        current_time_seconds=current_time_seconds,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
    )


# ----------------------------------------------------------------------
# Outcome / state / result contract
# ----------------------------------------------------------------------


class OutcomeContractTests(unittest.TestCase):
    def test_outcome_constants_are_unique(self) -> None:
        self.assertEqual(len(FRAME_OUTCOMES), len(set(FRAME_OUTCOMES)))

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
        self.assertTrue(required.issubset(FRAME_OUTCOMES))

    def test_interior_blocked_id_is_stable(self) -> None:
        self.assertEqual(OUTCOME_INTERIOR_BLOCKED, "interior_blocked")
        self.assertIn(OUTCOME_INTERIOR_BLOCKED, FRAME_OUTCOMES)

    def test_partial_completion_id_is_stable(self) -> None:
        self.assertEqual(OUTCOME_PARTIAL_COMPLEMENT, "partial_completion")

    def test_target_cells_match_contract_offsets(self) -> None:
        expected = {
            (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1),
            (0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1),
            (0, 1, 1), (0, 2, 1), (0, 3, 1),
            (3, 1, 1), (3, 2, 1), (3, 3, 1),
        }
        self.assertEqual(set(CASTING_S_C3_FRAME_CELLS), expected)
        self.assertEqual(
            len(CASTING_S_C3_FRAME_CELLS), CASTING_S_C3_TARGET_CELL_COUNT
        )
        self.assertEqual(len(CASTING_S_C3_REQUIRED_CELLS), 10)
        self.assertEqual(CASTING_S_C3_REQUIRED_CELL_COUNT, 10)
        self.assertEqual(len(CASTING_S_C3_CORNER_CELLS), 4)
        self.assertEqual(CASTING_S_C3_CORNER_CELL_COUNT, 4)
        self.assertEqual(len(CASTING_S_C3_INTERIOR_CELLS), 6)
        self.assertEqual(CASTING_S_C3_INTERIOR_CELL_COUNT, 6)
        # Required cells = full ring \ corner cells.
        self.assertEqual(
            set(CASTING_S_C3_REQUIRED_CELLS),
            set(CASTING_S_C3_FRAME_CELLS) - set(CASTING_S_C3_CORNER_CELLS),
        )


class StateImmutabilityTests(unittest.TestCase):
    def test_action_evidence_rejects_non_casting_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_type"):
            FrozenFrameActionEvidence(
                episode_id=EPISODE_ID,
                step_id=1,
                agent_id=AGENT_ID,
                action_type="place_block",
                item="water_bucket",
                target_cell=CASTING_S_C3_FRAME_CELLS[0],
            )

    def test_action_evidence_rejects_non_fluid_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "item"):
            FrozenFrameActionEvidence(
                episode_id=EPISODE_ID,
                step_id=1,
                agent_id=AGENT_ID,
                action_type="use_item",
                item="obsidian",
                target_cell=CASTING_S_C3_FRAME_CELLS[0],
            )

    def test_state_rejects_missing_agent_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            _state(agent_id=None)

    def test_state_rejects_action_from_wrong_episode(self) -> None:
        cell = _success_cell(
            CASTING_S_C3_FRAME_CELLS[0],
            last_action_step=4,
            relevant_action_steps=(4,),
        )
        object.__setattr__(
            cell,
            "action_evidence",
            _actions(
                cell.target_cell, (4,), episode_id="different_episode"
            ),
        )
        with self.assertRaisesRegex(ValueError, "episode_id"):
            _state(cells=(cell,) + _all_success_cells()[1:])

    def test_state_rejects_action_from_wrong_agent(self) -> None:
        cell = _success_cell(
            CASTING_S_C3_FRAME_CELLS[0],
            last_action_step=4,
            relevant_action_steps=(4,),
        )
        object.__setattr__(
            cell,
            "action_evidence",
            _actions(cell.target_cell, (4,), agent_id=WRONG_AGENT_ID),
        )
        with self.assertRaisesRegex(ValueError, "agent_id"):
            _state(cells=(cell,) + _all_success_cells()[1:])

    def test_state_is_frozen(self) -> None:
        state = _state()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.step_id = 101  # type: ignore[misc]

    def test_cell_truth_is_frozen(self) -> None:
        cell = _success_cell(
            CASTING_S_C3_FRAME_CELLS[0],
            last_action_step=10,
            relevant_action_steps=(3, 5, 9, 10),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cell.current_block = "air"  # type: ignore[misc]

    def test_state_rejects_non_string_episode_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            FrozenFrameEvaluationState(
                episode_id="",  # type: ignore[arg-type]
                step_id=0,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_wrong_target_cell_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires exactly 14 target cells"
        ):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=(),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_wrong_target_cell_order(self) -> None:
        cells = list(_all_success_cells())
        # Swap the first two cells: 0 and 1.
        cells[0], cells[1] = cells[1], cells[0]
        with self.assertRaisesRegex(
            ValueError, "frozen casting_s_c3_fixed frame order"
        ):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=100,
                cells=tuple(cells),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_wrong_interior_cell_order(self) -> None:
        interior = list(_all_allowed_interior())
        interior[0], interior[1] = interior[1], interior[0]
        with self.assertRaisesRegex(
            ValueError, "frozen casting_s_c3_fixed interior order"
        ):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=100,
                cells=_all_success_cells(),
                interior_cells=tuple(interior),
            )

    def test_state_rejects_duplicate_relevant_actions(self) -> None:
        cell0 = _success_cell(
            CASTING_S_C3_FRAME_CELLS[0],
            last_action_step=20,
            relevant_action_steps=(3, 5, 9, 16),
        )
        cell1 = _success_cell(
            CASTING_S_C3_FRAME_CELLS[1],
            last_action_step=24,
            relevant_action_steps=(3, 6, 10, 20),  # step 3 collides
        )
        with self.assertRaisesRegex(
            ValueError, "disjoint across target cells"
        ):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=24,
                cells=(cell0, cell1) + _all_success_cells()[2:],
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_future_relevant_action(self) -> None:
        # The state's __post_init__ rejects cells whose
        # relevant_action_steps include a step beyond ``step_id``.
        with self.assertRaisesRegex(ValueError, "future step"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=5,
                cells=(
                    _success_cell(
                        CASTING_S_C3_FRAME_CELLS[0],
                        last_action_step=8,
                        relevant_action_steps=(3, 8),
                    ),
                )
                + _all_success_cells(start_step=20)[1:],
                interior_cells=_all_allowed_interior(),
            )

    def test_cell_truth_rejects_non_int_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            FrozenFrameCellTruth(
                target_cell=CASTING_S_C3_FRAME_CELLS[0],
                initial_block="air",
                current_block="obsidian",
                water_truth=CastingFluidTruth(present=True, evidence_step=4),
                lava_truth=CastingFluidTruth(present=True, evidence_step=4),
                transition_evidence=CastingTransitionEvidence(
                    before_block="air", after_block="obsidian", update_step=4
                ),
                relevant_action_steps=("not", "ints"),  # type: ignore[arg-type]
            )

    def test_state_rejects_zero_causality_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "causality_window_steps"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=100,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
                causality_window_steps=0,
            )

    def test_state_rejects_terminated_step_in_future(self) -> None:
        cells = _all_success_cells()
        with self.assertRaisesRegex(ValueError, "terminated_step"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=100,
                cells=cells,
                interior_cells=_all_allowed_interior(),
                episode_terminated=True,
                terminated_step=200,
                terminated_reason="driver_done",
            )

    def test_state_rejects_terminated_reason_without_terminated(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_terminated=True"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=100,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
                episode_terminated=False,
                terminated_reason="driver_done",
            )

    def test_state_rejects_nan_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _state(current_time_seconds=float("nan"))

    def test_state_rejects_non_positive_budgets(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_environment_steps"):
            _state(max_environment_steps=0, episode_terminated=False)
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            _state(max_game_time_seconds=0.0, episode_terminated=False)

    def test_interior_cell_requires_string_or_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            FrozenFrameInteriorCellTruth(
                target_cell=CASTING_S_C3_INTERIOR_CELLS[0],
                current_block="",
            )


class ResultImmutabilityTests(unittest.TestCase):
    def test_result_rejects_success_outcome_inconsistency(self) -> None:
        result = FrozenFrameEvaluator().evaluate(_state())
        with self.assertRaisesRegex(ValueError, "success must equal"):
            dataclasses.replace(result, success=False)

    def test_result_is_frozen(self) -> None:
        state = _state()
        result = FrozenFrameEvaluator().evaluate(state)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_result_as_dict_is_json_serializable(self) -> None:
        state = _state()
        result = FrozenFrameEvaluator().evaluate(state)
        snapshot = result.as_dict()
        encoded = json.dumps(snapshot, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["outcome"], result.outcome)

    def test_result_as_dict_is_detached(self) -> None:
        state = _state()
        result = FrozenFrameEvaluator().evaluate(state)
        snapshot = result.as_dict()
        snapshot["evidence"]["mutated"] = True
        snapshot["per_cell_outcomes"].append("tampered")
        snapshot["interior_blocker_cells"].append([0, 0, 0])
        self.assertFalse(result.evidence.get("mutated", False))


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_evaluate_is_deterministic(self) -> None:
        state = _state()
        first = FrozenFrameEvaluator().evaluate(state)
        second = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(first, second)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_priority_is_stable_for_same_input(self) -> None:
        # Build a state with one cell truth_missing and one wrong_block.
        truth_missing = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
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
            action_evidence=_actions(CASTING_S_C3_FRAME_CELLS[0], (4,)),
            transition_action_step=4,
        )
        wrong_block = _success_cell(
            CASTING_S_C3_FRAME_CELLS[1],
            last_action_step=10,
            relevant_action_steps=(6,),
            current_block="cobblestone",
        )
        cells = (truth_missing, wrong_block) + _all_success_cells()[2:]
        state = _state(cells=cells)
        first = FrozenFrameEvaluator().evaluate(state)
        second = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(first.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(second.outcome, OUTCOME_TRUTH_MISSING)
        self.assertEqual(first, second)


# ----------------------------------------------------------------------
# Success path
# ----------------------------------------------------------------------


class SuccessPathTests(unittest.TestCase):
    def test_full_14_cell_success(self) -> None:
        state = _state()
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertTrue(result.success)
        self.assertEqual(result.completed_cells, 14)
        self.assertEqual(result.total_cells, 14)
        self.assertEqual(result.completed_corner_cells, 4)
        self.assertEqual(result.total_corner_cells, 4)
        self.assertEqual(result.completed_interior_cells, 6)
        self.assertEqual(result.total_interior_cells, 6)
        self.assertEqual(result.interior_blocker_cells, ())
        self.assertEqual(result.first_failed_cell, None)
        self.assertEqual(result.failure_type, None)
        self.assertEqual(result.blocking_conditions, ())
        for verdict in result.per_cell_outcomes:
            self.assertEqual(verdict, PER_CELL_SUCCESS)
        for verdict in result.per_interior_cell_outcomes:
            self.assertEqual(verdict, PER_INTERIOR_CELL_ALLOWED)


# ----------------------------------------------------------------------
# Corner requirement: 10-cell vanilla frame is partial, never success
# ----------------------------------------------------------------------


class CornerRequirementTests(unittest.TestCase):
    def test_vanilla_10_cell_frame_missing_corners_is_partial(self) -> None:
        # 10 required cells succeed and four corners remain air. C3
        # requires the full ring, so this is partial rather than success.
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if target in set(CASTING_S_C3_CORNER_CELLS):
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="air",
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
        ordered_cells = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        state = _state(cells=ordered_cells, step_id=4 * 14)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 10)
        self.assertEqual(result.completed_corner_cells, 0)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertNotIn(
            "missing_truth",
            "|".join(result.blocking_conditions),
        )

    def test_vanilla_10_cell_with_one_corner_is_partial(self) -> None:
        # 10 required + 1 corner = 11 completed cells; three corners
        # remain air. Completion order is intentionally irrelevant.
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if target in set(CASTING_S_C3_CORNER_CELLS) and target != CASTING_S_C3_CORNER_CELLS[0]:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="air",
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
        ordered = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        state = _state(cells=ordered, step_id=4 * 14)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 11)
        self.assertEqual(result.completed_corner_cells, 1)
        self.assertFalse(result.success)

    def test_14_cell_full_ring_is_success(self) -> None:
        # Sanity: all 14 cells succeeded (including the four corners)
        # = the only way to get success.
        cells = tuple(
            _success_cell(
                target,
                last_action_step=4 * index + 3,
                relevant_action_steps=(
                    4 * index, 4 * index + 1, 4 * index + 2, 4 * index + 3
                ),
            )
            for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
        )
        state = _state(cells=cells, step_id=4 * 14)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.completed_cells, 14)
        self.assertEqual(result.completed_corner_cells, 4)
        self.assertTrue(result.success)


# ----------------------------------------------------------------------
# Partial completion
# ----------------------------------------------------------------------


class PartialCompletionTests(unittest.TestCase):
    def test_non_prefix_single_completed_cell_is_partial(self) -> None:
        cells = []
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            cell = _success_cell(
                target,
                last_action_step=4 * index + 3,
                relevant_action_steps=(
                    4 * index,
                    4 * index + 1,
                    4 * index + 2,
                    4 * index + 3,
                ),
                current_block="obsidian" if index == 1 else "air",
            )
            cells.append(cell)
        result = FrozenFrameEvaluator().evaluate(
            _state(cells=tuple(cells), step_id=70)
        )
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 1)
        self.assertEqual(result.per_cell_outcomes[0], PER_CELL_INCOMPLETE)
        self.assertEqual(result.per_cell_outcomes[1], PER_CELL_SUCCESS)

    def test_only_first_five_cells_succeed(self) -> None:
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if index < 5:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="air",
                )
        ordered = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        state = _state(cells=ordered, step_id=last_action + 10)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 5)
        self.assertEqual(result.first_failed_cell, 5)

    def test_one_cell_succeeds(self) -> None:
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if index == 0:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="air",
                )
        ordered = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        state = _state(cells=ordered, step_id=last_action + 10)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
        self.assertEqual(result.completed_cells, 1)

    def test_blocking_wrong_block_among_completed_cells_is_not_partial(self) -> None:
        # Cells 0..4 and 6..13 succeed, while cell 5 contains
        # cobblestone. Order is irrelevant; the blocker makes this
        # wrong_block rather than partial_completion.
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if index == 5:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="cobblestone",
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
        ordered = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        state = _state(cells=ordered, step_id=last_action + 10)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.completed_cells, 13)
        self.assertEqual(result.first_failed_cell, 5)

    def test_no_cells_succeed_is_wrong_block(self) -> None:
        cells = tuple(
            _success_cell(
                target,
                last_action_step=4 * index + 3,
                relevant_action_steps=(
                    4 * index,
                    4 * index + 1,
                    4 * index + 2,
                    4 * index + 3,
                ),
                current_block="cobblestone",
            )
            for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
        )
        state = _state(cells=cells, step_id=4 * 14)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.completed_cells, 0)


# ----------------------------------------------------------------------
# Baseline / initial state
# ----------------------------------------------------------------------


class InvalidInitialStateTests(unittest.TestCase):
    def test_baseline_target_obsidian_is_invalid_initial_state(self) -> None:
        cells = list(_all_success_cells())
        # Mutate cell 3 (a corner) to have initial_block=obsidian.
        object.__setattr__(cells[3], "initial_block", "obsidian")
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INVALID_INITIAL_STATE)
        self.assertEqual(result.failure_type, OUTCOME_INVALID_INITIAL_STATE)
        self.assertEqual(result.first_failed_cell, 3)


# ----------------------------------------------------------------------
# Wrong block / causality / truth missing
# ----------------------------------------------------------------------


class WrongBlockTests(unittest.TestCase):
    def test_one_cell_wrong_block_in_middle(self) -> None:
        cells = list(_all_success_cells())
        # Cell 7 is (0,4,1), a corner; mutate it to cobblestone.
        object.__setattr__(cells[7], "current_block", "cobblestone")
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.first_failed_cell, 7)
        self.assertEqual(result.failure_type, OUTCOME_WRONG_BLOCK)


class InteriorBlockedTests(unittest.TestCase):
    def test_one_interior_dirt_is_interior_blocked(self) -> None:
        interior = list(_all_allowed_interior())
        object.__setattr__(interior[0], "current_block", "dirt")
        state = _state(interior_cells=tuple(interior))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INTERIOR_BLOCKED)
        self.assertEqual(result.failure_type, OUTCOME_INTERIOR_BLOCKED)
        self.assertEqual(
            result.interior_blocker_cells, (CASTING_S_C3_INTERIOR_CELLS[0],)
        )
        # Even though all 14 target cells succeeded, the frame is not
        # a valid full ring because the interior is blocked.
        self.assertFalse(result.success)

    def test_interior_obsidian_is_interior_blocked(self) -> None:
        interior = list(_all_allowed_interior())
        object.__setattr__(interior[2], "current_block", "obsidian")
        state = _state(interior_cells=tuple(interior))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INTERIOR_BLOCKED)

    def test_interior_blocked_outranks_partial_completion(self) -> None:
        # 5 cells succeed, 9 cells are wrong_block, but interior also
        # has a blocker. The verdict must be interior_blocked, not
        # partial_completion.
        cell_by_target: dict[tuple[int, int, int], FrozenFrameCellTruth] = {}
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            if index < 5:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
            else:
                cell_by_target[target] = _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                    current_block="cobblestone",
                )
        ordered = tuple(cell_by_target[c] for c in CASTING_S_C3_FRAME_CELLS)
        interior = list(_all_allowed_interior())
        object.__setattr__(interior[0], "current_block", "dirt")
        state = _state(cells=ordered, interior_cells=tuple(interior), step_id=4 * 14)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INTERIOR_BLOCKED)
        self.assertFalse(result.success)


class TruthMissingTests(unittest.TestCase):
    def test_one_cell_missing_water(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[5],
            "water_truth",
            None,  # type: ignore[arg-type]
        )
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "target_5" in c and "water_truth" in c
                for c in result.blocking_conditions
            )
        )
        self.assertEqual(result.failure_type, None)

    def test_one_cell_missing_lava(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[5],
            "lava_truth",
            None,  # type: ignore[arg-type]
        )
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "target_5" in c and "lava_truth" in c
                for c in result.blocking_conditions
            )
        )

    def test_one_cell_missing_transition(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[5],
            "transition_evidence",
            None,  # type: ignore[arg-type]
        )
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "target_5" in c and "transition_evidence" in c
                for c in result.blocking_conditions
            )
        )

    def test_one_cell_missing_current_block(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(cells[5], "current_block", None)  # type: ignore[arg-type]
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_one_cell_missing_relevant_actions(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(cells[5], "relevant_action_steps", ())
        object.__setattr__(cells[5], "action_evidence", ())
        state = _state(cells=tuple(cells))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_interior_missing_truth(self) -> None:
        interior = list(_all_allowed_interior())
        object.__setattr__(interior[3], "current_block", None)  # type: ignore[arg-type]
        state = _state(interior_cells=tuple(interior))
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any(
                "interior_3" in c
                for c in result.blocking_conditions
            )
        )
        self.assertEqual(
            result.evidence["interior_extras"]["missing_interior_cells"],
            (3,),
        )


class CausalityTests(unittest.TestCase):
    def test_transition_without_action_attribution_is_truth_missing(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(cells[0], "transition_action_step", None)
        result = FrozenFrameEvaluator().evaluate(_state(cells=tuple(cells)))
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertTrue(
            any("transition_action_step" in condition
                for condition in result.blocking_conditions)
        )

    def test_transition_attributed_to_unlisted_action_fails(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(cells[0], "transition_action_step", 3)
        result = FrozenFrameEvaluator().evaluate(_state(cells=tuple(cells)))
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("transition_not_attributed_to_action" in condition
                for condition in result.blocking_conditions)
        )

    def test_transition_cannot_start_as_obsidian(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[0],
            "transition_evidence",
            CastingTransitionEvidence(
                before_block="obsidian",
                after_block="obsidian",
                update_step=7,
            ),
        )
        result = FrozenFrameEvaluator().evaluate(_state(cells=tuple(cells)))
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("transition_started_as_obsidian" in condition
                for condition in result.blocking_conditions)
        )

    def test_transition_must_change_block(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[0],
            "transition_evidence",
            CastingTransitionEvidence(
                before_block="air", after_block="air", update_step=7
            ),
        )
        result = FrozenFrameEvaluator().evaluate(_state(cells=tuple(cells)))
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("transition_did_not_change_block" in condition
                for condition in result.blocking_conditions)
        )

    def test_fluid_evidence_cannot_follow_transition(self) -> None:
        cells = list(_all_success_cells())
        object.__setattr__(
            cells[0],
            "water_truth",
            CastingFluidTruth(present=True, evidence_step=8),
        )
        result = FrozenFrameEvaluator().evaluate(_state(cells=tuple(cells)))
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("fluid_evidence_after_transition" in condition
                for condition in result.blocking_conditions)
        )
    def test_transition_outside_window(self) -> None:
        cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=20
            ),
            relevant_action_steps=(3, 5, 9, 10),
            action_evidence=_actions(
                CASTING_S_C3_FRAME_CELLS[0], (3, 5, 9, 10)
            ),
            transition_action_step=10,
        )
        cells = (cell,) + _all_success_cells(start_step=24)[1:]
        state = _state(cells=cells)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any("outside_window" in c for c in result.blocking_conditions)
        )

    def test_transition_before_any_action(self) -> None:
        cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=8),
            lava_truth=CastingFluidTruth(present=True, evidence_step=8),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(7, 8, 9, 10),
            action_evidence=_actions(
                CASTING_S_C3_FRAME_CELLS[0], (7, 8, 9, 10)
            ),
            transition_action_step=10,
        )
        cells = (cell,) + _all_success_cells(start_step=20)[1:]
        state = _state(cells=cells)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)

    def test_water_not_present(self) -> None:
        cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=False, evidence_step=4),
            lava_truth=CastingFluidTruth(present=True, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(3, 4, 5, 6),
            action_evidence=_actions(
                CASTING_S_C3_FRAME_CELLS[0], (3, 4, 5, 6)
            ),
            transition_action_step=5,
        )
        cells = (cell,) + _all_success_cells(start_step=20)[1:]
        state = _state(cells=cells)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any(
                "water_not_present" in c
                for c in result.blocking_conditions
            )
        )

    def test_lava_not_present(self) -> None:
        cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=4),
            lava_truth=CastingFluidTruth(present=False, evidence_step=4),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=5
            ),
            relevant_action_steps=(3, 4, 5, 6),
            action_evidence=_actions(
                CASTING_S_C3_FRAME_CELLS[0], (3, 4, 5, 6)
            ),
            transition_action_step=5,
        )
        cells = (cell,) + _all_success_cells(start_step=20)[1:]
        state = _state(cells=cells)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertTrue(
            any(
                "lava_not_present" in c
                for c in result.blocking_conditions
            )
        )


# ----------------------------------------------------------------------
# Episode / step / agent id consistency
# ----------------------------------------------------------------------


class IdentityConsistencyTests(unittest.TestCase):
    def test_wrong_episode_id_in_state_fails(self) -> None:
        # The state accepts any non-empty string episode_id; the
        # wrong-episode-id rejection lives in the FakeBackend's
        # set_frame_evaluation_state. We cover that path in
        # FakeBackendSetGetClearTests.test_wrong_episode_rejected.
        # The state only validates the *type* of episode_id.
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            FrozenFrameEvaluationState(
                episode_id="",  # type: ignore[arg-type]
                step_id=200,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_negative_step_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_id"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=-1,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_bool_step_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_id"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=True,  # type: ignore[arg-type]
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
            )

    def test_state_rejects_empty_agent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=10,
                cells=_all_success_cells(),
                interior_cells=_all_allowed_interior(),
                agent_id="",
            )


# ----------------------------------------------------------------------
# Budget / abnormal termination
# ----------------------------------------------------------------------


class BudgetTests(unittest.TestCase):
    def test_terminated_episode_requires_reason_for_evaluation(self) -> None:
        state = _state(terminated_reason=None)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:terminated_reason", result.blocking_conditions
        )

    def test_step_budget_exceeded(self) -> None:
        state = _state(step_id=641, terminated_step=641)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)
        self.assertEqual(result.failure_type, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_time_budget_exceeded(self) -> None:
        state = _state(current_time_seconds=601.0)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TIME_BUDGET_EXCEEDED)
        self.assertEqual(result.failure_type, OUTCOME_TIME_BUDGET_EXCEEDED)

    def test_in_progress_when_not_terminated(self) -> None:
        state = _state(episode_terminated=False)
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)
        self.assertEqual(result.failure_type, None)

    def test_abnormal_termination(self) -> None:
        state = _state(terminated_reason="universe_collapsed")
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_ABNORMAL_TERMINATION)
        self.assertEqual(result.failure_type, OUTCOME_ABNORMAL_TERMINATION)

    def test_normal_termination_reasons_accepted(self) -> None:
        for reason in NORMAL_TERMINATION_REASONS:
            with self.subTest(reason=reason):
                state = _state(terminated_reason=reason)
                result = FrozenFrameEvaluator().evaluate(state)
                # All 14 cells succeed and interior is air => success
                # unless interior was set to None previously.
                self.assertEqual(result.outcome, OUTCOME_SUCCESS)


# ----------------------------------------------------------------------
# FakeBackend set / get / clear
# ----------------------------------------------------------------------


class FakeBackendSetGetClearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeEnvironmentBackend()
        self.backend.open()
        self.task = _task()
        self.backend.reset(self.task)

    def tearDown(self) -> None:
        self.backend.close()

    def _build_state(
        self, *, step_id: int, agent_id: str | None = AGENT_ID
    ) -> FrozenFrameEvaluationState:
        # Build a state whose cells fit within the requested step_id.
        # This means every cell's relevant_action_steps are <= step_id
        # and last_action_step <= step_id.
        if step_id == 0:
            cells = tuple(
                _success_cell(
                    target,
                    last_action_step=0,
                    relevant_action_steps=(0,) if index == 0 else (),
                )
                for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
            )
            return FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
                cells=cells,
                interior_cells=_all_allowed_interior(),
                agent_id=agent_id,
                causality_window_steps=4,
                episode_terminated=False,
                current_time_seconds=0.0,
                max_environment_steps=640,
                max_game_time_seconds=600.0,
            )
        return _state(step_id=step_id, agent_id=agent_id)

    def test_set_then_get_returns_same_state(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        self.assertEqual(self.backend.get_frame_evaluation_state(), state)

    def test_get_without_set_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "frame evaluation state is unavailable"):
            self.backend.get_frame_evaluation_state()

    def test_clear_drops_state(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        self.backend.clear_frame_evaluation_state()
        with self.assertRaisesRegex(RuntimeError, "frame evaluation state is unavailable"):
            self.backend.get_frame_evaluation_state()

    def test_reset_clears_state(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        self.backend.reset(self.task)
        with self.assertRaisesRegex(RuntimeError, "frame evaluation state is unavailable"):
            self.backend.get_frame_evaluation_state()

    def test_step_clears_state(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        actions = {
            AGENT_ID: MacroAction.wait(),
        }
        self.backend.step(actions)
        with self.assertRaisesRegex(RuntimeError, "frame evaluation state is unavailable"):
            self.backend.get_frame_evaluation_state()

    def test_close_clears_state(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        self.backend.close()
        # After close, the backend is no longer open; the only way
        # to test the cleared state is via ``clear_frame_evaluation_state``
        # being a no-op without raising.
        self.assertIsNone(self.backend._frame_evaluation_state)

    def test_wrong_episode_rejected(self) -> None:
        # Build a state with the wrong episode id.
        bad = self._build_state(step_id=0)
        object.__setattr__(bad, "episode_id", "casting_c1_fixed_seed_0")
        with self.assertRaisesRegex(ValueError, "episode_id must match"):
            self.backend.set_frame_evaluation_state(bad)

    def test_wrong_step_rejected(self) -> None:
        # The backend is at step_id=0 after reset. A state with
        # step_id=99 must be rejected. Build the state at step_id=99
        # with disjoint per-cell steps so it is otherwise valid.
        cells = []
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            last_action = 4 * index + 3
            steps = (4 * index, 4 * index + 1, 4 * index + 2, last_action)
            cells.append(
                _success_cell(
                    target,
                    last_action_step=last_action,
                    relevant_action_steps=steps,
                )
            )
        bad = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=4 * 14,
            cells=tuple(cells),
            interior_cells=_all_allowed_interior(),
        )
        with self.assertRaisesRegex(ValueError, "step_id must match"):
            self.backend.set_frame_evaluation_state(bad)

    def test_wrong_type_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "FrozenFrameEvaluationState"):
            self.backend.set_frame_evaluation_state("not a state")  # type: ignore[arg-type]

    def test_wrong_workflow_rejected(self) -> None:
        wrong_task = dataclasses.replace(
            self.task, workflow="casting_c1_fixed"
        )
        self.backend.reset(wrong_task)
        state = self._build_state(step_id=0)
        with self.assertRaisesRegex(ValueError, "casting_s_c3_fixed workflow"):
            self.backend.set_frame_evaluation_state(state)

    def test_wrong_agent_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            self._build_state(step_id=0, agent_id=WRONG_AGENT_ID)

    def test_set_after_step_advance_rejected(self) -> None:
        actions = {AGENT_ID: MacroAction.wait()}
        self.backend.step(actions)
        # backend is now at step 1; state with step 0 must be rejected.
        bad = self._build_state(step_id=0)
        with self.assertRaisesRegex(ValueError, "step_id must match"):
            self.backend.set_frame_evaluation_state(bad)

    def test_injecting_state_at_step_zero_and_advancing(self) -> None:
        state = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state)
        # Re-inject a fresh state at the same step should be fine.
        state2 = self._build_state(step_id=0)
        self.backend.set_frame_evaluation_state(state2)
        self.assertEqual(self.backend.get_frame_evaluation_state(), state2)
        # step() advances; the slot is cleared.
        self.backend.step({AGENT_ID: MacroAction.wait()})
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.backend.get_frame_evaluation_state()


# ----------------------------------------------------------------------
# Observation leakage regression
# ----------------------------------------------------------------------


class ObservationLeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeEnvironmentBackend()
        self.backend.open()
        self.task = _task()
        self.backend.reset(self.task)

    def tearDown(self) -> None:
        self.backend.close()

    def test_observation_does_not_leak_frame_state(self) -> None:
        # Build a state whose cells fit within step_id=0 by hand.
        cells = tuple(
            _success_cell(
                target,
                last_action_step=0,
                relevant_action_steps=(0,) if index == 0 else (),
            )
            for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
        )
        interior = _all_allowed_interior()
        state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=cells,
            interior_cells=interior,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=False,
            current_time_seconds=0.0,
            max_environment_steps=640,
            max_game_time_seconds=600.0,
        )
        self.backend.set_frame_evaluation_state(state)
        observations = self.backend._observations()
        agent_obs = observations[AGENT_ID]
        # Walk every key on the observation; none must contain
        # evaluator-only truth tokens.
        forbidden_tokens = (
            "FrozenFrame",
            "frame_evaluation",
            "frame_evaluator",
            "casting_s_c3_fixed",
            "target_cell",
            "transition_evidence",
            "water_truth",
            "lava_truth",
            "relevant_action_steps",
            "interior_blocker",
            "interior_cell",
            "interior_allowlist",
            "corner_cell",
            "completed_cells",
            "per_cell_outcomes",
            "per_interior_cell_outcomes",
            "first_failed_cell",
            "completed_corner_cells",
            "completed_interior_cells",
            "casting_frame_evaluator",
        )
        for key, value in _walk(agent_obs):
            with self.subTest(key=key):
                if isinstance(value, str):
                    for token in forbidden_tokens:
                        self.assertNotIn(token, value)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            for token in forbidden_tokens:
                                self.assertNotIn(token, item)

    def test_fakebackend_observations_are_observation_only(self) -> None:
        observations = self.backend._observations()
        for agent_id, obs in observations.items():
            with self.subTest(agent_id=agent_id):
                self.assertEqual(obs.episode_id, self.task.task_id)
                self.assertEqual(obs.step_id, 0)
                self.assertEqual(obs.agent_id, agent_id)
                # Observation must only carry the public schema
                # fields. No evaluator-only state ever leaks onto it.
                self.assertEqual(
                    set(_attribute_names(obs)),
                    {
                        "episode_id",
                        "agent_id",
                        "step_id",
                        "timestamp",
                        "frame",
                        "visible_inventory",
                        "messages",
                        "workflow_stage",
                    },
                )

    def test_step_observation_does_not_leak_frame_state(self) -> None:
        cells = tuple(
            _success_cell(
                target,
                last_action_step=0,
                relevant_action_steps=(0,) if index == 0 else (),
            )
            for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
        )
        state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=cells,
            interior_cells=_all_allowed_interior(),
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=False,
            current_time_seconds=0.0,
            max_environment_steps=640,
            max_game_time_seconds=600.0,
        )
        self.backend.set_frame_evaluation_state(state)
        actions = {AGENT_ID: MacroAction.wait()}
        result = self.backend.step(actions)
        for agent_id, obs in result.observations.items():
            with self.subTest(agent_id=agent_id):
                self.assertEqual(obs.step_id, 1)
                # The frame dict stays a tiny dict, no evaluator truth.
                self.assertEqual(
                    obs.frame,
                    {"backend": "fake", "step_id": 1},
                )


# ----------------------------------------------------------------------
# Task-origin / grid-origin coordinate conversion
# ----------------------------------------------------------------------


class OriginAnchorTests(unittest.TestCase):
    def test_default_anchor_converts_known_offsets(self) -> None:
        anchor = default_c3_anchor()
        # Public min_corner = [0, 0, 1] should map to grid (0, 0, 1).
        self.assertEqual(anchor.convert((0, 0, 1)), (0, 0, 1))
        # All 14 frozen cells should be in-bounds.
        for offset in CASTING_S_C3_FRAME_CELLS:
            grid_offset = anchor.convert(offset)
            for axis, (value, low, high) in enumerate(
                zip(grid_offset, anchor.grid_min, anchor.grid_max)
            ):
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)
        # All 6 interior cells should be in-bounds.
        for offset in CASTING_S_C3_INTERIOR_CELLS:
            grid_offset = anchor.convert(offset)
            for axis, (value, low, high) in enumerate(
                zip(grid_offset, anchor.grid_min, anchor.grid_max)
            ):
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

    def test_convert_all_returns_tuple(self) -> None:
        anchor = default_c3_anchor()
        offsets = ((0, 0, 0), (0, 0, 1), (3, 4, 1))
        converted = anchor.convert_all(offsets)
        self.assertEqual(converted, ((0, 0, 0), (0, 0, 1), (3, 4, 1)))

    def test_out_of_bounds_raises(self) -> None:
        anchor = FrozenFrameOriginAnchor(
            task_origin_in_grid=(0, 0, 0),
            grid_min=(-3, -1, 0),
            grid_max=(3, 5, 6),
        )
        # (4, 0, 0) is outside the grid x range (-3..3).
        with self.assertRaisesRegex(ValueError, "outside the truth grid"):
            anchor.convert((4, 0, 0))

    def test_origin_in_grid_offsets_within_bounds(self) -> None:
        # Anchor that places the origin at (3, 0, 0) - all 14 cells
        # end up out of bounds (x=3..6, but max is 3).
        anchor = FrozenFrameOriginAnchor(
            task_origin_in_grid=(3, 0, 0),
            grid_min=(-3, -1, 0),
            grid_max=(3, 5, 6),
        )
        with self.assertRaisesRegex(ValueError, "outside the truth grid"):
            anchor.convert((1, 0, 1))

    def test_missing_origin_anchor_rejected(self) -> None:
        # The FrozenFrameOriginAnchor's three fields are all required
        # and type-checked.
        with self.assertRaisesRegex((TypeError, ValueError), "strict integers"):
            FrozenFrameOriginAnchor(
                task_origin_in_grid=None,  # type: ignore[arg-type]
                grid_min=(-3, -1, 0),
                grid_max=(3, 5, 6),
            )

    def test_bad_xyz_type_rejected(self) -> None:
        anchor = default_c3_anchor()
        with self.assertRaisesRegex(ValueError, "strict integers"):
            anchor.convert((0.5, 0, 1))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "strict integers"):
            anchor.convert((True, 0, 1))  # type: ignore[arg-type]

    def test_non_xyz_tuple_rejected(self) -> None:
        anchor = default_c3_anchor()
        with self.assertRaisesRegex(ValueError, "strict integers"):
            anchor.convert((0, 0))  # type: ignore[arg-type]

    def test_inverted_bounds_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid bound invalid"):
            FrozenFrameOriginAnchor(
                task_origin_in_grid=(0, 0, 0),
                grid_min=(3, 5, 6),
                grid_max=(-3, -1, 0),
            )

    def test_convert_all_propagates_errors(self) -> None:
        anchor = default_c3_anchor()
        with self.assertRaisesRegex(ValueError, "offsets\\[0\\]"):
            anchor.convert_all(((4, 0, 0),))
        with self.assertRaisesRegex(TypeError, "non-None sequence"):
            anchor.convert_all(None)  # type: ignore[arg-type]

    def test_default_anchor_matches_portal_grid_constants(self) -> None:
        from obsidianlink.env.portal_spec import (
            PORTAL_GRID_MAX,
            PORTAL_GRID_MIN,
        )

        anchor = default_c3_anchor()
        self.assertEqual(anchor.grid_min, PORTAL_GRID_MIN)
        self.assertEqual(anchor.grid_max, PORTAL_GRID_MAX)


# ----------------------------------------------------------------------
# Information isolation: AST / import graph / module surface
# ----------------------------------------------------------------------


class InformationIsolationTests(unittest.TestCase):
    def test_evaluator_source_does_not_import_agents_workflows_drivers(
        self,
    ) -> None:
        forbidden_modules = (
            "obsidianlink.agents",
            "obsidianlink.workflows",
            "obsidianlink.drivers",
        )
        tree = ast.parse(EVALUATOR_SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith(forbidden_modules),
                        f"forbidden import: {alias.name}",
                    )
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                self.assertFalse(
                    node.module.startswith(forbidden_modules),
                    f"forbidden import: from {node.module}",
                )

    def test_evaluator_source_does_not_read_prompt_or_observation(self) -> None:
        forbidden_attrs = (
            "scenario_parameters",
            "evaluator_contract",
            "instruction",
        )
        tree = ast.parse(EVALUATOR_SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_attrs:
                    self.fail(
                        f"forbidden attribute access: {node.attr} at "
                        f"line {node.lineno}"
                    )

    def test_evaluator_signature_takes_only_evaluator_state(self) -> None:
        tree = ast.parse(EVALUATOR_SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                if node.args.args:
                    self.assertEqual(
                        node.args.args[0].arg, "self"
                    )
                if node.args.args and len(node.args.args) > 1:
                    self.assertEqual(
                        node.args.args[1].arg, "state"
                    )
                # The second parameter (state) must be type-annotated
                # with the FrozenFrameEvaluationState class.
                if len(node.args.args) > 1:
                    annotation = node.args.args[1].annotation
                    if annotation is not None and isinstance(annotation, ast.Name):
                        self.assertEqual(
                            annotation.id,
                            "FrozenFrameEvaluationState",
                        )


# ----------------------------------------------------------------------
# C1 / C2 / portal regression
# ----------------------------------------------------------------------


class C1C2PortalRegressionTests(unittest.TestCase):
    def test_r3_casting_evaluator_still_works(self) -> None:
        from obsidianlink.evaluation.casting import CastingEvaluationState

        state = CastingEvaluationState(
            episode_id="casting_c1_fixed_seed_0",
            step_id=10,
            target_cell=(0, 0, 0),
            initial_target_block="air",
            current_target_block="obsidian",
            target_update_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=10
            ),
            water_truth=CastingFluidTruth(present=True, evidence_step=8),
            lava_truth=CastingFluidTruth(present=True, evidence_step=9),
            relevant_action_steps=(8, 9, 10),
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=10,
            terminated_reason="driver_done",
            max_environment_steps=240,
            max_game_time_seconds=180.0,
        )
        result = CastingEvaluator().evaluate(state)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)

    def test_r5_continuous_casting_evaluator_still_works(self) -> None:
        from obsidianlink.evaluation.continuous_casting import (
            ContinuousCastingCellTruth,
        )

        # Each cell uses disjoint steps so the R5 disjointness check
        # passes.
        cells = tuple(
            ContinuousCastingCellTruth(
                target_cell=target,
                initial_block="air",
                current_block="obsidian",
                water_truth=CastingFluidTruth(present=True, evidence_step=4 + 3 * index),
                lava_truth=CastingFluidTruth(present=True, evidence_step=3 + 3 * index),
                transition_evidence=CastingTransitionEvidence(
                    before_block="air", after_block="obsidian", update_step=5 + 3 * index
                ),
                relevant_action_steps=(3 + 3 * index, 4 + 3 * index, 5 + 3 * index),
            )
            for index, target in enumerate(((2, 4, 3), (3, 4, 3), (4, 4, 3)))
        )
        state = ContinuousCastingEvaluationState(
            episode_id="casting_c3_fixed_seed_0",
            step_id=12,
            cells=cells,
            episode_terminated=True,
            terminated_step=12,
            terminated_reason="driver_done",
            max_environment_steps=240,
            max_game_time_seconds=180.0,
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)

    def test_portal_evaluator_still_works(self) -> None:
        state = EvaluationState(
            episode_id="a0_test",
            step_id=10,
            portal_built_by_episode=True,
            valid_portal_frame=True,
            portal_activated=True,
            agents_in_nether=frozenset({"agent_1"}),
            entered_via_episode_portal_by_agent={"agent_1": True},
            first_valid_frame_step=5,
            first_activation_step=6,
            first_nether_step_by_agent={"agent_1": 7},
            episode_terminated=True,
            terminated_step=10,
            terminated_reason="driver_done",
            latched_timestamps={
                "task_reset": 0.0,
                "first_obsidian_placed": 0.5,
                "build_site_selected": 0.7,
                "valid_portal_frame": 1.0,
                "portal_activated": 2.0,
                "agent_entered_nether:agent_1": 3.0,
            },
        )
        result = PortalEvaluator().evaluate(state)
        self.assertTrue(result.success)

    def test_fakebackend_c1_c2_c3_surfaces_coexist(self) -> None:
        # Verify the FakeBackend keeps the C1 / C2 / C3 surfaces
        # side by side and that they each reject cross-type
        # injection.
        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(_task())
        # Inject a C3 state whose cells fit step_id=0 with disjoint
        # per-cell steps. Only the first cell gets a relevant step;
        # the rest stay empty (truth_missing later, but the FakeBackend
        # only enforces type guards, not evaluator truth).
        cells = tuple(
            _success_cell(
                target,
                last_action_step=0,
                relevant_action_steps=(0,) if index == 0 else (),
            )
            for index, target in enumerate(CASTING_S_C3_FRAME_CELLS)
        )
        c3 = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=cells,
            interior_cells=_all_allowed_interior(),
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=False,
            current_time_seconds=0.0,
            max_environment_steps=640,
            max_game_time_seconds=600.0,
        )
        backend.set_frame_evaluation_state(c3)
        self.assertIsNotNone(backend.get_frame_evaluation_state())
        with self.assertRaisesRegex(RuntimeError, "casting evaluation state is unavailable"):
            backend.get_casting_evaluation_state()
        with self.assertRaisesRegex(RuntimeError, "continuous casting evaluation state is unavailable"):
            backend.get_continuous_casting_evaluation_state()
        backend.close()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _walk(obj: Any) -> list[tuple[str, Any]]:
    """Return a flat list of ``(key, value)`` pairs for a Mapping.

    Used by the leakage test to walk an Observation record.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.append((str(key), value))
            out.extend(_walk(value))
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            out.append((f"[{index}]", item))
            out.extend(_walk(item))
    return out


def _attribute_names(obj: Any) -> list[str]:
    return [field.name for field in dataclasses.fields(obj)]


if __name__ == "__main__":
    unittest.main()
