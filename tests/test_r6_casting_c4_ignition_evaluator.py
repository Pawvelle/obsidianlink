"""Offline tests for the R6 Casting-S-C4 ignition evaluator.

These tests prove, in code, that:

* :class:`FrozenFrameIdentity`, :class:`IgnitionActionEvidence`,
  :class:`PortalActivationEvidence`,
  :class:`FrozenIgnitionEvaluationState`, and
  :class:`FrozenIgnitionEvaluationResult` are frozen,
  type-strict, and JSON-serializable, with a recursive frozen
  evidence tree.
* :class:`FrozenIgnitionEvaluator` returns the same result for
  the same state on repeated calls, never reads Agent text /
  images / Planner input, and never imports the driver /
  planner / workflow / Agent / model surface.
* The closed outcome set and the locked priority order are
  honoured for every required scenario: full success path +
  deterministic replay; C3 not built; typed frame identity
  geometry mismatch; arbitrary equal mappings cannot success;
  truth missing; wrong agent / action / item / target reached
  through the public construction API; wrong activation agent;
  ignition action/activation step order errors; 4-step
  boundary (0 / 1 / 4 / 5); latched frame identity missing and
  mismatched; external activation reached through the public
  construction API; episode / step / agent identity
  inconsistencies; step / time budget exceeded; abnormal
  termination; bool impersonating int; ``as_dict()`` stable
  snapshot; evaluator / Agent / driver information isolation;
  FakeBackend C1 / C2 / C3 / C4 truth slot cross-isolation.
* Malformed evidence (bool impersonating int, non-xyz tuple,
  empty / non-string identifier, future steps, etc.) is
  rejected at construction time.
* The :class:`FakeEnvironmentBackend` exposes a
  ``set_ignition_evaluation_state`` /
  ``get_ignition_evaluation_state`` /
  ``clear_ignition_evaluation_state`` surface that is
  identity-guarded and that does not leak into
  :class:`Observation`.
* The C1 / C2 / portal / C3 frame surfaces keep working.

The tests never start Minecraft, MineRL, or Gradle. All
normal-business negative paths use the public construction API
of the evidence / identity / state classes. Only the dataclass
immutability tests rely on ``object.__setattr__`` to verify that
mutation raises :class:`dataclasses.FrozenInstanceError`; no test
constructs a "well-typed but invalid" instance via mutation.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from pathlib import Path
from typing import Any

from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation import (
    ACTIVATION_OFFSET_VERDICT_EXTERNAL,
    ACTIVATION_OFFSET_VERDICT_INTERNAL,
    ACTIVATION_VERDICT_MISSING,
    ACTIVATION_VERDICT_OBSERVED,
    ACTIVATION_WINDOW_VERDICT_BEFORE,
    ACTIVATION_WINDOW_VERDICT_OK,
    ACTIVATION_WINDOW_VERDICT_OUTSIDE,
    CASTING_S_C3_FRAME_CELLS,
    CASTING_S_C3_INTERIOR_CELLS,
    CASTING_S_C4_AGENT_ID,
    CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
    CASTING_S_C4_FRAME_HEIGHT,
    CASTING_S_C4_FRAME_INTERIOR_CELLS,
    CASTING_S_C4_FRAME_INTERIOR_SET,
    CASTING_S_C4_FRAME_MAX_CORNER,
    CASTING_S_C4_FRAME_MIN_CORNER,
    CASTING_S_C4_FRAME_ORIENTATION,
    CASTING_S_C4_FRAME_WIDTH,
    CASTING_S_C4_IGNITION_ACTION_TYPE,
    CASTING_S_C4_IGNITION_ITEM,
    CASTING_S_C4_PUBLIC_IGNITION_TARGET,
    FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH,
    FRAME_IDENTITY_VERDICT_MATCH,
    FRAME_IDENTITY_VERDICT_MISSING,
    FRAME_IDENTITY_VERDICT_MISMATCH,
    FrozenFrameActionEvidence,
    FrozenFrameCellTruth,
    FrozenFrameEvaluationState,
    FrozenFrameEvaluator,
    FrozenFrameIdentity,
    FrozenFrameInteriorCellTruth,
    FrozenIgnitionEvaluationResult,
    FrozenIgnitionEvaluationState,
    FrozenIgnitionEvaluator,
    IGNITION_AGENT_VERDICT_OK,
    IGNITION_AGENT_VERDICT_WRONG,
    IGNITION_OUTCOMES,
    IGNITION_VERDICT_MISSING,
    IGNITION_VERDICT_OBSERVED,
    IgnitionActionEvidence,
    OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_ACTIVATION_BEFORE_IGNITION,
    OUTCOME_ACTIVATION_MISSING,
    OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
    OUTCOME_EXTERNAL_ACTIVATION,
    OUTCOME_FRAME_IDENTITY_MISMATCH,
    OUTCOME_FRAME_NOT_BUILT,
    OUTCOME_IGNITION_ACTION_MISSING,
    OUTCOME_IN_PROGRESS,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_IGNITION_ACTION,
    OUTCOME_WRONG_IGNITION_AGENT,
    OUTCOME_WRONG_IGNITION_ITEM,
    OUTCOME_WRONG_IGNITION_TARGET,
    PortalActivationEvidence,
    build_c4_c3_frame_identity,
)
from obsidianlink.evaluation.casting import (
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
)
from obsidianlink.evaluation.casting_frame_evaluator import (
    OUTCOME_SUCCESS as FRAME_OUTCOME_SUCCESS,
)
from obsidianlink.evaluation.continuous_casting import (
    ContinuousCastingCellTruth,
    ContinuousCastingEvaluationState,
    ContinuousCastingEvaluator,
)
from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator


EPISODE_ID = "casting_s_c4_fixed_seed_0"
AGENT_ID = CASTING_S_C4_AGENT_ID
WRONG_AGENT_ID = "agent_2"
MAX_ENVIRONMENT_STEPS = 700
STEP_ID = 602
IGNITION_STEP = 600
ACTIVATION_STEP = 602
IDENTITY_STEP = 602

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE = (
    ROOT / "obsidianlink/evaluation/casting_ignition_evaluator.py"
)


# ----------------------------------------------------------------------
# Helpers (only the public construction API)
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
                "Casting-S-C4 ignition evaluator unit-test task."
            ),
            "spawn_positions": {AGENT_ID: [0, 4, 0]},
            "initial_inventories": {
                AGENT_ID: {
                    "water_bucket": 14,
                    "lava_bucket": 14,
                    "cobblestone": 28,
                    "flint_and_steel": 1,
                }
            },
            "workflow": "casting_s_c4_fixed",
            "milestones": [
                "task_reset",
                "first_obsidian_cast",
                "build_site_selected",
                "valid_portal_frame",
                "portal_activated",
            ],
            "limits": {
                "max_environment_steps": MAX_ENVIRONMENT_STEPS,
                "max_model_calls": 1,
                "max_game_time_seconds": 640,
            },
            "split": "development",
        }
    )


def _success_cell(
    target_cell: tuple[int, int, int],
    *,
    last_action_step: int,
    relevant_action_steps: tuple[int, ...],
) -> FrozenFrameCellTruth:
    return FrozenFrameCellTruth(
        target_cell=target_cell,
        initial_block="air",
        current_block="obsidian",
        water_truth=CastingFluidTruth(
            present=True, evidence_step=last_action_step
        ),
        lava_truth=CastingFluidTruth(
            present=True, evidence_step=min(relevant_action_steps)
        ),
        transition_evidence=CastingTransitionEvidence(
            before_block="air",
            after_block="obsidian",
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


def _all_success_cells() -> tuple[FrozenFrameCellTruth, ...]:
    """14 success cells in the canonical order, disjoint steps.

    All cells' last steps are <= ``STEP_ID`` and the embedded C3
    frame state shares ``STEP_ID`` with the C4 state. The first
    cell starts at step 4 and the last cell's last step is at
    ``4 + 14*4 - 1 = 59``.
    """
    step_cursor = 4
    cells: list[FrozenFrameCellTruth] = []
    for target_cell in CASTING_S_C3_FRAME_CELLS:
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


def _all_allowed_interior() -> tuple[FrozenFrameInteriorCellTruth, ...]:
    return tuple(
        FrozenFrameInteriorCellTruth(
            target_cell=cell, current_block="air"
        )
        for cell in CASTING_S_C3_INTERIOR_CELLS
    )


def _build_frame_state(
    *,
    step_id: int = STEP_ID,
    terminated_step: int | None = STEP_ID,
    terminated_reason: str | None = "driver_done",
    episode_terminated: bool = True,
    max_environment_steps: int = MAX_ENVIRONMENT_STEPS,
    max_game_time_seconds: int = 640,
    interior_block: str = "air",
    cells: tuple[FrozenFrameCellTruth, ...] | None = None,
) -> FrozenFrameEvaluationState:
    if cells is None:
        cells = _compress_success_cells(step_id)
    interior = tuple(
        FrozenFrameInteriorCellTruth(
            target_cell=cell, current_block=interior_block
        )
        for cell in CASTING_S_C3_INTERIOR_CELLS
    )
    if not episode_terminated:
        terminated_step = None
        terminated_reason = None
    return FrozenFrameEvaluationState(
        episode_id=EPISODE_ID,
        step_id=step_id,
        cells=cells,
        interior_cells=interior,
        agent_id=AGENT_ID,
        causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
        episode_terminated=episode_terminated,
        terminated_step=terminated_step,
        terminated_reason=terminated_reason,
        current_time_seconds=0.0,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
    )


def _compress_success_cells(
    last_step: int,
) -> tuple[FrozenFrameCellTruth, ...]:
    """Build 14 success cells whose last action step <= ``last_step``."""
    budget = max(last_step - 4 + 1, 1)
    per_cell = max(budget // 14, 1)
    cursor = 4
    cells: list[FrozenFrameCellTruth] = []
    for target_cell in CASTING_S_C3_FRAME_CELLS:
        if per_cell == 1:
            steps = (cursor,)
            last = cursor
        else:
            last = cursor + per_cell - 1
            steps = tuple(range(cursor, last + 1))
        cells.append(
            _success_cell(
                target_cell,
                last_action_step=last,
                relevant_action_steps=steps,
            )
        )
        cursor = last + 1
    return tuple(cells)


def _ignition(
    *,
    step_id: int = IGNITION_STEP,
    agent_id: str = AGENT_ID,
    action_type: str = CASTING_S_C4_IGNITION_ACTION_TYPE,
    item: str = CASTING_S_C4_IGNITION_ITEM,
    target_cell: tuple[int, int, int] = CASTING_S_C4_PUBLIC_IGNITION_TARGET,
    episode_id: str = EPISODE_ID,
) -> IgnitionActionEvidence:
    return IgnitionActionEvidence(
        episode_id=episode_id,
        step_id=step_id,
        agent_id=agent_id,
        action_type=action_type,
        item=item,
        target_cell=target_cell,
    )


def _activation(
    *,
    update_step: int = ACTIVATION_STEP,
    nether_portal_offset: tuple[int, int, int] = (1, 1, 1),
    latched_frame_identity: FrozenFrameIdentity | None = None,
    agent_id: str = AGENT_ID,
    episode_id: str = EPISODE_ID,
) -> PortalActivationEvidence:
    if latched_frame_identity is None:
        latched_frame_identity = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID, step_id=IDENTITY_STEP,
        )
    return PortalActivationEvidence(
        episode_id=episode_id,
        update_step=update_step,
        agent_id=agent_id,
        nether_portal_offset=nether_portal_offset,
        latched_frame_identity=latched_frame_identity,
    )


def _identity(
    *,
    step_id: int = IDENTITY_STEP,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
    activation_offsets: tuple[tuple[int, int, int], ...] = (
        CASTING_S_C4_PUBLIC_IGNITION_TARGET,
    ),
) -> FrozenFrameIdentity:
    return FrozenFrameIdentity(
        orientation=CASTING_S_C4_FRAME_ORIENTATION,
        min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
        max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
        width=CASTING_S_C4_FRAME_WIDTH,
        height=CASTING_S_C4_FRAME_HEIGHT,
        target_offsets=tuple(CASTING_S_C3_FRAME_CELLS),
        interior_offsets=tuple(CASTING_S_C3_INTERIOR_CELLS),
        required_corner_count=4,
        required_full_ring_count=14,
        activation_offsets=activation_offsets,
        episode_id=episode_id,
        step_id=step_id,
        agent_id=agent_id,
    )


def _identity_with(
    *,
    step_id: int = IDENTITY_STEP,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
    orientation: str = CASTING_S_C4_FRAME_ORIENTATION,
    min_corner: tuple[int, int, int] = CASTING_S_C4_FRAME_MIN_CORNER,
    max_corner: tuple[int, int, int] = CASTING_S_C4_FRAME_MAX_CORNER,
    width: int = CASTING_S_C4_FRAME_WIDTH,
    height: int = CASTING_S_C4_FRAME_HEIGHT,
    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C3_FRAME_CELLS,
    interior_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C3_INTERIOR_CELLS,
    required_corner_count: int = 4,
    required_full_ring_count: int = 14,
    activation_offsets: tuple[tuple[int, int, int], ...] = (
        CASTING_S_C4_PUBLIC_IGNITION_TARGET,
    ),
) -> FrozenFrameIdentity:
    """Build a ``FrozenFrameIdentity`` overriding one or more fields.

    Used by the typed-identity negative tests where one field
    must differ from the canonical C3 / C4 contract. The override
    uses the public construction API (no mutation).
    """
    return FrozenFrameIdentity(
        orientation=orientation,
        min_corner=min_corner,
        max_corner=max_corner,
        width=width,
        height=height,
        target_offsets=tuple(target_offsets),
        interior_offsets=tuple(interior_offsets),
        required_corner_count=required_corner_count,
        required_full_ring_count=required_full_ring_count,
        activation_offsets=activation_offsets,
        episode_id=episode_id,
        step_id=step_id,
        agent_id=agent_id,
    )


_DEFAULT: Any = object()


def _state(
    *,
    frame_state: FrozenFrameEvaluationState | None = None,
    ignition_action: IgnitionActionEvidence | None | object = _DEFAULT,
    activation_evidence: PortalActivationEvidence | None | object = _DEFAULT,
    latched_frame_identity: FrozenFrameIdentity | None | object = _DEFAULT,
    step_id: int = STEP_ID,
    terminated_step: int | None = STEP_ID,
    terminated_reason: str | None = "driver_done",
    episode_terminated: bool = True,
    current_time_seconds: float = 0.0,
    max_environment_steps: int = MAX_ENVIRONMENT_STEPS,
    max_game_time_seconds: int = 640,
    causality_window_steps: int = CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
    ignition_step_id: int = IGNITION_STEP,
    activation_update_step: int = ACTIVATION_STEP,
) -> FrozenIgnitionEvaluationState:
    if frame_state is None:
        frame_state = _build_frame_state(
            step_id=step_id,
            terminated_step=terminated_step,
            terminated_reason=terminated_reason,
            episode_terminated=episode_terminated,
            max_environment_steps=max_environment_steps,
            max_game_time_seconds=max_game_time_seconds,
        )
    if latched_frame_identity is _DEFAULT:
        latched_frame_identity = _identity(step_id=step_id)
    if ignition_action is _DEFAULT:
        ignition_action = _ignition(step_id=ignition_step_id)
    if activation_evidence is _DEFAULT:
        # The activation's identity must match the state's
        # identity, so re-build it from ``latched_frame_identity``
        # when the caller supplied a custom state step_id.
        activation_identity = latched_frame_identity
        if (
            not isinstance(activation_identity, FrozenFrameIdentity)
            or activation_identity.step_id != step_id
        ):
            activation_identity = build_c4_c3_frame_identity(
                episode_id=EPISODE_ID, step_id=step_id,
            )
        activation_evidence = _activation(
            update_step=activation_update_step,
            latched_frame_identity=activation_identity,
        )
    if not episode_terminated:
        terminated_step = None
        terminated_reason = None
    return FrozenIgnitionEvaluationState(
        episode_id=EPISODE_ID,
        step_id=step_id,
        frame_state=frame_state,
        latched_frame_identity=latched_frame_identity,
        ignition_action=ignition_action,
        activation_evidence=activation_evidence,
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
# Outcome / contract / constants
# ----------------------------------------------------------------------


class OutcomeContractTests(unittest.TestCase):
    def test_outcome_constants_are_unique(self) -> None:
        self.assertEqual(len(IGNITION_OUTCOMES), len(set(IGNITION_OUTCOMES)))

    def test_outcome_constants_cover_required_set(self) -> None:
        required = {
            OUTCOME_SUCCESS,
            OUTCOME_IN_PROGRESS,
            OUTCOME_FRAME_NOT_BUILT,
            OUTCOME_IGNITION_ACTION_MISSING,
            OUTCOME_WRONG_IGNITION_AGENT,
            OUTCOME_WRONG_IGNITION_ACTION,
            OUTCOME_WRONG_IGNITION_ITEM,
            OUTCOME_WRONG_IGNITION_TARGET,
            OUTCOME_ACTIVATION_MISSING,
            OUTCOME_ACTIVATION_BEFORE_IGNITION,
            OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
            OUTCOME_EXTERNAL_ACTIVATION,
            OUTCOME_FRAME_IDENTITY_MISMATCH,
            OUTCOME_TRUTH_MISSING,
            OUTCOME_STEP_BUDGET_EXCEEDED,
            OUTCOME_TIME_BUDGET_EXCEEDED,
            OUTCOME_ABNORMAL_TERMINATION,
        }
        self.assertTrue(required.issubset(IGNITION_OUTCOMES))

    def test_outcome_ids_are_stable_strings(self) -> None:
        for outcome in (
            OUTCOME_SUCCESS,
            OUTCOME_IN_PROGRESS,
            OUTCOME_FRAME_NOT_BUILT,
            OUTCOME_IGNITION_ACTION_MISSING,
            OUTCOME_WRONG_IGNITION_AGENT,
            OUTCOME_WRONG_IGNITION_ACTION,
            OUTCOME_WRONG_IGNITION_ITEM,
            OUTCOME_WRONG_IGNITION_TARGET,
            OUTCOME_ACTIVATION_MISSING,
            OUTCOME_ACTIVATION_BEFORE_IGNITION,
            OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
            OUTCOME_EXTERNAL_ACTIVATION,
            OUTCOME_FRAME_IDENTITY_MISMATCH,
            OUTCOME_TRUTH_MISSING,
            OUTCOME_STEP_BUDGET_EXCEEDED,
            OUTCOME_TIME_BUDGET_EXCEEDED,
            OUTCOME_ABNORMAL_TERMINATION,
        ):
            with self.subTest(outcome=outcome):
                self.assertIsInstance(outcome, str)
                self.assertTrue(outcome)
                self.assertNotIn(" ", outcome)
                self.assertEqual(outcome, outcome.lower())

    def test_public_ignition_target_is_frozen(self) -> None:
        self.assertEqual(CASTING_S_C4_PUBLIC_IGNITION_TARGET, (1, 1, 1))
        self.assertEqual(
            len(CASTING_S_C4_PUBLIC_IGNITION_TARGET), 3
        )
        for value in CASTING_S_C4_PUBLIC_IGNITION_TARGET:
            self.assertIsInstance(value, int)
            self.assertFalse(isinstance(value, bool))

    def test_public_ignition_target_is_in_frame_interior(self) -> None:
        self.assertIn(
            CASTING_S_C4_PUBLIC_IGNITION_TARGET,
            CASTING_S_C4_FRAME_INTERIOR_SET,
        )
        self.assertEqual(len(CASTING_S_C4_FRAME_INTERIOR_CELLS), 6)
        for cell in CASTING_S_C4_FRAME_INTERIOR_CELLS:
            self.assertEqual(len(cell), 3)
            for coordinate in cell:
                self.assertIsInstance(coordinate, int)
                self.assertFalse(isinstance(coordinate, bool))

    def test_default_causality_window_matches_c3(self) -> None:
        self.assertEqual(
            CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
            DEFAULT_CAUSALITY_WINDOW_STEPS,
        )
        self.assertEqual(CASTING_S_C4_CAUSALITY_WINDOW_STEPS, 4)


# ----------------------------------------------------------------------
# FrozenFrameIdentity structural tests
# ----------------------------------------------------------------------


class FrozenFrameIdentityContractTests(unittest.TestCase):
    def test_rejects_empty_orientation(self) -> None:
        with self.assertRaisesRegex(ValueError, "orientation"):
            FrozenFrameIdentity(
                orientation="",
                min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
                max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
                width=CASTING_S_C4_FRAME_WIDTH,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=CASTING_S_C3_FRAME_CELLS,
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=4,
                required_full_ring_count=14,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id=EPISODE_ID,
                step_id=IDENTITY_STEP,
                agent_id=AGENT_ID,
            )

    def test_rejects_inverted_corners(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_corner"):
            FrozenFrameIdentity(
                orientation=CASTING_S_C4_FRAME_ORIENTATION,
                min_corner=(3, 4, 1),
                max_corner=(0, 0, 1),
                width=CASTING_S_C4_FRAME_WIDTH,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=CASTING_S_C3_FRAME_CELLS,
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=4,
                required_full_ring_count=14,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id=EPISODE_ID,
                step_id=IDENTITY_STEP,
                agent_id=AGENT_ID,
            )

    def test_rejects_zero_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            FrozenFrameIdentity(
                orientation=CASTING_S_C4_FRAME_ORIENTATION,
                min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
                max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
                width=0,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=CASTING_S_C3_FRAME_CELLS,
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=4,
                required_full_ring_count=14,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id=EPISODE_ID,
                step_id=IDENTITY_STEP,
                agent_id=AGENT_ID,
            )

    def test_rejects_non_xyz_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict integers"):
            FrozenFrameIdentity(
                orientation=CASTING_S_C4_FRAME_ORIENTATION,
                min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
                max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
                width=CASTING_S_C4_FRAME_WIDTH,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=((0, 0, 1.5),),  # type: ignore[arg-type]
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=1,
                required_full_ring_count=1,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id=EPISODE_ID,
                step_id=IDENTITY_STEP,
                agent_id=AGENT_ID,
            )

    def test_rejects_non_string_episode_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            FrozenFrameIdentity(
                orientation=CASTING_S_C4_FRAME_ORIENTATION,
                min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
                max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
                width=CASTING_S_C4_FRAME_WIDTH,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=CASTING_S_C3_FRAME_CELLS,
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=4,
                required_full_ring_count=14,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id="",
                step_id=IDENTITY_STEP,
                agent_id=AGENT_ID,
            )

    def test_rejects_bool_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            FrozenFrameIdentity(
                orientation=CASTING_S_C4_FRAME_ORIENTATION,
                min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
                max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
                width=CASTING_S_C4_FRAME_WIDTH,
                height=CASTING_S_C4_FRAME_HEIGHT,
                target_offsets=CASTING_S_C3_FRAME_CELLS,
                interior_offsets=CASTING_S_C3_INTERIOR_CELLS,
                required_corner_count=4,
                required_full_ring_count=14,
                activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
                episode_id=EPISODE_ID,
                step_id=True,  # type: ignore[arg-type]
                agent_id=AGENT_ID,
            )

    def test_is_frozen(self) -> None:
        identity = _identity()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.episode_id = "x"  # type: ignore[misc]

    def test_as_dict_is_json_serializable(self) -> None:
        identity = _identity()
        snapshot = identity.as_dict()
        encoded = json.dumps(snapshot, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["orientation"], identity.orientation)
        self.assertEqual(decoded["width"], identity.width)


# ----------------------------------------------------------------------
# IgnitionActionEvidence structural tests
# ----------------------------------------------------------------------


class IgnitionActionEvidenceTests(unittest.TestCase):
    def test_rejects_negative_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _ignition(step_id=-1)

    def test_rejects_bool_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _ignition(step_id=True)  # type: ignore[arg-type]

    def test_rejects_empty_episode_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            _ignition(episode_id="")

    def test_rejects_empty_agent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            _ignition(agent_id="")

    def test_rejects_empty_action_type(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "action_type"
        ):
            _ignition(action_type="")

    def test_rejects_empty_item(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "item"
        ):
            _ignition(item="")

    def test_rejects_non_xyz_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "strict integers"
        ):
            _ignition(target_cell=(0, 0, 1.5))  # type: ignore[arg-type]

    def test_rejects_list_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "strict integers"
        ):
            _ignition(target_cell=[1, 1, 1])  # type: ignore[arg-type]

    def test_accepts_any_non_empty_string_for_semantic_fields(
        self,
    ) -> None:
        # The constructor is now deliberately lenient on
        # ``agent_id`` / ``action_type`` / ``item`` / ``target_cell``
        # values; only the structural / type / format checks
        # remain. The semantic check is the evaluator's job.
        action = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id="some_other_agent",
            action_type="mine_target",
            item="diamond_pickaxe",
            target_cell=(5, 5, 5),
        )
        self.assertEqual(action.agent_id, "some_other_agent")
        self.assertEqual(action.action_type, "mine_target")
        self.assertEqual(action.item, "diamond_pickaxe")
        self.assertEqual(action.target_cell, (5, 5, 5))

    def test_is_frozen(self) -> None:
        action = _ignition()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            action.step_id = 100  # type: ignore[misc]


# ----------------------------------------------------------------------
# PortalActivationEvidence structural tests
# ----------------------------------------------------------------------


class PortalActivationEvidenceTests(unittest.TestCase):
    def test_rejects_non_xyz_offset(self) -> None:
        identity = _identity()
        with self.assertRaisesRegex(
            ValueError, "strict integers"
        ):
            _activation(
                nether_portal_offset=(0, 0, 1.5),  # type: ignore[arg-type]
                latched_frame_identity=identity,
            )

    def test_rejects_negative_update_step(self) -> None:
        identity = _identity()
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _activation(
                update_step=-1,
                latched_frame_identity=identity,
            )

    def test_rejects_bool_update_step(self) -> None:
        identity = _identity()
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _activation(
                update_step=True,  # type: ignore[arg-type]
                latched_frame_identity=identity,
            )

    def test_rejects_empty_episode_id(self) -> None:
        identity = _identity()
        with self.assertRaisesRegex(ValueError, "episode_id"):
            _activation(episode_id="", latched_frame_identity=identity)

    def test_rejects_empty_agent_id(self) -> None:
        identity = _identity()
        with self.assertRaisesRegex(ValueError, "agent_id"):
            _activation(agent_id="", latched_frame_identity=identity)

    def test_rejects_non_frame_identity_latched_frame_identity(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "latched_frame_identity"
        ):
            _activation(
                latched_frame_identity={  # type: ignore[arg-type]
                    "orientation": "plane_z",
                }
            )

    def test_accepts_external_offset_for_evaluator(self) -> None:
        # The constructor is now lenient on the interior-set
        # membership; ``external_activation`` is the evaluator's
        # verdict for non-interior offsets. Construction must
        # succeed; classification is left to the evaluator.
        external = _activation(
            nether_portal_offset=(0, 0, 1),
        )
        self.assertEqual(external.nether_portal_offset, (0, 0, 1))

    def test_is_frozen(self) -> None:
        activation = _activation()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            activation.update_step = 100  # type: ignore[misc]


# ----------------------------------------------------------------------
# State validation
# ----------------------------------------------------------------------


class StateValidationTests(unittest.TestCase):
    def test_state_rejects_wrong_agent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=WRONG_AGENT_ID,  # type: ignore[arg-type]
            )

    def test_state_rejects_non_frozen_frame_state(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "frame_state must be a FrozenFrameEvaluationState"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID,
                frame_state="not a state",  # type: ignore[arg-type]
                latched_frame_identity=_identity(),
            )

    def test_state_rejects_non_frozen_identity(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "latched_frame_identity must be a FrozenFrameIdentity"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID,
                frame_state=_build_frame_state(),
                latched_frame_identity={  # type: ignore[arg-type]
                    "orientation": "plane_z"
                },
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_frame_state_with_mismatched_episode_id(
        self,
    ) -> None:
        # The C4 wrapper checks that the embedded C3 frame state's
        # ``episode_id`` matches its own. We construct a C4 state
        # with a *different* ``episode_id`` from the C3 frame state
        # (the C3 frame state is still valid on its own).
        with self.assertRaisesRegex(
            ValueError, "frame_state.episode_id"
        ):
            FrozenIgnitionEvaluationState(
                episode_id="other_episode",
                step_id=STEP_ID,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_frame_state_with_mismatched_step_id(self) -> None:
        # The C4 wrapper checks that the embedded C3 frame state's
        # ``step_id`` matches its own. Use a different C4 step_id
        # against a C3 frame state built at the original step.
        with self.assertRaisesRegex(
            ValueError, "frame_state.step_id"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID - 1,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(step_id=STEP_ID - 1),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_frame_state_with_mismatched_budget(self) -> None:
        # The C4 wrapper checks that the embedded C3 frame state's
        # ``max_environment_steps`` matches its own. Pass a
        # different C4 ``max_environment_steps``.
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS - 1,
                max_game_time_seconds=640,
            )

    def test_state_rejects_ignition_with_mismatched_episode_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_action.episode_id"
        ):
            _state(ignition_action=_ignition(episode_id="other"))

    def test_state_rejects_ignition_in_future(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_action.step_id"
        ):
            _state(ignition_action=_ignition(step_id=STEP_ID + 5))

    def test_state_rejects_activation_with_mismatched_episode_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "activation_evidence.episode_id"
        ):
            _state(activation_evidence=_activation(episode_id="other"))

    def test_state_rejects_activation_in_future(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "activation_evidence.update_step"
        ):
            _state(activation_evidence=_activation(update_step=STEP_ID + 5))

    def test_state_rejects_identity_with_mismatched_episode_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "latched_frame_identity.episode_id"
        ):
            _state(latched_frame_identity=_identity(episode_id="other"))

    def test_state_rejects_identity_with_mismatched_agent_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "latched_frame_identity.agent_id"
        ):
            _state(latched_frame_identity=_identity(agent_id=WRONG_AGENT_ID))

    def test_state_rejects_zero_causality_window(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "causality_window_steps"
        ):
            _state(causality_window_steps=0)

    def test_state_rejects_bool_causality_window(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "causality_window_steps"
        ):
            _state(causality_window_steps=True)  # type: ignore[arg-type]

    def test_state_rejects_terminated_step_in_future(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "terminated_step"
        ):
            _state(
                terminated_step=STEP_ID + 5,
                frame_state=_build_frame_state(
                    terminated_step=STEP_ID + 5,
                ),
            )

    def test_state_rejects_mismatched_terminated_step_with_frame(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "frame_state.terminated_step"
        ):
            _state(
                terminated_step=STEP_ID - 1,
                frame_state=_build_frame_state(
                    terminated_step=STEP_ID,
                ),
            )

    def test_state_rejects_mismatched_terminated_reason_with_frame(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "frame_state.terminated_reason"
        ):
            _state(
                terminated_reason="task_complete",
                frame_state=_build_frame_state(
                    terminated_reason="driver_done",
                ),
            )

    def test_state_rejects_terminated_reason_without_terminated(self) -> None:
        frame = _build_frame_state(episode_terminated=False)
        with self.assertRaisesRegex(
            ValueError, "episode_terminated=True"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=STEP_ID,
                frame_state=frame,
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                episode_terminated=False,
                terminated_step=None,
                terminated_reason="driver_done",
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_non_positive_budgets(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps"
        ):
            _state(max_environment_steps=0)
        with self.assertRaisesRegex(
            ValueError, "max_game_time_seconds"
        ):
            _state(max_game_time_seconds=0)

    def test_state_rejects_nan_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _state(current_time_seconds=float("nan"))

    def test_state_rejects_non_terminated_while_frame_terminated(self) -> None:
        frame = _build_frame_state(terminated_step=STEP_ID)
        with self.assertRaisesRegex(
            ValueError, "frame_state.episode_terminated"
        ):
            _state(
                episode_terminated=False,
                terminated_step=None,
                terminated_reason=None,
                frame_state=frame,
            )

    def test_state_is_frozen(self) -> None:
        state = _state()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.step_id = 999  # type: ignore[misc]


# ----------------------------------------------------------------------
# Result immutability / as_dict
# ----------------------------------------------------------------------


class ResultImmutabilityTests(unittest.TestCase):
    def test_result_rejects_success_outcome_inconsistency(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        with self.assertRaisesRegex(ValueError, "success must equal"):
            dataclasses.replace(result, success=False)

    def test_result_rejects_unknown_outcome(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        with self.assertRaisesRegex(ValueError, "unknown outcome"):
            dataclasses.replace(result, outcome="not_a_real_outcome")

    def test_result_is_frozen(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_as_dict_is_json_serializable(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        snapshot = result.as_dict()
        encoded = json.dumps(snapshot, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["outcome"], result.outcome)
        self.assertEqual(decoded["success"], result.success)
        self.assertEqual(decoded["frame_outcome"], result.frame_outcome)

    def test_as_dict_is_detached(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        snapshot = result.as_dict()
        snapshot["evidence"]["mutated"] = True
        snapshot["blocking_conditions"].append("tampered")
        snapshot["activation_observed_offset"] = [9, 9, 9]
        # Re-fetch the result: it must not reflect the mutation.
        fresh = result.as_dict()
        self.assertNotIn("mutated", fresh["evidence"])
        self.assertNotIn("tampered", fresh["blocking_conditions"])

    def test_as_dict_offsets_are_lists(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        snapshot = result.as_dict()
        if result.activation_observed_offset is not None:
            self.assertIsInstance(
                snapshot["activation_observed_offset"], list
            )
            self.assertEqual(
                snapshot["activation_observed_offset"],
                list(result.activation_observed_offset),
            )

    def test_evaluate_returns_same_result_for_same_state(self) -> None:
        state = _state()
        r1 = FrozenIgnitionEvaluator().evaluate(state)
        r2 = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.as_dict(), r2.as_dict())


# ----------------------------------------------------------------------
# Determinism and priority
# ----------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_priority_is_stable_for_same_input(self) -> None:
        # Wrong item via the public API.
        state = _state(ignition_action=_ignition(item="water_bucket"))
        r1 = FrozenIgnitionEvaluator().evaluate(state)
        r2 = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(r1.outcome, OUTCOME_WRONG_IGNITION_ITEM)
        self.assertEqual(r2.outcome, OUTCOME_WRONG_IGNITION_ITEM)
        self.assertEqual(r1.as_dict(), r2.as_dict())

    def test_priority_step_budget_beats_truth(self) -> None:
        state = _state(
            ignition_action=None,
            activation_evidence=None,
            step_id=MAX_ENVIRONMENT_STEPS + 1,
            max_environment_steps=MAX_ENVIRONMENT_STEPS,
            terminated_step=MAX_ENVIRONMENT_STEPS + 1,
            terminated_reason="driver_done",
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_priority_time_budget_beats_frame(self) -> None:
        state = _state(
            current_time_seconds=641.0,
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TIME_BUDGET_EXCEEDED)

    def test_priority_frame_identity_geometry_beats_c3(self) -> None:
        # The C3 frame is success but the frame identity geometry
        # is wrong. ``frame_identity_mismatch`` is reported even
        # though C3 itself passes.
        bad_geo = FrozenFrameIdentity(
            orientation=CASTING_S_C4_FRAME_ORIENTATION,
            min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
            max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
            width=CASTING_S_C4_FRAME_WIDTH,
            height=CASTING_S_C4_FRAME_HEIGHT,
            target_offsets=((0, 0, 1),),  # only 1 cell instead of 14
            interior_offsets=((1, 1, 1),),
            required_corner_count=0,
            required_full_ring_count=1,
            activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
            episode_id=EPISODE_ID,
            step_id=IDENTITY_STEP,
            agent_id=AGENT_ID,
        )
        state = _state(latched_frame_identity=bad_geo)
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)
        self.assertEqual(
            result.frame_identity_verdict,
            FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH,
        )

    def test_priority_frame_not_built_beats_ignition_missing(self) -> None:
        # C3 frame not built. C4 must report ``frame_not_built``
        # before any ignition verdict.
        partial_cells = list(_compress_success_cells(STEP_ID))
        # Build a fresh, valid cell whose ``current_block`` is
        # ``air`` (not ``obsidian``) so the C3 evaluator reports
        # ``truth_missing``.
        incomplete_cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=CastingFluidTruth(
                present=True, evidence_step=STEP_ID,
            ),
            lava_truth=CastingFluidTruth(
                present=True, evidence_step=STEP_ID,
            ),
            transition_evidence=CastingTransitionEvidence(
                before_block="air",
                after_block="air",
                update_step=STEP_ID,
            ),
            relevant_action_steps=(STEP_ID,),
            action_evidence=(
                FrozenFrameActionEvidence(
                    episode_id=EPISODE_ID,
                    step_id=STEP_ID,
                    agent_id=AGENT_ID,
                    action_type="use_item",
                    item="water_bucket",
                    target_cell=CASTING_S_C3_FRAME_CELLS[0],
                ),
            ),
            transition_action_step=STEP_ID,
        )
        partial_cells[0] = incomplete_cell
        frame_state = _build_frame_state(cells=tuple(partial_cells))
        state = _state(
            frame_state=frame_state,
            ignition_action=None,
            activation_evidence=None,
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_FRAME_NOT_BUILT)

    def test_c3_in_progress_propagates(self) -> None:
        # C3 frame state not yet terminated; C4 must propagate
        # ``in_progress`` so the C4 evaluator reports
        # ``in_progress`` too.
        result = FrozenIgnitionEvaluator().evaluate(
            _state(episode_terminated=False)
        )
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)


# ----------------------------------------------------------------------
# Success path
# ----------------------------------------------------------------------


class SuccessPathTests(unittest.TestCase):
    def test_full_success_path(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertTrue(result.success)
        self.assertEqual(result.frame_outcome, FRAME_OUTCOME_SUCCESS)
        self.assertEqual(
            result.ignition_verdict, IGNITION_VERDICT_OBSERVED
        )
        self.assertEqual(
            result.activation_verdict, ACTIVATION_VERDICT_OBSERVED
        )
        self.assertEqual(
            result.activation_window_verdict,
            ACTIVATION_WINDOW_VERDICT_OK,
        )
        self.assertEqual(
            result.activation_offset_verdict,
            ACTIVATION_OFFSET_VERDICT_INTERNAL,
        )
        self.assertEqual(
            result.frame_identity_verdict, FRAME_IDENTITY_VERDICT_MATCH
        )
        self.assertEqual(result.activation_delta_steps, 2)
        self.assertEqual(result.activation_observed_offset, (1, 1, 1))
        self.assertEqual(result.blocking_conditions, ())
        self.assertEqual(result.failure_type, None)
        self.assertEqual(result.failure_step, None)

    def test_success_with_zero_delta(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                ignition_action=_ignition(step_id=ACTIVATION_STEP),
                activation_evidence=_activation(
                    update_step=ACTIVATION_STEP,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 0)
        self.assertEqual(
            result.activation_window_verdict,
            ACTIVATION_WINDOW_VERDICT_OK,
        )

    def test_success_with_one_step_delta(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 1,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 1)

    def test_success_with_four_step_delta(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                step_id=650,
                terminated_step=650,
                latched_frame_identity=_identity(step_id=650),
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 4,
                    latched_frame_identity=_identity(step_id=650),
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 4)

    def test_activation_at_interior_cell_not_public_target(self) -> None:
        identity = _identity(activation_offsets=((2, 2, 1),))
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=identity,
                activation_evidence=_activation(
                    nether_portal_offset=(2, 2, 1),
                    latched_frame_identity=identity,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(
            result.activation_offset_verdict,
            ACTIVATION_OFFSET_VERDICT_INTERNAL,
        )

    def test_in_progress_when_not_terminated(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(episode_terminated=False)
        )
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)
        self.assertFalse(result.success)
        self.assertIn(
            "episode_not_terminated", result.blocking_conditions
        )


# ----------------------------------------------------------------------
# C3 not built
# ----------------------------------------------------------------------


class FrameNotBuiltTests(unittest.TestCase):
    def test_frame_partial_completion_is_frame_not_built(self) -> None:
        # Construct a C3 frame where cell 0 is left as ``air``
        # (not obsidian) via the public construction API. The
        # C3 evaluator reports ``truth_missing`` (because the
        # other 13 cells have full evidence but cell 0 does not)
        # and the C4 evaluator reports ``frame_not_built``.
        partial_cells = list(_compress_success_cells(STEP_ID))
        # Build a fresh, valid cell whose ``current_block`` is
        # ``air`` (not ``obsidian``) and whose other evidence is
        # missing, so the C3 evaluator reports ``truth_missing``.
        incomplete_cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="air",
            water_truth=CastingFluidTruth(
                present=True, evidence_step=STEP_ID,
            ),
            lava_truth=CastingFluidTruth(
                present=True, evidence_step=STEP_ID,
            ),
            transition_evidence=CastingTransitionEvidence(
                before_block="air",
                after_block="air",
                update_step=STEP_ID,
            ),
            relevant_action_steps=(STEP_ID,),
            action_evidence=(
                FrozenFrameActionEvidence(
                    episode_id=EPISODE_ID,
                    step_id=STEP_ID,
                    agent_id=AGENT_ID,
                    action_type="use_item",
                    item="water_bucket",
                    target_cell=CASTING_S_C3_FRAME_CELLS[0],
                ),
            ),
            transition_action_step=STEP_ID,
        )
        partial_cells[0] = incomplete_cell
        frame_state = _build_frame_state(cells=tuple(partial_cells))
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                frame_state=frame_state,
                ignition_action=None,
                activation_evidence=None,
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_NOT_BUILT)
        self.assertIn("frame_not_built", result.blocking_conditions[0])

    def test_frame_interior_blocked_is_frame_not_built(self) -> None:
        frame_state = _build_frame_state(interior_block="obsidian")
        result = FrozenIgnitionEvaluator().evaluate(
            _state(frame_state=frame_state, ignition_action=None)
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_NOT_BUILT)


# ----------------------------------------------------------------------
# Truth-missing
# ----------------------------------------------------------------------


class TruthMissingTests(unittest.TestCase):
    def test_ignition_action_missing(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(ignition_action=None, activation_evidence=None)
        )
        self.assertEqual(result.outcome, OUTCOME_IGNITION_ACTION_MISSING)
        self.assertEqual(
            result.ignition_verdict, IGNITION_VERDICT_MISSING
        )
        self.assertIn(
            "ignition_action_missing", result.blocking_conditions
        )

    def test_activation_missing(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=None)
        )
        self.assertEqual(result.outcome, OUTCOME_ACTIVATION_MISSING)
        self.assertIn(
            "portal_activation_missing", result.blocking_conditions
        )


# ----------------------------------------------------------------------
# Wrong ignition (via the public construction API)
# ----------------------------------------------------------------------


class WrongIgnitionComponentTests(unittest.TestCase):
    def test_wrong_agent(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(ignition_action=_ignition(agent_id=WRONG_AGENT_ID))
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_AGENT)
        self.assertEqual(
            result.ignition_agent_verdict,
            IGNITION_AGENT_VERDICT_WRONG,
        )
        self.assertIn(
            "ignition_agent_mismatch", result.blocking_conditions
        )

    def test_wrong_action_type(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(ignition_action=_ignition(action_type="place_block"))
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_ACTION)
        self.assertIn(
            "ignition_action_type_mismatch", result.blocking_conditions
        )

    def test_wrong_item(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(ignition_action=_ignition(item="water_bucket"))
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_ITEM)
        self.assertIn(
            "ignition_item_mismatch", result.blocking_conditions
        )

    def test_wrong_target(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(ignition_action=_ignition(target_cell=(2, 1, 1)))
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_TARGET)
        self.assertIn(
            "ignition_target_mismatch", result.blocking_conditions
        )

    def test_wrong_activation_agent(self) -> None:
        # The activation evidence is bound to the same agent who
        # performed the ignition. A non-``agent_1`` activation
        # agent is rejected even when the ignition itself is
        # correct.
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=_activation(agent_id=WRONG_AGENT_ID))
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_AGENT)
        self.assertIn(
            "ignition_agent_mismatch", result.blocking_conditions
        )


# ----------------------------------------------------------------------
# Step order and 4-step boundary
# ----------------------------------------------------------------------


class StepOrderTests(unittest.TestCase):
    def test_activation_before_ignition(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                ignition_action=_ignition(step_id=600),
                activation_evidence=_activation(update_step=590),
            )
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_BEFORE_IGNITION
        )
        self.assertEqual(
            result.activation_window_verdict,
            ACTIVATION_WINDOW_VERDICT_BEFORE,
        )
        self.assertIn(
            "activation_before_ignition", result.blocking_conditions
        )

    def test_activation_outside_window_five_steps(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                step_id=650,
                terminated_step=650,
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 5,
                ),
            )
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_OUTSIDE_WINDOW
        )
        self.assertEqual(
            result.activation_window_verdict,
            ACTIVATION_WINDOW_VERDICT_OUTSIDE,
        )
        self.assertIn(
            "activation_outside_window", result.blocking_conditions
        )

    def test_activation_outside_window_far_in_future(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                step_id=680,
                terminated_step=680,
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 50,
                ),
            )
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_OUTSIDE_WINDOW
        )

    def test_boundary_four_step_delta_is_success(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                step_id=650,
                terminated_step=650,
                latched_frame_identity=_identity(step_id=650),
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 4,
                    latched_frame_identity=_identity(step_id=650),
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 4)

    def test_boundary_five_step_delta_is_outside(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                step_id=650,
                terminated_step=650,
                latched_frame_identity=_identity(step_id=650),
                ignition_action=_ignition(step_id=IGNITION_STEP),
                activation_evidence=_activation(
                    update_step=IGNITION_STEP + 5,
                    latched_frame_identity=_identity(step_id=650),
                ),
            )
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_OUTSIDE_WINDOW
        )
        self.assertEqual(result.activation_delta_steps, 5)


# ----------------------------------------------------------------------
# Typed frame identity
# ----------------------------------------------------------------------


class FrameIdentityTests(unittest.TestCase):
    def test_correct_c4_c3_identity_success(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(_state())
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(
            result.frame_identity_verdict, FRAME_IDENTITY_VERDICT_MATCH
        )

    def test_arbitrary_equal_mappings_cannot_success(self) -> None:
        # A well-typed identity with the SAME target_offsets /
        # interior_offsets / etc. but different activation_offsets
        # must fail closed.
        different_activation_offsets = _identity(
            activation_offsets=((2, 2, 1),),
        )
        activation = _activation(
            latched_frame_identity=different_activation_offsets,
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )
        self.assertEqual(
            result.frame_identity_verdict, FRAME_IDENTITY_VERDICT_MISMATCH
        )

    def test_orientation_mismatch(self) -> None:
        # Use a different orientation; the C3 frozen public plan
        # is plane_z only. The ``FrozenFrameIdentity`` is
        # constructed directly through the public API.
        bad = _identity_with(
            orientation="plane_x",
        )
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )
        self.assertEqual(
            result.frame_identity_verdict, FRAME_IDENTITY_VERDICT_MISMATCH
        )

    def test_width_mismatch(self) -> None:
        bad = _identity_with(width=5)
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_height_mismatch(self) -> None:
        bad = _identity_with(height=6)
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_min_corner_mismatch(self) -> None:
        bad = _identity_with(min_corner=(1, 0, 1))
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_max_corner_mismatch(self) -> None:
        bad = _identity_with(max_corner=(3, 4, 2))
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_target_offsets_mismatch(self) -> None:
        # Replace one cell to make the offset set different.
        bad = _identity_with(
            target_offsets=tuple(
                c for c in CASTING_S_C3_FRAME_CELLS if c != (0, 0, 1)
            )
        )
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_target_offsets_reordered_is_not_canonical(self) -> None:
        bad = _identity_with(
            target_offsets=tuple(reversed(CASTING_S_C3_FRAME_CELLS)),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_target_offsets_duplicate_is_not_canonical(self) -> None:
        bad = _identity_with(
            target_offsets=(
                *CASTING_S_C3_FRAME_CELLS,
                CASTING_S_C3_FRAME_CELLS[0],
            ),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_interior_offsets_mismatch(self) -> None:
        # Remove one interior cell.
        bad = _identity_with(
            interior_offsets=tuple(
                c for c in CASTING_S_C3_INTERIOR_CELLS if c != (2, 3, 1)
            )
        )
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_interior_offsets_reordered_is_not_canonical(self) -> None:
        bad = _identity_with(
            interior_offsets=tuple(reversed(CASTING_S_C3_INTERIOR_CELLS)),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_interior_offsets_duplicate_is_not_canonical(self) -> None:
        bad = _identity_with(
            interior_offsets=(
                *CASTING_S_C3_INTERIOR_CELLS,
                CASTING_S_C3_INTERIOR_CELLS[0],
            ),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_activation_offsets_empty_is_not_canonical(self) -> None:
        bad = _identity_with(activation_offsets=())
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_activation_offsets_duplicate_is_not_canonical(self) -> None:
        bad = _identity_with(
            activation_offsets=(
                CASTING_S_C4_PUBLIC_IGNITION_TARGET,
                CASTING_S_C4_PUBLIC_IGNITION_TARGET,
            ),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=bad,
                activation_evidence=_activation(
                    latched_frame_identity=bad,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_observed_activation_must_be_in_latched_offsets(self) -> None:
        identity = _identity(
            activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=identity,
                activation_evidence=_activation(
                    nether_portal_offset=(2, 2, 1),
                    latched_frame_identity=identity,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH)

    def test_multiple_activation_offsets_use_canonical_interior_order(self) -> None:
        activation_offsets = (
            CASTING_S_C3_INTERIOR_CELLS[0],
            CASTING_S_C3_INTERIOR_CELLS[3],
        )
        identity = _identity(activation_offsets=activation_offsets)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                latched_frame_identity=identity,
                activation_evidence=_activation(
                    nether_portal_offset=activation_offsets[1],
                    latched_frame_identity=identity,
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)

    def test_required_full_ring_count_mismatch(self) -> None:
        bad = _identity_with(required_full_ring_count=10)
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_required_corner_count_mismatch(self) -> None:
        bad = _identity_with(required_corner_count=3)
        activation = _activation(latched_frame_identity=bad)
        result = FrozenIgnitionEvaluator().evaluate(
            _state(activation_evidence=activation)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_state_identity_geometry_mismatch(self) -> None:
        # The wrapping state's identity has the wrong geometry; the
        # C4 evaluator must fail closed before any activation /
        # ignition check.
        bad_state_geo = FrozenFrameIdentity(
            orientation=CASTING_S_C4_FRAME_ORIENTATION,
            min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
            max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
            width=CASTING_S_C4_FRAME_WIDTH,
            height=CASTING_S_C4_FRAME_HEIGHT,
            target_offsets=((0, 0, 1),),  # only 1 cell
            interior_offsets=((1, 1, 1),),
            required_corner_count=0,
            required_full_ring_count=1,
            activation_offsets=(CASTING_S_C4_PUBLIC_IGNITION_TARGET,),
            episode_id=EPISODE_ID,
            step_id=IDENTITY_STEP,
            agent_id=AGENT_ID,
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(latched_frame_identity=bad_state_geo)
        )
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )
        self.assertEqual(
            result.frame_identity_verdict,
            FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH,
        )

    def test_extra_key_in_identity_via_dict_must_be_rejected_at_state(
        self,
    ) -> None:
        # A ``FrozenFrameIdentity`` is a typed dataclass; you
        # cannot add extra keys. Verified by the type system.
        identity = _identity()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.extra_field = "x"  # type: ignore[attr-defined]

    def test_identity_geometry_succeeds_with_correct_c3_plan(self) -> None:
        # build_c4_c3_frame_identity uses the frozen C3 plan;
        # the evaluator must accept it.
        identity = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID, step_id=IDENTITY_STEP,
        )
        result = FrozenIgnitionEvaluator().evaluate(
            _state(latched_frame_identity=identity)
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(
            result.frame_identity_verdict, FRAME_IDENTITY_VERDICT_MATCH
        )


# ----------------------------------------------------------------------
# External activation (via the public construction API)
# ----------------------------------------------------------------------


class ExternalActivationTests(unittest.TestCase):
    def test_activation_outside_frame_interior(self) -> None:
        # (0, 0, 1) is a corner cell, not in the frame interior.
        # The constructor accepts it; the evaluator reports
        # ``external_activation``.
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                activation_evidence=_activation(
                    nether_portal_offset=(0, 0, 1),
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_EXTERNAL_ACTIVATION)
        self.assertEqual(
            result.activation_offset_verdict,
            ACTIVATION_OFFSET_VERDICT_EXTERNAL,
        )
        self.assertIn("external_activation", result.blocking_conditions)

    def test_activation_far_outside_frame(self) -> None:
        result = FrozenIgnitionEvaluator().evaluate(
            _state(
                activation_evidence=_activation(
                    nether_portal_offset=(5, 0, 1),
                ),
            )
        )
        self.assertEqual(result.outcome, OUTCOME_EXTERNAL_ACTIVATION)

    def test_activation_within_interior_yields_success(self) -> None:
        for offset in CASTING_S_C4_FRAME_INTERIOR_CELLS:
            with self.subTest(offset=offset):
                identity = _identity(activation_offsets=(offset,))
                result = FrozenIgnitionEvaluator().evaluate(
                    _state(
                        latched_frame_identity=identity,
                        activation_evidence=_activation(
                            nether_portal_offset=offset,
                            latched_frame_identity=identity,
                        ),
                    )
                )
                self.assertEqual(result.outcome, OUTCOME_SUCCESS)
                self.assertEqual(
                    result.activation_offset_verdict,
                    ACTIVATION_OFFSET_VERDICT_INTERNAL,
                )


# ----------------------------------------------------------------------
# Budget / abnormal termination
# ----------------------------------------------------------------------


class BudgetTests(unittest.TestCase):
    def test_step_budget_exceeded(self) -> None:
        state = _state(
            step_id=MAX_ENVIRONMENT_STEPS + 1,
            terminated_step=MAX_ENVIRONMENT_STEPS + 1,
            ignition_action=_ignition(),
            activation_evidence=_activation(),
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(
            result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED
        )
        self.assertIn(
            "step_budget_exceeded", result.blocking_conditions
        )
        self.assertEqual(result.failure_type, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_time_budget_exceeded(self) -> None:
        state = _state(
            current_time_seconds=641.0,
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(
            result.outcome, OUTCOME_TIME_BUDGET_EXCEEDED
        )
        self.assertIn(
            "time_budget_exceeded", result.blocking_conditions
        )
        self.assertEqual(result.failure_type, OUTCOME_TIME_BUDGET_EXCEEDED)

    def test_abnormal_termination(self) -> None:
        # The C3 frame state and the C4 wrapper do not enforce
        # NORMAL_TERMINATION_REASONS at construction time; they
        # delegate that check to the C4 evaluator's outcome
        # classification. A non-NORMAL ``terminated_reason`` can
        # be constructed through the public API, and the
        # evaluator reports ``abnormal_termination``.
        state = _state(terminated_reason="explosion")
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(
            result.outcome, OUTCOME_ABNORMAL_TERMINATION
        )
        self.assertIn(
            "abnormal_termination", result.blocking_conditions
        )

    def test_terminated_with_normal_reason_passes(self) -> None:
        for reason in NORMAL_TERMINATION_REASONS:
            with self.subTest(reason=reason):
                state = _state(terminated_reason=reason)
                result = FrozenIgnitionEvaluator().evaluate(state)
                self.assertEqual(result.outcome, OUTCOME_SUCCESS)


# ----------------------------------------------------------------------
# Type / identity consistency
# ----------------------------------------------------------------------


class IdentityConsistencyTests(unittest.TestCase):
    def test_state_rejects_non_int_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id="not_an_int",  # type: ignore[arg-type]
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_bool_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=True,  # type: ignore[arg-type]
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_negative_step_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            FrozenIgnitionEvaluationState(
                episode_id=EPISODE_ID,
                step_id=-1,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_empty_episode_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            FrozenIgnitionEvaluationState(
                episode_id="",  # type: ignore[arg-type]
                step_id=STEP_ID,
                frame_state=_build_frame_state(),
                latched_frame_identity=_identity(),
                agent_id=AGENT_ID,
                causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
                max_environment_steps=MAX_ENVIRONMENT_STEPS,
                max_game_time_seconds=640,
            )

    def test_state_rejects_non_int_max_env_steps(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps"
        ):
            _state(max_environment_steps=10.5)  # type: ignore[arg-type]

    def test_state_rejects_bool_max_env_steps(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps"
        ):
            _state(max_environment_steps=True)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# FakeBackend C4 slot
# ----------------------------------------------------------------------


class FakeBackendSetGetClearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeEnvironmentBackend()
        self.backend.open()
        self.task = _task()
        self.backend.reset(self.task)

    def tearDown(self) -> None:
        self.backend.close()

    def _build_state(self) -> FrozenIgnitionEvaluationState:
        return _build_state_at_step_zero_for_fakebackend(
            episode_id=EPISODE_ID
        )

    def test_set_then_get_returns_same_state(self) -> None:
        state = self._build_state()
        self.backend.set_ignition_evaluation_state(state)
        self.assertEqual(
            self.backend.get_ignition_evaluation_state(), state
        )

    def test_get_without_set_raises(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "ignition evaluation state is unavailable"
        ):
            self.backend.get_ignition_evaluation_state()

    def test_clear_drops_state(self) -> None:
        state = self._build_state()
        self.backend.set_ignition_evaluation_state(state)
        self.backend.clear_ignition_evaluation_state()
        with self.assertRaisesRegex(
            RuntimeError, "ignition evaluation state is unavailable"
        ):
            self.backend.get_ignition_evaluation_state()

    def test_reset_clears_state(self) -> None:
        state = self._build_state()
        self.backend.set_ignition_evaluation_state(state)
        self.backend.reset(self.task)
        with self.assertRaisesRegex(
            RuntimeError, "ignition evaluation state is unavailable"
        ):
            self.backend.get_ignition_evaluation_state()

    def test_step_clears_state(self) -> None:
        state = self._build_state()
        self.backend.set_ignition_evaluation_state(state)
        self.backend.step({AGENT_ID: MacroAction.wait()})
        with self.assertRaisesRegex(
            RuntimeError, "ignition evaluation state is unavailable"
        ):
            self.backend.get_ignition_evaluation_state()

    def test_close_clears_state(self) -> None:
        state = self._build_state()
        self.backend.set_ignition_evaluation_state(state)
        self.backend.close()
        self.assertIsNone(self.backend._ignition_evaluation_state)

    def test_wrong_episode_rejected(self) -> None:
        # Construct a C4 state with a *different* ``episode_id``
        # from the current task via the public API. The C4 wrapper
        # accepts it (it only checks internal consistency); the
        # FakeBackend's episode check rejects it.
        bad = _build_state_at_step_zero_for_fakebackend(
            episode_id="other_episode"
        )
        with self.assertRaisesRegex(
            ValueError, "episode_id must match"
        ):
            self.backend.set_ignition_evaluation_state(bad)

    def test_wrong_step_rejected(self) -> None:
        # Construct a C4 state with step_id=14 (different from
        # the FakeBackend's current step_id=0).
        cells = []
        for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
            cells.append(
                _success_cell(
                    target,
                    last_action_step=index,
                    relevant_action_steps=(index,),
                )
            )
        frame_state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=14,
            cells=tuple(cells),
            interior_cells=_all_allowed_interior(),
            agent_id=AGENT_ID,
            causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
            episode_terminated=False,
            current_time_seconds=0.0,
            max_environment_steps=MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=640,
        )
        bad = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=14,
            frame_state=frame_state,
            latched_frame_identity=build_c4_c3_frame_identity(
                episode_id=EPISODE_ID, step_id=14,
            ),
            agent_id=AGENT_ID,
            causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
            max_environment_steps=MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=640,
        )
        with self.assertRaisesRegex(
            ValueError, "step_id must match"
        ):
            self.backend.set_ignition_evaluation_state(bad)

    def test_wrong_type_rejected(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "FrozenIgnitionEvaluationState"
        ):
            self.backend.set_ignition_evaluation_state(
                "not a state"  # type: ignore[arg-type]
            )

    def test_wrong_workflow_rejected(self) -> None:
        c1_task = TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": "casting_c1_fixed_seed_0",
                "route": "lava_casting",
                "difficulty": 1,
                "agent_ids": [AGENT_ID],
                "world_seed": 0,
                "instruction": "test",
                "spawn_positions": {AGENT_ID: [0, 4, 0]},
                "initial_inventories": {
                    AGENT_ID: {
                        "water_bucket": 1,
                        "lava_bucket": 1,
                        "cobblestone": 8,
                    }
                },
                "workflow": "casting_c1_fixed",
                "milestones": ["task_reset"],
                "limits": {
                    "max_environment_steps": 160,
                    "max_model_calls": 1,
                    "max_game_time_seconds": 120,
                },
                "split": "development",
            }
        )
        self.backend.reset(c1_task)
        state = self._build_state()
        with self.assertRaisesRegex(
            ValueError, "casting_s_c4_fixed workflow"
        ):
            self.backend.set_ignition_evaluation_state(state)


# ----------------------------------------------------------------------
# Observation leakage
# ----------------------------------------------------------------------


class ObservationLeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeEnvironmentBackend()
        self.backend.open()
        self.task = _task()
        self.backend.reset(self.task)

    def tearDown(self) -> None:
        self.backend.close()

    def test_observation_does_not_leak_ignition_state(self) -> None:
        state = _build_minimal_c4_state_at_step_zero()
        self.backend.set_ignition_evaluation_state(state)
        observations = self.backend._observations()
        agent_obs = observations[AGENT_ID]
        forbidden_tokens = (
            "FrozenIgnition",
            "ignition_evaluation",
            "ignition_evaluator",
            "casting_s_c4_fixed",
            "FrozenFrameIdentity",
            "frame_identity",
            "ignition_action",
            "activation_evidence",
            "ignition_target",
            "ignition_item",
            "ignition_agent",
            "nether_portal",
            "flint_and_steel",
            "wrong_ignition",
            "frame_not_built",
            "ignition_verdict",
            "activation_verdict",
            "activation_window",
            "activation_offset",
            "portal_activation",
            "external_activation",
            "frame_identity_verdict",
            "public_ignition_target",
            "frame_interior",
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

    def test_observation_schema_is_unchanged(self) -> None:
        observations = self.backend._observations()
        for agent_id, obs in observations.items():
            with self.subTest(agent_id=agent_id):
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

    def test_step_observation_does_not_leak_ignition_state(self) -> None:
        state = _build_minimal_c4_state_at_step_zero()
        self.backend.set_ignition_evaluation_state(state)
        result = self.backend.step({AGENT_ID: MacroAction.wait()})
        for agent_id, obs in result.observations.items():
            with self.subTest(agent_id=agent_id):
                self.assertEqual(obs.step_id, 1)
                self.assertEqual(
                    obs.frame, {"backend": "fake", "step_id": 1}
                )


# ----------------------------------------------------------------------
# FakeBackend C1/C2/C3/C4 coexistence
# ----------------------------------------------------------------------


class FakeBackendCrossSlotTests(unittest.TestCase):
    def test_c4_surface_coexists_with_c1_c2_on_c4_task(self) -> None:
        from obsidianlink.evaluation.casting import CastingEvaluationState
        from obsidianlink.evaluation.continuous_casting import (
            ContinuousCastingEvaluationState,
        )

        backend = FakeEnvironmentBackend()
        backend.open()
        c4_task = _task()
        backend.reset(c4_task)
        c1 = CastingEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            target_cell=(0, 0, 0),
            initial_target_block="air",
            current_target_block="obsidian",
        )
        c2 = ContinuousCastingEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=tuple(
                ContinuousCastingCellTruth(
                    target_cell=target,
                    initial_block="air",
                    current_block="air",
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
                for target in ((2, 4, 3), (3, 4, 3), (4, 4, 3))
            ),
            max_environment_steps=MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=640,
        )
        c4 = _build_minimal_c4_state_at_step_zero()
        backend.set_casting_evaluation_state(c1)
        backend.set_continuous_casting_evaluation_state(c2)
        backend.set_ignition_evaluation_state(c4)
        self.assertIs(backend.get_casting_evaluation_state(), c1)
        self.assertIs(
            backend.get_continuous_casting_evaluation_state(), c2
        )
        self.assertIs(backend.get_ignition_evaluation_state(), c4)
        with self.assertRaisesRegex(
            RuntimeError, "frame evaluation state is unavailable"
        ):
            backend.get_frame_evaluation_state()
        backend.close()

    def test_c3_surface_does_not_leak_into_c4(self) -> None:
        c3_task = TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": "casting_s_c3_fixed_seed_0",
                "route": "lava_casting",
                "difficulty": 3,
                "agent_ids": [AGENT_ID],
                "world_seed": 0,
                "instruction": "test",
                "spawn_positions": {AGENT_ID: [0, 4, 0]},
                "initial_inventories": {
                    AGENT_ID: {
                        "water_bucket": 14,
                        "lava_bucket": 14,
                        "cobblestone": 28,
                    }
                },
                "workflow": "casting_s_c3_fixed",
                "milestones": ["task_reset", "valid_portal_frame"],
                "limits": {
                    "max_environment_steps": 640,
                    "max_model_calls": 1,
                    "max_game_time_seconds": 600,
                },
                "split": "development",
            }
        )
        c3_frame = _build_minimal_c3_frame_state_at_step_zero(
            episode_id="casting_s_c3_fixed_seed_0"
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(c3_task)
        backend.set_frame_evaluation_state(c3_frame)
        self.assertIs(backend.get_frame_evaluation_state(), c3_frame)
        c4_state = _build_minimal_c4_state_at_step_zero()
        with self.assertRaisesRegex(
            ValueError, "casting_s_c4_fixed workflow"
        ):
            backend.set_ignition_evaluation_state(c4_state)
        backend.close()


# ----------------------------------------------------------------------
# Information isolation (AST / import graph)
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

    def test_evaluator_source_does_not_read_scenario_or_instruction(
        self,
    ) -> None:
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
                    self.assertEqual(node.args.args[0].arg, "self")
                if node.args.args and len(node.args.args) > 1:
                    self.assertEqual(node.args.args[1].arg, "state")
                if len(node.args.args) > 1:
                    annotation = node.args.args[1].annotation
                    if annotation is not None and isinstance(
                        annotation, ast.Name
                    ):
                        self.assertEqual(
                            annotation.id,
                            "FrozenIgnitionEvaluationState",
                        )


# ----------------------------------------------------------------------
# C1 / C2 / C3 / portal regression
# ----------------------------------------------------------------------


class C1C2C3PortalRegressionTests(unittest.TestCase):
    def test_r3_casting_evaluator_still_works(self) -> None:
        state = {
            "episode_id": "casting_c1_fixed_seed_0",
            "step_id": 10,
            "target_cell": (0, 0, 0),
            "initial_target_block": "air",
            "current_target_block": "obsidian",
            "target_update_evidence": CastingTransitionEvidence(
                before_block="air",
                after_block="obsidian",
                update_step=10,
            ),
            "water_truth": CastingFluidTruth(present=True, evidence_step=8),
            "lava_truth": CastingFluidTruth(present=True, evidence_step=9),
            "relevant_action_steps": (8, 9, 10),
            "causality_window_steps": 4,
            "episode_terminated": True,
            "terminated_step": 10,
            "terminated_reason": "driver_done",
            "max_environment_steps": 240,
            "max_game_time_seconds": 180.0,
        }
        from obsidianlink.evaluation.casting import CastingEvaluationState

        result = CastingEvaluator().evaluate(
            CastingEvaluationState(**state)
        )
        self.assertTrue(result.success)

    def test_r5_continuous_casting_evaluator_still_works(self) -> None:
        cells = tuple(
            ContinuousCastingCellTruth(
                target_cell=target,
                initial_block="air",
                current_block="obsidian",
                water_truth=CastingFluidTruth(
                    present=True, evidence_step=4 + 3 * index
                ),
                lava_truth=CastingFluidTruth(
                    present=True, evidence_step=3 + 3 * index
                ),
                transition_evidence=CastingTransitionEvidence(
                    before_block="air",
                    after_block="obsidian",
                    update_step=5 + 3 * index,
                ),
                relevant_action_steps=(
                    3 + 3 * index,
                    4 + 3 * index,
                    5 + 3 * index,
                ),
            )
            for index, target in enumerate(
                ((2, 4, 3), (3, 4, 3), (4, 4, 3))
            )
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

    def test_c3_frame_evaluator_still_works(self) -> None:
        frame_state = _build_frame_state()
        result = FrozenFrameEvaluator().evaluate(frame_state)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, FRAME_OUTCOME_SUCCESS)

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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _walk(obj: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
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


def _build_state_at_step_zero_for_fakebackend(
    *,
    episode_id: str = EPISODE_ID,
) -> FrozenIgnitionEvaluationState:
    """Build a minimal C4 state at step_id=0 for FakeBackend tests.

    The C3 frame state is partially supplied; the C4 wrapper only
    requires the type and identity guards. We do not run the
    evaluator on this state — it exists only to exercise the
    FakeBackend's set/get surface.
    """
    cells = []
    for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
        if index == 0:
            # Build a cell whose action_evidence shares the
            # requested ``episode_id`` so the C3 frame state
            # validator accepts it.
            cells.append(
                FrozenFrameCellTruth(
                    target_cell=target,
                    initial_block="air",
                    current_block="obsidian",
                    water_truth=CastingFluidTruth(
                        present=True, evidence_step=0,
                    ),
                    lava_truth=CastingFluidTruth(
                        present=True, evidence_step=0,
                    ),
                    transition_evidence=CastingTransitionEvidence(
                        before_block="air",
                        after_block="obsidian",
                        update_step=0,
                    ),
                    relevant_action_steps=(0,),
                    action_evidence=(
                        FrozenFrameActionEvidence(
                            episode_id=episode_id,
                            step_id=0,
                            agent_id=AGENT_ID,
                            action_type="use_item",
                            item="water_bucket",
                            target_cell=target,
                        ),
                    ),
                    transition_action_step=0,
                )
            )
        else:
            # truth_missing at step 0: but the C3 evaluator
            # would reject this. We use a cell that has empty
            # relevant actions and current_block = air so the
            # C3 evaluator reports truth_missing. We only need
            # the state to be constructable here; we don't run
            # the evaluator inside the FakeBackend.
            cells.append(
                FrozenFrameCellTruth(
                    target_cell=target,
                    initial_block="air",
                    current_block=None,
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
            )
    interior = _all_allowed_interior()
    frame_state = FrozenFrameEvaluationState(
        episode_id=episode_id,
        step_id=0,
        cells=tuple(cells),
        interior_cells=interior,
        agent_id=AGENT_ID,
        causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
        episode_terminated=False,
        current_time_seconds=0.0,
        max_environment_steps=MAX_ENVIRONMENT_STEPS,
        max_game_time_seconds=640,
    )
    return FrozenIgnitionEvaluationState(
        episode_id=episode_id,
        step_id=0,
        frame_state=frame_state,
        latched_frame_identity=build_c4_c3_frame_identity(
            episode_id=episode_id, step_id=0,
        ),
        agent_id=AGENT_ID,
        causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
        max_environment_steps=MAX_ENVIRONMENT_STEPS,
        max_game_time_seconds=640,
    )


def _build_minimal_c4_state_at_step_zero() -> FrozenIgnitionEvaluationState:
    """Build a minimal C4 state at step_id=0 for FakeBackend tests.

    The C3 frame state is partially supplied; the C4 wrapper only
    requires the type and identity guards. We do not run the
    evaluator on this state — it exists only to exercise the
    FakeBackend's set/get surface.
    """
    cells = []
    for index, target in enumerate(CASTING_S_C3_FRAME_CELLS):
        if index == 0:
            cells.append(
                _success_cell(
                    target,
                    last_action_step=0,
                    relevant_action_steps=(0,),
                )
            )
        else:
            cells.append(
                FrozenFrameCellTruth(
                    target_cell=target,
                    initial_block="air",
                    current_block=None,
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
            )
    interior = _all_allowed_interior()
    frame_state = FrozenFrameEvaluationState(
        episode_id=EPISODE_ID,
        step_id=0,
        cells=tuple(cells),
        interior_cells=interior,
        agent_id=AGENT_ID,
        causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
        episode_terminated=False,
        current_time_seconds=0.0,
        max_environment_steps=MAX_ENVIRONMENT_STEPS,
        max_game_time_seconds=640,
    )
    return FrozenIgnitionEvaluationState(
        episode_id=EPISODE_ID,
        step_id=0,
        frame_state=frame_state,
        latched_frame_identity=build_c4_c3_frame_identity(
            episode_id=EPISODE_ID, step_id=0,
        ),
        agent_id=AGENT_ID,
        causality_window_steps=CASTING_S_C4_CAUSALITY_WINDOW_STEPS,
        max_environment_steps=MAX_ENVIRONMENT_STEPS,
        max_game_time_seconds=640,
    )


def _build_minimal_c3_frame_state_at_step_zero(
    episode_id: str = "casting_s_c3_fixed_seed_0",
) -> FrozenFrameEvaluationState:
    """Build a minimal C3 frame state at step_id=0."""
    cell0 = FrozenFrameCellTruth(
        target_cell=CASTING_S_C3_FRAME_CELLS[0],
        initial_block="air",
        current_block="obsidian",
        water_truth=CastingFluidTruth(present=True, evidence_step=0),
        lava_truth=CastingFluidTruth(present=True, evidence_step=0),
        transition_evidence=CastingTransitionEvidence(
            before_block="air", after_block="obsidian", update_step=0
        ),
        relevant_action_steps=(0,),
        action_evidence=(
            FrozenFrameActionEvidence(
                episode_id=episode_id,
                step_id=0,
                agent_id=AGENT_ID,
                action_type="use_item",
                item="water_bucket",
                target_cell=CASTING_S_C3_FRAME_CELLS[0],
            ),
        ),
        transition_action_step=0,
    )
    other_cells = tuple(
        FrozenFrameCellTruth(
            target_cell=tgt,
            initial_block="air",
            current_block=None,
            water_truth=None,
            lava_truth=None,
            transition_evidence=None,
            relevant_action_steps=(),
        )
        for tgt in CASTING_S_C3_FRAME_CELLS[1:]
    )
    return FrozenFrameEvaluationState(
        episode_id=episode_id,
        step_id=0,
        cells=(cell0,) + other_cells,
        interior_cells=tuple(
            FrozenFrameInteriorCellTruth(
                target_cell=c, current_block="air"
            )
            for c in CASTING_S_C3_INTERIOR_CELLS
        ),
        agent_id=AGENT_ID,
        causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
        max_environment_steps=MAX_ENVIRONMENT_STEPS,
        max_game_time_seconds=640,
    )


if __name__ == "__main__":
    unittest.main()
