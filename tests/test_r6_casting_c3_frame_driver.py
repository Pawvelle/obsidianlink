"""Offline tests for the R6 Casting-S-C3 deterministic frame driver.

These tests prove, in code, that:

* :class:`PublicC3FrameDriverContext` is a strictly-typed, frozen,
  immutable public driver context; the driver never reads
  ``scenario_parameters`` or ``evaluator_contract`` from the
  original task.
* :func:`build_casting_s_c3_frame_action_plan` builds a fixed,
  deterministic, ordered 14-cell plan whose cell offsets exactly
  match the locked :data:`CASTING_S_C3_FRAME_CELLS` order and
  whose default length / wait count fit within the 640-step,
  600-second, 320-wait, 336-step-plan budgets.
* :func:`run_casting_s_c3_frame_driver` walks the plan on the
  :class:`FakeEnvironmentBackend`, never imports the frame
  evaluator, never calls
  :meth:`FakeEnvironmentBackend.set_frame_evaluation_state` /
  :meth:`FakeEnvironmentBackend.get_frame_evaluation_state` /
  :meth:`FakeEnvironmentBackend.clear_frame_evaluation_state`,
  and never reads ``scenario_parameters`` / ``evaluator_contract``
  / ``FrozenFrameEvaluationState` or any evaluator-only field.
* Every step / time / wait / plan / total-recovery budget has a
  hard, fail-closed bound.
* The driver's recovery protocol retries the typed
  :class:`RecoverableBackendError` deterministically and fails
  closed on any non-recoverable exception.
* The driver's events carry ``episode_id`` / ``step_id`` /
  ``agent_id`` / ``cell_index`` / ``target_offset`` /
  ``relevant_action``; the result is deeply immutable, the
  ``as_dict()`` snapshot is JSON-serializable, and the same input
  yields the same action sequence / events / ``as_dict()`` snapshot
  on repeated runs.
* An Observation ``__getattribute__`` guard fails closed if the
  driver ever tries to read a hidden ``target_cell`` /
  ``current_block`` / ``outcome`` / ``success`` / ``failure_type``
  / ``blocking_conditions`` / ``per_cell_outcomes`` /
  ``first_failed_cell`` / ``completed_cells`` field.
* The test orchestrator (this file) is the *only* place that
  injects the R6 C3 frame evaluator truth via
  :meth:`FakeEnvironmentBackend.set_frame_evaluation_state`,
  and the :class:`FrozenFrameEvaluator` correctly returns
  ``success`` for 14 success cells + 6 allowed interior cells and
  ``partial_completion`` for 1–13 success cells.
* Wrong block, interior blocker, missing causality evidence, and
  identity mismatches all produce the closed-set failure outcomes
  the C3 frame evaluator documents.
* The C1, C2, portal, and existing R6 C3 frame evaluator
  regression tests all stay green.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
import time as _time
import unittest
from pathlib import Path
from typing import Any, Mapping

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)
from obsidianlink.drivers.casting_s_c3_frame import (
    AGENT_ID,
    ALLOWED_C3_FRAME_ACTION_TYPES,
    ALLOWED_C3_FRAME_FAMILIES,
    ALLOWED_C3_FRAME_LAYOUTS,
    ALLOWED_C3_FRAME_LEVELS,
    ALLOWED_C3_FRAME_MODES,
    ALLOWED_C3_FRAME_TARGETS,
    C3_FRAME_GRID_X_MAX,
    C3_FRAME_GRID_X_MIN,
    C3_FRAME_GRID_Y_MAX,
    C3_FRAME_GRID_Y_MIN,
    C3_FRAME_GRID_Z_MAX,
    C3_FRAME_GRID_Z_MIN,
    CASTING_S_C3_FRAME_CELLS,
    CASTING_S_C3_TARGET_CELL_COUNT,
    DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    DEFAULT_MAX_WAIT_STEPS,
    DEFAULT_OBSIDIAN_WAIT_STEPS,
    DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    DRIVER_STATUS_BLOCKED,
    DRIVER_STATUS_COMPLETED,
    DRIVER_STATUS_FAILED,
    DRIVER_STATUSES,
    FAMILY_C3_FRAME,
    LAYOUT_C3_FRAME,
    LEVEL_C3_FRAME,
    MAX_FRAME_PLAN_STEPS,
    MAX_FRAME_PLAN_WAIT_STEPS,
    MAX_RECOVERIES_PER_ACTION,
    MAX_TOTAL_RECOVERY_BUDGET,
    MODE_C3_FRAME,
    PHASE_PLACE_LAVA,
    PHASE_PLACE_SUPPORT,
    PHASE_PLACE_WATER,
    PHASE_PREPARE,
    PHASE_RECOVERY,
    PHASE_VALUES,
    PHASE_WAIT_FOR_OBSIDIAN,
    RECOVERIES_PER_USE_ITEM_DEFAULT,
    TOTAL_RECOVERY_BUDGET_DEFAULT,
    WORKFLOW_C3_FRAME,
    CastingC3FrameDriverResult,
    CastingC3FramePlanStep,
    PublicC3FrameDriverContext,
    build_casting_s_c3_frame_action_plan,
    run_casting_s_c3_frame_driver,
)
from obsidianlink.core.casting_s_c3_frame_context import (
    build_public_c3_frame_driver_context_from_task,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.env.capabilities import (
    BackendCapabilities,
    CapabilityMismatchError,
)
from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
)
from obsidianlink.evaluation.casting_frame_evaluator import (
    CASTING_S_C3_INTERIOR_CELLS,
    FrozenFrameActionEvidence,
    FrozenFrameCellTruth,
    FrozenFrameEvaluationState,
    FrozenFrameEvaluator,
    FrozenFrameInteriorCellTruth,
    OUTCOME_INTERIOR_BLOCKED,
    OUTCOME_PARTIAL_COMPLEMENT,
    OUTCOME_SUCCESS,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    PER_CELL_INCOMPLETE,
    PER_CELL_SUCCESS,
    PER_INTERIOR_CELL_ALLOWED,
)


EPISODE_ID = "casting_s_c3_fixed_seed_0"
WRONG_EPISODE_ID = "casting_s_c3_fixed_seed_99"
WRONG_AGENT_ID = "agent_2"
INVENTORY: dict[str, int] = {
    "water_bucket": 14,
    "lava_bucket": 14,
    "cobblestone": 28,
}


# ----------------------------------------------------------------------
# Task / context helpers
# ----------------------------------------------------------------------


def _task_dict(
    *,
    inventory: dict[str, int] | None = None,
    workflow: str = WORKFLOW_C3_FRAME,
    family: str = FAMILY_C3_FRAME,
    mode: str = MODE_C3_FRAME,
    level: str = LEVEL_C3_FRAME,
    layout: str = LAYOUT_C3_FRAME,
    episode_id: str = EPISODE_ID,
    max_environment_steps: int = 640,
    max_game_time_seconds: int = 600,
    public_spec: dict[str, Any] | None = None,
    include_evaluator_contract: bool = False,
) -> dict[str, Any]:
    if public_spec is None:
        public_spec = {
            "coordinate_space": "task_origin_relative",
            "task_origin_marker": "visible",
            "frame_plan": {
                "orientation": "plane_z",
                "min_corner": [0, 0, 1],
                "width": 4,
                "height": 5,
                "require_full_ring": True,
                "minecraft_minimum_required_block_count": 10,
                "benchmark_required_full_ring_block_count": 14,
                "required_corner_count": 4,
                "interior_allowlist": ["air", "nether_portal", "fire"],
                "fixed_offsets": [list(cell) for cell in CASTING_S_C3_FRAME_CELLS],
            },
            "ignition_plan": {
                "required": False,
                "action": None,
                "item": None,
                "target_offset": None,
            },
            "nether_entry_goal": {
                "required": False,
                "designated_agent_ids": [],
                "target_dimension": None,
            },
        }
    scenario_parameters: dict[str, Any] = {
        "task_family": family,
        "agent_mode": mode,
        "task_level": level,
        "layout_type": layout,
        "compatibility_task_name": WORKFLOW_C3_FRAME,
        "implementation_status": "contract_only",
        "world_dimension": "minecraft:overworld",
        "layout": "fixed_controlled",
        "mechanics_required": "vanilla_water_lava_block_update",
        "public_task_spec": public_spec,
        "allow_minecraft_commands": False,
        "allow_evaluator_world_mutation": False,
        "allow_live_run": False,
        "requires_explicit_live_run_approval": True,
    }
    if include_evaluator_contract:
        scenario_parameters["evaluator_contract"] = {
            "frame_attribution": {
                "baseline_policy": "all_benchmark_frame_cells_non_obsidian",
                "required_mechanism": "vanilla_water_lava_block_update",
                "required_action_type": "use_item",
                "required_items": ["water_bucket", "lava_bucket"],
                "direct_obsidian_placement_allowed": False,
                "causality_window_steps": 4,
                "fail_closed_on_missing_truth": True,
                "require_episode_and_step_identity": True,
            },
            "activation_attribution": {"required": False},
            "nether_entry_attribution": {"required": False},
        }
    return {
        "schema_version": "0.1",
        "task_id": episode_id,
        "route": "lava_casting",
        "difficulty": 3,
        "agent_ids": [AGENT_ID],
        "world_seed": 0,
        "instruction": "R6 C3 frame driver unit-test task.",
        "spawn_positions": {AGENT_ID: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID: dict(inventory if inventory is not None else INVENTORY)
        },
        "workflow": workflow,
        "milestones": [
            "task_reset",
            "first_obsidian_cast",
            "build_site_selected",
            "valid_portal_frame",
        ],
        "limits": {
            "max_environment_steps": max_environment_steps,
            "max_model_calls": 1,
            "max_game_time_seconds": max_game_time_seconds,
        },
        "split": "development",
        "scenario_parameters": scenario_parameters,
    }


def _task(
    *,
    inventory: dict[str, int] | None = None,
    workflow: str = WORKFLOW_C3_FRAME,
    family: str = FAMILY_C3_FRAME,
    mode: str = MODE_C3_FRAME,
    level: str = LEVEL_C3_FRAME,
    layout: str = LAYOUT_C3_FRAME,
    episode_id: str = EPISODE_ID,
    max_environment_steps: int = 640,
    max_game_time_seconds: int = 600,
    public_spec: dict[str, Any] | None = None,
    include_evaluator_contract: bool = False,
) -> TaskInstance:
    return TaskInstance.from_dict(
        _task_dict(
            inventory=inventory,
            workflow=workflow,
            family=family,
            mode=mode,
            level=level,
            layout=layout,
            episode_id=episode_id,
            max_environment_steps=max_environment_steps,
            max_game_time_seconds=max_game_time_seconds,
            public_spec=public_spec,
            include_evaluator_contract=include_evaluator_contract,
        )
    )


def _context(
    *,
    inventory: dict[str, int] | None = None,
    family: str = FAMILY_C3_FRAME,
    mode: str = MODE_C3_FRAME,
    level: str = LEVEL_C3_FRAME,
    layout: str = LAYOUT_C3_FRAME,
    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C3_FRAME_CELLS,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
    workflow: str = WORKFLOW_C3_FRAME,
    task_step_limit: int = 640,
    task_time_limit: float = 600.0,
) -> PublicC3FrameDriverContext:
    return PublicC3FrameDriverContext(
        episode_id=episode_id,
        workflow=workflow,
        family=family,
        mode=mode,
        level=level,
        layout=layout,
        agent_id=agent_id,
        target_offsets=target_offsets,
        initial_inventory=(
            inventory if inventory is not None else dict(INVENTORY)
        ),
        task_step_limit=task_step_limit,
        task_time_limit=task_time_limit,
    )


# ----------------------------------------------------------------------
# C3 frame world truth (test orchestrator only)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class C3FrameWorldTruth:
    """Test-only description of the R6 C3 frame casting world.

    The orchestrator (functions below) is the *only* component that
    consumes a :class:`C3FrameWorldTruth`. The driver never sees this
    object. Each target cell carries the evaluator truth required by
    :class:`FrozenFrameCellTruth`; the 6 interior cells carry the
    truth required by :class:`FrozenFrameInteriorCellTruth`.

    The default values match the R6 C3 frame driver's default plan
    (24 steps per cell, 14 cells). The first ``use_item(lava_bucket)``
    is at global step ``9 + 24 * cell_index``; the first
    ``use_item(water_bucket)`` is at global step ``16 + 24 *
    cell_index``; the block update is at ``20 + 24 * cell_index``,
    which is within the 4-step causality window of the last relevant
    action (the water ``use_item``).
    """

    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C3_FRAME_CELLS
    initial_blocks: tuple[str, ...] = ("air",) * 14
    current_blocks: tuple[str, ...] = ("obsidian",) * 14
    interior_current_blocks: tuple[str | None, ...] = ("air",) * 6
    water_truth: tuple[tuple[bool, int | None], ...] = tuple(
        (True, 16 + 24 * index) for index in range(14)
    )
    lava_truth: tuple[tuple[bool, int | None], ...] = tuple(
        (True, 9 + 24 * index) for index in range(14)
    )
    transition_steps: tuple[int | None, ...] = tuple(
        20 + 24 * index for index in range(14)
    )
    transition_before_blocks: tuple[str | None, ...] = ("air",) * 14
    transition_after_blocks: tuple[str | None, ...] = ("obsidian",) * 14
    per_cell_relevant_action_steps: tuple[tuple[int, ...], ...] = tuple(
        (9 + 24 * index, 16 + 24 * index) for index in range(14)
    )
    terminated_step: int = 14 * 24
    terminated_reason: str = "driver_done"
    current_time_seconds: float = 0.0
    causality_window_steps: int = 4


def _actions(
    target_cell: tuple[int, int, int],
    records: tuple[tuple[int, str], ...],
    *,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
) -> tuple[FrozenFrameActionEvidence, ...]:
    return tuple(
        FrozenFrameActionEvidence(
            episode_id=episode_id,
            step_id=step_id,
            agent_id=agent_id,
            action_type="use_item",
            item=item,
            target_cell=target_cell,
        )
        for step_id, item in records
    )


def _state(
    world: C3FrameWorldTruth,
    *,
    task: TaskInstance,
    relevant_records: tuple[tuple[tuple[int, int, str], ...], ...] | None = None,
    current_time_seconds: float | None = None,
) -> FrozenFrameEvaluationState:
    """Build a :class:`FrozenFrameEvaluationState` from a world truth.

    ``relevant_records`` is a 14-tuple of ``(step_id, item)`` pairs
    for each target cell. When ``None``, the world supplies the
    evidence steps from ``per_cell_relevant_action_steps`` and the
    items are derived from the per-cell record tuples.
    """
    cells: list[FrozenFrameCellTruth] = []
    if relevant_records is None:
        relevant_records = tuple(
            tuple((step, "lava_bucket" if step % 2 == 0 else "water_bucket")
                  for step in world.per_cell_relevant_action_steps[index])
            for index in range(14)
        )
    for index, target_cell in enumerate(world.target_offsets):
        records = relevant_records[index]
        steps = tuple(step for step, _item in records)
        if not steps:
            cells.append(
                FrozenFrameCellTruth(
                    target_cell=target_cell,
                    initial_block=world.initial_blocks[index],
                    current_block=world.current_blocks[index],
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                    action_evidence=(),
                )
            )
            continue
        last_action = max(steps)
        cells.append(
            FrozenFrameCellTruth(
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
                relevant_action_steps=steps,
                action_evidence=_actions(target_cell, records),
                transition_action_step=last_action,
            )
        )
    interior = tuple(
        FrozenFrameInteriorCellTruth(
            target_cell=cell, current_block=block
        )
        for cell, block in zip(
            CASTING_S_C3_INTERIOR_CELLS, world.interior_current_blocks
        )
    )
    return FrozenFrameEvaluationState(
        episode_id=task.task_id,
        step_id=world.terminated_step,
        cells=tuple(cells),
        interior_cells=interior,
        agent_id=AGENT_ID,
        causality_window_steps=world.causality_window_steps,
        episode_terminated=True,
        terminated_step=world.terminated_step,
        terminated_reason=world.terminated_reason,
        current_time_seconds=(
            current_time_seconds
            if current_time_seconds is not None
            else world.current_time_seconds
        ),
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )


def _state_from_driver(
    driver_result: CastingC3FrameDriverResult,
    world: C3FrameWorldTruth,
    *,
    task: TaskInstance,
    current_time_seconds: float | None = None,
) -> FrozenFrameEvaluationState:
    """Build a :class:`FrozenFrameEvaluationState` from a driver result.

    The orchestrator derives per-cell ``(step_id, item)`` records
    from ``driver_result.per_cell_relevant_action_records`` and
    hands them to :func:`_state`. The driver never sees this code
    path.
    """
    records: list[tuple[tuple[int, int, str], ...]] = []
    for index in range(14):
        cell_records = driver_result.per_cell_relevant_action_records.get(
            index, ()
        )
        records.append(
            tuple((int(step_id), str(item)) for step_id, item in cell_records)
        )
    return _state(
        world,
        task=task,
        relevant_records=tuple(records),
        current_time_seconds=current_time_seconds,
    )


def run_orchestrator(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC3FrameDriverResult,
    world: C3FrameWorldTruth,
    *,
    task: TaskInstance,
    current_time_seconds: float | None = None,
):
    """Build the orchestrator-side state and call the frame evaluator.

    The orchestrator (this function) is the *only* place in the R6
    C3 frame driver test suite that calls
    ``set_frame_evaluation_state`` /
    ``get_frame_evaluation_state`` and the
    :class:`FrozenFrameEvaluator`. The driver never sees the truth
    surface.
    """
    state = _state_from_driver(
        driver_result,
        world,
        task=task,
        current_time_seconds=current_time_seconds,
    )
    backend.set_frame_evaluation_state(state)
    return FrozenFrameEvaluator().evaluate(
        backend.get_frame_evaluation_state()
    )


# ----------------------------------------------------------------------
# Public context / contract tests
# ----------------------------------------------------------------------


class PublicContextContractTests(unittest.TestCase):
    """Static contract: identity, immutability, fail-closed validation."""

    def test_action_allowlist_is_closed(self) -> None:
        self.assertEqual(
            ALLOWED_C3_FRAME_ACTION_TYPES,
            frozenset(
                {"equip_item", "use_item", "place_block", "wait"}
            ),
        )
        self.assertEqual(
            ALLOWED_C3_FRAME_TARGETS,
            frozenset({"water_bucket", "lava_bucket", "cobblestone"}),
        )
        self.assertEqual(ALLOWED_C3_FRAME_FAMILIES, frozenset({"casting"}))
        self.assertEqual(ALLOWED_C3_FRAME_MODES, frozenset({"single"}))
        self.assertEqual(ALLOWED_C3_FRAME_LEVELS, frozenset({"C3"}))
        self.assertEqual(ALLOWED_C3_FRAME_LAYOUTS, frozenset({"fixed"}))
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
        self.assertEqual(AGENT_ID, "agent_1")
        self.assertEqual(WORKFLOW_C3_FRAME, "casting_s_c3_fixed")
        self.assertEqual(FAMILY_C3_FRAME, "casting")
        self.assertEqual(MODE_C3_FRAME, "single")
        self.assertEqual(LEVEL_C3_FRAME, "C3")
        self.assertEqual(LAYOUT_C3_FRAME, "fixed")

    def test_default_target_cells_match_public_spec(self) -> None:
        self.assertEqual(len(CASTING_S_C3_FRAME_CELLS), 14)
        self.assertEqual(
            CASTING_S_C3_TARGET_CELL_COUNT, 14
        )
        # The driver must agree with the frame evaluator on the
        # canonical 14-cell order. Both modules have their own copy
        # of this constant; the duplication is intentional.
        from obsidianlink.evaluation.casting_frame_evaluator import (
            CASTING_S_C3_FRAME_CELLS as EVAL_FRAME_CELLS,
        )

        self.assertEqual(CASTING_S_C3_FRAME_CELLS, EVAL_FRAME_CELLS)

    def test_context_rejects_wrong_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "workflow"):
            _context(workflow="casting_c1_fixed")

    def test_context_rejects_wrong_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "family"):
            _context(family="ruined")

    def test_context_rejects_wrong_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            _context(mode="multi")

    def test_context_rejects_wrong_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "level"):
            _context(level="C2")

    def test_context_rejects_wrong_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "layout"):
            _context(layout="random")

    def test_context_rejects_wrong_agent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id"):
            _context(agent_id="agent_2")

    def test_context_rejects_wrong_target_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "must contain exactly 14"
        ):
            _context(target_offsets=CASTING_S_C3_FRAME_CELLS[:7])

    def test_context_rejects_reordered_target_offsets(self) -> None:
        # Swap the first two cells; both are still in the public
        # grid so the order check (not the grid check) must fire.
        reordered = (
            CASTING_S_C3_FRAME_CELLS[1],
            CASTING_S_C3_FRAME_CELLS[0],
        ) + CASTING_S_C3_FRAME_CELLS[2:]
        with self.assertRaisesRegex(
            ValueError, "must exactly match the locked"
        ):
            _context(target_offsets=reordered)

    def test_context_rejects_duplicate_target_offsets(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "must not contain duplicates"
        ):
            _context(
                target_offsets=(
                    (0, 0, 1),
                    (0, 0, 1),
                )
                + CASTING_S_C3_FRAME_CELLS[2:14]
            )

    def test_context_rejects_out_of_grid_target_offset(self) -> None:
        out_of_grid = CASTING_S_C3_FRAME_CELLS[:13] + ((99, 99, 99),)
        with self.assertRaisesRegex(ValueError, "outside"):
            _context(target_offsets=out_of_grid)

    def test_context_rejects_non_int_offset(self) -> None:
        bad = CASTING_S_C3_FRAME_CELLS[:13] + ((1, 2, "1"),)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "strict integers"):
            _context(target_offsets=bad)

    def test_context_rejects_bool_offset(self) -> None:
        bad = CASTING_S_C3_FRAME_CELLS[:13] + ((True, False, 1),)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "strict integers"):
            _context(target_offsets=bad)

    def test_context_rejects_empty_episode_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            _context(episode_id="")

    def test_context_rejects_non_positive_task_step_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_step_limit"):
            _context(task_step_limit=0)

    def test_context_rejects_task_step_limit_below_cell_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "task_step_limit must be at least"
        ):
            _context(task_step_limit=10)

    def test_context_rejects_inventory_missing_water_bucket(self) -> None:
        inventory = {"lava_bucket": 14, "cobblestone": 28}
        with self.assertRaisesRegex(ValueError, "water_bucket"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_missing_lava_bucket(self) -> None:
        inventory = {"water_bucket": 14, "cobblestone": 28}
        with self.assertRaisesRegex(ValueError, "lava_bucket"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_missing_cobblestone(self) -> None:
        inventory = {"water_bucket": 14, "lava_bucket": 14}
        with self.assertRaisesRegex(ValueError, "cobblestone"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_negative_quantity(self) -> None:
        inventory = {"water_bucket": -1, "lava_bucket": 14, "cobblestone": 28}
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_context_rejects_inventory_bool_quantity(self) -> None:
        inventory = {"water_bucket": True, "lava_bucket": 14, "cobblestone": 28}
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_context_rejects_inventory_unknown_item(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "cobblestone": 28,
            "obsidian": 1,
        }
        with self.assertRaisesRegex(ValueError, "forbidden item"):
            _context(inventory=inventory)

    def test_context_is_frozen(self) -> None:
        context = _context()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.episode_id = "tampered"  # type: ignore[misc]

    def test_context_inventory_is_mapping_proxy(self) -> None:
        context = _context()
        self.assertIsInstance(
            context.initial_inventory, Mapping
        )
        with self.assertRaises(TypeError):
            context.initial_inventory["water_bucket"] = 0  # type: ignore[index]

    def test_context_target_offsets_are_frozen_tuple(self) -> None:
        context = _context()
        self.assertIsInstance(context.target_offsets, tuple)
        with self.assertRaises(Exception):
            context.target_offsets[0] = (1, 1, 1)  # type: ignore[index]

    def test_context_grid_bounds_are_documented(self) -> None:
        self.assertEqual(C3_FRAME_GRID_X_MIN, 0)
        self.assertEqual(C3_FRAME_GRID_X_MAX, 3)
        self.assertEqual(C3_FRAME_GRID_Y_MIN, 0)
        self.assertEqual(C3_FRAME_GRID_Y_MAX, 4)
        self.assertEqual(C3_FRAME_GRID_Z_MIN, 1)
        self.assertEqual(C3_FRAME_GRID_Z_MAX, 1)


# ----------------------------------------------------------------------
# Build-public-context-from-task tests
# ----------------------------------------------------------------------


class BuildContextFromTaskTests(unittest.TestCase):
    """The orchestrator helper builds a strict public context."""

    def test_build_context_from_task_succeeds(self) -> None:
        task = _task()
        context = build_public_c3_frame_driver_context_from_task(task)
        self.assertEqual(context.episode_id, task.task_id)
        self.assertEqual(context.workflow, WORKFLOW_C3_FRAME)
        self.assertEqual(context.family, FAMILY_C3_FRAME)
        self.assertEqual(context.mode, MODE_C3_FRAME)
        self.assertEqual(context.level, LEVEL_C3_FRAME)
        self.assertEqual(context.layout, LAYOUT_C3_FRAME)
        self.assertEqual(context.agent_id, AGENT_ID)
        self.assertEqual(context.target_offsets, CASTING_S_C3_FRAME_CELLS)
        self.assertEqual(context.task_step_limit, 640)
        self.assertEqual(context.task_time_limit, 600.0)

    def test_build_context_rejects_wrong_workflow(self) -> None:
        task = _task(workflow="casting_c1_fixed")
        with self.assertRaisesRegex(ValueError, "workflow"):
            build_public_c3_frame_driver_context_from_task(task)

    def test_build_context_ignores_evaluator_contract(self) -> None:
        # The helper must build the context purely from the
        # public task spec; ``evaluator_contract`` is evaluator-only
        # and must not affect the result.
        task = _task(include_evaluator_contract=True)
        context = build_public_c3_frame_driver_context_from_task(task)
        # No exception is raised; the context is built from
        # public_task_spec only. evaluator_contract lives in
        # ``task.scenario_parameters`` but the helper does not
        # read it.
        self.assertEqual(context.target_offsets, CASTING_S_C3_FRAME_CELLS)

    def test_build_context_rejects_bool_coordinate_before_normalization(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["frame_plan"][
            "fixed_offsets"
        ][1][0] = True
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "strict integers"):
            build_public_c3_frame_driver_context_from_task(task)

    def test_build_context_rejects_numeric_string_before_normalization(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["frame_plan"][
            "fixed_offsets"
        ][1][0] = "1"
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "strict integers"):
            build_public_c3_frame_driver_context_from_task(task)

    def test_public_context_helper_imports_in_clean_interpreter(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from obsidianlink.core.casting_s_c3_frame_context "
                    "import build_public_c3_frame_driver_context_from_task; "
                    "print(build_public_c3_frame_driver_context_from_task.__name__)"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            "build_public_c3_frame_driver_context_from_task",
        )


# ----------------------------------------------------------------------
# Plan builder tests
# ----------------------------------------------------------------------


class PlanBuilderContractTests(unittest.TestCase):
    """The 14-cell plan is fixed, ordered, and bounded."""

    def test_default_plan_length(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        # 24 steps per cell × 14 cells = 336.
        self.assertEqual(len(plan), 336)
        # Plan length is well under the 640-step task limit.
        self.assertLess(len(plan), 640)
        # Plan length is well under the hard cap.
        self.assertLessEqual(len(plan), MAX_FRAME_PLAN_STEPS)

    def test_default_plan_uses_phases(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        seen_phases: set[str] = set()
        for step in plan:
            self.assertIn(step.phase, PHASE_VALUES)
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

    def test_default_plan_action_allowlist(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        for step in plan:
            self.assertIsInstance(step, CastingC3FramePlanStep)
            self.assertIn(step.action.action_type, ALLOWED_C3_FRAME_ACTION_TYPES)
            if step.action.target is not None:
                self.assertIn(step.action.target, ALLOWED_C3_FRAME_TARGETS)

    def test_default_plan_target_offsets_match_contract(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        # The first step of each cell's segment must carry the
        # cell's target offset.
        cell_offsets: list[tuple[int, int, int]] = []
        for step in plan:
            if step.label.endswith(".prepare.select_lava"):
                cell_offsets.append(step.target_offset)
        self.assertEqual(
            tuple(cell_offsets), CASTING_S_C3_FRAME_CELLS
        )

    def test_default_plan_relevant_action_count(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        relevant_count = 0
        per_cell_count: dict[int, int] = {i: 0 for i in range(14)}
        for step in plan:
            if step.relevant_action:
                relevant_count += 1
                per_cell_count[step.cell_index] += 1
        # 2 relevant actions per cell × 14 cells = 28.
        self.assertEqual(relevant_count, 28)
        for cell_index in range(14):
            self.assertEqual(per_cell_count[cell_index], 2)

    def test_default_plan_recoveries_are_bounded(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        for step in plan:
            self.assertGreaterEqual(step.recoveries_allowed, 0)
            self.assertLessEqual(
                step.recoveries_allowed, MAX_RECOVERIES_PER_ACTION
            )
            if step.recoveries_allowed > 0:
                self.assertEqual(step.action.action_type, "use_item")

    def test_default_plan_actions_accepted_by_public_protocol(self) -> None:
        for step in build_casting_s_c3_frame_action_plan():
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

    def test_default_plan_wait_count_fits_budget(self) -> None:
        plan = build_casting_s_c3_frame_action_plan()
        wait_count = sum(
            step.action.action_type == "wait" for step in plan
        )
        # 17 waits / cell × 14 cells = 238.
        self.assertEqual(wait_count, 14 * 17)
        # Default budget fits.
        self.assertLessEqual(wait_count, DEFAULT_MAX_WAIT_STEPS)
        # Hard cap fits.
        self.assertLessEqual(wait_count, MAX_FRAME_PLAN_WAIT_STEPS)

    def test_plan_rejects_oversized_wait_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "C3 frame plan wait steps exceed"
        ):
            build_casting_s_c3_frame_action_plan(
                obsidian_wait_steps=20
            )

    def test_plan_rejects_oversized_recoveries(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_use_item"
        ):
            build_casting_s_c3_frame_action_plan(
                recoveries_per_use_item=MAX_RECOVERIES_PER_ACTION + 1
            )

    def test_plan_rejects_negative_wait_count(self) -> None:
        with self.assertRaises(ValueError):
            build_casting_s_c3_frame_action_plan(
                support_block_wait_steps=-1  # type: ignore[arg-type]
            )

    def test_plan_rejects_reordered_target_offsets(self) -> None:
        reordered = tuple(
            (y, x, z) for (x, y, z) in CASTING_S_C3_FRAME_CELLS
        )
        with self.assertRaisesRegex(
            ValueError, "must match the locked"
        ):
            build_casting_s_c3_frame_action_plan(
                target_offsets=reordered
            )

    def test_plan_recovery_constants_are_documented(self) -> None:
        self.assertGreaterEqual(RECOVERIES_PER_USE_ITEM_DEFAULT, 1)
        self.assertLessEqual(
            RECOVERIES_PER_USE_ITEM_DEFAULT, MAX_RECOVERIES_PER_ACTION
        )
        self.assertGreaterEqual(TOTAL_RECOVERY_BUDGET_DEFAULT, 1)
        self.assertLessEqual(
            TOTAL_RECOVERY_BUDGET_DEFAULT, MAX_TOTAL_RECOVERY_BUDGET
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


# ----------------------------------------------------------------------
# Driver static / source contract tests
# ----------------------------------------------------------------------


class DriverSourceContractTests(unittest.TestCase):
    """The driver source must respect the C3 frame information
    isolation contract: no frame evaluator import, no truth
    surface calls, no scenario_parameters / evaluator_contract
    reads, no FrozenFrameEvaluationState surface."""

    def _driver_source(self) -> str:
        import obsidianlink.drivers.casting_s_c3_frame as driver_module

        with open(
            driver_module.__file__, "r", encoding="utf-8"
        ) as handle:
            return handle.read()

    def test_driver_source_does_not_import_frame_evaluator(self) -> None:
        source = self._driver_source()
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                if "casting_frame_evaluator" in node.module:
                    self.fail(
                        "driver source must not import from "
                        f"casting_frame_evaluator (module={node.module!r})"
                    )
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
        forbidden = {
            "FrozenFrameEvaluator",
            "FrozenFrameEvaluationState",
            "FrozenFrameEvaluationResult",
            "FrozenFrameCellTruth",
            "FrozenFrameActionEvidence",
            "FrozenFrameInteriorCellTruth",
            "FrozenFrameOriginAnchor",
            "FRAME_OUTCOMES",
            "CASTING_S_C3_FRAME_CELLS",
            "INTERIOR_ALLOWED",
            "default_c3_anchor",
        }
        for name in forbidden:
            self.assertNotIn(
                name,
                imported_names,
                f"driver module must not import {name!r}",
            )

    def test_driver_source_does_not_call_frame_truth_surface(self) -> None:
        # Walk the AST to find any ``Attribute`` access on the
        # frame truth methods or the backend's private slot. This
        # ignores docstring / comment references and only catches
        # real code paths.
        source = self._driver_source()
        tree = ast.parse(source)
        forbidden_attrs = {
            "set_frame_evaluation_state",
            "get_frame_evaluation_state",
            "clear_frame_evaluation_state",
            "_frame_evaluation_state",
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in forbidden_attrs
            ):
                self.fail(
                    f"driver source must not reference attribute "
                    f"{node.attr!r}"
                )

    def test_driver_source_does_not_read_scenario_parameters(self) -> None:
        # Walk the AST to find any ``Attribute`` access to
        # ``scenario_parameters`` (in real code, not in a
        # docstring / comment). The driver may not pull any field
        # off ``task.scenario_parameters``.
        source = self._driver_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "scenario_parameters":
                    self.fail(
                        "driver must not reference "
                        f"{node.attr!r} as a code attribute"
                    )
        # Also block subscript access on a ``scenario_parameters``
        # attribute.
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "scenario_parameters"
                ):
                    self.fail(
                        "driver must not read task.scenario_parameters"
                    )

    def test_driver_source_does_not_read_evaluator_contract(self) -> None:
        # Walk the AST to find any code-level reference to
        # ``evaluator_contract``. Docstring / comment occurrences
        # are ignored (they are explicitly allowed for the
        # information-isolation rationale in the module
        # docstring).
        source = self._driver_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "evaluator_contract":
                    self.fail(
                        "driver must not reference "
                        f"{node.attr!r} as a code attribute"
                    )
            elif isinstance(node, ast.Name):
                if node.id == "evaluator_contract":
                    self.fail(
                        "driver must not reference "
                        f"{node.id!r} as a code name"
                    )

    def test_driver_source_does_not_use_dict_getattr_tricks(self) -> None:
        # The driver must not use ``getattr(backend, ...)`` to
        # smuggle access to frame truth methods.
        source = self._driver_source()
        self.assertNotIn(
            "getattr(backend",
            source,
            "driver must not use getattr(backend, ...)",
        )

    def test_driver_function_signatures_explicit(self) -> None:
        import inspect

        sig = inspect.signature(run_casting_s_c3_frame_driver)
        self.assertIn("backend", sig.parameters)
        self.assertIn("context", sig.parameters)
        # The driver must not accept a ``task`` parameter; the
        # public context is the only TaskInstance-shaped input.
        self.assertNotIn("task", sig.parameters)


# ----------------------------------------------------------------------
# Driver on FakeBackend tests
# ----------------------------------------------------------------------


class DriverFakeBackendTests(unittest.TestCase):
    """The driver walks the 14-cell plan on the FakeBackend."""

    def _run_driver(
        self,
        *,
        context: PublicC3FrameDriverContext | None = None,
        task: TaskInstance | None = None,
        max_environment_steps: int | None = None,
        max_game_time_seconds: float | None = None,
        max_wait_steps: int = DEFAULT_MAX_WAIT_STEPS,
        total_recovery_budget: int = TOTAL_RECOVERY_BUDGET_DEFAULT,
        recoveries_per_use_item: int = RECOVERIES_PER_USE_ITEM_DEFAULT,
        plan: tuple[CastingC3FramePlanStep, ...] | None = None,
        backend: FakeEnvironmentBackend | None = None,
        event_sink=None,
    ) -> CastingC3FrameDriverResult:
        effective_task = task or _task()
        effective_context = context or (
            build_public_c3_frame_driver_context_from_task(effective_task)
        )
        if backend is None:
            backend = FakeEnvironmentBackend()
            backend.open()
            owns_backend = True
        else:
            owns_backend = False
        try:
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
            return run_casting_s_c3_frame_driver(
                backend, effective_context, **kwargs
            )
        finally:
            if owns_backend:
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
        # 2 relevant actions per cell × 14 cells = 28.
        self.assertEqual(
            sum(
                len(records)
                for records in result.per_cell_relevant_action_records.values()
            ),
            28,
        )
        self.assertEqual(
            set(result.per_cell_relevant_action_records.keys()),
            set(range(14)),
        )
        self.assertEqual(
            set(result.per_cell_target_offset.keys()),
            set(range(14)),
        )
        self.assertEqual(
            result.final_observation.step_id, result.planned_steps
        )

    def test_driver_rejects_incomplete_backend_capabilities_before_reset(self) -> None:
        backend = FakeEnvironmentBackend.with_capabilities(BackendCapabilities())
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError) as caught:
                run_casting_s_c3_frame_driver(backend, _context())
            self.assertEqual(caught.exception.task_id, EPISODE_ID)
            self.assertEqual(
                caught.exception.missing,
                (
                    "select_water_bucket",
                    "select_lava_bucket",
                    "use_water_bucket",
                    "use_lava_bucket",
                    "public_inventory",
                    "selected_item",
                    "target_block_truth",
                    "fluid_truth",
                ),
            )
            self.assertIsNone(backend._task)
            self.assertEqual(backend._step_id, 0)
        finally:
            backend.close()

    def test_driver_relevant_records_have_items(self) -> None:
        result = self._run_driver()
        for records in result.per_cell_relevant_action_records.values():
            for step_id, item in records:
                self.assertIn(item, {"water_bucket", "lava_bucket"})
                self.assertIsInstance(step_id, int)
                self.assertGreaterEqual(step_id, 0)

    def test_driver_per_cell_target_offsets_match_context(self) -> None:
        result = self._run_driver()
        for cell_index, offset in result.per_cell_target_offset.items():
            self.assertEqual(offset, CASTING_S_C3_FRAME_CELLS[cell_index])

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

    def test_driver_event_cell_index_and_target_offset_are_set(self) -> None:
        result = self._run_driver()
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["cell_index"], -1)
                self.assertIsNone(event["target_offset"])
                continue
            self.assertIn(event["cell_index"], set(range(14)))
            self.assertEqual(
                tuple(event["target_offset"]),
                CASTING_S_C3_FRAME_CELLS[event["cell_index"]],
            )

    def test_driver_event_relevant_action_uses_use_item(self) -> None:
        result = self._run_driver()
        for event in result.events:
            if event.get("relevant_action"):
                self.assertEqual(event["action_type"], "use_item")
                self.assertIn(event["target"], {"water_bucket", "lava_bucket"})

    def test_driver_event_sink_cannot_mutate_evidence(self) -> None:
        def mutate(event):  # type: ignore[no-untyped-def]
            event["label"] = "tampered"
            if "visible_inventory" in event:
                event["visible_inventory"]["water_bucket"] = 0

        result = self._run_driver(event_sink=mutate)
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 14
        )

    def test_driver_result_evidence_is_deeply_immutable(self) -> None:
        result = self._run_driver()
        with self.assertRaises(TypeError):
            result.events[0]["label"] = "tampered"  # type: ignore[index]
        snapshot = result.as_dict()
        snapshot["events"][0]["label"] = "tampered"
        snapshot["events"][0]["visible_inventory"]["water_bucket"] = 0
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 14
        )

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
            [event["target"] for event in first.events],
            [event["target"] for event in second.events],
        )
        self.assertEqual(
            first.per_cell_relevant_action_records,
            second.per_cell_relevant_action_records,
        )
        self.assertEqual(
            first.per_cell_target_offset, second.per_cell_target_offset
        )
        # as_dict() snapshot must be identical too.
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_driver_fires_step_budget(self) -> None:
        result = self._run_driver(max_environment_steps=10)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("step budget exhausted", result.blocked_reason or "")
        self.assertLessEqual(result.steps_executed, 10)
        self.assertLess(result.steps_executed, result.planned_steps)
        self.assertEqual(result.events[-1].get("budget_exceeded"), "step")

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
            result = self._run_driver(
                backend=backend, max_game_time_seconds=120.0
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("time budget exceeded", result.blocked_reason or "")
        self.assertEqual(result.events[-1].get("budget_exceeded"), "time")

    def test_driver_fires_wait_budget(self) -> None:
        result = self._run_driver(max_wait_steps=2)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("wait budget exhausted", result.blocked_reason or "")
        self.assertEqual(result.events[-1].get("budget_exceeded"), "wait")
        self.assertEqual(result.wait_steps, 2)

    def test_driver_fires_plan_length_budget(self) -> None:
        too_long_plan = tuple(
            CastingC3FramePlanStep(
                cell_index=0,
                target_offset=CASTING_S_C3_FRAME_CELLS[0],
                label=f"step.{index}",
                phase=PHASE_PREPARE,
                action=MacroAction.wait(),
            )
            for index in range(641)
        )
        with self.assertRaisesRegex(
            ValueError, "plan length cannot exceed"
        ):
            self._run_driver(plan=too_long_plan)

    def test_driver_fires_recovery_budget_too_large(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "total_recovery_budget"
        ):
            self._run_driver(
                total_recovery_budget=MAX_TOTAL_RECOVERY_BUDGET + 1
            )

    def test_driver_fires_max_wait_steps_too_large(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            self._run_driver(
                max_wait_steps=MAX_FRAME_PLAN_WAIT_STEPS + 1
            )

    def test_driver_fires_max_environment_steps_above_task_limit(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps cannot exceed"
        ):
            self._run_driver(max_environment_steps=10_000)

    def test_driver_fires_max_game_time_above_task_limit(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_game_time_seconds cannot exceed"
        ):
            self._run_driver(max_game_time_seconds=10_000.0)

    def test_driver_marks_early_backend_termination_as_blocked(self) -> None:
        class _EarlyTerminatingBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override]
                step = super().step(actions)
                return dataclasses.replace(step, terminated=True)

        backend = _EarlyTerminatingBackend()
        backend.open()
        try:
            result = self._run_driver(backend=backend)
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertIn("backend termination", result.blocked_reason or "")

    def test_driver_marks_early_truncation_as_blocked(self) -> None:
        class _EarlyTruncatingBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override]
                step = super().step(actions)
                return dataclasses.replace(step, truncated=True)

        backend = _EarlyTruncatingBackend()
        backend.open()
        try:
            result = self._run_driver(backend=backend)
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertTrue(result.truncated)
        self.assertIn("backend truncation", result.blocked_reason or "")

    def test_driver_blocks_on_missing_required_item(self) -> None:
        # The public context validator requires cobblestone for
        # the canonical C3 frame contract. We test the runtime
        # block path by patching the backend to expose a
        # visible_inventory that omits cobblestone while keeping
        # the two buckets.
        class _NoCobbleBackend(FakeEnvironmentBackend):
            def _observations(self):  # type: ignore[override]
                task = self._require_task()
                inventory = dict(task.initial_inventories[AGENT_ID])
                inventory.pop("cobblestone", None)
                return {
                    agent_id: Observation(
                        episode_id=task.task_id,
                        agent_id=agent_id,
                        step_id=self._step_id,
                        timestamp=0.0,
                        frame={
                            "backend": "fake_no_cobble",
                            "step_id": self._step_id,
                        },
                        visible_inventory=inventory,
                        workflow_stage=task.workflow,
                    )
                    for agent_id in task.agent_ids
                }

        backend = _NoCobbleBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(backend, context)
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("cobblestone", result.blocked_reason or "")
        # No relevant action should have been submitted because
        # the first place_block was refused.
        self.assertEqual(
            result.per_cell_relevant_action_records, {}
        )

    def test_driver_does_not_call_frame_truth_surface(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(
                _task()
            )
            set_calls: list[Any] = []
            get_calls: list[int] = []
            clear_calls: list[int] = []
            original_set = backend.set_frame_evaluation_state
            original_get = backend.get_frame_evaluation_state
            original_clear = backend.clear_frame_evaluation_state
            backend.set_frame_evaluation_state = (  # type: ignore[method-assign]
                lambda state: set_calls.append(state) or original_set(state)
            )
            backend.get_frame_evaluation_state = (  # type: ignore[method-assign]
                lambda: get_calls.append(1) or original_get()
            )
            backend.clear_frame_evaluation_state = (  # type: ignore[method-assign]
                lambda: clear_calls.append(1) or original_clear()
            )
            run_casting_s_c3_frame_driver(backend, context)
        finally:
            backend.close()
        self.assertEqual(set_calls, [])
        self.assertEqual(get_calls, [])
        self.assertEqual(clear_calls, [])

    def test_driver_does_not_leak_truth_into_observation(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(
                _task()
            )
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
                "frame_evaluator",
                "frame_outcome",
                "success",
                "blocking_conditions",
                "outcome",
                "failure_type",
                "per_cell_outcomes",
                "first_failed_cell",
                "completed_cells",
                "completed_interior_cells",
                "interior_blocker_cells",
                "interior_current_blocks",
            }

            def guarded(self, name):  # type: ignore[no-untyped-def]
                if name in forbidden:
                    raise AssertionError(
                        f"driver attempted to read Observation.{name}"
                    )
                return original(self, name)

            Observation.__getattribute__ = guarded  # type: ignore[assignment]
            try:
                result = run_casting_s_c3_frame_driver(backend, context)
            finally:
                Observation.__getattribute__ = original  # type: ignore[assignment]
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)

    def test_driver_rejects_non_context_argument(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            with self.assertRaisesRegex(
                ValueError, "PublicC3FrameDriverContext"
            ):
                run_casting_s_c3_frame_driver(  # type: ignore[arg-type]
                    backend, "not a context"
                )
        finally:
            backend.close()


# ----------------------------------------------------------------------
# Recovery protocol tests
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

    def test_recoverable_error_is_retried(self) -> None:
        # The first use_item step (cell 0, lava) is at plan_index 8
        # and corresponds to backend step 9 (after the reset at
        # step 0 and 8 wait/place steps). Force one transient raise
        # on step 9; the driver must retry and complete the plan.
        backend = self._make_recovery_backend(raise_spec={9: 1})
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(
                backend,
                context,
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
        # Three different use_item steps raise once each. With
        # total_recovery_budget=2 the third raise must block.
        backend = self._make_recovery_backend(raise_spec={9: 1, 33: 1, 57: 1})
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(
                backend,
                context,
                total_recovery_budget=2,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertEqual(result.recovery_attempts, 3)
        self.assertIn(
            "recovery budget exhausted", result.blocked_reason or ""
        )

    def test_recoverable_error_blocked_when_per_step_budget_exhausted(self) -> None:
        # The first use_item step raises twice. With
        # recoveries_per_use_item=1 the second raise must block.
        backend = self._make_recovery_backend(raise_spec={9: 2})
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(
                backend,
                context,
                total_recovery_budget=6,
                recoveries_per_use_item=1,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertEqual(result.recovery_attempts, 2)
        self.assertIn(
            "per-step recovery budget exhausted",
            result.blocked_reason or "",
        )

    def test_recovery_event_includes_recoverable_metadata(self) -> None:
        backend = self._make_recovery_backend(
            raise_spec={9: 1}, recoverable_kind="custom_kind"
        )
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(
                backend,
                context,
                total_recovery_budget=3,
            )
        finally:
            backend.close()
        recovery_events = [
            event for event in result.events
            if event.get("phase") == PHASE_RECOVERY
        ]
        self.assertEqual(len(recovery_events), 1)
        self.assertEqual(
            recovery_events[0]["recoverable_kind"], "custom_kind"
        )
        self.assertIn("recoverable_message", recovery_events[0])

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
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(backend, context)
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("RuntimeError", result.blocked_reason or "")

    def test_recovery_budget_zero_still_runs(self) -> None:
        result = DriverFakeBackendTests()._run_driver(
            total_recovery_budget=0
        )
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.recovery_attempts, 0)


# ----------------------------------------------------------------------
# Stale step / replay tests
# ----------------------------------------------------------------------


class StaleStepAndReplayTests(unittest.TestCase):
    def test_driver_rejects_backend_step_jump(self) -> None:
        class _JumpingBackend:
            def __init__(self) -> None:
                self.task_id = None

            def reset(self, task):  # type: ignore[no-untyped-def]
                self.task_id = task.task_id
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
                    episode_id=self.task_id,
                    agent_id=AGENT_ID,
                    step_id=2,
                    timestamp=1.0,
                    frame={},
                    visible_inventory={},
                    workflow_stage=WORKFLOW_C3_FRAME,
                )
                return BackendStep(
                    episode_id=self.task_id,
                    step_id=2,
                    observations={AGENT_ID: observation},
                    rewards={AGENT_ID: 0.0},
                    terminated=False,
                    truncated=False,
                )

        with self.assertRaisesRegex(ValueError, "advance exactly once"):
            run_casting_s_c3_frame_driver(
                _JumpingBackend(),  # type: ignore[arg-type]
                build_public_c3_frame_driver_context_from_task(_task()),
            )

    def test_driver_event_step_ids_are_unique_and_monotonic(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            result = run_casting_s_c3_frame_driver(backend, context)
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
        def run_once() -> tuple:
            backend = FakeEnvironmentBackend()
            backend.open()
            try:
                task = _task()
                context = build_public_c3_frame_driver_context_from_task(task)
                driver_result = run_casting_s_c3_frame_driver(
                    backend, context
                )
                result = run_orchestrator(
                    backend, driver_result, C3FrameWorldTruth(), task=task
                )
                snapshot = (
                    dict(driver_result.per_cell_relevant_action_records),
                    dict(driver_result.per_cell_target_offset),
                    result.outcome,
                    result.success,
                )
                return snapshot
            finally:
                backend.close()

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)


# ----------------------------------------------------------------------
# Driver ↔ frame evaluator end-to-end
# ----------------------------------------------------------------------


class DriverFrameEvaluatorEndToEndTests(unittest.TestCase):
    """The driver + orchestrator + frame evaluator end-to-end."""

    def test_normal_path_yields_success(self) -> None:
        task = _task()
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(task)
            driver_result = run_casting_s_c3_frame_driver(backend, context)
            result = run_orchestrator(
                backend,
                driver_result,
                C3FrameWorldTruth(),
                task=task,
            )
        finally:
            backend.close()
        self.assertEqual(driver_result.status, DRIVER_STATUS_COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.completed_cells, 14)
        self.assertEqual(result.total_cells, 14)
        self.assertEqual(result.completed_interior_cells, 6)
        self.assertEqual(result.total_interior_cells, 6)
        self.assertEqual(
            tuple(result.per_cell_outcomes),
            (PER_CELL_SUCCESS,) * 14,
        )
        self.assertEqual(
            tuple(result.per_interior_cell_outcomes),
            (PER_INTERIOR_CELL_ALLOWED,) * 6,
        )
        self.assertIsNone(result.first_failed_cell)
        self.assertEqual(result.failure_type, None)
        self.assertEqual(result.blocking_conditions, ())

    def test_completed_cells_is_fourteen(self) -> None:
        task = _task()
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(task)
            driver_result = run_casting_s_c3_frame_driver(backend, context)
            result = run_orchestrator(
                backend,
                driver_result,
                C3FrameWorldTruth(),
                task=task,
            )
        finally:
            backend.close()
        self.assertEqual(result.completed_cells, 14)
        self.assertEqual(result.total_cells, 14)
        self.assertEqual(
            tuple(result.per_cell_outcomes),
            (PER_CELL_SUCCESS,) * 14,
        )

    def _partial_completion_world(
        self, completed: int
    ) -> C3FrameWorldTruth:
        current_blocks = ("obsidian",) * completed + (
            "air",
        ) * (14 - completed)
        per_cell_relevant = tuple(
            (9 + 24 * index, 16 + 24 * index) for index in range(completed)
        ) + ((),) * (14 - completed)
        return C3FrameWorldTruth(
            current_blocks=current_blocks,
            per_cell_relevant_action_steps=per_cell_relevant,
        )

    def test_partial_completion_one_to_thirteen(self) -> None:
        for completed in range(1, 14):
            with self.subTest(completed=completed):
                task = _task()
                backend = FakeEnvironmentBackend()
                backend.open()
                try:
                    context = (
                        build_public_c3_frame_driver_context_from_task(task)
                    )
                    driver_result = run_casting_s_c3_frame_driver(
                        backend, context
                    )
                    result = run_orchestrator(
                        backend,
                        driver_result,
                        self._partial_completion_world(completed),
                        task=task,
                    )
                finally:
                    backend.close()
                self.assertEqual(result.outcome, OUTCOME_PARTIAL_COMPLEMENT)
                self.assertEqual(result.completed_cells, completed)
                self.assertEqual(result.total_cells, 14)
                self.assertEqual(
                    result.first_failed_cell, completed
                )

    def test_wrong_block_on_incomplete_target(self) -> None:
        # Cell 7 is a non-corner cell; replace its current_block
        # with cobblestone. The evaluator must report wrong_block
        # for the cell and overall wrong_block (because some
        # incomplete cell is in a wrong block).
        current_blocks = (
            "obsidian",
        ) * 7 + ("cobblestone",) + ("air",) * 6
        per_cell_relevant = tuple(
            (9 + 24 * index, 16 + 24 * index) for index in range(7)
        ) + ((),) * 7
        world = C3FrameWorldTruth(
            current_blocks=current_blocks,
            per_cell_relevant_action_steps=per_cell_relevant,
        )
        task = _task()
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(task)
            driver_result = run_casting_s_c3_frame_driver(backend, context)
            result = run_orchestrator(
                backend, driver_result, world, task=task
            )
        finally:
            backend.close()
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.completed_cells, 7)
        self.assertEqual(result.first_failed_cell, 7)

    def test_interior_blocker_fails_closed(self) -> None:
        # An interior cell ends up as ``dirt``; the evaluator
        # must report ``interior_blocked`` and outrank
        # partial_completion.
        interior = ("dirt",) + ("air",) * 5
        world = C3FrameWorldTruth(interior_current_blocks=interior)
        task = _task()
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(task)
            driver_result = run_casting_s_c3_frame_driver(backend, context)
            result = run_orchestrator(
                backend, driver_result, world, task=task
            )
        finally:
            backend.close()
        self.assertEqual(result.outcome, OUTCOME_INTERIOR_BLOCKED)
        self.assertEqual(result.completed_interior_cells, 5)
        self.assertEqual(result.total_interior_cells, 6)
        self.assertEqual(
            result.interior_blocker_cells,
            (CASTING_S_C3_INTERIOR_CELLS[0],),
        )

    def test_truth_missing_via_orchestrator(self) -> None:
        # The orchestrator builds a state where cell 0 is missing
        # its transition evidence. The evaluator must report
        # truth_missing, outranking partial_completion.
        records: list[tuple[tuple[int, int, str], ...]] = []
        for index in range(14):
            records.append(
                (
                    (9 + 24 * index, "lava_bucket"),
                    (16 + 24 * index, "water_bucket"),
                )
            )
        cells_truth: list[FrozenFrameCellTruth] = []
        for index, target_cell in enumerate(CASTING_S_C3_FRAME_CELLS):
            if index == 0:
                # Cell 0 is reported as obsidian but the
                # transition_evidence is missing. The other 13
                # cells get a full success record.
                cells_truth.append(
                    FrozenFrameCellTruth(
                        target_cell=target_cell,
                        initial_block="air",
                        current_block="obsidian",
                        water_truth=CastingFluidTruth(
                            present=True, evidence_step=16
                        ),
                        lava_truth=CastingFluidTruth(
                            present=True, evidence_step=9
                        ),
                        transition_evidence=None,
                        relevant_action_steps=(9, 16),
                        action_evidence=_actions(
                            target_cell, records[index]
                        ),
                        transition_action_step=16,
                    )
                )
            else:
                cells_truth.append(
                    FrozenFrameCellTruth(
                        target_cell=target_cell,
                        initial_block="air",
                        current_block="obsidian",
                        water_truth=CastingFluidTruth(
                            present=True, evidence_step=16 + 24 * index
                        ),
                        lava_truth=CastingFluidTruth(
                            present=True, evidence_step=9 + 24 * index
                        ),
                        transition_evidence=CastingTransitionEvidence(
                            before_block="air",
                            after_block="obsidian",
                            update_step=20 + 24 * index,
                        ),
                        relevant_action_steps=(
                            9 + 24 * index,
                            16 + 24 * index,
                        ),
                        action_evidence=_actions(
                            target_cell, records[index]
                        ),
                        transition_action_step=16 + 24 * index,
                    )
                )
        state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=14 * 24,
            cells=tuple(cells_truth),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(
                    target_cell=cell, current_block="air"
                )
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=14 * 24,
            terminated_reason="driver_done",
            current_time_seconds=0.0,
            max_environment_steps=640,
            max_game_time_seconds=600,
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            # Advance the backend to the right step before
            # injecting the state. ``set_frame_evaluation_state``
            # requires ``state.step_id == backend._step_id``.
            for _ in range(14 * 24):
                backend.step({AGENT_ID: MacroAction.wait()})
            backend.set_frame_evaluation_state(state)
            result = FrozenFrameEvaluator().evaluate(
                backend.get_frame_evaluation_state()
            )
        finally:
            backend.close()
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_causality_missing_via_orchestrator(self) -> None:
        # Cell 0 has a transition update_step far outside the
        # causality window from its last relevant action. The
        # other 13 cells are normal.
        records: list[tuple[tuple[int, int, str], ...]] = []
        for index in range(14):
            records.append(
                (
                    (9 + 24 * index, "lava_bucket"),
                    (16 + 24 * index, "water_bucket"),
                )
            )
        cells_truth: list[FrozenFrameCellTruth] = []
        for index, target_cell in enumerate(CASTING_S_C3_FRAME_CELLS):
            if index == 0:
                # Force the update_step well past the 4-step
                # window of the last relevant action (16).
                cells_truth.append(
                    FrozenFrameCellTruth(
                        target_cell=target_cell,
                        initial_block="air",
                        current_block="obsidian",
                        water_truth=CastingFluidTruth(
                            present=True, evidence_step=16
                        ),
                        lava_truth=CastingFluidTruth(
                            present=True, evidence_step=9
                        ),
                        transition_evidence=CastingTransitionEvidence(
                            before_block="air",
                            after_block="obsidian",
                            update_step=200,
                        ),
                        relevant_action_steps=(9, 16),
                        action_evidence=_actions(
                            target_cell, records[index]
                        ),
                        transition_action_step=16,
                    )
                )
            else:
                cells_truth.append(
                    FrozenFrameCellTruth(
                        target_cell=target_cell,
                        initial_block="air",
                        current_block="obsidian",
                        water_truth=CastingFluidTruth(
                            present=True, evidence_step=16 + 24 * index
                        ),
                        lava_truth=CastingFluidTruth(
                            present=True, evidence_step=9 + 24 * index
                        ),
                        transition_evidence=CastingTransitionEvidence(
                            before_block="air",
                            after_block="obsidian",
                            update_step=20 + 24 * index,
                        ),
                        relevant_action_steps=(
                            9 + 24 * index,
                            16 + 24 * index,
                        ),
                        action_evidence=_actions(
                            target_cell, records[index]
                        ),
                        transition_action_step=16 + 24 * index,
                    )
                )
        state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=14 * 24,
            cells=tuple(cells_truth),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(
                    target_cell=cell, current_block="air"
                )
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=14 * 24,
            terminated_reason="driver_done",
            current_time_seconds=0.0,
            max_environment_steps=640,
            max_game_time_seconds=600,
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(_task())
            # Advance the backend to the right step before
            # injecting the state. ``set_frame_evaluation_state``
            # requires ``state.step_id == backend._step_id``.
            for _ in range(14 * 24):
                backend.step({AGENT_ID: MacroAction.wait()})
            backend.set_frame_evaluation_state(state)
            result = FrozenFrameEvaluator().evaluate(
                backend.get_frame_evaluation_state()
            )
        finally:
            backend.close()
        self.assertEqual(result.outcome, "causality_missing")

    def test_identity_mismatch_via_orchestrator(self) -> None:
        # A wrong agent_id on the cell evidence is rejected at
        # state construction time. The orchestrator cannot inject
        # identity-mismatched truth at all.
        records: list[tuple[tuple[int, int, str], ...]] = []
        for index in range(14):
            records.append(
                (
                    (9 + 24 * index, "lava_bucket"),
                    (16 + 24 * index, "water_bucket"),
                )
            )
        cell = FrozenFrameCellTruth(
            target_cell=CASTING_S_C3_FRAME_CELLS[0],
            initial_block="air",
            current_block="obsidian",
            water_truth=CastingFluidTruth(present=True, evidence_step=16),
            lava_truth=CastingFluidTruth(present=True, evidence_step=9),
            transition_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=20
            ),
            relevant_action_steps=(9, 16),
            action_evidence=(
                FrozenFrameActionEvidence(
                    episode_id=EPISODE_ID,
                    step_id=9,
                    agent_id=WRONG_AGENT_ID,
                    action_type="use_item",
                    item="lava_bucket",
                    target_cell=CASTING_S_C3_FRAME_CELLS[0],
                ),
                FrozenFrameActionEvidence(
                    episode_id=EPISODE_ID,
                    step_id=16,
                    agent_id=AGENT_ID,
                    action_type="use_item",
                    item="water_bucket",
                    target_cell=CASTING_S_C3_FRAME_CELLS[0],
                ),
            ),
            transition_action_step=16,
        )
        with self.assertRaisesRegex(ValueError, "agent_id"):
            FrozenFrameEvaluationState(
                episode_id=EPISODE_ID,
                step_id=14 * 24,
                cells=(cell,) + tuple(
                    FrozenFrameCellTruth(
                        target_cell=offset,
                        initial_block="air",
                        current_block="obsidian",
                        water_truth=CastingFluidTruth(
                            present=True, evidence_step=16 + 24 * cell_index
                        ),
                        lava_truth=CastingFluidTruth(
                            present=True, evidence_step=9 + 24 * cell_index
                        ),
                        transition_evidence=CastingTransitionEvidence(
                            before_block="air",
                            after_block="obsidian",
                            update_step=20 + 24 * cell_index,
                        ),
                        relevant_action_steps=(
                            9 + 24 * cell_index,
                            16 + 24 * cell_index,
                        ),
                        action_evidence=_actions(
                            offset, records[cell_index]
                        ),
                        transition_action_step=16 + 24 * cell_index,
                    )
                    for cell_index, offset in enumerate(
                        CASTING_S_C3_FRAME_CELLS[1:], start=1
                    )
                ),
                interior_cells=tuple(
                    FrozenFrameInteriorCellTruth(
                        target_cell=cell, current_block="air"
                    )
                    for cell in CASTING_S_C3_INTERIOR_CELLS
                ),
                agent_id=AGENT_ID,
                causality_window_steps=4,
                episode_terminated=True,
                terminated_step=14 * 24,
                terminated_reason="driver_done",
                current_time_seconds=0.0,
                max_environment_steps=640,
                max_game_time_seconds=600,
            )


# ----------------------------------------------------------------------
# Observation leakage / observability
# ----------------------------------------------------------------------


class ObservationLeakageTests(unittest.TestCase):
    def test_observation_does_not_carry_frame_truth_after_driver(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            context = build_public_c3_frame_driver_context_from_task(_task())
            driver_result = run_casting_s_c3_frame_driver(backend, context)
            # After the driver, the orchestrator injects the full
            # truth. The driver result's final observation was
            # captured *before* the orchestrator ran, so it must
            # still be clean.
            run_orchestrator(
                backend,
                driver_result,
                C3FrameWorldTruth(),
                task=_task(),
            )
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
                "frame_evaluator",
                "frame_outcome",
                "success",
                "blocking_conditions",
                "outcome",
                "failure_type",
                "per_cell_outcomes",
                "first_failed_cell",
                "completed_cells",
                "completed_interior_cells",
                "interior_blocker_cells",
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


# ----------------------------------------------------------------------
# Driver backend shape tests
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
                            "water_bucket": 14,
                            "lava_bucket": 14,
                            "cobblestone": 28,
                        },
                        workflow_stage=WORKFLOW_C3_FRAME,
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
                        "water_bucket": 14,
                        "lava_bucket": 14,
                        "cobblestone": 28,
                    },
                    workflow_stage=WORKFLOW_C3_FRAME,
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
        context = build_public_c3_frame_driver_context_from_task(_task())
        result = run_casting_s_c3_frame_driver(backend, context)
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)


if __name__ == "__main__":
    unittest.main()
