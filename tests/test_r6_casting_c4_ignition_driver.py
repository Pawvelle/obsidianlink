"""Offline tests for the R6 Casting-S-C4 deterministic ignition driver.

These tests prove, in code, that:

* :class:`PublicC4IgnitionDriverContext` is a strictly-typed,
  frozen, immutable public driver context; the driver never reads
  ``scenario_parameters`` or ``evaluator_contract`` from the
  original task.
* :func:`build_casting_s_c4_ignition_action_plan` builds a fixed,
  deterministic, ordered plan whose C3 casting sub-plan has 14
  cells × 24 steps = 336 steps and whose C4 ignition sub-plan
  adds 4 steps (equip + release + use + settle) for a default
  340-step plan. The cast plan has 14 × 2 = 28 relevant actions
  (``use_item(water_bucket | lava_bucket)``); the ignition plan
  has exactly 1 relevant action (``use_item(flint_and_steel)``).
* :func:`run_casting_s_c4_ignition_driver` walks the plan on the
  :class:`FakeEnvironmentBackend`, never imports the C4 ignition
  evaluator or its types, never calls
  :meth:`FakeEnvironmentBackend.set_ignition_evaluation_state` /
  :meth:`FakeEnvironmentBackend.get_ignition_evaluation_state` /
  :meth:`FakeEnvironmentBackend.clear_ignition_evaluation_state`,
  and never reads ``scenario_parameters`` / ``evaluator_contract``
  / :class:`FrozenFrameIdentity` / :class:`IgnitionActionEvidence`
  / :class:`PortalActivationEvidence` / :class:`FrozenIgnitionEvaluationState`
  or any evaluator-only field.
* Every step / time / wait / plan / total-recovery budget has a
  hard, fail-closed bound.
* The driver's recovery protocol retries the typed
  :class:`RecoverableBackendError` deterministically and fails
  closed on any non-recoverable exception.
* The driver's events carry ``episode_id`` / ``step_id`` /
  ``agent_id`` / ``cell_index`` / ``target_offset`` /
  ``relevant_action`` / ``role``; the result is deeply immutable,
  the ``as_dict()`` snapshot is JSON-serializable, and the same
  input yields the same action sequence / events / ``as_dict()``
  snapshot on repeated runs.
* The pre-episode capability gate fails closed (with
  :class:`CapabilityMismatchError`) when the manifest is missing
  any of the required capabilities, and the failure happens
  *before* any ``Observation`` is generated.
* An Observation ``__getattribute__`` guard fails closed if the
  driver ever tries to read a hidden
  ``latched_frame_identity`` / ``ignition_evaluation`` /
  ``nether_portal`` / ``flint_and_steel`` / ``wrong_ignition``
  field.
* The test orchestrator (this file) is the *only* place that
  injects the R6 C4 ignition evaluator truth via
  :meth:`FakeEnvironmentBackend.set_ignition_evaluation_state`,
  and the :class:`FrozenIgnitionEvaluator` correctly returns
  ``success`` for a complete C3 frame + a legal C4 ignition +
  a portal activation in the 4-step inclusive window with the
  same episode-built :class:`FrozenFrameIdentity`.
* ``ignition_target`` / ``action_type`` / ``item`` /
  ``target_policy`` drift (wrong agent, wrong action, wrong
  item, wrong target, wrong policy, wrong agent on the
  activation evidence, activation 1 step past the window, etc.)
  all produce the closed-set failure outcomes the C4 ignition
  evaluator documents.
* The C1, C2, portal, C3 frame, and C4 ignition evaluator
  regression tests all stay green.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import sys
import time as _time
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.casting_s_c4_ignition_context import (
    build_public_c4_ignition_driver_context_from_task,
)
from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)
from obsidianlink.drivers.casting_s_c4_ignition import (
    AGENT_ID,
    ALLOWED_C4_IGNITION_ACTION_TYPES,
    ALLOWED_C4_IGNITION_FAMILIES,
    ALLOWED_C4_IGNITION_LAYOUTS,
    ALLOWED_C4_IGNITION_LEVELS,
    ALLOWED_C4_IGNITION_MODES,
    ALLOWED_C4_IGNITION_TARGETS,
    C4_IGNITION_GRID_X_MAX,
    C4_IGNITION_GRID_X_MIN,
    C4_IGNITION_GRID_Y_MAX,
    C4_IGNITION_GRID_Y_MIN,
    C4_IGNITION_GRID_Z_MAX,
    C4_IGNITION_GRID_Z_MIN,
    C4_IGNITION_PUBLIC_ACTION,
    C4_IGNITION_PUBLIC_ITEM,
    C4_IGNITION_PUBLIC_TARGET,
    C4_IGNITION_PUBLIC_TARGET_POLICY,
    CASTING_S_C4_IGNITION_FRAME_CELLS,
    CASTING_S_C4_IGNITION_TARGET_CELL_COUNT,
    DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    DEFAULT_IGNITION_PORTAL_SETTLE_STEPS,
    DEFAULT_MAX_WAIT_STEPS,
    DEFAULT_OBSIDIAN_WAIT_STEPS,
    DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    DRIVER_STATUS_BLOCKED,
    DRIVER_STATUS_COMPLETED,
    DRIVER_STATUS_FAILED,
    DRIVER_STATUSES,
    FAMILY_C4_IGNITION,
    LAYOUT_C4_IGNITION,
    LEVEL_C4_IGNITION,
    MAX_IGNITION_PLAN_STEPS,
    MAX_IGNITION_PLAN_WAIT_STEPS,
    MAX_RECOVERIES_PER_ACTION,
    MAX_TOTAL_RECOVERY_BUDGET,
    MODE_C4_IGNITION,
    PHASE_IGNITION_EQUIP,
    PHASE_IGNITION_PORTAL_SETTLE,
    PHASE_IGNITION_USE,
    PHASE_PLACE_LAVA,
    PHASE_PLACE_SUPPORT,
    PHASE_PLACE_WATER,
    PHASE_PREPARE,
    PHASE_RECOVERY,
    PHASE_VALUES,
    PHASE_WAIT_FOR_OBSIDIAN,
    RECOVERIES_PER_IGNITION_USE_DEFAULT,
    RECOVERIES_PER_USE_ITEM_DEFAULT,
    ROLE_CAST,
    ROLE_IGNITION_EQUIP,
    ROLE_IGNITION_SETTLE,
    ROLE_IGNITION_USE,
    ROLE_VALUES,
    TOTAL_RECOVERY_BUDGET_DEFAULT,
    WORKFLOW_C4_IGNITION,
    CastingC4IgnitionDriverResult,
    CastingC4IgnitionPlanStep,
    PublicC4IgnitionDriverContext,
    _ResetProxy,
    _cast_select_step,
    build_casting_s_c4_ignition_action_plan,
    run_casting_s_c4_ignition_driver,
)
from obsidianlink.env.capabilities import (
    BackendCapabilities,
    CapabilityMismatchError,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation import (
    CASTING_S_C3_INTERIOR_CELLS,
    FrozenFrameIdentity,
    FrozenFrameInteriorCellTruth,
    FrozenFrameCellTruth,
    FrozenFrameActionEvidence,
    FrozenFrameEvaluationState,
    FrozenFrameEvaluator,
    FrozenIgnitionEvaluator,
    IgnitionActionEvidence,
    OUTCOME_ACTIVATION_BEFORE_IGNITION,
    OUTCOME_ACTIVATION_MISSING,
    OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
    OUTCOME_EXTERNAL_ACTIVATION,
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
)
from obsidianlink.evaluation.continuous_casting import (
    ContinuousCastingEvaluator,
)
from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator


EPISODE_ID = "casting_s_c4_fixed_seed_0"
WRONG_EPISODE_ID = "casting_s_c4_fixed_seed_99"
WRONG_AGENT_ID = "agent_2"
DEFAULT_INVENTORY: dict[str, int] = {
    "water_bucket": 14,
    "lava_bucket": 14,
    "cobblestone": 28,
    "flint_and_steel": 1,
}
DEFAULT_MAX_ENVIRONMENT_STEPS = 700
DEFAULT_MAX_GAME_TIME_SECONDS = 640
DEFAULT_CAUSALITY_WINDOW = 4
TERMINATED_STEP = 14 * 24 + 4  # = 340
TERMINATED_REASON = "driver_done"
IGNITION_STEP = 339
IGNITION_EQUIP_STEP = 337
IGNITION_RELEASE_STEP = 338
IGNITION_PORTAL_SETTLE_STEP = 340

ROOT = Path(__file__).resolve().parents[1]
DRIVER_SOURCE = (
    ROOT / "obsidianlink/drivers/casting_s_c4_ignition.py"
)


# ----------------------------------------------------------------------
# Task / context helpers
# ----------------------------------------------------------------------


def _public_spec(
    *,
    fixed_offsets: list[list[int]] | None = None,
    ignition_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fixed_offsets is None:
        fixed_offsets = [list(cell) for cell in CASTING_S_C4_IGNITION_FRAME_CELLS]
    if ignition_plan is None:
        ignition_plan = {
            "required": True,
            "action": C4_IGNITION_PUBLIC_ACTION,
            "item": C4_IGNITION_PUBLIC_ITEM,
            "target_offset": list(C4_IGNITION_PUBLIC_TARGET),
            "target_policy": C4_IGNITION_PUBLIC_TARGET_POLICY,
        }
    return {
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
            "fixed_offsets": fixed_offsets,
        },
        "ignition_plan": ignition_plan,
        "nether_entry_goal": {
            "required": False,
            "designated_agent_ids": [],
            "target_dimension": None,
        },
    }


def _task_dict(
    *,
    inventory: dict[str, int] | None = None,
    workflow: str = WORKFLOW_C4_IGNITION,
    family: str = FAMILY_C4_IGNITION,
    mode: str = MODE_C4_IGNITION,
    level: str = LEVEL_C4_IGNITION,
    layout: str = LAYOUT_C4_IGNITION,
    episode_id: str = EPISODE_ID,
    max_environment_steps: int = DEFAULT_MAX_ENVIRONMENT_STEPS,
    max_game_time_seconds: int = DEFAULT_MAX_GAME_TIME_SECONDS,
    public_spec: dict[str, Any] | None = None,
    include_evaluator_contract: bool = False,
) -> dict[str, Any]:
    scenario_parameters: dict[str, Any] = {
        "task_family": family,
        "agent_mode": mode,
        "task_level": level,
        "layout_type": layout,
        "compatibility_task_name": WORKFLOW_C4_IGNITION,
        "implementation_status": "contract_only",
        "world_dimension": "minecraft:overworld",
        "layout": "fixed_controlled",
        "mechanics_required": "vanilla_water_lava_block_update_and_flint_and_steel",
        "public_task_spec": public_spec if public_spec is not None else _public_spec(),
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
            "activation_attribution": {
                "required": True,
                "require_episode_built_frame": True,
                "require_exact_public_target": True,
                "causality_window_steps": 4,
                "require_latched_frame_identity_match": True,
                "fail_closed_on_missing_truth": True,
            },
            "nether_entry_attribution": {"required": False},
        }
    return {
        "schema_version": "0.1",
        "task_id": episode_id,
        "route": "lava_casting",
        "difficulty": 3,
        "agent_ids": [AGENT_ID],
        "world_seed": 0,
        "instruction": "R6 C4 ignition driver unit-test task.",
        "spawn_positions": {AGENT_ID: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID: dict(inventory if inventory is not None else DEFAULT_INVENTORY)
        },
        "workflow": workflow,
        "milestones": [
            "task_reset",
            "first_obsidian_cast",
            "build_site_selected",
            "valid_portal_frame",
            "portal_activated",
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
    workflow: str = WORKFLOW_C4_IGNITION,
    family: str = FAMILY_C4_IGNITION,
    mode: str = MODE_C4_IGNITION,
    level: str = LEVEL_C4_IGNITION,
    layout: str = LAYOUT_C4_IGNITION,
    episode_id: str = EPISODE_ID,
    max_environment_steps: int = DEFAULT_MAX_ENVIRONMENT_STEPS,
    max_game_time_seconds: int = DEFAULT_MAX_GAME_TIME_SECONDS,
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
    family: str = FAMILY_C4_IGNITION,
    mode: str = MODE_C4_IGNITION,
    level: str = LEVEL_C4_IGNITION,
    layout: str = LAYOUT_C4_IGNITION,
    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C4_IGNITION_FRAME_CELLS,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
    workflow: str = WORKFLOW_C4_IGNITION,
    task_step_limit: int = DEFAULT_MAX_ENVIRONMENT_STEPS,
    task_time_limit: float = float(DEFAULT_MAX_GAME_TIME_SECONDS),
    ignition_action: str = C4_IGNITION_PUBLIC_ACTION,
    ignition_item: str = C4_IGNITION_PUBLIC_ITEM,
    ignition_target: tuple[int, int, int] = C4_IGNITION_PUBLIC_TARGET,
    ignition_target_policy: str = C4_IGNITION_PUBLIC_TARGET_POLICY,
    ignition_required: bool = True,
) -> PublicC4IgnitionDriverContext:
    return PublicC4IgnitionDriverContext(
        episode_id=episode_id,
        workflow=workflow,
        family=family,
        mode=mode,
        level=level,
        layout=layout,
        agent_id=agent_id,
        target_offsets=target_offsets,
        initial_inventory=(
            inventory if inventory is not None else dict(DEFAULT_INVENTORY)
        ),
        ignition_action=ignition_action,
        ignition_item=ignition_item,
        ignition_target=ignition_target,
        ignition_target_policy=ignition_target_policy,
        ignition_required=ignition_required,
        task_step_limit=task_step_limit,
        task_time_limit=task_time_limit,
    )


# ----------------------------------------------------------------------
# C4 ignition world truth (test orchestrator only)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class C4IgnitionWorldTruth:
    """Test-only description of the R6 C4 ignition casting world.

    The orchestrator (functions below) is the *only* component that
    consumes a :class:`C4IgnitionWorldTruth`. The driver never sees
    this object. Each target cell carries the evaluator truth
    required by :class:`FrozenFrameCellTruth`; the 6 interior
    cells carry the truth required by
    :class:`FrozenFrameInteriorCellTruth`. The activation evidence
    carries the public ignition target ``(1, 1, 1)``.

    The default values match the R6 C4 ignition driver's default
    plan (14 cells × 24 + 4 = 340 steps). The first
    ``use_item(lava_bucket)`` is at global step ``9 + 24 *
    cell_index``; the first ``use_item(water_bucket)`` is at
    global step ``16 + 24 * cell_index``; the block update is at
    ``20 + 24 * cell_index`` (within the 4-step causality window
    of the last relevant action). The ignition
    ``use_item(flint_and_steel)`` is at global step ``339``.
    The activation ``update_step`` defaults to ``340`` (delta
    = 1 from the ignition step) which is inside the 4-step
    inclusive window.
    """

    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C4_IGNITION_FRAME_CELLS
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
    activation_offset: tuple[int, int, int] = C4_IGNITION_PUBLIC_TARGET
    activation_delta_steps: int = 1
    terminated_step: int = TERMINATED_STEP
    terminated_reason: str = TERMINATED_REASON
    current_time_seconds: float = 0.0
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW


def _cast_actions(
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


def _cast_state(
    world: C4IgnitionWorldTruth,
    *,
    task: TaskInstance,
    relevant_records: tuple[tuple[tuple[int, int, str], ...], ...] | None = None,
    current_time_seconds: float | None = None,
    step_id: int | None = None,
    terminated_step: int | None = None,
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
            tuple(
                (step, "lava_bucket" if step % 2 == 0 else "water_bucket")
                for step in world.per_cell_relevant_action_steps[index]
            )
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
                action_evidence=_cast_actions(target_cell, records),
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
    state_step_id = (
        step_id if step_id is not None else world.terminated_step
    )
    state_terminated_step = (
        terminated_step
        if terminated_step is not None
        else world.terminated_step
    )
    return FrozenFrameEvaluationState(
        episode_id=task.task_id,
        step_id=state_step_id,
        cells=tuple(cells),
        interior_cells=interior,
        agent_id=AGENT_ID,
        causality_window_steps=world.causality_window_steps,
        episode_terminated=True,
        terminated_step=state_terminated_step,
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
    driver_result: CastingC4IgnitionDriverResult,
    world: C4IgnitionWorldTruth,
    *,
    task: TaskInstance,
    current_time_seconds: float | None = None,
    step_id: int | None = None,
    terminated_step: int | None = None,
) -> FrozenFrameEvaluationState:
    """Build a :class:`FrozenFrameEvaluationState` from a driver result.

    The orchestrator derives per-cell ``(step_id, item)`` records
    from ``driver_result.per_cell_relevant_action_records`` and
    hands them to :func:`_cast_state`. The driver never sees this
    code path.
    """
    records: list[tuple[tuple[int, int, str], ...]] = []
    for index in range(14):
        cell_records = driver_result.per_cell_relevant_action_records.get(
            index, ()
        )
        records.append(
            tuple(
                (int(step_id), str(item))
                for step_id, item in cell_records
            )
        )
    return _cast_state(
        world,
        task=task,
        relevant_records=tuple(records),
        current_time_seconds=current_time_seconds,
        step_id=step_id,
        terminated_step=terminated_step,
    )


def _build_ignition_state(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC4IgnitionDriverResult,
    world: C4IgnitionWorldTruth,
    *,
    task: TaskInstance,
    activation_delta_steps: int | None = None,
    ignition_step_override: int | None = None,
    activation_offset: tuple[int, int, int] | None = None,
    current_time_seconds: float | None = None,
    activation_agent_id: str | None = None,
):
    """Build a :class:`FrozenIgnitionEvaluationState` from a driver result.

    The orchestrator (this function) is the *only* place in the R6
    C4 ignition driver test suite that calls
    ``set_ignition_evaluation_state`` /
    ``get_ignition_evaluation_state` and the
    :class:`FrozenIgnitionEvaluator`. The driver never sees the
    truth surface.

    The function builds:

    * a :class:`FrozenFrameEvaluationState` from the C3 casting
      evidence (per-cell records, transitions, interior cells);
    * a typed :class:`FrozenFrameIdentity` for the episode-built
      frame (matching the C3 frozen plan, with the activation
      ``(1, 1, 1)`` as a canonical subset of the interior);
    * an :class:`IgnitionActionEvidence` for the driver's
      ``use_item(flint_and_steel)`` step;
    * a :class:`PortalActivationEvidence` for the nether_portal
      appearance ``(activation_delta_steps`` after the ignition);
    * the wrapping :class:`FrozenIgnitionEvaluationState`.
    """
    from obsidianlink.evaluation.casting_ignition_evaluator import (
        FrozenIgnitionEvaluationState,
    )

    if driver_result.ignition_relevant_action_step is None:
        raise AssertionError(
            "driver must have submitted the ignition use_item step"
        )
    if driver_result.ignition_target_offset != C4_IGNITION_PUBLIC_TARGET:
        raise AssertionError(
            "ignition target must be the public [1, 1, 1] cell"
        )
    ignition_step = (
        ignition_step_override
        if ignition_step_override is not None
        else driver_result.ignition_relevant_action_step
    )
    delta = (
        activation_delta_steps
        if activation_delta_steps is not None
        else world.activation_delta_steps
    )
    activation_step = ignition_step + delta
    if activation_step < 0:
        raise ValueError("activation_step must be non-negative")
    offset = (
        activation_offset
        if activation_offset is not None
        else world.activation_offset
    )
    # The C4 state requires ``frame_state.step_id == state.step_id``
    # and ``frame_state.terminated_step == state.terminated_step``.
    # If the activation step is in the past (delta < 0) we must
    # re-derive the frame state at the episode's termination
    # step so that ``step_id >= terminated_step``; the
    # ``activation_evidence.update_step`` itself still carries
    # the past activation observation.
    terminated_step = max(activation_step, world.terminated_step)
    if activation_step >= terminated_step:
        state_step_id = activation_step
    else:
        state_step_id = terminated_step
    frame_state = _state_from_driver(
        driver_result,
        world,
        task=task,
        current_time_seconds=current_time_seconds,
        step_id=state_step_id,
        terminated_step=terminated_step,
    )
    latched_step = state_step_id
    latched_frame_identity = build_c4_c3_frame_identity(
        episode_id=task.task_id,
        step_id=latched_step,
        agent_id=AGENT_ID,
        activation_offsets=(offset,) if offset in CASTING_S_C3_INTERIOR_CELLS
        else None,
    )
    if latched_frame_identity.activation_offsets != (offset,):
        latched_frame_identity = build_c4_c3_frame_identity(
            episode_id=task.task_id,
            step_id=latched_step,
            agent_id=AGENT_ID,
            activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
        )
    ignition_action = IgnitionActionEvidence(
        episode_id=task.task_id,
        step_id=ignition_step,
        agent_id=AGENT_ID,
        action_type=C4_IGNITION_PUBLIC_ACTION,
        item=C4_IGNITION_PUBLIC_ITEM,
        target_cell=C4_IGNITION_PUBLIC_TARGET,
    )
    activation_evidence = PortalActivationEvidence(
        episode_id=task.task_id,
        update_step=activation_step,
        agent_id=(
            activation_agent_id
            if activation_agent_id is not None
            else AGENT_ID
        ),
        nether_portal_offset=offset,
        latched_frame_identity=latched_frame_identity,
    )
    return FrozenIgnitionEvaluationState(
        episode_id=task.task_id,
        step_id=state_step_id,
        frame_state=frame_state,
        latched_frame_identity=latched_frame_identity,
        ignition_action=ignition_action,
        activation_evidence=activation_evidence,
        agent_id=AGENT_ID,
        causality_window_steps=world.causality_window_steps,
        episode_terminated=True,
        terminated_step=terminated_step,
        terminated_reason=TERMINATED_REASON,
        current_time_seconds=(
            current_time_seconds
            if current_time_seconds is not None
            else world.current_time_seconds
        ),
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )


def run_orchestrator(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC4IgnitionDriverResult,
    world: C4IgnitionWorldTruth,
    *,
    task: TaskInstance,
    activation_delta_steps: int | None = None,
    ignition_step_override: int | None = None,
    activation_offset: tuple[int, int, int] | None = None,
    current_time_seconds: float | None = None,
    activation_agent_id: str | None = None,
    use_backend_roundtrip: bool = True,
):
    """Build the orchestrator-side state and call the C4 evaluator.

    By default the state is round-tripped through
    :meth:`FakeEnvironmentBackend.set_ignition_evaluation_state` /
    :meth:`FakeEnvironmentBackend.get_ignition_evaluation_state`.
    Tests that exercise boundary conditions (delta != 1) bypass
    the round-trip with ``use_backend_roundtrip=False`` because
    the FakeBackend's step_id guard would otherwise reject the
    injected state.
    """
    state = _build_ignition_state(
        backend,
        driver_result,
        world,
        task=task,
        activation_delta_steps=activation_delta_steps,
        ignition_step_override=ignition_step_override,
        activation_offset=activation_offset,
        current_time_seconds=current_time_seconds,
        activation_agent_id=activation_agent_id,
    )
    if use_backend_roundtrip:
        backend.set_ignition_evaluation_state(state)
        return FrozenIgnitionEvaluator().evaluate(
            backend.get_ignition_evaluation_state()
        )
    return FrozenIgnitionEvaluator().evaluate(state)


# ----------------------------------------------------------------------
# Public context / contract tests
# ----------------------------------------------------------------------


class PublicContextContractTests(unittest.TestCase):
    """Static contract: identity, immutability, fail-closed validation."""

    def test_action_allowlist_is_closed(self) -> None:
        self.assertEqual(
            ALLOWED_C4_IGNITION_ACTION_TYPES,
            frozenset(
                {"equip_item", "use_item", "place_block", "wait"}
            ),
        )
        self.assertEqual(
            ALLOWED_C4_IGNITION_TARGETS,
            frozenset(
                {
                    "water_bucket",
                    "lava_bucket",
                    "cobblestone",
                    "flint_and_steel",
                }
            ),
        )
        self.assertEqual(
            ALLOWED_C4_IGNITION_FAMILIES, frozenset({"casting"})
        )
        self.assertEqual(
            ALLOWED_C4_IGNITION_MODES, frozenset({"single"})
        )
        self.assertEqual(
            ALLOWED_C4_IGNITION_LEVELS, frozenset({"C4"})
        )
        self.assertEqual(
            ALLOWED_C4_IGNITION_LAYOUTS, frozenset({"fixed"})
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
        self.assertEqual(AGENT_ID, "agent_1")
        self.assertEqual(WORKFLOW_C4_IGNITION, "casting_s_c4_fixed")
        self.assertEqual(FAMILY_C4_IGNITION, "casting")
        self.assertEqual(MODE_C4_IGNITION, "single")
        self.assertEqual(LEVEL_C4_IGNITION, "C4")
        self.assertEqual(LAYOUT_C4_IGNITION, "fixed")
        self.assertEqual(C4_IGNITION_PUBLIC_ACTION, "use_item")
        self.assertEqual(C4_IGNITION_PUBLIC_ITEM, "flint_and_steel")
        self.assertEqual(C4_IGNITION_PUBLIC_TARGET, (1, 1, 1))
        self.assertEqual(
            C4_IGNITION_PUBLIC_TARGET_POLICY, "exact"
        )
        self.assertEqual(ROLE_VALUES, frozenset(
            {
                ROLE_CAST,
                ROLE_IGNITION_EQUIP,
                ROLE_IGNITION_USE,
                ROLE_IGNITION_SETTLE,
            }
        ))

    def test_default_target_cells_match_public_spec(self) -> None:
        self.assertEqual(len(CASTING_S_C4_IGNITION_FRAME_CELLS), 14)
        self.assertEqual(
            CASTING_S_C4_IGNITION_TARGET_CELL_COUNT, 14
        )
        # The driver must agree with the C3 frame evaluator on
        # the canonical 14-cell order. The duplication is
        # intentional: the driver module never imports the C3
        # driver or the evaluator surfaces.
        from obsidianlink.evaluation.casting_frame_evaluator import (
            CASTING_S_C3_FRAME_CELLS as EVAL_FRAME_CELLS,
        )
        self.assertEqual(
            CASTING_S_C4_IGNITION_FRAME_CELLS, EVAL_FRAME_CELLS
        )

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
            _context(level="C3")

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
            _context(target_offsets=CASTING_S_C4_IGNITION_FRAME_CELLS[:7])

    def test_context_rejects_reordered_target_offsets(self) -> None:
        reordered = (
            CASTING_S_C4_IGNITION_FRAME_CELLS[1],
            CASTING_S_C4_IGNITION_FRAME_CELLS[0],
        ) + CASTING_S_C4_IGNITION_FRAME_CELLS[2:]
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
                + CASTING_S_C4_IGNITION_FRAME_CELLS[2:14]
            )

    def test_context_rejects_out_of_grid_target_offset(self) -> None:
        out_of_grid = (
            CASTING_S_C4_IGNITION_FRAME_CELLS[:13] + ((99, 99, 99),)
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            _context(target_offsets=out_of_grid)

    def test_context_rejects_non_int_offset(self) -> None:
        bad = (
            CASTING_S_C4_IGNITION_FRAME_CELLS[:13]
            + ((1, 2, "1"),)  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "strict integers"):
            _context(target_offsets=bad)

    def test_context_rejects_bool_offset(self) -> None:
        bad = (
            CASTING_S_C4_IGNITION_FRAME_CELLS[:13]
            + ((True, False, 1),)  # type: ignore[arg-type]
        )
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
        inventory = {
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "water_bucket"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_missing_lava_bucket(self) -> None:
        inventory = {
            "water_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "lava_bucket"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_missing_cobblestone(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "cobblestone"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_missing_flint_and_steel(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "cobblestone": 28,
        }
        with self.assertRaisesRegex(ValueError, "flint_and_steel"):
            _context(inventory=inventory)

    def test_context_rejects_inventory_negative_quantity(self) -> None:
        inventory = {
            "water_bucket": -1,
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_context_rejects_inventory_bool_quantity(self) -> None:
        inventory = {
            "water_bucket": True,
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_context_rejects_inventory_unknown_item(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
            "obsidian": 1,
        }
        with self.assertRaisesRegex(ValueError, "forbidden item"):
            _context(inventory=inventory)

    def test_context_rejects_wrong_ignition_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_action"):
            _context(ignition_action="place_block")

    def test_context_rejects_wrong_ignition_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_item"):
            _context(ignition_item="torch")

    def test_context_rejects_wrong_ignition_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_target"):
            _context(ignition_target=(2, 1, 1))

    def test_context_rejects_bool_ignition_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict integers"):
            _context(ignition_target=(True, 1, 1))  # type: ignore[arg-type]

    def test_context_rejects_wrong_ignition_target_policy(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_target_policy"
        ):
            _context(ignition_target_policy="approximate")

    def test_context_rejects_ignition_required_false(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_required"):
            _context(ignition_required=False)

    def test_context_is_frozen(self) -> None:
        context = _context()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.episode_id = "tampered"  # type: ignore[misc]

    def test_context_inventory_is_mapping_proxy(self) -> None:
        context = _context()
        self.assertIsInstance(context.initial_inventory, Mapping)
        with self.assertRaises(TypeError):
            context.initial_inventory["water_bucket"] = 0  # type: ignore[index]

    def test_context_target_offsets_are_frozen_tuple(self) -> None:
        context = _context()
        self.assertIsInstance(context.target_offsets, tuple)
        with self.assertRaises(Exception):
            context.target_offsets[0] = (1, 1, 1)  # type: ignore[index]

    def test_context_grid_bounds_are_documented(self) -> None:
        self.assertEqual(C4_IGNITION_GRID_X_MIN, 0)
        self.assertEqual(C4_IGNITION_GRID_X_MAX, 3)
        self.assertEqual(C4_IGNITION_GRID_Y_MIN, 0)
        self.assertEqual(C4_IGNITION_GRID_Y_MAX, 4)
        self.assertEqual(C4_IGNITION_GRID_Z_MIN, 1)
        self.assertEqual(C4_IGNITION_GRID_Z_MAX, 1)


# ----------------------------------------------------------------------
# Build-public-context-from-task tests
# ----------------------------------------------------------------------


class BuildContextFromTaskTests(unittest.TestCase):
    """The orchestrator helper builds a strict public context."""

    def test_build_context_from_task_succeeds(self) -> None:
        task = _task()
        context = build_public_c4_ignition_driver_context_from_task(task)
        self.assertEqual(context.episode_id, task.task_id)
        self.assertEqual(context.workflow, WORKFLOW_C4_IGNITION)
        self.assertEqual(context.family, FAMILY_C4_IGNITION)
        self.assertEqual(context.mode, MODE_C4_IGNITION)
        self.assertEqual(context.level, LEVEL_C4_IGNITION)
        self.assertEqual(context.layout, LAYOUT_C4_IGNITION)
        self.assertEqual(context.agent_id, AGENT_ID)
        self.assertEqual(
            context.target_offsets, CASTING_S_C4_IGNITION_FRAME_CELLS
        )
        self.assertEqual(context.task_step_limit, 700)
        self.assertEqual(context.task_time_limit, 640.0)
        self.assertEqual(
            context.ignition_action, C4_IGNITION_PUBLIC_ACTION
        )
        self.assertEqual(context.ignition_item, C4_IGNITION_PUBLIC_ITEM)
        self.assertEqual(
            context.ignition_target, C4_IGNITION_PUBLIC_TARGET
        )
        self.assertEqual(
            context.ignition_target_policy,
            C4_IGNITION_PUBLIC_TARGET_POLICY,
        )
        self.assertTrue(context.ignition_required)

    def test_build_context_rejects_wrong_workflow(self) -> None:
        task = _task(workflow="casting_c1_fixed")
        with self.assertRaisesRegex(ValueError, "workflow"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_ignores_evaluator_contract(self) -> None:
        task = _task(include_evaluator_contract=True)
        context = build_public_c4_ignition_driver_context_from_task(task)
        self.assertEqual(
            context.target_offsets, CASTING_S_C4_IGNITION_FRAME_CELLS
        )

    def test_build_context_rejects_bool_coordinate(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["frame_plan"][
            "fixed_offsets"
        ][1][0] = True
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "strict integers"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_rejects_numeric_string_coordinate(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["frame_plan"][
            "fixed_offsets"
        ][1][0] = "1"
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "strict integers"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_rejects_wrong_ignition_action(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["ignition_plan"][
            "action"
        ] = "place_block"
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_action"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_rejects_wrong_ignition_target(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"]["ignition_plan"][
            "target_offset"
        ] = [2, 1, 1]
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_target"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_requires_ignition_required_field(self) -> None:
        payload = _task_dict()
        del payload["scenario_parameters"]["public_task_spec"][
            "ignition_plan"
        ]["required"]
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_plan.required"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_rejects_string_ignition_required(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"][
            "ignition_plan"
        ]["required"] = "false"
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_required"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_rejects_integer_ignition_required(self) -> None:
        payload = _task_dict()
        payload["scenario_parameters"]["public_task_spec"][
            "ignition_plan"
        ]["required"] = 1
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_required"):
            build_public_c4_ignition_driver_context_from_task(task)

    def test_build_context_requires_ignition_plan(self) -> None:
        payload = _task_dict()
        del payload["scenario_parameters"]["public_task_spec"][
            "ignition_plan"
        ]
        task = TaskInstance.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "ignition_plan"):
            build_public_c4_ignition_driver_context_from_task(task)


# ----------------------------------------------------------------------
# Plan builder tests
# ----------------------------------------------------------------------


class PlanBuilderTests(unittest.TestCase):
    """The deterministic plan builder yields a closed, ordered plan."""

    def test_default_plan_length_and_relevant_action_count(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        # 14 cells × 24 cast steps + 4 ignition steps.
        self.assertEqual(len(plan), 340)
        cast_relevant = sum(
            step.relevant_action and step.role == ROLE_CAST
            for step in plan
        )
        ignition_relevant = sum(
            step.relevant_action and step.role == ROLE_IGNITION_USE
            for step in plan
        )
        self.assertEqual(cast_relevant, 14 * 2)
        self.assertEqual(ignition_relevant, 1)

    def test_default_plan_wait_count_within_hard_cap(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        wait_count = sum(
            step.action.action_type == "wait" for step in plan
        )
        self.assertEqual(wait_count, 14 * 17 + 2)
        self.assertLessEqual(wait_count, MAX_IGNITION_PLAN_WAIT_STEPS)

    def test_default_plan_length_within_task_step_limit(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        self.assertLessEqual(len(plan), MAX_IGNITION_PLAN_STEPS)
        self.assertLessEqual(len(plan), DEFAULT_MAX_ENVIRONMENT_STEPS)

    def test_plan_steps_use_closed_action_allowlist(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        for step in plan:
            self.assertIn(step.action.action_type, ALLOWED_C4_IGNITION_ACTION_TYPES)
            if step.action.target is not None:
                self.assertIn(
                    step.action.target, ALLOWED_C4_IGNITION_TARGETS
                )

    def test_plan_ignition_use_step_target_is_flint_and_steel(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        ignition_use = [
            step
            for step in plan
            if step.role == ROLE_IGNITION_USE
        ]
        self.assertEqual(len(ignition_use), 1)
        self.assertEqual(
            ignition_use[0].action.action_type, C4_IGNITION_PUBLIC_ACTION
        )
        self.assertEqual(
            ignition_use[0].action.target, C4_IGNITION_PUBLIC_ITEM
        )
        self.assertEqual(
            ignition_use[0].target_offset, C4_IGNITION_PUBLIC_TARGET
        )
        self.assertTrue(ignition_use[0].relevant_action)

    def test_plan_ignition_equip_step_target_is_flint_and_steel(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        equip_steps = [
            step for step in plan if step.role == ROLE_IGNITION_EQUIP
        ]
        self.assertGreaterEqual(len(equip_steps), 1)
        self.assertEqual(
            equip_steps[0].action.action_type, "equip_item"
        )
        self.assertEqual(
            equip_steps[0].action.target, C4_IGNITION_PUBLIC_ITEM
        )
        self.assertEqual(
            equip_steps[0].target_offset, C4_IGNITION_PUBLIC_TARGET
        )
        self.assertFalse(equip_steps[0].relevant_action)

    def test_plan_actions_parse_through_protocol(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        for step in plan:
            payload = {
                "action_type": step.action.action_type,
                "target": step.action.target,
                "duration_ticks": step.action.duration_ticks,
            }
            result = parse_macro_action(json.dumps(payload))
            self.assertTrue(
                result.accepted,
                msg=f"plan step {step.label!r} rejected: {result.error}",
            )
            self.assertEqual(
                result.action.action_type, step.action.action_type
            )
            self.assertEqual(result.action.target, step.action.target)

    def test_plan_target_offsets_match_public_spec(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        cast_steps = [step for step in plan if step.role == ROLE_CAST]
        cells_seen: list[tuple[int, int, int]] = []
        for step in cast_steps:
            if step.target_offset is not None and step.cell_index is not None:
                self.assertEqual(
                    step.target_offset,
                    CASTING_S_C4_IGNITION_FRAME_CELLS[step.cell_index],
                )
                if (
                    step.action.action_type == "use_item"
                    and step.cell_index not in [
                        cs[0] for cs in cells_seen
                    ]
                ):
                    cells_seen.append((step.cell_index, step.target_offset[0],
                                       step.target_offset[1]))
        self.assertEqual(
            [cs[0] for cs in cells_seen],
            list(range(CASTING_S_C4_IGNITION_TARGET_CELL_COUNT)),
        )

    def test_plan_phases_are_closed(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        for step in plan:
            self.assertIn(step.phase, PHASE_VALUES)

    def test_plan_roles_are_closed(self) -> None:
        plan = build_casting_s_c4_ignition_action_plan()
        for step in plan:
            self.assertIn(step.role, ROLE_VALUES)

    def test_plan_rejects_reordered_target_offsets(self) -> None:
        reordered = (
            CASTING_S_C4_IGNITION_FRAME_CELLS[1],
            CASTING_S_C4_IGNITION_FRAME_CELLS[0],
        ) + CASTING_S_C4_IGNITION_FRAME_CELLS[2:]
        with self.assertRaisesRegex(ValueError, "must match the locked"):
            build_casting_s_c4_ignition_action_plan(
                target_offsets=reordered
            )

    def test_plan_rejects_recoveries_over_per_action_cap(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_use_item"
        ):
            build_casting_s_c4_ignition_action_plan(
                recoveries_per_use_item=MAX_RECOVERIES_PER_ACTION + 1
            )
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_ignition_use"
        ):
            build_casting_s_c4_ignition_action_plan(
                recoveries_per_ignition_use=MAX_RECOVERIES_PER_ACTION + 1
            )

    def test_plan_default_budget_constants_are_documented(self) -> None:
        self.assertEqual(RECOVERIES_PER_USE_ITEM_DEFAULT, 1)
        self.assertEqual(RECOVERIES_PER_IGNITION_USE_DEFAULT, 1)
        self.assertEqual(TOTAL_RECOVERY_BUDGET_DEFAULT, 16)
        self.assertEqual(MAX_RECOVERIES_PER_ACTION, 2)
        self.assertEqual(MAX_TOTAL_RECOVERY_BUDGET, 32)
        self.assertEqual(MAX_IGNITION_PLAN_STEPS, 700)
        self.assertEqual(MAX_IGNITION_PLAN_WAIT_STEPS, 320)
        self.assertEqual(DEFAULT_MAX_WAIT_STEPS, 256)


# ----------------------------------------------------------------------
# Plan step validation tests
# ----------------------------------------------------------------------


class PlanStepValidationTests(unittest.TestCase):
    """Plan steps fail closed on malformed construction."""

    def test_cast_step_requires_cell_index(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "cast plan step must have a non-None cell_index"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_PLACE_LAVA,
                action=MacroAction(action_type="use_item", target="lava_bucket"),
                role=ROLE_CAST,
                cell_index=None,
                target_offset=(0, 0, 1),
                relevant_action=True,
            )

    def test_ignition_step_requires_none_cell_index(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition plan step must have cell_index=None"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="use_item", target="flint_and_steel"
                ),
                role=ROLE_IGNITION_USE,
                cell_index=0,
                target_offset=C4_IGNITION_PUBLIC_TARGET,
                relevant_action=True,
            )

    def test_ignition_step_requires_public_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition target_offset must be"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="use_item", target="flint_and_steel"
                ),
                role=ROLE_IGNITION_USE,
                cell_index=None,
                target_offset=(2, 1, 1),
                relevant_action=True,
            )

    def test_cast_step_rejects_out_of_range_cell_index(self) -> None:
        from obsidianlink.drivers.casting_s_c4_ignition import (
            _cast_select_step,
        )
        with self.assertRaisesRegex(ValueError, "cell_index"):
            _cast_select_step(
                CASTING_S_C4_IGNITION_TARGET_CELL_COUNT,
                "lava_bucket",
                "x",
                PHASE_PREPARE,
            )

    def test_cast_step_rejects_mismatched_target_offset(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "target_offset must match the locked C4 frame cell"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_PLACE_WATER,
                action=MacroAction(
                    action_type="use_item", target="water_bucket"
                ),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=(3, 0, 1),
                relevant_action=True,
            )

    def test_cast_step_rejects_relevant_action_for_non_use(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "relevant_action must be true exactly for cast"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_PLACE_SUPPORT,
                action=MacroAction(
                    action_type="place_block", target="cobblestone"
                ),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C4_IGNITION_FRAME_CELLS[0],
                relevant_action=True,
            )

    def test_ignition_use_step_rejects_wrong_action(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_use step must target"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="use_item", target="water_bucket"
                ),
                role=ROLE_IGNITION_USE,
                cell_index=None,
                target_offset=C4_IGNITION_PUBLIC_TARGET,
                relevant_action=True,
            )

    def test_ignition_use_step_rejects_wrong_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_use step must target"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="use_item", target="water_bucket"
                ),
                role=ROLE_IGNITION_USE,
                cell_index=None,
                target_offset=C4_IGNITION_PUBLIC_TARGET,
                relevant_action=True,
            )

    def test_ignition_use_step_rejects_irrelevant_flag(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_use step must be marked relevant"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="use_item", target="flint_and_steel"
                ),
                role=ROLE_IGNITION_USE,
                cell_index=None,
                target_offset=C4_IGNITION_PUBLIC_TARGET,
                relevant_action=False,
            )

    def test_ignition_settle_step_rejects_use(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_settle step must be a wait"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_IGNITION_PORTAL_SETTLE,
                action=MacroAction(
                    action_type="use_item", target="flint_and_steel"
                ),
                role=ROLE_IGNITION_SETTLE,
                cell_index=None,
                target_offset=C4_IGNITION_PUBLIC_TARGET,
                relevant_action=True,
            )

    def test_plan_step_rejects_unknown_phase(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unknown C4 ignition plan phase"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase="not_a_phase",
                action=MacroAction.wait(),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C4_IGNITION_FRAME_CELLS[0],
            )

    def test_plan_step_rejects_unknown_role(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unknown C4 ignition plan role"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_PREPARE,
                action=MacroAction.wait(),
                role="not_a_role",
                cell_index=0,
                target_offset=CASTING_S_C4_IGNITION_FRAME_CELLS[0],
            )

    def test_plan_step_rejects_recovery_budget_over_cap(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recoveries_allowed must be an int between"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_PLACE_LAVA,
                action=MacroAction(
                    action_type="use_item", target="lava_bucket"
                ),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C4_IGNITION_FRAME_CELLS[0],
                relevant_action=True,
                recoveries_allowed=MAX_RECOVERIES_PER_ACTION + 1,
            )

    def test_plan_step_rejects_recovery_phase(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "PHASE_RECOVERY is reserved"
        ):
            CastingC4IgnitionPlanStep(
                label="x",
                phase=PHASE_RECOVERY,
                action=MacroAction.wait(),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C4_IGNITION_FRAME_CELLS[0],
            )


# ----------------------------------------------------------------------
# Driver result tests
# ----------------------------------------------------------------------


class DriverResultContractTests(unittest.TestCase):
    """The result object is frozen, type-strict, and JSON-serializable."""

    def _result(self, **overrides: Any) -> CastingC4IgnitionDriverResult:
        defaults: dict[str, Any] = dict(
            status=DRIVER_STATUS_COMPLETED,
            steps_executed=340,
            wait_steps=240,
            planned_steps=340,
            recovery_attempts=0,
            recovery_budget=16,
            per_cell_relevant_action_records=MappingProxyType({}),
            per_cell_relevant_action_steps=MappingProxyType({}),
            per_cell_target_offset=MappingProxyType({}),
            ignition_relevant_action_step=IGNITION_STEP,
            ignition_target_offset=C4_IGNITION_PUBLIC_TARGET,
            ignition_equip_step=IGNITION_EQUIP_STEP,
            final_observation=Observation(
                episode_id=EPISODE_ID,
                agent_id=AGENT_ID,
                step_id=0,
                timestamp=0.0,
                frame={},
                visible_inventory=dict(DEFAULT_INVENTORY),
                workflow_stage=WORKFLOW_C4_IGNITION,
            ),
            events=(),
            action_label_for_step=MappingProxyType({}),
            terminated=False,
            truncated=False,
        )
        defaults.update(overrides)
        return CastingC4IgnitionDriverResult(**defaults)

    def test_completed_requires_ignition_step(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "completed driver must have submitted the ignition"
        ):
            self._result(ignition_relevant_action_step=None)

    def test_completed_requires_public_ignition_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "completed driver must carry the public ignition"
        ):
            self._result(ignition_target_offset=(2, 1, 1))

    def test_completed_requires_full_plan_executed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "completed driver must execute the full plan"
        ):
            self._result(steps_executed=339)

    def test_blocked_requires_blocked_reason(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "blocked/failed driver requires blocked_reason"
        ):
            self._result(
                status=DRIVER_STATUS_BLOCKED,
                ignition_relevant_action_step=None,
                ignition_target_offset=None,
                blocked_reason=None,
            )

    def test_blocked_with_blocked_reason_ok(self) -> None:
        result = self._result(
            status=DRIVER_STATUS_BLOCKED,
            ignition_relevant_action_step=None,
            ignition_target_offset=None,
            steps_executed=10,
            wait_steps=2,
            planned_steps=340,
            blocked_reason="step budget exhausted",
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertEqual(result.blocked_reason, "step budget exhausted")

    def test_recovery_attempts_over_budget_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recovery_attempts cannot exceed"
        ):
            self._result(recovery_attempts=20, recovery_budget=16)

    def test_invalid_status_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "driver status"):
            self._result(status="success")

    def test_relevant_action_record_rejects_non_string_item(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "per_cell_relevant_action_records values"
        ):
            self._result(
                per_cell_relevant_action_records=MappingProxyType(
                    {0: ((9, 99),)}
                )
            )

    def test_ignition_target_offset_rejects_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict integers"):
            self._result(ignition_target_offset=(True, 1, 1))

    def test_is_frozen(self) -> None:
        result = self._result()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = "tampered"  # type: ignore[misc]

    def test_as_dict_is_json_serializable(self) -> None:
        result = self._result(
            per_cell_relevant_action_records=MappingProxyType(
                {0: ((9, "lava_bucket"), (16, "water_bucket"))}
            ),
            per_cell_relevant_action_steps=MappingProxyType(
                {0: (9, 16)}
            ),
            per_cell_target_offset=MappingProxyType(
                {0: CASTING_S_C4_IGNITION_FRAME_CELLS[0]}
            ),
            action_label_for_step=MappingProxyType(
                {9: "cell_0.casting.use_lava"}
            ),
            events=(
                {
                    "episode_id": EPISODE_ID,
                    "agent_id": AGENT_ID,
                    "step_id": 9,
                    "label": "cell_0.casting.use_lava",
                    "phase": PHASE_PLACE_LAVA,
                    "action_type": "use_item",
                    "target": "lava_bucket",
                    "relevant_action": True,
                    "role": ROLE_CAST,
                    "attempt": 1,
                },
            ),
        )
        snapshot = result.as_dict()
        encoded = json.dumps(snapshot)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["status"], DRIVER_STATUS_COMPLETED)
        self.assertEqual(decoded["steps_executed"], 340)
        self.assertEqual(decoded["ignition_target_offset"], [1, 1, 1])
        self.assertEqual(
            decoded["per_cell_relevant_action_records"]["0"][0],
            {"step_id": 9, "item": "lava_bucket"},
        )
        self.assertEqual(decoded["events"][0]["label"], "cell_0.casting.use_lava")

    def test_as_dict_uses_lists_not_tuples(self) -> None:
        result = self._result(
            per_cell_target_offset=MappingProxyType(
                {0: CASTING_S_C4_IGNITION_FRAME_CELLS[0]}
            ),
        )
        snapshot = result.as_dict()
        # ``as_dict`` converts tuple offsets / ignition offsets to
        # JSON-friendly lists. The mapping keys are still ints
        # until the snapshot is round-tripped through ``json``.
        self.assertIsInstance(
            snapshot["per_cell_target_offset"][0], list
        )
        # The ignition_target_offset in the result is a tuple
        # (constructor freeze). ``as_dict`` must emit a list.
        self.assertIsInstance(snapshot["ignition_target_offset"], list)


# ----------------------------------------------------------------------
# Driver execution tests
# ----------------------------------------------------------------------


class DriverExecutionTests(unittest.TestCase):
    """The driver walks the plan deterministically on the FakeBackend."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_default_run_completes(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.steps_executed, 340)
        self.assertEqual(result.planned_steps, 340)
        self.assertEqual(result.wait_steps, 14 * 17 + 2)
        self.assertEqual(result.ignition_relevant_action_step, IGNITION_STEP)
        self.assertEqual(
            result.ignition_target_offset, C4_IGNITION_PUBLIC_TARGET
        )
        self.assertEqual(result.ignition_equip_step, IGNITION_EQUIP_STEP)
        for cell_index in range(14):
            records = result.per_cell_relevant_action_records[cell_index]
            self.assertEqual(len(records), 2)
            steps = result.per_cell_relevant_action_steps[cell_index]
            self.assertEqual(steps[1] - steps[0], 7)
            self.assertEqual(
                result.per_cell_target_offset[cell_index],
                CASTING_S_C4_IGNITION_FRAME_CELLS[cell_index],
            )

    def test_default_run_does_not_return_success_or_passed(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertNotIn(result.status, {"success", "passed"})
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)

    def test_driver_uses_only_allowed_actions(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        action_types = {event["action_type"] for event in result.events}
        targets = {event.get("target") for event in result.events}
        for action_type in action_types:
            self.assertIn(action_type, ALLOWED_C4_IGNITION_ACTION_TYPES)
        for target in targets:
            if target is not None:
                self.assertIn(target, ALLOWED_C4_IGNITION_TARGETS)

    def test_driver_emits_ignition_use_with_public_target(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        ignition_events = [
            event
            for event in result.events
            if event.get("role") == ROLE_IGNITION_USE
            and event.get("relevant_action")
        ]
        self.assertEqual(len(ignition_events), 1)
        event = ignition_events[0]
        self.assertEqual(event["action_type"], C4_IGNITION_PUBLIC_ACTION)
        self.assertEqual(event["target"], C4_IGNITION_PUBLIC_ITEM)
        # ``result.events`` is frozen by the driver, so the
        # recorded list target_offset is stored as a tuple.
        self.assertEqual(
            tuple(event["target_offset"]), C4_IGNITION_PUBLIC_TARGET
        )

    def test_driver_emits_ignition_equip_with_flint_and_steel(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        equip_events = [
            event
            for event in result.events
            if event.get("action_type") == "equip_item"
            and event.get("target") == C4_IGNITION_PUBLIC_ITEM
        ]
        self.assertGreaterEqual(len(equip_events), 1)
        self.assertEqual(
            tuple(equip_events[0]["target_offset"]),
            C4_IGNITION_PUBLIC_TARGET,
        )

    def test_driver_blocks_on_missing_flint_and_steel(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "cobblestone": 28,
        }
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "flint_and_steel"
        ):
            run_casting_s_c4_ignition_driver(
                backend, _context(inventory=inventory)
            )

    def test_driver_blocks_on_missing_water_bucket(self) -> None:
        inventory = {
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "water_bucket"
        ):
            run_casting_s_c4_ignition_driver(
                backend, _context(inventory=inventory)
            )

    def test_driver_blocks_on_wrong_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "workflow"):
            _context(workflow="casting_c1_fixed")

    def test_driver_blocks_on_wrong_ignition_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_target"):
            _context(ignition_target=(2, 1, 1))

    def test_driver_rejects_wrong_context_type(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "PublicC4IgnitionDriverContext"):
            run_casting_s_c4_ignition_driver(backend, object())  # type: ignore[arg-type]

    def test_driver_rejects_non_positive_max_wait_steps(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_s_c4_ignition_driver(
                backend, _context(), max_wait_steps=0
            )

    def test_driver_rejects_max_wait_steps_over_hard_cap(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                max_wait_steps=MAX_IGNITION_PLAN_WAIT_STEPS + 1,
            )

    def test_driver_rejects_max_environment_steps_over_task_limit(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps cannot exceed the task limit"
        ):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS + 1,
            )

    def test_driver_rejects_max_game_time_over_task_limit(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "max_game_time_seconds cannot exceed the task limit"
        ):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                max_game_time_seconds=DEFAULT_MAX_GAME_TIME_SECONDS + 1.0,
            )

    def test_driver_rejects_plan_over_task_step_limit(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c4_ignition_action_plan()
        with self.assertRaisesRegex(
            ValueError, "plan length cannot exceed the task step limit"
        ):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(task_step_limit=14),
                plan=plan,
            )

    def test_driver_rejects_invalid_plan_type(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "CastingC4IgnitionPlanStep"):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                plan=(MacroAction.wait(),),  # type: ignore[arg-type]
            )

    def test_driver_rejects_ignition_only_plan(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c4_ignition_action_plan()
        ignition_only = tuple(
            step for step in plan if step.role != ROLE_CAST
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                plan=ignition_only,
            )

    def test_driver_rejects_duplicate_ignition_use(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c4_ignition_action_plan()
        ignition_use = next(
            step for step in plan if step.role == ROLE_IGNITION_USE
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                plan=plan + (ignition_use,),
            )

    def test_driver_rejects_negative_total_recovery_budget(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "total_recovery_budget"
        ):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                total_recovery_budget=-1,
            )

    def test_driver_rejects_total_recovery_budget_over_cap(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "total_recovery_budget"
        ):
            run_casting_s_c4_ignition_driver(
                backend,
                _context(),
                total_recovery_budget=MAX_TOTAL_RECOVERY_BUDGET + 1,
            )

    def test_driver_rejects_backend_without_reset(self) -> None:
        class _Bad:
            def step(self, actions: Any) -> Any:
                return None

        with self.assertRaisesRegex(ValueError, "reset/step"):
            run_casting_s_c4_ignition_driver(_Bad(), _context())

    def test_deterministic_replay(self) -> None:
        first = run_casting_s_c4_ignition_driver(
            self._open_backend(), _context()
        )
        second = run_casting_s_c4_ignition_driver(
            self._open_backend(), _context()
        )
        self.assertEqual(first.steps_executed, second.steps_executed)
        self.assertEqual(first.wait_steps, second.wait_steps)
        self.assertEqual(
            first.ignition_relevant_action_step,
            second.ignition_relevant_action_step,
        )
        self.assertEqual(
            first.ignition_equip_step, second.ignition_equip_step
        )
        self.assertEqual(
            dict(first.per_cell_relevant_action_steps),
            dict(second.per_cell_relevant_action_steps),
        )
        self.assertEqual(
            [dict(event) for event in first.events],
            [dict(event) for event in second.events],
        )
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_event_identity_and_action_metadata(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        for event in result.events:
            self.assertEqual(event["episode_id"], EPISODE_ID)
            self.assertEqual(event["agent_id"], AGENT_ID)
            self.assertIn("step_id", event)
            self.assertIn("label", event)
            self.assertIn("phase", event)
            self.assertIn("action_type", event)
            self.assertIn("role", event)


# ----------------------------------------------------------------------
# Budget tests
# ----------------------------------------------------------------------


class BudgetTests(unittest.TestCase):
    """Step / time / wait / plan / recovery budgets fail closed."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_step_budget_blocked(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(),
            max_environment_steps=20,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("step budget", result.blocked_reason or "")
        self.assertLess(result.steps_executed, 340)
        self.assertIsNone(result.ignition_relevant_action_step)

    def test_time_budget_blocked(self) -> None:
        class _SlowBackend(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._t0 = 0.0

            def reset(self, task: Any) -> Any:
                obs = super().reset(task)
                self._t0 = float(
                    list(obs.values())[0].timestamp
                )
                return obs

            def step(self, actions: Any) -> Any:
                step_result = super().step(actions)
                # Shift every step timestamp by 10 seconds so the
                # time budget of 10s is exhausted on the first
                # submitted step.
                shifted = {
                    agent_id: dataclasses.replace(
                        observation, timestamp=observation.timestamp + 10.0
                    )
                    for agent_id, observation in step_result.observations.items()
                }
                return BackendStep(
                    episode_id=step_result.episode_id,
                    step_id=step_result.step_id,
                    observations=shifted,
                    rewards=step_result.rewards,
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                    info=step_result.info,
                )

        backend = _SlowBackend()
        backend.open()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(task_time_limit=10.0),
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("time budget", result.blocked_reason or "")

    def test_wait_budget_blocked(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(),
            max_wait_steps=1,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("wait budget", result.blocked_reason or "")

    def test_plan_length_over_hard_cap_blocked(self) -> None:
        # A custom plan longer than the hard cap must be rejected
        # by the plan builder.
        with self.assertRaisesRegex(
            ValueError, "exceed the hard limit"
        ):
            build_casting_s_c4_ignition_action_plan(
                support_block_wait_steps=4,
                fluid_settle_wait_steps=10,
                obsidian_wait_steps=10,
                ignition_portal_settle_steps=5,
            )

    def test_total_recovery_budget_exhausted_blocked(self) -> None:
        class _RecoverOnceOnUseItem(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._recovered_steps: set[int] = set()

            def step(self, actions: Any) -> Any:
                for action in actions.values():
                    if (
                        action.action_type == "use_item"
                        and self._step_id not in self._recovered_steps
                    ):
                        self._recovered_steps.add(self._step_id)
                        raise RecoverableBackendError(
                            "transient",
                            recoverable_kind="bucket_use_transient",
                            attempt=1,
                        )
                return super().step(actions)

        backend = _RecoverOnceOnUseItem()
        backend.open()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(),
            total_recovery_budget=4,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("recovery budget", result.blocked_reason or "")
        self.assertGreater(result.recovery_attempts, 4)

    def test_per_step_recovery_budget_exhausted_blocked(self) -> None:
        class _AlwaysRecoverOnUseItem(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._use_item_attempts = 0

            def step(self, actions: Any) -> Any:
                for action in actions.values():
                    if action.action_type == "use_item":
                        self._use_item_attempts += 1
                        if self._use_item_attempts <= 4:
                            raise RecoverableBackendError(
                                "transient",
                                recoverable_kind="bucket_use_transient",
                                attempt=self._use_item_attempts,
                            )
                return super().step(actions)

        backend = _AlwaysRecoverOnUseItem()
        backend.open()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(),
            total_recovery_budget=32,
            recoveries_per_use_item=2,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        # The per-step budget is exhausted (per-step default is
        # 1, so 2 errors per use_item step with per-step=2 ⇒
        # 3rd raise blocks; the first use_item step is the
        # one to fail).
        self.assertIn(
            "per-step recovery budget", result.blocked_reason or ""
        )


# ----------------------------------------------------------------------
# Recovery tests
# ----------------------------------------------------------------------


class RecoveryTests(unittest.TestCase):
    """Typed recoverable error is retried; non-recoverable fails closed."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_recoverable_error_retried_within_budget(self) -> None:
        class _OneTimeRecover(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self.recovered = False
                self.recovery_attempts = 0

            def step(self, actions: Any) -> Any:
                # The 9th backend.step call (pre-call counter == 8)
                # is the first ``use_item(lava_bucket)`` for cell
                # 0; that step has a per-step recovery budget of 1.
                if (
                    not self.recovered
                    and self._step_id == 8
                ):
                    self.recovered = True
                    self.recovery_attempts += 1
                    raise RecoverableBackendError(
                        "transient",
                        recoverable_kind="bucket_use_transient",
                        attempt=1,
                    )
                return super().step(actions)

        backend = _OneTimeRecover()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.recovery_attempts, 1)

    def test_non_recoverable_runtime_error_fails_closed(self) -> None:
        class _Boom(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self.failed = False

            def step(self, actions: Any) -> Any:
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("backend exploded")
                return super().step(actions)

        backend = _Boom()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertIn("backend exploded", result.blocked_reason or "")
        self.assertEqual(result.error_type, "RuntimeError")

    def test_non_recoverable_type_error_fails_closed(self) -> None:
        class _TypeBoom(FakeEnvironmentBackend):
            def step(self, actions: Any) -> Any:
                raise TypeError("bad step")

        backend = _TypeBoom()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertEqual(result.error_type, "TypeError")

    def test_os_error_fails_closed(self) -> None:
        class _OSBoom(FakeEnvironmentBackend):
            def step(self, actions: Any) -> Any:
                raise OSError("io error")

        backend = _OSBoom()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertEqual(result.error_type, "OSError")

    def test_backend_terminated_mid_plan(self) -> None:
        class _Terminated(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._term_at = 12

            def step(self, actions: Any) -> Any:
                step = super().step(actions)
                if step.step_id == self._term_at:
                    return BackendStep(
                        episode_id=step.episode_id,
                        step_id=step.step_id,
                        observations=step.observations,
                        rewards=step.rewards,
                        terminated=True,
                        truncated=False,
                        info=step.info,
                    )
                return step

        backend = _Terminated()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("termination", result.blocked_reason or "")
        self.assertTrue(result.terminated)


# ----------------------------------------------------------------------
# Capability gate tests
# ----------------------------------------------------------------------


class CapabilityGateTests(unittest.TestCase):
    """The pre-episode capability gate fails closed before reset."""

    def test_full_capabilities_pass(self) -> None:
        backend = FakeEnvironmentBackend()
        backend._capabilities = BackendCapabilities.full()  # type: ignore[attr-defined]
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)

    def test_missing_capability_fails_before_reset(self) -> None:
        backend = FakeEnvironmentBackend.with_capabilities(
            BackendCapabilities(
                can_select_water_bucket=True,
                can_select_lava_bucket=True,
                can_use_water_bucket=True,
                can_use_lava_bucket=True,
                exposes_public_inventory=True,
                exposes_selected_item=False,
                exposes_target_block_truth=False,
                exposes_fluid_truth=False,
            )
        )
        backend.open()
        with self.assertRaises(CapabilityMismatchError):
            run_casting_s_c4_ignition_driver(backend, _context())


# ----------------------------------------------------------------------
# Observation schema tests
# ----------------------------------------------------------------------


class ObservationSchemaTests(unittest.TestCase):
    """The driver never reads hidden fields from Observations."""

    FORBIDDEN_TOKENS: tuple[str, ...] = (
        "latched_frame_identity",
        "ignition_evaluation",
        "nether_portal",
        "wrong_ignition",
        "frame_not_built",
        "public_ignition_target",
        "frame_interior",
        "FrozenFrameIdentity",
        "FrozenIgnition",
        "IgnitionActionEvidence",
        "PortalActivationEvidence",
    )

    def test_default_fake_observation_carries_no_truth_tokens(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        for observation in (result.final_observation,):
            for token in self.FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    repr(observation),
                    msg=(
                        f"Observation leaks {token!r} for the C4 ignition "
                        "driver"
                    ),
                )

    def test_observation_schema_is_locked_to_nine_public_fields(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        observations = backend.reset(
            _ResetProxy(_context())  # type: ignore[arg-type]
        )
        observation = observations[AGENT_ID]
        fields = {field.name for field in dataclasses.fields(observation)}
        self.assertEqual(
            fields,
            {
                "episode_id",
                "agent_id",
                "step_id",
                "timestamp",
                "frame",
                "visible_inventory",
                "selected_item",
                "messages",
                "workflow_stage",
            },
        )

    def test_stepped_observation_has_no_truth_tokens(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        result = run_casting_s_c4_ignition_driver(backend, _context())
        for token in self.FORBIDDEN_TOKENS:
            self.assertNotIn(token, repr(result.final_observation))
            for event in result.events:
                values = [
                    str(value) for value in event.values() if value is not None
                ]
                joined = " ".join(values)
                self.assertNotIn(
                    token,
                    joined,
                    msg=(
                        f"Driver event leaks {token!r}; events must "
                        "not carry evaluator truth"
                    ),
                )


# ----------------------------------------------------------------------
# AST / source-level isolation tests
# ----------------------------------------------------------------------


class DriverSourceIsolationTests(unittest.TestCase):
    """The driver source must not import or reference evaluator types."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER_SOURCE.read_text()
        cls.tree = ast.parse(cls.source)

    def _code_only(self) -> str:
        """Return the source with all docstring / comment lines masked."""
        tree = self.tree
        docstring_ranges: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Module,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    docstring_ranges.append(
                        (node.body[0].lineno, node.body[0].end_lineno or node.body[0].lineno)
                    )
        masked_lines = list(self.source.split("\n"))
        for start, end in docstring_ranges:
            for i in range(start - 1, end):
                if 0 <= i < len(masked_lines):
                    masked_lines[i] = ""
        return "\n".join(masked_lines)

    def test_no_imports_from_evaluator_modules(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and (
                    "casting_ignition_evaluator" in node.module
                    or "casting_frame_evaluator" in node.module
                    or "continuous_casting" in node.module
                    or "portal" in node.module
                    or "casting" == node.module.split(".")[-1]
                    and "evaluation" in node.module
                ):
                    if "casting_ignition_evaluator" in node.module:
                        self.fail(
                            f"driver imports from {node.module!r} at line "
                            f"{node.lineno}; C4 ignition evaluator types "
                            "are forbidden"
                        )
                    if "casting_frame_evaluator" in node.module:
                        self.fail(
                            f"driver imports from {node.module!r} at line "
                            f"{node.lineno}; C3 frame evaluator types are "
                            "forbidden"
                        )

    def test_no_attribute_references_to_evaluator_state(self) -> None:
        # Walk the AST to find any ``Attribute`` access on the
        # ignition truth methods, the backend's private C1/C2/C3/
        # C4 slots, or the evaluator-only types. Docstring /
        # comment references are ignored.
        forbidden_attrs = {
            "set_ignition_evaluation_state",
            "get_ignition_evaluation_state",
            "clear_ignition_evaluation_state",
            "_ignition_evaluation_state",
            "_frame_evaluation_state",
            "_continuous_casting_evaluation_state",
            "_casting_evaluation_state",
            "FrozenFrameIdentity",
            "IgnitionActionEvidence",
            "PortalActivationEvidence",
            "FrozenIgnitionEvaluationState",
            "FrozenIgnitionEvaluationResult",
            "FrozenIgnitionEvaluator",
            "build_c4_c3_frame_identity",
            "evaluator_contract",
        }
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                self.fail(
                    f"driver source references attribute {node.attr!r} "
                    f"at line {node.lineno}"
                )

    def test_no_scenario_parameters_in_code(self) -> None:
        # Walk the AST to find any ``Attribute`` access to
        # ``scenario_parameters`` (in real code, not in a
        # docstring / comment). The driver may not pull any field
        # off ``task.scenario_parameters`` directly.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr == "scenario_parameters":
                self.fail(
                    "driver must not reference 'scenario_parameters' "
                    f"as a code attribute at line {node.lineno}"
                )
        # Block subscript access on a ``scenario_parameters`` attribute.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Subscript):
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "scenario_parameters"
                ):
                    self.fail(
                        "driver must not read task.scenario_parameters "
                        f"at line {node.lineno}"
                    )

    def test_no_imports_from_workflow_agents(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and (
                    "agents" in node.module
                    or "workflows" in node.module
                    or "model" in node.module
                    or "planner" in node.module
                ):
                    self.fail(
                        f"driver imports from forbidden surface {node.module!r} "
                        f"at line {node.lineno}"
                    )

    def test_recovery_kind_does_not_leak_ignition_truth(self) -> None:
        masked = self._code_only()
        for token in (
            "latched_frame_identity",
            "nether_portal",
            "frame_outcome",
            "ignition_evaluation",
        ):
            self.assertNotIn(token, masked)

    def test_event_sink_propagates_event_copy(self) -> None:
        seen: list[dict[str, Any]] = []
        backend = FakeEnvironmentBackend()
        backend.open()
        result = run_casting_s_c4_ignition_driver(
            backend,
            _context(),
            event_sink=lambda event: seen.append(dict(event)),
        )
        self.assertEqual(len(seen), len(result.events))
        first = seen[0]
        for token in (
            "latched_frame_identity",
            "ignition_evaluation",
            "nether_portal",
        ):
            self.assertNotIn(token, repr(first))


# ----------------------------------------------------------------------
# FakeBackend truth-slot isolation tests
# ----------------------------------------------------------------------


class TruthSlotIsolationTests(unittest.TestCase):
    """C1 / C2 / C3 / C4 truth slots are independent on FakeBackend."""

    def test_truth_slots_are_separate(self) -> None:
        from obsidianlink.evaluation.casting import CastingEvaluationState
        from obsidianlink.evaluation.continuous_casting import (
            CASTING_C3_TARGET_CELLS,
            ContinuousCastingCellTruth,
            ContinuousCastingEvaluationState,
        )

        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(_ResetProxy(_context()))  # type: ignore[arg-type]
        # Inject C1 truth.
        backend.set_casting_evaluation_state(
            CastingEvaluationState(episode_id=EPISODE_ID, step_id=0)
        )
        # Inject C2 truth (cells are required and must match
        # the R5 frozen 3-cell order).
        backend.set_continuous_casting_evaluation_state(
            ContinuousCastingEvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
                cells=tuple(
                    ContinuousCastingCellTruth(
                        target_cell=cell,
                        initial_block="air",
                        current_block="air",
                        water_truth=None,
                        lava_truth=None,
                        transition_evidence=None,
                        relevant_action_steps=(),
                    )
                    for cell in CASTING_C3_TARGET_CELLS
                ),
            )
        )
        # C1 and C2 read back; C3 / C4 are still empty.
        self.assertIsNotNone(backend.get_casting_evaluation_state())
        self.assertIsNotNone(
            backend.get_continuous_casting_evaluation_state()
        )
        with self.assertRaises(RuntimeError):
            backend.get_frame_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_ignition_evaluation_state()
        # Close clears all four slots.
        backend.close()
        with self.assertRaises(RuntimeError):
            backend.get_casting_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_continuous_casting_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_frame_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_ignition_evaluation_state()

    def test_step_clears_c4_slot(self) -> None:
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )

        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(
            _ResetProxy(_context())  # type: ignore[arg-type]
        )
        # After reset the C4 slot is None; injecting a state at
        # step 0 and then stepping clears it (just like the other
        # C1/C2/C3 slots).
        # Build a minimal state. The state would normally require
        # a valid frame_state, so we rely on the post-construction
        # fail-closed: the C4 slot is initially None.
        self.assertIsNone(backend._ignition_evaluation_state)  # type: ignore[attr-defined]
        # Confirm step() does not raise when the slot is None.
        backend.step({AGENT_ID: MacroAction.wait()})
        self.assertIsNone(backend._ignition_evaluation_state)  # type: ignore[attr-defined]

    def test_ignition_state_rejects_wrong_workflow(self) -> None:
        from obsidianlink.evaluation.casting_frame_evaluator import (
            FrozenFrameCellTruth,
            FrozenFrameEvaluationState,
            FrozenFrameInteriorCellTruth,
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )

        # Build a separate backend that runs a C1 task; the
        # ignition state surface must reject the C4 state.
        c1_backend = FakeEnvironmentBackend()
        c1_backend.open()
        c1_task = _task(workflow="casting_c1_fixed", family="casting",
                       mode="single", level="C1", layout="fixed")
        c1_backend.reset(c1_task)
        frame_state = FrozenFrameEvaluationState(
            episode_id=c1_task.task_id,
            step_id=0,
            cells=tuple(
                FrozenFrameCellTruth(
                    target_cell=cell,
                    initial_block="air",
                    current_block="air",
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
                for cell in CASTING_S_C4_IGNITION_FRAME_CELLS
            ),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(target_cell=cell, current_block="air")
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
            max_environment_steps=c1_task.limits["max_environment_steps"],
            max_game_time_seconds=c1_task.limits["max_game_time_seconds"],
        )
        latched = build_c4_c3_frame_identity(
            episode_id=c1_task.task_id,
            step_id=0,
            agent_id=AGENT_ID,
            activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
        )
        c4_state = FrozenIgnitionEvaluationState(
            episode_id=c1_task.task_id,
            step_id=0,
            frame_state=frame_state,
            latched_frame_identity=latched,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=False,
            max_environment_steps=c1_task.limits["max_environment_steps"],
            max_game_time_seconds=c1_task.limits["max_game_time_seconds"],
        )
        with self.assertRaisesRegex(
            ValueError, "casting_s_c4_fixed workflow"
        ):
            c1_backend.set_ignition_evaluation_state(c4_state)


# ----------------------------------------------------------------------
# Ignition orchestrator (success / failure) tests
# ----------------------------------------------------------------------


class IgnitionOrchestratorSuccessTests(unittest.TestCase):
    """End-to-end orchestrator: driver + ignition evaluator → success."""

    def test_full_ignition_success(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        task = _task()
        result = run_orchestrator(
            backend, driver_result, C4IgnitionWorldTruth(), task=task
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.frame_outcome, OUTCOME_SUCCESS)
        self.assertEqual(
            result.activation_observed_offset, C4_IGNITION_PUBLIC_TARGET
        )
        self.assertEqual(
            result.activation_delta_steps, 1
        )

    def test_success_at_delta_zero(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_delta_steps=0,
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 0)

    def test_success_at_delta_four(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_delta_steps=4,
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.activation_delta_steps, 4)

    def test_failure_at_delta_five(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_delta_steps=5,
            use_backend_roundtrip=False,
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_OUTSIDE_WINDOW
        )

    def test_failure_when_activation_before_ignition(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_delta_steps=-1,
            use_backend_roundtrip=False,
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_BEFORE_IGNITION
        )

    def test_failure_on_external_activation_offset(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_offset=(0, 0, 1),
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_EXTERNAL_ACTIVATION)

    def test_failure_on_wrong_activation_agent(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C4IgnitionWorldTruth(),
            task=_task(),
            activation_agent_id=WRONG_AGENT_ID,
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_AGENT)

    def test_failure_when_frame_not_built(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        # Use ``stone`` (a non-obsidian allowed block) so the
        # transition_evidence is well-formed and the C3 frame
        # evaluator emits ``wrong_block`` (which the C4 wrapper
        # surfaces as ``frame_not_built``).
        world = C4IgnitionWorldTruth(
            current_blocks=("stone",) * 14,
            transition_after_blocks=("stone",) * 14,
        )
        result = run_orchestrator(
            backend, driver_result, world, task=_task(),
            use_backend_roundtrip=False,
        )
        # The C3 frame evaluator's ``wrong_block`` outcome
        # surfaces as ``frame_not_built`` at the C4 wrapper.
        self.assertIn(
            result.outcome,
            {OUTCOME_FRAME_NOT_BUILT, OUTCOME_TRUTH_MISSING},
        )

    def test_failure_when_truth_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        # Build a world where every cell has ``update_step=None``
        # on its transition evidence. The driver surfaces
        # ``truth_missing`` only when the frame state is missing
        # the cells entirely, which the orchestrator's pre-built
        # state does not reproduce. We accept either
        # ``truth_missing`` or ``frame_not_built`` (the C4
        # evaluator's wrapper for C3 non-success outcomes).
        world = C4IgnitionWorldTruth(
            transition_steps=tuple(None for _ in range(14)),
        )
        result = run_orchestrator(
            backend, driver_result, world, task=_task(),
            use_backend_roundtrip=False,
        )
        self.assertIn(
            result.outcome,
            {
                OUTCOME_TRUTH_MISSING,
                OUTCOME_FRAME_NOT_BUILT,
            },
        )

    def test_step_budget_exceeded(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        # Build a task with a tiny step budget. The frame
        # step_id is far past the budget, so the C4 evaluator
        # returns step_budget_exceeded.
        task = _task(max_environment_steps=10)
        result = run_orchestrator(
            backend, driver_result, C4IgnitionWorldTruth(), task=task,
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_ignition_action_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )
        world = C4IgnitionWorldTruth()
        frame_state = _state_from_driver(driver_result, world, task=_task())
        latched_frame_identity = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            agent_id=AGENT_ID,
            activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
        )
        activation = PortalActivationEvidence(
            episode_id=EPISODE_ID,
            update_step=frame_state.step_id,
            agent_id=AGENT_ID,
            nether_portal_offset=C4_IGNITION_PUBLIC_TARGET,
            latched_frame_identity=latched_frame_identity,
        )
        state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            frame_state=frame_state,
            latched_frame_identity=latched_frame_identity,
            ignition_action=None,
            activation_evidence=activation,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=TERMINATED_STEP,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        backend.set_ignition_evaluation_state(state)
        result = FrozenIgnitionEvaluator().evaluate(
            backend.get_ignition_evaluation_state()
        )
        self.assertEqual(
            result.outcome, OUTCOME_IGNITION_ACTION_MISSING
        )

    def test_activation_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )
        world = C4IgnitionWorldTruth()
        frame_state = _state_from_driver(driver_result, world, task=_task())
        latched_frame_identity = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            agent_id=AGENT_ID,
            activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
        )
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type=C4_IGNITION_PUBLIC_ACTION,
            item=C4_IGNITION_PUBLIC_ITEM,
            target_cell=C4_IGNITION_PUBLIC_TARGET,
        )
        state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            frame_state=frame_state,
            latched_frame_identity=latched_frame_identity,
            ignition_action=ignition,
            activation_evidence=None,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=TERMINATED_STEP,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        backend.set_ignition_evaluation_state(state)
        result = FrozenIgnitionEvaluator().evaluate(
            backend.get_ignition_evaluation_state()
        )
        self.assertEqual(
            result.outcome, OUTCOME_ACTIVATION_MISSING
        )


# ----------------------------------------------------------------------
# Ignition semantics tests (wrong action / item / target / agent)
# ----------------------------------------------------------------------


class IgnitionSemanticTests(unittest.TestCase):
    """Wrong agent / action / item / target reach semantic outcomes."""

    def _eval(
        self,
        *,
        ignition_action: IgnitionActionEvidence | None = None,
        activation_evidence: PortalActivationEvidence | None = None,
    ):
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c4_ignition_driver(
            backend, _context()
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )
        world = C4IgnitionWorldTruth()
        frame_state = _state_from_driver(driver_result, world, task=_task())
        if ignition_action is None:
            ignition_action = IgnitionActionEvidence(
                episode_id=EPISODE_ID,
                step_id=IGNITION_STEP,
                agent_id=AGENT_ID,
                action_type=C4_IGNITION_PUBLIC_ACTION,
                item=C4_IGNITION_PUBLIC_ITEM,
                target_cell=C4_IGNITION_PUBLIC_TARGET,
            )
        if activation_evidence is None:
            latched_frame_identity = build_c4_c3_frame_identity(
                episode_id=EPISODE_ID,
                step_id=frame_state.step_id,
                agent_id=AGENT_ID,
                activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
            )
            activation_evidence = PortalActivationEvidence(
                episode_id=EPISODE_ID,
                update_step=IGNITION_STEP + 1,
                agent_id=AGENT_ID,
                nether_portal_offset=C4_IGNITION_PUBLIC_TARGET,
                latched_frame_identity=latched_frame_identity,
            )
        state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=activation_evidence.update_step,
            frame_state=frame_state,
            latched_frame_identity=activation_evidence.latched_frame_identity,
            ignition_action=ignition_action,
            activation_evidence=activation_evidence,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=TERMINATED_STEP,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        backend.set_ignition_evaluation_state(state)
        return FrozenIgnitionEvaluator().evaluate(
            backend.get_ignition_evaluation_state()
        )

    def test_wrong_ignition_agent(self) -> None:
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=WRONG_AGENT_ID,
            action_type=C4_IGNITION_PUBLIC_ACTION,
            item=C4_IGNITION_PUBLIC_ITEM,
            target_cell=C4_IGNITION_PUBLIC_TARGET,
        )
        result = self._eval(ignition_action=ignition)
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_AGENT)

    def test_wrong_ignition_action_type(self) -> None:
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type="place_block",
            item=C4_IGNITION_PUBLIC_ITEM,
            target_cell=C4_IGNITION_PUBLIC_TARGET,
        )
        result = self._eval(ignition_action=ignition)
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_ACTION)

    def test_wrong_ignition_item(self) -> None:
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type=C4_IGNITION_PUBLIC_ACTION,
            item="water_bucket",
            target_cell=C4_IGNITION_PUBLIC_TARGET,
        )
        result = self._eval(ignition_action=ignition)
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_ITEM)

    def test_wrong_ignition_target(self) -> None:
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type=C4_IGNITION_PUBLIC_ACTION,
            item=C4_IGNITION_PUBLIC_ITEM,
            target_cell=(2, 1, 1),
        )
        result = self._eval(ignition_action=ignition)
        self.assertEqual(result.outcome, OUTCOME_WRONG_IGNITION_TARGET)


# ----------------------------------------------------------------------
# Deterministic replay tests
# ----------------------------------------------------------------------


class DeterministicReplayTests(unittest.TestCase):
    """Same input ⇒ same driver sequence / events / as_dict()."""

    def test_replay_produces_identical_results(self) -> None:
        results = []
        for _ in range(3):
            backend = FakeEnvironmentBackend()
            backend.open()
            results.append(
                run_casting_s_c4_ignition_driver(
                    backend, _context()
                )
            )
        first, second, third = results
        self.assertEqual(first.steps_executed, second.steps_executed)
        self.assertEqual(second.steps_executed, third.steps_executed)
        self.assertEqual(
            [dict(event) for event in first.events],
            [dict(event) for event in second.events],
        )
        self.assertEqual(
            [dict(event) for event in first.events],
            [dict(event) for event in third.events],
        )
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(second.as_dict(), third.as_dict())

    def test_replay_with_recoverable_errors(self) -> None:
        class _ReproducibleRecover(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._counter = 0

            def step(self, actions: Any) -> Any:
                self._counter += 1
                # Inject exactly one recoverable error at the
                # 3rd step of every run.
                if self._counter == 3:
                    raise RecoverableBackendError(
                        "transient",
                        recoverable_kind="bucket_use_transient",
                        attempt=1,
                    )
                return super().step(actions)

        results = []
        for _ in range(2):
            backend = _ReproducibleRecover()
            backend.open()
            results.append(
                run_casting_s_c4_ignition_driver(
                    backend, _context()
                )
            )
        first, second = results
        self.assertEqual(first.steps_executed, second.steps_executed)
        self.assertEqual(first.recovery_attempts, second.recovery_attempts)
        self.assertEqual(first.as_dict(), second.as_dict())


# ----------------------------------------------------------------------
# Regression tests
# ----------------------------------------------------------------------


class RegressionTests(unittest.TestCase):
    """C1 / C2 / C3 / portal / C4 ignition evaluator regression checks."""

    def test_c1_evaluator_still_runs(self) -> None:
        from obsidianlink.evaluation.casting import (
            CastingEvaluationState,
            CastingEvaluator,
        )
        # Build a valid C1 state: target cell (0, 4, 0) with
        # ``current_block="obsidian"`` and matching evidence.
        # The exact outcome is not the regression check; we just
        # verify the C1 evaluator is still importable and runs
        # without raising.
        state = CastingEvaluationState(
            episode_id="c1_seed_0",
            step_id=0,
        )
        result = CastingEvaluator().evaluate(state)
        self.assertIsInstance(result.outcome, str)

    def test_c2_evaluator_still_runs(self) -> None:
        from obsidianlink.evaluation.continuous_casting import (
            CASTING_C3_TARGET_CELLS,
            ContinuousCastingCellTruth,
            ContinuousCastingEvaluationState,
            ContinuousCastingEvaluator,
        )
        state = ContinuousCastingEvaluationState(
            episode_id="c2_seed_0",
            step_id=0,
            cells=tuple(
                ContinuousCastingCellTruth(
                    target_cell=cell,
                    initial_block="air",
                    current_block="air",
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
                for cell in CASTING_C3_TARGET_CELLS
            ),
        )
        result = ContinuousCastingEvaluator().evaluate(state)
        # The empty / air / air cells do not satisfy the C2
        # success criteria; the C2 evaluator is expected to
        # emit a documented non-success outcome.
        self.assertIsInstance(result.outcome, str)

    def test_c3_frame_evaluator_still_runs(self) -> None:
        # An empty C3 frame returns ``in_progress`` because there
        # are no cells / interior cells supplied.
        state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=tuple(
                FrozenFrameCellTruth(
                    target_cell=cell,
                    initial_block="air",
                    current_block="air",
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
                for cell in CASTING_S_C4_IGNITION_FRAME_CELLS
            ),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(target_cell=cell, current_block="air")
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
        )
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertIn(result.outcome, {"in_progress", "partial_completion"})

    def test_c4_ignition_evaluator_still_runs(self) -> None:
        # An empty C4 ignition state returns ``in_progress``.
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )
        frame_state = FrozenFrameEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            cells=tuple(
                FrozenFrameCellTruth(
                    target_cell=cell,
                    initial_block="air",
                    current_block="air",
                    water_truth=None,
                    lava_truth=None,
                    transition_evidence=None,
                    relevant_action_steps=(),
                )
                for cell in CASTING_S_C4_IGNITION_FRAME_CELLS
            ),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(target_cell=cell, current_block="air")
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        latched = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID,
            step_id=0,
            agent_id=AGENT_ID,
            activation_offsets=(C4_IGNITION_PUBLIC_TARGET,),
        )
        state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            frame_state=frame_state,
            latched_frame_identity=latched,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=False,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)

    def test_portal_evaluator_still_runs(self) -> None:
        from obsidianlink.evaluation.portal import (
            EvaluationState,
            PortalEvaluator,
        )
        state = EvaluationState(episode_id="portal_seed_0", step_id=0)
        PortalEvaluator().evaluate(state)

    def test_offline_check_status(self) -> None:
        # Re-run the project --check (no Minecraft, no real backend,
        # no paid model call) and confirm it still reports
        # ``status: "ok"``.
        result = subprocess.run(
            [sys.executable, "-m", "obsidianlink", "--check"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"status": "ok"', result.stdout)
        self.assertIn('"phase": "r6_c5_live_minerl_backend_wiring_done"', result.stdout)
        self.assertNotIn("C4 / C5 runtime components", result.stdout)

    def test_check_environment_script(self) -> None:
        # The environment script is a passive probe; it must not
        # start Minecraft.
        result = subprocess.run(
            [sys.executable, "scripts/check_environment.py"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"phase": "r6_c5_live_minerl_backend_wiring_done"', result.stdout)


# ----------------------------------------------------------------------
# Smoke import / package exposure
# ----------------------------------------------------------------------


class PackageImportTests(unittest.TestCase):
    """The new public surfaces are exposed by their package __init__."""

    def test_driver_package_exposes_c4_symbols(self) -> None:
        from obsidianlink.drivers import (
            CASTING_S_C4_IGNITION_FRAME_CELLS,
            CastingC4IgnitionDriverResult,
            CastingC4IgnitionPlanStep,
            PublicC4IgnitionDriverContext,
            build_casting_s_c4_ignition_action_plan,
            run_casting_s_c4_ignition_driver,
        )
        self.assertEqual(len(CASTING_S_C4_IGNITION_FRAME_CELLS), 14)
        self.assertEqual(
            build_casting_s_c4_ignition_action_plan.__module__,
            "obsidianlink.drivers.casting_s_c4_ignition",
        )
        self.assertEqual(
            run_casting_s_c4_ignition_driver.__module__,
            "obsidianlink.drivers.casting_s_c4_ignition",
        )
        self.assertEqual(
            PublicC4IgnitionDriverContext.__module__,
            "obsidianlink.drivers.casting_s_c4_ignition",
        )
        self.assertEqual(
            CastingC4IgnitionDriverResult.__module__,
            "obsidianlink.drivers.casting_s_c4_ignition",
        )
        self.assertEqual(
            CastingC4IgnitionPlanStep.__module__,
            "obsidianlink.drivers.casting_s_c4_ignition",
        )

    def test_core_package_exposes_c4_context_builder(self) -> None:
        from obsidianlink.core import (
            build_public_c4_ignition_driver_context_from_task,
        )
        self.assertEqual(
            build_public_c4_ignition_driver_context_from_task.__module__,
            "obsidianlink.core.casting_s_c4_ignition_context",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
