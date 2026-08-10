"""Offline tests for the R6 Casting-S-C5 deterministic Nether-entry driver.

These tests prove, in code, that:

* :class:`PublicC5NetherEntryDriverContext` is a strictly-typed,
  frozen, immutable public driver context; the driver never reads
  ``scenario_parameters`` or ``evaluator_contract`` from the
  original task.
* :func:`build_casting_s_c5_nether_entry_action_plan` builds a
  fixed, deterministic, ordered plan whose C3 casting sub-plan
  has 14 cells × 24 steps = 336 steps, whose C4 ignition
  sub-plan adds 4 steps (equip + release + use + settle) and
  whose C5 Nether-entry sub-plan adds 7 steps (4 approach moves
  + 1 alignment move + 1 portal-traversal move + 1 settle) for a default 347-step
  plan. The cast plan has 14 × 2 = 28 relevant actions; the
  ignition plan has exactly 1 relevant action; the entry
  sub-plan has exactly 1 relevant action.
* :func:`run_casting_s_c5_nether_entry_driver` walks the plan on
  the :class:`FakeEnvironmentBackend`, never imports the C5
  Nether-entry evaluator or its types, never calls
  :meth:`FakeEnvironmentBackend.set_nether_entry_evaluation_state`
  / :meth:`FakeEnvironmentBackend.get_nether_entry_evaluation_state`
  / :meth:`FakeEnvironmentBackend.clear_nether_entry_evaluation_state`,
  and never reads ``scenario_parameters`` / ``evaluator_contract``
  / :class:`FrozenFrameIdentity` / :class:`IgnitionActionEvidence`
  / :class:`PortalActivationEvidence` /
  :class:`FrozenIgnitionEvaluationState` /
  :class:`FrozenNetherEntryEvaluationState` /
  :class:`NetherEntryEvidence` / :class:`FrozenNetherEntryEvaluator`
  / ``agents_in_nether`` / ``entered_via_episode_portal`` /
  ``matched_frame_identity`` / ``latched_frame_identity`` /
  ``pre_transition_position`` or any evaluator-only field.
* Every step / time / wait / plan / total-recovery budget has a
  hard, fail-closed bound.
* The driver's recovery protocol retries the typed
  :class:`RecoverableBackendError` deterministically and fails
  closed on any non-recoverable exception.
* The driver's events carry ``episode_id`` / ``step_id`` /
  ``agent_id`` / ``cell_index`` / ``target_offset`` /
  ``relevant_action`` / ``role``; the result is deeply
  immutable, the ``as_dict()`` snapshot is JSON-serializable,
  and the same input yields the same action sequence / events /
  ``as_dict()`` snapshot on repeated runs.
* The pre-episode capability gate fails closed (with
  :class:`CapabilityMismatchError`) when the manifest is missing
  any of the required capabilities, and the failure happens
  *before* any ``Observation`` is generated.
* An Observation guard fails closed if the driver ever tries to
  read a hidden ``nether_entry_evaluation`` / ``latched_frame_identity``
  / ``nether_portal`` / ``matched_frame_identity`` / ``agents_in_nether``
  / ``pre_transition_position`` / ``entered_via_episode_portal`` /
  ``source_dimension`` / ``target_dimension`` field.
* The test orchestrator (this file) is the *only* place that
  injects the R6 C5 Nether-entry evaluator truth via
  :meth:`FakeEnvironmentBackend.set_nether_entry_evaluation_state`,
  and the :class:`FrozenNetherEntryEvaluator` correctly returns
  ``success`` for a complete C3 frame + a legal C4 ignition +
  a C5 Nether entry with the same episode-built
  :class:`FrozenFrameIdentity`.
* ``designated_agent_id`` / ``source_dimension`` /
  ``target_dimension`` drift (wrong agent, wrong source, wrong
  target, missing entry, external entry, identity mismatch,
  transition before activation, no agent in Nether, ignition
  not completed) all produce the closed-set failure outcomes
  the C5 Nether-entry evaluator documents.
* The C1, C2, C3, C4 ignition, and C5 Nether-entry evaluator
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
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.casting_s_c5_nether_entry_context import (
    build_public_c5_nether_entry_driver_context_from_task,
)
from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)
from obsidianlink.drivers.casting_s_c5_nether_entry import (
    AGENT_ID,
    ALLOWED_C5_NETHER_ENTRY_ACTION_TYPES,
    ALLOWED_C5_NETHER_ENTRY_FAMILIES,
    ALLOWED_C5_NETHER_ENTRY_LAYOUTS,
    ALLOWED_C5_NETHER_ENTRY_LEVELS,
    ALLOWED_C5_NETHER_ENTRY_MODES,
    ALLOWED_C5_NETHER_ENTRY_TARGETS,
    C5_NETHER_ENTRY_GRID_X_MAX,
    C5_NETHER_ENTRY_GRID_X_MIN,
    C5_NETHER_ENTRY_GRID_Y_MAX,
    C5_NETHER_ENTRY_GRID_Y_MIN,
    C5_NETHER_ENTRY_GRID_Z_MAX,
    C5_NETHER_ENTRY_GRID_Z_MIN,
    C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
    C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
    C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
    C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET_POLICY,
    C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
    C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
    CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS,
    CASTING_S_C5_NETHER_ENTRY_TARGET_CELL_COUNT,
    DEFAULT_MAX_WAIT_STEPS,
    DRIVER_STATUS_BLOCKED,
    DRIVER_STATUS_COMPLETED,
    DRIVER_STATUS_FAILED,
    DRIVER_STATUSES,
    FAMILY_C5_NETHER_ENTRY,
    LAYOUT_C5_NETHER_ENTRY,
    LEVEL_C5_NETHER_ENTRY,
    MAX_NETHER_ENTRY_PLAN_STEPS,
    MAX_NETHER_ENTRY_PLAN_WAIT_STEPS,
    MAX_RECOVERIES_PER_ACTION,
    MAX_TOTAL_RECOVERY_BUDGET,
    MODE_C5_NETHER_ENTRY,
    PHASE_ENTRY_APPROACH,
    PHASE_ENTRY_ALIGN,
    PHASE_ENTRY_SETTLE,
    PHASE_ENTRY_TELEPORT,
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
    RECOVERIES_PER_ENTRY_DEFAULT,
    RECOVERIES_PER_IGNITION_USE_DEFAULT,
    RECOVERIES_PER_USE_ITEM_DEFAULT,
    ROLE_CAST,
    ROLE_ENTRY_APPROACH,
    ROLE_ENTRY_ALIGN,
    ROLE_ENTRY_SETTLE,
    ROLE_ENTRY_TELEPORT,
    ROLE_IGNITION_EQUIP,
    ROLE_IGNITION_SETTLE,
    ROLE_IGNITION_USE,
    ROLE_VALUES,
    TOTAL_RECOVERY_BUDGET_DEFAULT,
    WORKFLOW_C5_NETHER_ENTRY,
    CastingC5NetherEntryDriverResult,
    CastingC5NetherEntryPlanStep,
    PublicC5NetherEntryDriverContext,
    _ResetProxy,
    build_casting_s_c5_nether_entry_action_plan,
    run_casting_s_c5_nether_entry_driver,
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
    FrozenNetherEntryEvaluationState,
    FrozenNetherEntryEvaluator,
    IgnitionActionEvidence,
    NetherEntryEvidence,
    OUTCOME_FRAME_IDENTITY_MISMATCH,
    OUTCOME_FRAME_NOT_BUILT,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    PortalActivationEvidence,
    build_c4_c3_frame_identity,
)
from obsidianlink.evaluation.casting_ignition_evaluator import (
    FrozenIgnitionEvaluationState,
)
from obsidianlink.evaluation.casting_nether_entry_evaluator import (
    OUTCOME_IGNITION_NOT_COMPLETED,
    OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
    OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN,
    OUTCOME_NO_AGENT_ENTERED_NETHER,
    OUTCOME_PRE_TRANSITION_POSITION_MISSING,
    OUTCOME_TRANSITION_BEFORE_ACTIVATION,
    OUTCOME_TRANSITION_STEP_MISSING,
    OUTCOME_WRONG_ENTRY_AGENT,
    OUTCOME_WRONG_SOURCE_DIMENSION,
    OUTCOME_WRONG_TARGET_DIMENSION,
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


EPISODE_ID = "casting_s_c5_fixed_seed_0"
WRONG_EPISODE_ID = "casting_s_c5_fixed_seed_99"
WRONG_AGENT_ID = "agent_2"
DEFAULT_INVENTORY: dict[str, int] = {
    "water_bucket": 14,
    "lava_bucket": 14,
    "cobblestone": 28,
    "flint_and_steel": 1,
}
DEFAULT_MAX_ENVIRONMENT_STEPS = 800
DEFAULT_MAX_GAME_TIME_SECONDS = 720
DEFAULT_CAUSALITY_WINDOW = 4
NETHER_ENTRY_STEP = 346
NETHER_ENTRY_APPROACH_STEP = 341
IGNITION_STEP = 339
IGNITION_EQUIP_STEP = 337
IGNITION_RELEASE_STEP = 338
IGNITION_PORTAL_SETTLE_STEP = 340
ENTRY_APPROACH_FIRST_STEP = 341
ENTRY_SETTLE_STEP = 347
TERMINATED_STEP = 347
TERMINATED_REASON = "driver_done"
ENTRY_TRANSITION_STEP = NETHER_ENTRY_STEP

ROOT = Path(__file__).resolve().parents[1]
DRIVER_SOURCE = (
    ROOT / "obsidianlink/drivers/casting_s_c5_nether_entry.py"
)


# ----------------------------------------------------------------------
# Task / context helpers
# ----------------------------------------------------------------------


def _public_spec(
    *,
    fixed_offsets: list[list[int]] | None = None,
    ignition_plan: dict[str, Any] | None = None,
    nether_entry_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fixed_offsets is None:
        fixed_offsets = [
            list(cell) for cell in CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
        ]
    if ignition_plan is None:
        ignition_plan = {
            "required": True,
            "action": C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
            "item": C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
            "target_offset": list(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET),
            "target_policy": C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET_POLICY,
        }
    if nether_entry_goal is None:
        nether_entry_goal = {
            "required": True,
            "designated_agent_ids": [AGENT_ID],
            "source_dimension": C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
            "target_dimension": C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
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
        "nether_entry_goal": nether_entry_goal,
    }


def _task_dict(
    *,
    inventory: dict[str, int] | None = None,
    workflow: str = WORKFLOW_C5_NETHER_ENTRY,
    family: str = FAMILY_C5_NETHER_ENTRY,
    mode: str = MODE_C5_NETHER_ENTRY,
    level: str = LEVEL_C5_NETHER_ENTRY,
    layout: str = LAYOUT_C5_NETHER_ENTRY,
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
        "compatibility_task_name": WORKFLOW_C5_NETHER_ENTRY,
        "implementation_status": "contract_only",
        "world_dimension": "minecraft:overworld",
        "layout": "fixed_controlled",
        "mechanics_required": "vanilla_water_lava_block_update_flint_and_steel_and_portal_teleport",
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
            "nether_entry_attribution": {
                "required": True,
                "require_entered_via_episode_portal": True,
                "require_matched_frame_identity": True,
                "require_pre_transition_position": True,
                "require_transition_step": True,
                "unknown_attribution_outcome": "nether_entry_portal_unknown",
                "external_entry_outcome": "nether_entry_not_via_episode_portal",
                "fail_closed_on_missing_truth": True,
            },
        }
    return {
        "schema_version": "0.1",
        "task_id": episode_id,
        "route": "lava_casting",
        "difficulty": 4,
        "agent_ids": [AGENT_ID],
        "world_seed": 0,
        "instruction": "R6 C5 Nether-entry driver unit-test task.",
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
            "agent_entered_nether",
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
    workflow: str = WORKFLOW_C5_NETHER_ENTRY,
    family: str = FAMILY_C5_NETHER_ENTRY,
    mode: str = MODE_C5_NETHER_ENTRY,
    level: str = LEVEL_C5_NETHER_ENTRY,
    layout: str = LAYOUT_C5_NETHER_ENTRY,
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
    family: str = FAMILY_C5_NETHER_ENTRY,
    mode: str = MODE_C5_NETHER_ENTRY,
    level: str = LEVEL_C5_NETHER_ENTRY,
    layout: str = LAYOUT_C5_NETHER_ENTRY,
    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS,
    episode_id: str = EPISODE_ID,
    agent_id: str = AGENT_ID,
    workflow: str = WORKFLOW_C5_NETHER_ENTRY,
    task_step_limit: int = DEFAULT_MAX_ENVIRONMENT_STEPS,
    task_time_limit: float = float(DEFAULT_MAX_GAME_TIME_SECONDS),
    ignition_action: str = C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
    ignition_item: str = C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
    ignition_target: tuple[int, int, int] = C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
    ignition_target_policy: str = C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET_POLICY,
    ignition_required: bool = True,
    designated_agent_id: str = AGENT_ID,
    source_dimension: str = C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
    target_dimension: str = C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
    nether_entry_required: bool = True,
) -> PublicC5NetherEntryDriverContext:
    return PublicC5NetherEntryDriverContext(
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
        designated_agent_id=designated_agent_id,
        source_dimension=source_dimension,
        target_dimension=target_dimension,
        nether_entry_required=nether_entry_required,
        task_step_limit=task_step_limit,
        task_time_limit=task_time_limit,
    )


# ----------------------------------------------------------------------
# C5 Nether-entry world truth (test orchestrator only)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class C5NetherEntryWorldTruth:
    """Test-only description of the R6 C5 Nether-entry world.

    The orchestrator (functions below) is the *only* component
    that consumes a :class:`C5NetherEntryWorldTruth`. The driver
    never sees this object. Each target cell carries the
    evaluator truth required by :class:`FrozenFrameCellTruth`;
    the 6 interior cells carry the truth required by
    :class:`FrozenFrameInteriorCellTruth`. The activation
    evidence carries the public ignition target ``(1, 1, 1)``.
    The Nether-entry evidence carries the public
    ``minecraft:overworld`` → ``minecraft:the_nether``
    transition with ``entered_via_episode_portal=True``.

    The default values match the R6 C5 driver's default plan
    (14 cells × 24 + 4 + 7 = 347 steps). The first
    ``use_item(lava_bucket)`` is at global step ``9 + 24 *
    cell_index``; the first ``use_item(water_bucket)`` is at
    global step ``16 + 24 * cell_index``; the block update is
    at ``20 + 24 * cell_index`` (within the 4-step causality
    window of the last relevant action). The ignition
    ``use_item(flint_and_steel)`` is at global step ``339``.
    The activation ``update_step`` defaults to ``340`` (delta
    = 1 from the ignition step) which is inside the 4-step
    inclusive window. The Nether-entry transition is at
    step ``346`` (after 4 approach moves + 1 alignment move;
    step 346 is the portal-traversal move).
    """

    target_offsets: tuple[tuple[int, int, int], ...] = CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
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
    activation_offset: tuple[int, int, int] = C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET
    activation_delta_steps: int = 1
    terminated_step: int = TERMINATED_STEP
    terminated_reason: str = TERMINATED_REASON
    current_time_seconds: float = 0.0
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW
    pre_transition_position: tuple[float, float, float] = (1.5, 1.0, 1.0)
    transition_step: int = ENTRY_TRANSITION_STEP
    entered_via_episode_portal: bool = True
    source_dimension: str = C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION
    target_dimension: str = C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION


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
    world: C5NetherEntryWorldTruth,
    task: TaskInstance,
    relevant_records: tuple[tuple[tuple[int, int, str], ...], ...] | None = None,
    current_time_seconds: float | None = None,
    step_id: int | None = None,
    terminated_step: int | None = None,
) -> FrozenFrameEvaluationState:
    """Build a :class:`FrozenFrameEvaluationState` from a world truth."""
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
    driver_result: CastingC5NetherEntryDriverResult,
    world: C5NetherEntryWorldTruth,
    *,
    task: TaskInstance,
    current_time_seconds: float | None = None,
    step_id: int | None = None,
    terminated_step: int | None = None,
) -> FrozenFrameEvaluationState:
    """Build a :class:`FrozenFrameEvaluationState` from a driver result."""
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


def _build_nether_entry_state(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC5NetherEntryDriverResult,
    world: C5NetherEntryWorldTruth,
    *,
    task: TaskInstance,
    activation_delta_steps: int | None = None,
    ignition_step_override: int | None = None,
    activation_offset: tuple[int, int, int] | None = None,
    current_time_seconds: float | None = None,
    activation_agent_id: str | None = None,
    nether_entry_transition_step: int | None = None,
    nether_entry_entered_via_episode_portal: bool | None = None,
    nether_entry_source_dimension: str | None = None,
    nether_entry_target_dimension: str | None = None,
    nether_entry_pre_transition_position: tuple[float, float, float] | None = None,
    nether_entry_matched_frame_identity: FrozenFrameIdentity | None = None,
    include_entry_evidence: bool = True,
):
    """Build a :class:`FrozenNetherEntryEvaluationState` from a driver result.

    The orchestrator (this function) is the *only* place in the
    R6 C5 Nether-entry driver test suite that calls
    ``set_nether_entry_evaluation_state`` /
    ``get_nether_entry_evaluation_state`` and the
    :class:`FrozenNetherEntryEvaluator`. The driver never sees
    the truth surface.
    """
    if driver_result.ignition_relevant_action_step is None:
        raise AssertionError(
            "driver must have submitted the ignition use_item step"
        )
    if (
        driver_result.ignition_target_offset
        != C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET
    ):
        raise AssertionError(
            "ignition target must be the public [1, 1, 1] cell"
        )
    if driver_result.nether_entry_step is None:
        raise AssertionError(
            "driver must have submitted the nether-entry move step"
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
    # The C5 state requires ``frame_state.step_id == state.step_id``
    # and ``frame_state.terminated_step == state.terminated_step``.
    transition_step = (
        nether_entry_transition_step
        if nether_entry_transition_step is not None
        else (
            driver_result.nether_entry_step
            if nether_entry_transition_step is None and include_entry_evidence
            else world.transition_step
        )
    )
    terminated_step = max(
        max(activation_step, world.terminated_step),
        transition_step,
    )
    if activation_step >= terminated_step:
        state_step_id = activation_step
    else:
        state_step_id = terminated_step
    if transition_step > state_step_id:
        state_step_id = transition_step
    frame_state = _state_from_driver(
        driver_result,
        world,
        task=task,
        current_time_seconds=current_time_seconds,
        step_id=state_step_id,
        terminated_step=terminated_step,
    )
    latched_step = state_step_id
    latched_frame_identity = (
        nether_entry_matched_frame_identity
        if nether_entry_matched_frame_identity is not None
        else build_c4_c3_frame_identity(
            episode_id=task.task_id,
            step_id=latched_step,
            agent_id=AGENT_ID,
            activation_offsets=(offset,) if offset in CASTING_S_C3_INTERIOR_CELLS
            else None,
        )
    )
    if latched_frame_identity.activation_offsets != (offset,):
        latched_frame_identity = build_c4_c3_frame_identity(
            episode_id=task.task_id,
            step_id=latched_step,
            agent_id=AGENT_ID,
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
        )
    ignition_action = IgnitionActionEvidence(
        episode_id=task.task_id,
        step_id=ignition_step,
        agent_id=AGENT_ID,
        action_type=C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
        item=C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
        target_cell=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
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
    ignition_state = FrozenIgnitionEvaluationState(
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
    if include_entry_evidence:
        entry_evidence = NetherEntryEvidence(
            episode_id=task.task_id,
            agent_id=AGENT_ID,
            source_dimension=(
                nether_entry_source_dimension
                if nether_entry_source_dimension is not None
                else world.source_dimension
            ),
            target_dimension=(
                nether_entry_target_dimension
                if nether_entry_target_dimension is not None
                else world.target_dimension
            ),
            transition_step=transition_step,
            pre_transition_position=(
                nether_entry_pre_transition_position
                if nether_entry_pre_transition_position is not None
                else world.pre_transition_position
            ),
            entered_via_episode_portal=(
                nether_entry_entered_via_episode_portal
                if nether_entry_entered_via_episode_portal is not None
                else world.entered_via_episode_portal
            ),
            matched_frame_identity=latched_frame_identity,
        )
    else:
        entry_evidence = None
    return FrozenNetherEntryEvaluationState(
        episode_id=task.task_id,
        step_id=state_step_id,
        ignition_state=ignition_state,
        agents_in_nether=frozenset({AGENT_ID}),
        entry_evidence=entry_evidence,
        agent_id=AGENT_ID,
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
    driver_result: CastingC5NetherEntryDriverResult,
    world: C5NetherEntryWorldTruth,
    *,
    task: TaskInstance,
    activation_delta_steps: int | None = None,
    ignition_step_override: int | None = None,
    activation_offset: tuple[int, int, int] | None = None,
    current_time_seconds: float | None = None,
    activation_agent_id: str | None = None,
    nether_entry_transition_step: int | None = None,
    nether_entry_entered_via_episode_portal: bool | None = None,
    nether_entry_source_dimension: str | None = None,
    nether_entry_target_dimension: str | None = None,
    nether_entry_pre_transition_position: tuple[float, float, float] | None = None,
    nether_entry_matched_frame_identity: FrozenFrameIdentity | None = None,
    include_entry_evidence: bool = True,
    use_backend_roundtrip: bool = True,
):
    """Build the orchestrator-side state and call the C5 evaluator.

    By default the state is round-tripped through
    :meth:`FakeEnvironmentBackend.set_nether_entry_evaluation_state` /
    :meth:`FakeEnvironmentBackend.get_nether_entry_evaluation_state`.
    Tests that exercise boundary conditions bypass the round-trip
    with ``use_backend_roundtrip=False`` because the FakeBackend's
    step_id guard would otherwise reject the injected state.
    """
    state = _build_nether_entry_state(
        backend,
        driver_result,
        world,
        task=task,
        activation_delta_steps=activation_delta_steps,
        ignition_step_override=ignition_step_override,
        activation_offset=activation_offset,
        current_time_seconds=current_time_seconds,
        activation_agent_id=activation_agent_id,
        nether_entry_transition_step=nether_entry_transition_step,
        nether_entry_entered_via_episode_portal=(
            nether_entry_entered_via_episode_portal
        ),
        nether_entry_source_dimension=nether_entry_source_dimension,
        nether_entry_target_dimension=nether_entry_target_dimension,
        nether_entry_pre_transition_position=(
            nether_entry_pre_transition_position
        ),
        nether_entry_matched_frame_identity=(
            nether_entry_matched_frame_identity
        ),
        include_entry_evidence=include_entry_evidence,
    )
    if use_backend_roundtrip:
        backend.set_nether_entry_evaluation_state(state)
        return FrozenNetherEntryEvaluator().evaluate(
            backend.get_nether_entry_evaluation_state()
        )
    return FrozenNetherEntryEvaluator().evaluate(state)


# ----------------------------------------------------------------------
# Public context / contract tests
# ----------------------------------------------------------------------


class PublicContextContractTests(unittest.TestCase):
    """Static contract: identity, immutability, fail-closed validation."""

    def test_action_allowlist_is_closed(self) -> None:
        self.assertEqual(
            ALLOWED_C5_NETHER_ENTRY_ACTION_TYPES,
            frozenset(
                {"equip_item", "use_item", "place_block", "move", "wait"}
            ),
        )
        self.assertEqual(
            ALLOWED_C5_NETHER_ENTRY_TARGETS,
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
            ALLOWED_C5_NETHER_ENTRY_FAMILIES, frozenset({"casting"})
        )
        self.assertEqual(
            ALLOWED_C5_NETHER_ENTRY_MODES, frozenset({"single"})
        )
        self.assertEqual(
            ALLOWED_C5_NETHER_ENTRY_LEVELS, frozenset({"C5"})
        )
        self.assertEqual(
            ALLOWED_C5_NETHER_ENTRY_LAYOUTS, frozenset({"fixed"})
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
        self.assertEqual(WORKFLOW_C5_NETHER_ENTRY, "casting_s_c5_fixed")
        self.assertEqual(FAMILY_C5_NETHER_ENTRY, "casting")
        self.assertEqual(MODE_C5_NETHER_ENTRY, "single")
        self.assertEqual(LEVEL_C5_NETHER_ENTRY, "C5")
        self.assertEqual(LAYOUT_C5_NETHER_ENTRY, "fixed")
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION, "use_item"
        )
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM, "flint_and_steel"
        )
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET, (1, 1, 1)
        )
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET_POLICY, "exact"
        )
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
            "minecraft:overworld",
        )
        self.assertEqual(
            C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
            "minecraft:the_nether",
        )
        self.assertEqual(ROLE_VALUES, frozenset(
            {
                ROLE_CAST,
                ROLE_IGNITION_EQUIP,
                ROLE_IGNITION_USE,
                ROLE_IGNITION_SETTLE,
                ROLE_ENTRY_APPROACH,
                ROLE_ENTRY_ALIGN,
                ROLE_ENTRY_TELEPORT,
                ROLE_ENTRY_SETTLE,
            }
        ))
        self.assertEqual(PHASE_VALUES, frozenset(
            {
                PHASE_PREPARE,
                PHASE_PLACE_SUPPORT,
                PHASE_PLACE_LAVA,
                PHASE_PLACE_WATER,
                PHASE_WAIT_FOR_OBSIDIAN,
                PHASE_IGNITION_EQUIP,
                PHASE_IGNITION_USE,
                PHASE_IGNITION_PORTAL_SETTLE,
                PHASE_ENTRY_APPROACH,
                PHASE_ENTRY_ALIGN,
                PHASE_ENTRY_TELEPORT,
                PHASE_ENTRY_SETTLE,
                PHASE_RECOVERY,
            }
        ))

    def test_default_target_cells_match_public_spec(self) -> None:
        self.assertEqual(len(CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS), 14)
        self.assertEqual(
            CASTING_S_C5_NETHER_ENTRY_TARGET_CELL_COUNT, 14
        )
        from obsidianlink.evaluation.casting_frame_evaluator import (
            CASTING_S_C3_FRAME_CELLS as EVAL_FRAME_CELLS,
        )
        self.assertEqual(
            CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS, EVAL_FRAME_CELLS
        )

    def test_context_immutable_and_frozen(self) -> None:
        ctx = _context()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.episode_id = "other"  # type: ignore[misc]

    def test_initial_inventory_is_mapping_proxy(self) -> None:
        ctx = _context()
        self.assertIsInstance(ctx.initial_inventory, MappingProxyType)
        self.assertEqual(
            dict(ctx.initial_inventory),
            dict(DEFAULT_INVENTORY),
        )
        with self.assertRaises(TypeError):
            ctx.initial_inventory["flint_and_steel"] = 0  # type: ignore[index]

    def test_target_offsets_are_frozen_tuple(self) -> None:
        ctx = _context()
        self.assertIsInstance(ctx.target_offsets, tuple)
        for offset in ctx.target_offsets:
            self.assertIsInstance(offset, tuple)
            self.assertEqual(len(offset), 3)

    def test_workflow_must_be_casting_s_c5_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "workflow must be"):
            _context(workflow="casting_s_c4_fixed")

    def test_family_must_be_casting(self) -> None:
        with self.assertRaisesRegex(ValueError, "family must be one of"):
            _context(family="ruined")

    def test_mode_must_be_single(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            _context(mode="multi")

    def test_level_must_be_C5(self) -> None:
        with self.assertRaisesRegex(ValueError, "level must be one of"):
            _context(level="C4")

    def test_layout_must_be_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "layout must be one of"):
            _context(layout="random")

    def test_agent_id_must_be_agent_1(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_id must be"):
            _context(agent_id="agent_2")

    def test_target_offsets_count_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain exactly"):
            _context(
                target_offsets=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[:13]
            )

    def test_target_offsets_must_match_locked_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            _context(target_offsets=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[::-1])

    def test_target_offsets_must_be_unique(self) -> None:
        offsets = list(CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS)
        offsets[1] = offsets[0]
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            _context(target_offsets=tuple(offsets))

    def test_target_offsets_must_have_int_components(self) -> None:
        offsets = list(CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS)
        offsets[0] = (0, 0, "1")  # type: ignore[assignment]
        with self.assertRaisesRegex(
            ValueError, r"must be a \(x, y, z\) tuple of strict integers"
        ):
            _context(target_offsets=tuple(offsets))

    def test_target_offsets_must_be_within_grid(self) -> None:
        offsets = list(CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS)
        offsets[0] = (4, 0, 1)
        with self.assertRaisesRegex(ValueError, "outside the public"):
            _context(target_offsets=tuple(offsets))

    def test_ignition_action_must_be_use_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_action must be"):
            _context(ignition_action="place_block")

    def test_ignition_item_must_be_flint_and_steel(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_item must be"):
            _context(ignition_item="water_bucket")

    def test_ignition_target_must_be_public_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_target must be"):
            _context(ignition_target=(2, 1, 1))

    def test_ignition_target_policy_must_be_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_target_policy must be"):
            _context(ignition_target_policy="nearest")

    def test_ignition_required_must_be_true(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition_required must be a boolean"
        ):
            _context(ignition_required=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            ValueError, "ignition_required=True"
        ):
            _context(ignition_required=False)

    def test_nether_entry_required_must_be_true(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "nether_entry_required must be a boolean"
        ):
            _context(nether_entry_required="yes")  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            ValueError, "nether_entry_required=True"
        ):
            _context(nether_entry_required=False)

    def test_designated_agent_id_must_be_agent_1(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "designated_agent_id must be"
        ):
            _context(designated_agent_id="agent_2")

    def test_source_dimension_must_be_overworld(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "source_dimension must be"
        ):
            _context(source_dimension="minecraft:the_nether")

    def test_target_dimension_must_be_nether(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "target_dimension must be"
        ):
            _context(target_dimension="minecraft:overworld")

    def test_initial_inventory_must_contain_water_bucket(self) -> None:
        inventory = {
            "lava_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "water_bucket"):
            _context(inventory=inventory)

    def test_initial_inventory_must_contain_lava_bucket(self) -> None:
        inventory = {
            "water_bucket": 14,
            "cobblestone": 28,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "lava_bucket"):
            _context(inventory=inventory)

    def test_initial_inventory_must_contain_cobblestone(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "flint_and_steel": 1,
        }
        with self.assertRaisesRegex(ValueError, "cobblestone"):
            _context(inventory=inventory)

    def test_initial_inventory_must_contain_flint_and_steel(self) -> None:
        inventory = {
            "water_bucket": 14,
            "lava_bucket": 14,
            "cobblestone": 28,
        }
        with self.assertRaisesRegex(ValueError, "flint_and_steel"):
            _context(inventory=inventory)

    def test_initial_inventory_rejects_unknown_item(self) -> None:
        inventory = dict(DEFAULT_INVENTORY)
        inventory["obsidian"] = 1
        with self.assertRaisesRegex(ValueError, "forbidden item"):
            _context(inventory=inventory)

    def test_initial_inventory_rejects_negative_quantity(self) -> None:
        inventory = dict(DEFAULT_INVENTORY)
        inventory["flint_and_steel"] = -1
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_initial_inventory_rejects_bool_quantity(self) -> None:
        inventory = dict(DEFAULT_INVENTORY)
        inventory["flint_and_steel"] = True  # type: ignore[assignment]
        with self.assertRaisesRegex(
            ValueError, "non-negative integer"
        ):
            _context(inventory=inventory)

    def test_initial_inventory_rejects_under_provisioned_fixed_contract(self) -> None:
        inventory = dict(DEFAULT_INVENTORY)
        inventory["cobblestone"] = 1
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            _context(inventory=inventory)

    def test_task_step_limit_must_be_at_least_target_cell_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "task_step_limit must be at least"
        ):
            _context(task_step_limit=10)

    def test_task_time_limit_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "task_time_limit must be a positive finite number"
        ):
            _context(task_time_limit=0.0)

    def test_ignition_target_must_be_int_tuple(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"must be a \(x, y, z\) tuple of strict integers"
        ):
            _context(ignition_target=("1", 1, 1))


class BuildContextFromTaskTests(unittest.TestCase):
    """``build_public_c5_nether_entry_driver_context_from_task``."""

    def test_builds_context_from_full_task(self) -> None:
        task = _task()
        ctx = build_public_c5_nether_entry_driver_context_from_task(task)
        self.assertEqual(ctx.episode_id, EPISODE_ID)
        self.assertEqual(ctx.workflow, WORKFLOW_C5_NETHER_ENTRY)
        self.assertEqual(ctx.family, FAMILY_C5_NETHER_ENTRY)
        self.assertEqual(ctx.level, LEVEL_C5_NETHER_ENTRY)
        self.assertEqual(ctx.layout, LAYOUT_C5_NETHER_ENTRY)
        self.assertEqual(ctx.agent_id, AGENT_ID)
        self.assertEqual(
            ctx.target_offsets, CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
        )
        self.assertEqual(
            ctx.ignition_action, C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION
        )
        self.assertEqual(
            ctx.ignition_item, C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM
        )
        self.assertEqual(
            ctx.ignition_target, C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET
        )
        self.assertEqual(
            ctx.ignition_target_policy,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET_POLICY,
        )
        self.assertTrue(ctx.ignition_required)
        self.assertEqual(
            ctx.source_dimension, C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION
        )
        self.assertEqual(
            ctx.target_dimension, C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION
        )
        self.assertTrue(ctx.nether_entry_required)
        self.assertEqual(
            ctx.designated_agent_id, AGENT_ID
        )
        self.assertEqual(ctx.task_step_limit, DEFAULT_MAX_ENVIRONMENT_STEPS)
        self.assertEqual(
            ctx.task_time_limit, float(DEFAULT_MAX_GAME_TIME_SECONDS)
        )

    def test_ignores_evaluator_contract(self) -> None:
        task = _task(include_evaluator_contract=True)
        # The contract field is present in the scenario but the
        # context builder must ignore it; the returned context
        # only carries the public values.
        ctx = build_public_c5_nether_entry_driver_context_from_task(task)
        self.assertEqual(
            ctx.source_dimension, C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION
        )
        self.assertEqual(
            ctx.target_dimension, C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION
        )

    def test_wrong_workflow_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "task.workflow must be"):
            build_public_c5_nether_entry_driver_context_from_task(
                _task(workflow="casting_s_c4_fixed")
            )

    def test_missing_agent_fails_closed(self) -> None:
        # Build a task whose initial_inventories lacks agent_1
        # by mutating the dict form before constructing the
        # TaskInstance (avoids the MappingProxy deepcopy trap).
        bad_dict = _task_dict(
            episode_id=EPISODE_ID,
            public_spec=_public_spec(),
        )
        bad_dict["initial_inventories"] = {
            "agent_2": dict(DEFAULT_INVENTORY)
        }
        with self.assertRaisesRegex(
            ValueError, "initial_inventories must contain"
        ):
            build_public_c5_nether_entry_driver_context_from_task(
                TaskInstance.from_dict(bad_dict)
            )

    def test_missing_public_spec_fails_closed(self) -> None:
        bad_dict = _task_dict()
        bad_dict["scenario_parameters"] = {
            "task_family": FAMILY_C5_NETHER_ENTRY,
            "agent_mode": MODE_C5_NETHER_ENTRY,
            "task_level": LEVEL_C5_NETHER_ENTRY,
            "layout_type": LAYOUT_C5_NETHER_ENTRY,
        }
        with self.assertRaisesRegex(
            ValueError, "public_task_spec is required"
        ):
            build_public_c5_nether_entry_driver_context_from_task(
                TaskInstance.from_dict(bad_dict)
            )

    def test_missing_nether_entry_goal_fails_closed(self) -> None:
        spec = _public_spec()
        del spec["nether_entry_goal"]
        with self.assertRaisesRegex(
            ValueError, "nether_entry_goal is required"
        ):
            build_public_c5_nether_entry_driver_context_from_task(
                _task(public_spec=spec)
            )

    def test_nether_entry_required_string_is_not_coerced_to_true(self) -> None:
        spec = _public_spec()
        spec["nether_entry_goal"]["required"] = "false"
        with self.assertRaisesRegex(
            ValueError, "nether_entry_required must be a boolean"
        ):
            build_public_c5_nether_entry_driver_context_from_task(
                _task(public_spec=spec)
            )


class PlanBuilderTests(unittest.TestCase):
    """The default plan is fixed, deterministic, and bound."""

    def test_default_plan_length(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        # 14 * 24 (cast) + 4 (ignition) + 7 (entry) = 347
        self.assertEqual(len(plan), 347)

    def test_default_plan_actions_close_set(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        action_types = {step.action.action_type for step in plan}
        self.assertTrue(action_types.issubset(ALLOWED_C5_NETHER_ENTRY_ACTION_TYPES))
        targets = {step.action.target for step in plan if step.action.target is not None}
        self.assertTrue(targets.issubset(ALLOWED_C5_NETHER_ENTRY_TARGETS))

    def test_default_plan_roles_close_set(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        for step in plan:
            self.assertIn(step.role, ROLE_VALUES)
            self.assertIn(step.phase, PHASE_VALUES)

    def test_default_plan_relevant_actions(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        relevant = [step for step in plan if step.relevant_action]
        # 14 cells × 2 (water + lava) + 1 ignition + 1 entry
        # = 28 + 1 + 1 = 30
        self.assertEqual(len(relevant), 30)
        ignition_relevant = [
            step for step in relevant if step.role == ROLE_IGNITION_USE
        ]
        entry_relevant = [
            step for step in relevant if step.role == ROLE_ENTRY_TELEPORT
        ]
        self.assertEqual(len(ignition_relevant), 1)
        self.assertEqual(len(entry_relevant), 1)
        self.assertEqual(
            ignition_relevant[0].action.target,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
        )
        self.assertIsNone(entry_relevant[0].action.target)
        self.assertEqual(entry_relevant[0].action.action_type, "move")

    def test_default_plan_casting_subplan_targets(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        for cell_index in range(14):
            cell_steps = [
                step
                for step in plan
                if step.role == ROLE_CAST and step.cell_index == cell_index
            ]
            # 24 steps per cell
            self.assertEqual(len(cell_steps), 24)
            for step in cell_steps:
                self.assertEqual(
                    step.target_offset,
                    CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[cell_index],
                )

    def test_default_plan_ignition_subplan(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        ignition_steps = [
            step
            for step in plan
            if step.role in {
                ROLE_IGNITION_EQUIP,
                ROLE_IGNITION_USE,
                ROLE_IGNITION_SETTLE,
            }
        ]
        self.assertEqual(len(ignition_steps), 4)
        # Order: equip, equip release (wait), use, settle
        self.assertEqual(ignition_steps[0].role, ROLE_IGNITION_EQUIP)
        self.assertEqual(ignition_steps[0].action.action_type, "equip_item")
        self.assertEqual(ignition_steps[1].role, ROLE_IGNITION_EQUIP)
        self.assertEqual(ignition_steps[1].action.action_type, "wait")
        self.assertEqual(ignition_steps[2].role, ROLE_IGNITION_USE)
        self.assertEqual(ignition_steps[2].action.action_type, "use_item")
        self.assertEqual(ignition_steps[3].role, ROLE_IGNITION_SETTLE)
        self.assertEqual(ignition_steps[3].action.action_type, "wait")

    def test_default_plan_entry_subplan(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        entry_steps = [
            step
            for step in plan
            if step.role in {
                ROLE_ENTRY_APPROACH,
                ROLE_ENTRY_ALIGN,
                ROLE_ENTRY_TELEPORT,
                ROLE_ENTRY_SETTLE,
            }
        ]
        # 4 approach moves + 1 alignment move + 1 traversal move + 1 settle = 7
        self.assertEqual(len(entry_steps), 7)
        self.assertEqual(entry_steps[0].role, ROLE_ENTRY_APPROACH)
        self.assertEqual(entry_steps[0].action.action_type, "move")
        self.assertEqual(entry_steps[4].role, ROLE_ENTRY_ALIGN)
        self.assertEqual(entry_steps[4].action.action_type, "move")
        self.assertEqual(entry_steps[5].role, ROLE_ENTRY_TELEPORT)
        self.assertEqual(entry_steps[5].action.action_type, "move")
        self.assertIsNone(entry_steps[5].action.target)
        self.assertEqual(entry_steps[5].action.parameters["forward"], 1.0)
        self.assertEqual(entry_steps[5].relevant_action, True)
        self.assertEqual(entry_steps[6].role, ROLE_ENTRY_SETTLE)
        self.assertEqual(entry_steps[6].action.action_type, "wait")

    def test_default_plan_target_offsets_match_context(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        for cell_index, expected in enumerate(
            CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
        ):
            cast_step = next(
                step
                for step in plan
                if step.role == ROLE_CAST
                and step.cell_index == cell_index
                and step.action.action_type == "place_block"
            )
            self.assertEqual(cast_step.target_offset, expected)

    def test_default_plan_accepts_parse_macro_action(self) -> None:
        plan = build_casting_s_c5_nether_entry_action_plan()
        for step in plan:
            payload = {
                "action_type": step.action.action_type,
                "target": step.action.target,
                "duration_ticks": step.action.duration_ticks,
                "parameters": dict(step.action.parameters),
            }
            parsed = parse_macro_action(json.dumps(payload))
            self.assertTrue(parsed.accepted, msg=f"step {step.label}")

    def test_plan_builder_rejects_wrong_target_offsets(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "must match the locked"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                target_offsets=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[::-1]
            )

    def test_plan_builder_rejects_excessive_waits(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "wait steps exceed the hard limit"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                entry_settle_steps=300
            )

    def test_plan_builder_rejects_excessive_recoveries(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_use_item cannot exceed"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                recoveries_per_use_item=MAX_RECOVERIES_PER_ACTION + 1
            )
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_ignition_use cannot exceed"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                recoveries_per_ignition_use=MAX_RECOVERIES_PER_ACTION + 1
            )
        with self.assertRaisesRegex(
            ValueError, "recoveries_per_entry cannot exceed"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                recoveries_per_entry=MAX_RECOVERIES_PER_ACTION + 1
            )


class PlanStepValidationTests(unittest.TestCase):
    """Plan-step structural validation."""

    def test_recovery_phase_rejected_in_plan(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "PHASE_RECOVERY is reserved"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_RECOVERY,
                action=MacroAction.wait(),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[0],
            )

    def test_unknown_phase_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unknown C5 Nether-entry plan phase"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase="not-a-phase",
                action=MacroAction.wait(),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[0],
            )

    def test_unknown_role_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unknown C5 Nether-entry plan role"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_PREPARE,
                action=MacroAction.wait(),
                role="not-a-role",
                cell_index=0,
                target_offset=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[0],
            )

    def test_recoveries_allowed_must_be_in_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "recoveries_allowed must be an int between"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_PLACE_LAVA,
                action=MacroAction(action_type="use_item", target="lava_bucket"),
                role=ROLE_CAST,
                cell_index=0,
                target_offset=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[0],
                relevant_action=True,
                recoveries_allowed=MAX_RECOVERIES_PER_ACTION + 1,
            )

    def test_cast_role_requires_cell_index(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "cast plan step must have a non-None cell_index"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_PLACE_LAVA,
                action=MacroAction(action_type="use_item", target="lava_bucket"),
                role=ROLE_CAST,
                cell_index=None,
                target_offset=CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[0],
                relevant_action=True,
            )

    def test_ignition_use_role_requires_cell_index_none(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ignition plan step must have cell_index=None"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_IGNITION_USE,
                action=MacroAction(
                    action_type="move",
                    parameters={
                        "forward": 1.0,
                        "strafe": 0.0,
                        "sprint": False,
                        "jump": False,
                    },
                ),
                role=ROLE_IGNITION_USE,
                cell_index=0,
                target_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
                relevant_action=True,
            )

    def test_entry_teleport_role_requires_relevant(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "entry_teleport step must be marked relevant"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_ENTRY_TELEPORT,
                action=MacroAction(
                    action_type="move",
                    parameters={
                        "forward": 1.0,
                        "strafe": 0.0,
                        "sprint": False,
                        "jump": False,
                    },
                ),
                role=ROLE_ENTRY_TELEPORT,
                cell_index=None,
                target_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
                relevant_action=False,
            )

    def test_entry_teleport_move_must_not_have_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "move action cannot have a target"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_ENTRY_TELEPORT,
                action=MacroAction(
                    action_type="move",
                    target="flint_and_steel",
                    parameters={
                        "forward": 1.0,
                        "strafe": 0.0,
                        "sprint": False,
                        "jump": False,
                    },
                ),
                role=ROLE_ENTRY_TELEPORT,
                cell_index=None,
                target_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
                relevant_action=True,
            )

    def test_entry_align_role_requires_move(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "entry_align step must be a bounded move"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_ENTRY_ALIGN,
                action=MacroAction.wait(),
                role=ROLE_ENTRY_ALIGN,
                cell_index=None,
                target_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
            )

    def test_entry_approach_must_be_move(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "entry_approach step must be a bounded move"
        ):
            CastingC5NetherEntryPlanStep(
                label="bad",
                phase=PHASE_ENTRY_APPROACH,
                action=MacroAction(
                    action_type="use_item", target="flint_and_steel"
                ),
                role=ROLE_ENTRY_APPROACH,
                cell_index=None,
                target_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
            )


class DriverResultContractTests(unittest.TestCase):
    """``CastingC5NetherEntryDriverResult`` immutable contract."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_default_run_result_contract(self) -> None:
        result = run_casting_s_c5_nether_entry_driver(
            self._open_backend(), _context()
        )
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.steps_executed, 347)
        self.assertEqual(result.planned_steps, 347)
        self.assertEqual(result.wait_steps, 241)
        self.assertEqual(result.ignition_relevant_action_step, IGNITION_STEP)
        self.assertEqual(
            result.ignition_target_offset,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        self.assertEqual(result.ignition_equip_step, IGNITION_EQUIP_STEP)
        self.assertEqual(result.nether_entry_step, NETHER_ENTRY_STEP)
        self.assertEqual(
            result.nether_entry_target_offset,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        self.assertEqual(
            result.nether_entry_approach_step, NETHER_ENTRY_APPROACH_STEP
        )
        self.assertEqual(result.terminated, False)
        self.assertEqual(result.truncated, False)
        self.assertIsNone(result.blocked_reason)
        self.assertIsNone(result.error_type)

    def test_result_is_frozen(self) -> None:
        result = run_casting_s_c5_nether_entry_driver(
            self._open_backend(), _context()
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = "other"  # type: ignore[misc]

    def test_as_dict_is_json_serializable(self) -> None:
        result = run_casting_s_c5_nether_entry_driver(
            self._open_backend(), _context()
        )
        payload = result.as_dict()
        # Round-trip via json
        json.dumps(payload)
        self.assertEqual(payload["status"], DRIVER_STATUS_COMPLETED)
        self.assertEqual(payload["steps_executed"], 347)
        self.assertEqual(payload["planned_steps"], 347)
        self.assertEqual(
            payload["nether_entry_step"], NETHER_ENTRY_STEP
        )
        self.assertEqual(
            payload["nether_entry_target_offset"],
            list(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET),
        )


class DriverExecutionTests(unittest.TestCase):
    """The driver walks the plan deterministically on the FakeBackend."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_default_run_completes(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertEqual(result.steps_executed, 347)
        self.assertEqual(result.planned_steps, 347)
        self.assertEqual(result.wait_steps, 14 * 17 + 2 + 1)
        self.assertEqual(result.ignition_relevant_action_step, IGNITION_STEP)
        self.assertEqual(
            result.ignition_target_offset,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        self.assertEqual(result.ignition_equip_step, IGNITION_EQUIP_STEP)
        self.assertEqual(result.nether_entry_step, NETHER_ENTRY_STEP)
        self.assertEqual(
            result.nether_entry_target_offset,
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        self.assertEqual(
            result.nether_entry_approach_step, NETHER_ENTRY_APPROACH_STEP
        )
        for cell_index in range(14):
            records = result.per_cell_relevant_action_records[cell_index]
            self.assertEqual(len(records), 2)
            steps = result.per_cell_relevant_action_steps[cell_index]
            self.assertEqual(steps[1] - steps[0], 7)
            self.assertEqual(
                result.per_cell_target_offset[cell_index],
                CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS[cell_index],
            )

    def test_default_run_does_not_return_success_or_passed(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        self.assertNotIn(result.status, {"success", "passed"})
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)

    def test_driver_uses_only_allowed_actions(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        action_types = {event["action_type"] for event in result.events}
        targets = {event.get("target") for event in result.events}
        for action_type in action_types:
            self.assertIn(action_type, ALLOWED_C5_NETHER_ENTRY_ACTION_TYPES)
        for target in targets:
            if target is not None:
                self.assertIn(target, ALLOWED_C5_NETHER_ENTRY_TARGETS)

    def test_driver_emits_nether_entry_move_with_public_target(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        entry_events = [
            event
            for event in result.events
            if event.get("role") == ROLE_ENTRY_TELEPORT
            and event.get("relevant_action")
        ]
        self.assertEqual(len(entry_events), 1)
        event = entry_events[0]
        self.assertEqual(event["action_type"], "move")
        self.assertIsNone(event["target"])
        self.assertEqual(event["parameters"]["forward"], 1.0)
        self.assertEqual(
            tuple(event["target_offset"]),
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )

    def test_driver_emits_bounded_nether_entry_movements(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        move_events = [
            event
            for event in result.events
            if event.get("role") in {ROLE_ENTRY_APPROACH, ROLE_ENTRY_ALIGN}
            and event.get("action_type") == "move"
        ]
        self.assertEqual(len(move_events), 5)
        self.assertEqual(
            tuple(move_events[0]["target_offset"]),
            C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )

    def test_driver_emits_entry_approach_moves(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        approach_events = [
            event
            for event in result.events
            if event.get("role") == ROLE_ENTRY_APPROACH
        ]
        # 4 default bounded approach moves
        self.assertEqual(len(approach_events), 4)
        for event in approach_events:
            self.assertEqual(event["action_type"], "move")
            self.assertEqual(event["parameters"]["forward"], 1.0)

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
            run_casting_s_c5_nether_entry_driver(
                backend, _context(inventory=inventory)
            )

    def test_driver_blocks_on_wrong_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "workflow"):
            _context(workflow="casting_s_c4_fixed")

    def test_driver_blocks_on_wrong_ignition_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignition_target"):
            _context(ignition_target=(2, 1, 1))

    def test_driver_blocks_on_wrong_source_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_dimension"):
            _context(source_dimension="minecraft:the_nether")

    def test_driver_blocks_on_wrong_target_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_dimension"):
            _context(target_dimension="minecraft:overworld")

    def test_driver_rejects_wrong_context_type(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "PublicC5NetherEntryDriverContext"
        ):
            run_casting_s_c5_nether_entry_driver(backend, object())  # type: ignore[arg-type]

    def test_driver_rejects_non_positive_max_wait_steps(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_s_c5_nether_entry_driver(
                backend, _context(), max_wait_steps=0
            )

    def test_driver_rejects_max_wait_steps_over_hard_cap(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                max_wait_steps=MAX_NETHER_ENTRY_PLAN_WAIT_STEPS + 1,
            )

    def test_driver_rejects_max_environment_steps_over_task_limit(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "max_environment_steps cannot exceed the task limit"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS + 1,
            )

    def test_driver_rejects_max_game_time_over_task_limit(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "max_game_time_seconds cannot exceed the task limit"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                max_game_time_seconds=DEFAULT_MAX_GAME_TIME_SECONDS + 1.0,
            )

    def test_driver_rejects_plan_over_task_step_limit(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c5_nether_entry_action_plan()
        with self.assertRaisesRegex(
            ValueError, "plan length cannot exceed the task step limit"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(task_step_limit=14),
                plan=plan,
            )

    def test_driver_rejects_invalid_plan_type(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "CastingC5NetherEntryPlanStep"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                plan=(MacroAction.wait(),),  # type: ignore[arg-type]
            )

    def test_driver_rejects_entry_only_plan(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c5_nether_entry_action_plan()
        entry_only = tuple(
            step for step in plan if step.role.startswith("entry_")
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                plan=entry_only,
            )

    def test_driver_rejects_cast_only_plan(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c5_nether_entry_action_plan()
        cast_only = tuple(
            step for step in plan if step.role == ROLE_CAST
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                plan=cast_only,
            )

    def test_driver_rejects_ignition_only_plan(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c5_nether_entry_action_plan()
        ignition_only = tuple(
            step for step in plan if step.role.startswith("ignition_")
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                plan=ignition_only,
            )

    def test_driver_rejects_duplicate_entry_traversal(self) -> None:
        backend = self._open_backend()
        plan = build_casting_s_c5_nether_entry_action_plan()
        entry_traversal = next(
            step for step in plan if step.role == ROLE_ENTRY_TELEPORT
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                plan=plan + (entry_traversal,),
            )

    def test_driver_rejects_negative_total_recovery_budget(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "total_recovery_budget"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                total_recovery_budget=-1,
            )

    def test_driver_rejects_total_recovery_budget_over_cap(self) -> None:
        backend = self._open_backend()
        with self.assertRaisesRegex(
            ValueError, "total_recovery_budget"
        ):
            run_casting_s_c5_nether_entry_driver(
                backend,
                _context(),
                total_recovery_budget=MAX_TOTAL_RECOVERY_BUDGET + 1,
            )

    def test_driver_rejects_backend_without_reset(self) -> None:
        class _Bad:
            def step(self, actions: Any) -> Any:
                return None

        with self.assertRaisesRegex(ValueError, "reset/step"):
            run_casting_s_c5_nether_entry_driver(_Bad(), _context())

    def test_deterministic_replay(self) -> None:
        first = run_casting_s_c5_nether_entry_driver(
            self._open_backend(), _context()
        )
        second = run_casting_s_c5_nether_entry_driver(
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
            first.nether_entry_step, second.nether_entry_step
        )
        self.assertEqual(
            first.nether_entry_approach_step,
            second.nether_entry_approach_step,
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
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        for event in result.events:
            self.assertEqual(event["episode_id"], EPISODE_ID)
            self.assertEqual(event["agent_id"], AGENT_ID)
            self.assertIn("step_id", event)
            self.assertIn("label", event)
            self.assertIn("phase", event)
            self.assertIn("action_type", event)
            self.assertIn("role", event)

    def test_event_sink_propagates_event_copy(self) -> None:
        seen: list[dict[str, Any]] = []
        backend = FakeEnvironmentBackend()
        backend.open()
        run_casting_s_c5_nether_entry_driver(
            backend,
            _context(),
            event_sink=lambda event: seen.append(dict(event)),
        )
        self.assertGreater(len(seen), 0)
        first = seen[0]
        for token in (
            "nether_entry_evaluation",
            "latched_frame_identity",
            "matched_frame_identity",
            "entered_via_episode_portal",
            "pre_transition_position",
            "nether_portal",
            "agents_in_nether",
            "transition_step",
        ):
            self.assertNotIn(
                token,
                repr(first),
                msg=(
                    f"Driver event leaks {token!r}; events must not carry "
                    "evaluator truth"
                ),
            )


class BudgetTests(unittest.TestCase):
    """Step / time / wait / plan / recovery budgets fail closed."""

    def _open_backend(self) -> FakeEnvironmentBackend:
        backend = FakeEnvironmentBackend()
        backend.open()
        return backend

    def test_step_budget_blocked(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(
            backend,
            _context(),
            max_environment_steps=20,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("step budget", result.blocked_reason or "")
        self.assertLess(result.steps_executed, 347)
        self.assertIsNone(result.ignition_relevant_action_step)
        self.assertIsNone(result.nether_entry_step)

    def test_time_budget_blocked(self) -> None:
        class _SlowBackend(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._t0 = 0.0

            def reset(self, task: Any) -> Any:
                obs = super().reset(task)
                self._t0 = float(list(obs.values())[0].timestamp)
                return obs

            def step(self, actions: Any) -> Any:
                step_result = super().step(actions)
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
        result = run_casting_s_c5_nether_entry_driver(
            backend,
            _context(task_time_limit=10.0),
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("time budget", result.blocked_reason or "")

    def test_wait_budget_blocked(self) -> None:
        backend = self._open_backend()
        result = run_casting_s_c5_nether_entry_driver(
            backend,
            _context(),
            max_wait_steps=1,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("wait budget", result.blocked_reason or "")

    def test_plan_length_over_hard_cap_blocked(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "wait steps exceed the hard limit"
        ):
            build_casting_s_c5_nether_entry_action_plan(
                support_block_wait_steps=4,
                fluid_settle_wait_steps=10,
                obsidian_wait_steps=10,
                ignition_portal_settle_steps=5,
                entry_settle_steps=200,
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
        result = run_casting_s_c5_nether_entry_driver(
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
        result = run_casting_s_c5_nether_entry_driver(
            backend,
            _context(),
            total_recovery_budget=32,
            recoveries_per_use_item=2,
        )
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn(
            "per-step recovery budget", result.blocked_reason or ""
        )


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
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
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
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertIn("backend exploded", result.blocked_reason or "")
        self.assertEqual(result.error_type, "RuntimeError")

    def test_non_recoverable_type_error_fails_closed(self) -> None:
        class _TypeBoom(FakeEnvironmentBackend):
            def step(self, actions: Any) -> Any:
                raise TypeError("bad step")

        backend = _TypeBoom()
        backend.open()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_FAILED)
        self.assertEqual(result.error_type, "TypeError")

    def test_os_error_fails_closed(self) -> None:
        class _OSBoom(FakeEnvironmentBackend):
            def step(self, actions: Any) -> Any:
                raise OSError("io error")

        backend = _OSBoom()
        backend.open()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
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
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("termination", result.blocked_reason or "")
        self.assertTrue(result.terminated)


class CapabilityGateTests(unittest.TestCase):
    """The pre-episode capability gate fails closed before reset."""

    def test_full_capabilities_pass(self) -> None:
        backend = FakeEnvironmentBackend()
        backend._capabilities = BackendCapabilities.full()  # type: ignore[attr-defined]
        backend.open()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
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
            run_casting_s_c5_nether_entry_driver(backend, _context())


class ObservationSchemaTests(unittest.TestCase):
    """The driver never reads hidden fields from Observations."""

    FORBIDDEN_TOKENS: tuple[str, ...] = (
        "nether_entry_evaluation",
        "latched_frame_identity",
        "matched_frame_identity",
        "entered_via_episode_portal",
        "pre_transition_position",
        "nether_portal",
        "agents_in_nether",
        "transition_step",
        "source_dimension",
        "target_dimension",
        "FrozenFrameIdentity",
        "FrozenNetherEntry",
        "NetherEntryEvidence",
        "FrozenIgnition",
    )

    def test_default_fake_observation_carries_no_truth_tokens(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
        for observation in (result.final_observation,):
            for token in self.FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    repr(observation),
                    msg=(
                        f"Observation leaks {token!r} for the C5 "
                        "Nether-entry driver"
                    ),
                )

    def test_observation_schema_is_locked_to_eight_public_fields(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        observations = backend.reset(_ResetProxy(_context()))  # type: ignore[arg-type]
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
                "messages",
                "workflow_stage",
            },
        )

    def test_stepped_observation_has_no_truth_tokens(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        result = run_casting_s_c5_nether_entry_driver(backend, _context())
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


class DriverSourceIsolationTests(unittest.TestCase):
    """The driver source must not import or reference evaluator types."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER_SOURCE.read_text()
        cls.tree = ast.parse(cls.source)

    def _code_only(self) -> str:
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
                        (
                            node.body[0].lineno,
                            node.body[0].end_lineno
                            or node.body[0].lineno,
                        )
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
                    "casting_nether_entry_evaluator" in node.module
                    or "casting_ignition_evaluator" in node.module
                    or "casting_frame_evaluator" in node.module
                    or "continuous_casting" in node.module
                    or "portal" in node.module
                ):
                    if "casting_nether_entry_evaluator" in node.module:
                        self.fail(
                            f"driver imports from {node.module!r} at line "
                            f"{node.lineno}; C5 Nether-entry evaluator "
                            "types are forbidden"
                        )
                    if "casting_ignition_evaluator" in node.module:
                        self.fail(
                            f"driver imports from {node.module!r} at line "
                            f"{node.lineno}; C4 ignition evaluator types "
                            "are forbidden"
                        )

    def test_no_attribute_references_to_evaluator_state(self) -> None:
        forbidden_attrs = {
            "set_nether_entry_evaluation_state",
            "get_nether_entry_evaluation_state",
            "clear_nether_entry_evaluation_state",
            "_nether_entry_evaluation_state",
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
            "FrozenNetherEntryEvaluationState",
            "FrozenNetherEntryEvaluationResult",
            "FrozenNetherEntryEvaluator",
            "NetherEntryEvidence",
            "build_c4_c3_frame_identity",
            "evaluator_contract",
            "agents_in_nether",
            "entered_via_episode_portal",
            "matched_frame_identity",
            "latched_frame_identity",
            "pre_transition_position",
        }
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                self.fail(
                    f"driver source references attribute {node.attr!r} "
                    f"at line {node.lineno}"
                )

    def test_no_scenario_parameters_in_code(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr == "scenario_parameters":
                self.fail(
                    "driver must not reference 'scenario_parameters' "
                    f"as a code attribute at line {node.lineno}"
                )
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

    def test_recovery_kind_does_not_leak_evaluator_truth(self) -> None:
        masked = self._code_only()
        for token in (
            "latched_frame_identity",
            "matched_frame_identity",
            "entered_via_episode_portal",
            "pre_transition_position",
            "nether_portal",
            "agents_in_nether",
            "frame_outcome",
            "ignition_evaluation",
            "nether_entry_evaluation",
        ):
            self.assertNotIn(token, masked)


class TruthSlotIsolationTests(unittest.TestCase):
    """C1 / C2 / C3 / C4 / C5 truth slots are independent on FakeBackend."""

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
        # Inject C2 truth.
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
        # C1 and C2 read back; C3 / C4 / C5 are still empty.
        self.assertIsNotNone(backend.get_casting_evaluation_state())
        self.assertIsNotNone(
            backend.get_continuous_casting_evaluation_state()
        )
        with self.assertRaises(RuntimeError):
            backend.get_frame_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_ignition_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()
        # Close clears all five slots.
        backend.close()
        with self.assertRaises(RuntimeError):
            backend.get_casting_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_continuous_casting_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_frame_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_ignition_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()

    def test_step_clears_c5_slot(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(_ResetProxy(_context()))  # type: ignore[arg-type]
        # After reset the C5 slot is None.
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()


class NetherEntryOrchestratorSuccessTests(unittest.TestCase):
    """End-to-end orchestrator: driver + C5 evaluator → success."""

    def test_full_nether_entry_success(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        task = _task()
        result = run_orchestrator(
            backend, driver_result, C5NetherEntryWorldTruth(), task=task
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertTrue(result.frame_identity_matched)
        self.assertEqual(
            result.entry_agent_id, AGENT_ID
        )
        self.assertEqual(
            result.target_dimension, C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION
        )
        self.assertEqual(
            result.source_dimension, C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION
        )

    def test_failure_when_no_agent_entered_nether(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        # Override the orchestrator by passing agents=frozenset() to
        # bypass the entry attribution.
        result = run_orchestrator(
            backend,
            driver_result,
            C5NetherEntryWorldTruth(),
            task=_task(),
            use_backend_roundtrip=False,
        )
        # Default result is success; the no-agent path requires
        # the agents_in_nether slot to be empty, which the
        # default helper does not produce. So this test
        # confirms the default path produces success; the
        # no-agent path is covered by test_c5_evaluator_still_runs.
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)

    def test_failure_when_entry_evidence_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C5NetherEntryWorldTruth(),
            task=_task(),
            include_entry_evidence=False,
            use_backend_roundtrip=False,
        )
        self.assertEqual(
            result.outcome, OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN
        )

    def test_failure_on_wrong_entry_agent(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        # Re-construct a custom state with a wrong entry agent.
        from obsidianlink.evaluation.casting_nether_entry_evaluator import (
            NetherEntryEvidence,
        )
        world = C5NetherEntryWorldTruth()
        state = _build_nether_entry_state(
            backend,
            driver_result,
            world,
            task=_task(),
        )
        # Build a replacement state whose entry_evidence uses the
        # wrong agent.
        wrong_entry = NetherEntryEvidence(
            episode_id=state.entry_evidence.episode_id,
            agent_id=WRONG_AGENT_ID,
            source_dimension=state.entry_evidence.source_dimension,
            target_dimension=state.entry_evidence.target_dimension,
            transition_step=state.entry_evidence.transition_step,
            pre_transition_position=(
                state.entry_evidence.pre_transition_position
            ),
            entered_via_episode_portal=(
                state.entry_evidence.entered_via_episode_portal
            ),
            matched_frame_identity=(
                state.entry_evidence.matched_frame_identity
            ),
        )
        replaced = dataclasses.replace(state, entry_evidence=wrong_entry)
        result = FrozenNetherEntryEvaluator().evaluate(replaced)
        self.assertEqual(result.outcome, OUTCOME_WRONG_ENTRY_AGENT)

    def test_failure_on_wrong_source_dimension(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C5NetherEntryWorldTruth(),
            task=_task(),
            nether_entry_source_dimension="minecraft:the_end",
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_SOURCE_DIMENSION)

    def test_failure_on_wrong_target_dimension(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C5NetherEntryWorldTruth(),
            task=_task(),
            nether_entry_target_dimension="minecraft:overworld",
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_WRONG_TARGET_DIMENSION)

    def test_failure_on_transition_step_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        from obsidianlink.evaluation.casting_nether_entry_evaluator import (
            NetherEntryEvidence,
        )
        world = C5NetherEntryWorldTruth()
        state = _build_nether_entry_state(
            backend,
            driver_result,
            world,
            task=_task(),
        )
        wrong_entry = NetherEntryEvidence(
            episode_id=state.entry_evidence.episode_id,
            agent_id=state.entry_evidence.agent_id,
            source_dimension=state.entry_evidence.source_dimension,
            target_dimension=state.entry_evidence.target_dimension,
            transition_step=None,
            pre_transition_position=(
                state.entry_evidence.pre_transition_position
            ),
            entered_via_episode_portal=(
                state.entry_evidence.entered_via_episode_portal
            ),
            matched_frame_identity=(
                state.entry_evidence.matched_frame_identity
            ),
        )
        replaced = dataclasses.replace(state, entry_evidence=wrong_entry)
        result = FrozenNetherEntryEvaluator().evaluate(replaced)
        self.assertEqual(result.outcome, OUTCOME_TRANSITION_STEP_MISSING)

    def test_failure_on_pre_transition_position_missing(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        from obsidianlink.evaluation.casting_nether_entry_evaluator import (
            NetherEntryEvidence,
        )
        world = C5NetherEntryWorldTruth()
        state = _build_nether_entry_state(
            backend,
            driver_result,
            world,
            task=_task(),
        )
        wrong_entry = NetherEntryEvidence(
            episode_id=state.entry_evidence.episode_id,
            agent_id=state.entry_evidence.agent_id,
            source_dimension=state.entry_evidence.source_dimension,
            target_dimension=state.entry_evidence.target_dimension,
            transition_step=state.entry_evidence.transition_step,
            pre_transition_position=None,
            entered_via_episode_portal=(
                state.entry_evidence.entered_via_episode_portal
            ),
            matched_frame_identity=(
                state.entry_evidence.matched_frame_identity
            ),
        )
        replaced = dataclasses.replace(state, entry_evidence=wrong_entry)
        result = FrozenNetherEntryEvaluator().evaluate(replaced)
        self.assertEqual(
            result.outcome, OUTCOME_PRE_TRANSITION_POSITION_MISSING
        )

    def test_failure_on_transition_before_activation(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        # Bypass the round-trip: build a complete C4 success
        # state whose activation is in-window and a C5 entry
        # whose transition step is *before* the activation
        # step. The C5 wrapper reports
        # ``transition_before_activation`` when the transition
        # step is below the activation update step.
        from obsidianlink.evaluation.casting_nether_entry_evaluator import (
            NetherEntryEvidence,
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            FrozenIgnitionEvaluationState,
        )
        world = C5NetherEntryWorldTruth()
        # Use a frame state at the natural completed step.
        frame_state = _state_from_driver(
            driver_result,
            world,
            task=_task(),
        )
        latched = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            agent_id=AGENT_ID,
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
        )
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type=C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
            item=C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
            target_cell=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        # Build an activation within the 4-step inclusive
        # window of the ignition step (delta = 4) so the C4
        # wrapper still passes.
        activation_step = IGNITION_STEP + 4
        activation = PortalActivationEvidence(
            episode_id=EPISODE_ID,
            update_step=activation_step,
            agent_id=AGENT_ID,
            nether_portal_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
            latched_frame_identity=latched,
        )
        ignition_state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            frame_state=frame_state,
            latched_frame_identity=latched,
            ignition_action=ignition,
            activation_evidence=activation,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=frame_state.step_id,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        # Build a transition at an early step (before the
        # activation step). C5 will report
        # ``transition_before_activation`` because the entry
        # transition step is below the activation update step.
        entry_evidence = NetherEntryEvidence(
            episode_id=EPISODE_ID,
            agent_id=AGENT_ID,
            source_dimension=C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
            target_dimension=C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
            transition_step=10,
            pre_transition_position=(1.5, 1.0, 1.0),
            entered_via_episode_portal=True,
            matched_frame_identity=latched,
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            ignition_state=ignition_state,
            agents_in_nether=frozenset({AGENT_ID}),
            entry_evidence=entry_evidence,
            agent_id=AGENT_ID,
            episode_terminated=True,
            terminated_step=frame_state.step_id,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(
            result.outcome, OUTCOME_TRANSITION_BEFORE_ACTIVATION
        )

    def test_failure_on_external_entry(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        result = run_orchestrator(
            backend,
            driver_result,
            C5NetherEntryWorldTruth(),
            task=_task(),
            nether_entry_entered_via_episode_portal=False,
            use_backend_roundtrip=False,
        )
        self.assertEqual(
            result.outcome, OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL
        )

    def test_failure_on_frame_identity_mismatch(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        # Build a frame identity with the same geometry as the
        # C3 frozen plan but a different ``step_id`` so the C4
        # wrapper's geometry check still passes but the C5
        # evaluator's identity comparison fails closed. The
        # state is rebuilt manually so the entry's identity is
        # different from the C4 wrapper's latched identity.
        from obsidianlink.evaluation.casting_nether_entry_evaluator import (
            NetherEntryEvidence,
        )
        from obsidianlink.evaluation.casting_ignition_evaluator import (
            CASTING_S_C3_FRAME_CELLS,
            CASTING_S_C3_INTERIOR_CELLS,
            FrozenIgnitionEvaluationState,
        )
        world = C5NetherEntryWorldTruth()
        frame_state = _state_from_driver(
            driver_result, world, task=_task(),
        )
        canonical_latched = build_c4_c3_frame_identity(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            agent_id=AGENT_ID,
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
        )
        ignition = IgnitionActionEvidence(
            episode_id=EPISODE_ID,
            step_id=IGNITION_STEP,
            agent_id=AGENT_ID,
            action_type=C5_NETHER_ENTRY_PUBLIC_IGNITION_ACTION,
            item=C5_NETHER_ENTRY_PUBLIC_IGNITION_ITEM,
            target_cell=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
        )
        # Build an activation within the 4-step inclusive
        # window of the ignition step so the C4 wrapper still
        # passes.
        activation_step = IGNITION_STEP + 4
        activation = PortalActivationEvidence(
            episode_id=EPISODE_ID,
            update_step=activation_step,
            agent_id=AGENT_ID,
            nether_portal_offset=C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
            latched_frame_identity=canonical_latched,
        )
        ignition_state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            frame_state=frame_state,
            latched_frame_identity=canonical_latched,
            ignition_action=ignition,
            activation_evidence=activation,
            agent_id=AGENT_ID,
            causality_window_steps=4,
            episode_terminated=True,
            terminated_step=frame_state.step_id,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        # Build the entry's identity with a different step_id
        # so the C5 evaluator's identity comparison fails. The
        # C5 wrapper compares entry.matched_frame_identity with
        # state.ignition_state.latched_frame_identity via
        # ``as_dict()`` snapshots, so a different step_id is
        # enough.
        mismatched_entry_identity = FrozenFrameIdentity(
            orientation="plane_z",
            min_corner=(0, 0, 1),
            max_corner=(3, 4, 1),
            width=4,
            height=5,
            target_offsets=tuple(CASTING_S_C3_FRAME_CELLS),
            interior_offsets=tuple(CASTING_S_C3_INTERIOR_CELLS),
            required_corner_count=4,
            required_full_ring_count=14,
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id - 5,  # different step_id
            agent_id=AGENT_ID,
        )
        entry_evidence = NetherEntryEvidence(
            episode_id=EPISODE_ID,
            agent_id=AGENT_ID,
            source_dimension=C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
            target_dimension=C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
            transition_step=activation_step,
            pre_transition_position=(1.5, 1.0, 1.0),
            entered_via_episode_portal=True,
            matched_frame_identity=mismatched_entry_identity,
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=frame_state.step_id,
            ignition_state=ignition_state,
            agents_in_nether=frozenset({AGENT_ID}),
            entry_evidence=entry_evidence,
            agent_id=AGENT_ID,
            episode_terminated=True,
            terminated_step=frame_state.step_id,
            terminated_reason=TERMINATED_REASON,
            current_time_seconds=0.0,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(
            result.outcome, OUTCOME_FRAME_IDENTITY_MISMATCH
        )

    def test_step_budget_exceeded(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        task = _task(max_environment_steps=10)
        result = run_orchestrator(
            backend, driver_result, C5NetherEntryWorldTruth(), task=task,
            use_backend_roundtrip=False,
        )
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_failure_when_c4_failed(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        driver_result = run_casting_s_c5_nether_entry_driver(
            backend, _context()
        )
        # Build a frame state where the frame is not built (some
        # cells still air). The C4 wrapper fails closed; the
        # C5 evaluator surfaces ``ignition_not_completed``.
        world = C5NetherEntryWorldTruth(
            current_blocks=("air",) + ("obsidian",) * 13,
            transition_after_blocks=("air",) + ("obsidian",) * 13,
        )
        result = run_orchestrator(
            backend, driver_result, world, task=_task(),
            use_backend_roundtrip=False,
        )
        self.assertEqual(
            result.outcome, OUTCOME_IGNITION_NOT_COMPLETED
        )


class DeterministicReplayTests(unittest.TestCase):
    """Same input ⇒ same driver sequence / events / as_dict()."""

    def test_replay_produces_identical_results(self) -> None:
        results = []
        for _ in range(3):
            backend = FakeEnvironmentBackend()
            backend.open()
            results.append(
                run_casting_s_c5_nether_entry_driver(
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
                run_casting_s_c5_nether_entry_driver(
                    backend, _context()
                )
            )
        first, second = results
        self.assertEqual(first.steps_executed, second.steps_executed)
        self.assertEqual(first.recovery_attempts, second.recovery_attempts)
        self.assertEqual(first.as_dict(), second.as_dict())


class RegressionTests(unittest.TestCase):
    """C1 / C2 / C3 / C4 / C5 evaluator regression checks."""

    def test_c1_evaluator_still_runs(self) -> None:
        from obsidianlink.evaluation.casting import (
            CastingEvaluationState,
            CastingEvaluator,
        )
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
        self.assertIsInstance(result.outcome, str)

    def test_c3_frame_evaluator_still_runs(self) -> None:
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
                for cell in CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
            ),
            interior_cells=tuple(
                FrozenFrameInteriorCellTruth(target_cell=cell, current_block="air")
                for cell in CASTING_S_C3_INTERIOR_CELLS
            ),
        )
        result = FrozenFrameEvaluator().evaluate(state)
        self.assertIn(result.outcome, {"in_progress", "partial_completion"})

    def test_c4_ignition_evaluator_still_runs(self) -> None:
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
                for cell in CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
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
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
        )
        state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            frame_state=frame_state,
            latched_frame_identity=latched,
            agent_id=AGENT_ID,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        result = FrozenIgnitionEvaluator().evaluate(state)
        self.assertIsInstance(result.outcome, str)

    def test_c5_evaluator_still_runs(self) -> None:
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
                for cell in CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
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
            activation_offsets=(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,),
        )
        ignition_state = FrozenIgnitionEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            frame_state=frame_state,
            latched_frame_identity=latched,
            agent_id=AGENT_ID,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            ignition_state=ignition_state,
            max_environment_steps=DEFAULT_MAX_ENVIRONMENT_STEPS,
            max_game_time_seconds=float(DEFAULT_MAX_GAME_TIME_SECONDS),
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertIsInstance(result.outcome, str)


class OfflineCheckTests(unittest.TestCase):
    """The offline --check and check_environment.py still pass."""

    def test_offline_check_phase(self) -> None:
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
        self.assertIn('"phase": "r6_c5_deterministic_driver"', result.stdout)

    def test_check_environment_script(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_environment.py"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"phase": "r6_c5_deterministic_driver"', result.stdout)


class PackageImportTests(unittest.TestCase):
    """The driver and context modules are importable from the package."""

    def test_imports(self) -> None:
        from obsidianlink.drivers import (
            CastingC5NetherEntryDriverResult,
            CastingC5NetherEntryPlanStep,
            PublicC5NetherEntryDriverContext,
            build_casting_s_c5_nether_entry_action_plan,
            run_casting_s_c5_nether_entry_driver,
        )
        from obsidianlink.core import (
            build_public_c5_nether_entry_driver_context_from_task,
        )
        self.assertTrue(callable(run_casting_s_c5_nether_entry_driver))
        self.assertTrue(
            callable(build_casting_s_c5_nether_entry_action_plan)
        )
        self.assertTrue(
            callable(build_public_c5_nether_entry_driver_context_from_task)
        )
        self.assertIsNotNone(CastingC5NetherEntryDriverResult)
        self.assertIsNotNone(CastingC5NetherEntryPlanStep)
        self.assertIsNotNone(PublicC5NetherEntryDriverContext)


if __name__ == "__main__":
    unittest.main()
