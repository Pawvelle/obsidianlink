from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.task_catalog import load_task_catalog, validate_catalog_references
from obsidianlink.core.types import TaskInstance
from obsidianlink.drivers.casting_c1 import run_casting_c1_driver
from obsidianlink.drivers.casting_c3 import run_casting_c3_driver
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting import (
    CastingEvaluationState,
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
    OUTCOME_SUCCESS,
    OUTCOME_TRUTH_MISSING,
)
from obsidianlink.evaluation.continuous_casting import (
    ContinuousCastingCellTruth,
    ContinuousCastingEvaluationState,
    ContinuousCastingEvaluator,
    OUTCOME_SUCCESS as CONTINUOUS_OUTCOME_SUCCESS,
)
from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator


ROOT = Path(__file__).resolve().parents[1]
TASK_CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"


def _run_r4_contract_check(backend: FakeEnvironmentBackend, task: TaskInstance) -> dict[str, object]:
    """Run the R4 single-cell driver and the R3 evaluator end-to-end.

    The driver is invoked first; the orchestrator (this function) then
    injects evaluator-only truth via
    :meth:`FakeEnvironmentBackend.set_casting_evaluation_state` and
    calls :class:`CastingEvaluator`. The driver never sees the truth
    surface.
    """
    parsed = parse_macro_action(
        '{"action_type":"wait","target":null,"duration_ticks":1,"parameters":{}}'
    )
    observations = backend.reset(task)
    step = backend.step({"agent_1": parsed.action})
    backend.set_evaluation_state(
        EvaluationState(
            episode_id=task.task_id,
            step_id=step.step_id,
            portal_built_by_episode=True,
            valid_portal_frame=True,
            portal_activated=True,
            agents_in_nether=frozenset({"agent_1"}),
            entered_via_episode_portal_by_agent={"agent_1": True},
        )
    )
    portal_result = PortalEvaluator().evaluate(
        backend.get_evaluation_state()
    )
    # Casting-c1 path 1: the casting evaluator must fail closed
    # when only an empty state is injected.
    backend.set_casting_evaluation_state(
        CastingEvaluationState(
            episode_id=task.task_id,
            step_id=step.step_id,
            agent_id="agent_1",
            target_cell=(0, 4, 1),
            max_environment_steps=task.limits["max_environment_steps"],
            max_game_time_seconds=task.limits["max_game_time_seconds"],
        )
    )
    truth_missing_result = CastingEvaluator().evaluate(
        backend.get_casting_evaluation_state()
    )
    if truth_missing_result.outcome != OUTCOME_TRUTH_MISSING:
        raise RuntimeError(
            "casting evaluator must fail closed without truth"
        )

    # Casting-c1 path 2: the R4 deterministic driver walks
    # the bounded plan on the FakeBackend; the orchestrator
    # injects a fully-populated success state and the
    # evaluator reports ``OUTCOME_SUCCESS``. This proves the
    # driver and the evaluator are wired together end-to-end
    # without any truth leaking into the driver.
    driver_result = run_casting_c1_driver(backend, task)
    relevant = driver_result.relevant_action_steps
    if not relevant:
        raise RuntimeError(
            "R4 driver must submit at least one relevant action"
        )
    last_action = max(relevant)
    success_state = CastingEvaluationState(
        episode_id=task.task_id,
        step_id=driver_result.steps_executed,
        agent_id="agent_1",
        target_cell=(0, 4, 1),
        initial_target_block="air",
        current_target_block="obsidian",
        target_update_evidence=CastingTransitionEvidence(
            before_block="air",
            after_block="obsidian",
            update_step=last_action,
        ),
        water_truth=CastingFluidTruth(
            present=True, evidence_step=last_action - 1
        ),
        lava_truth=CastingFluidTruth(
            present=True, evidence_step=relevant[0]
        ),
        relevant_action_steps=relevant,
        episode_terminated=True,
        terminated_step=driver_result.steps_executed,
        terminated_reason="driver_done",
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )
    backend.set_casting_evaluation_state(success_state)
    success_result = CastingEvaluator().evaluate(
        backend.get_casting_evaluation_state()
    )

    return {
        "agent_ids": sorted(observations),
        "action_parser_accepted": parsed.accepted,
        "backend_step": step.step_id,
        "portal_evaluator_success": portal_result.success,
        "casting_evaluator_outcome": truth_missing_result.outcome,
        "driver_status": driver_result.status,
        "driver_steps_executed": driver_result.steps_executed,
        "driver_relevant_action_steps": len(driver_result.relevant_action_steps),
        "driver_success_outcome": success_result.outcome,
    }


def _run_r5_contract_check() -> dict[str, object]:
    """Run the R5 multi-cell driver and the continuous evaluator end-to-end.

    The orchestrator (this function) builds a small success world,
    injects per-cell evaluator-only truth into the backend, and asks
    the :class:`ContinuousCastingEvaluator` for the verdict. The
    driver never sees the truth surface; the test orchestrator
    owns it.
    """
    task = TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": "casting_c3_contract_check",
            "route": "lava_casting",
            "difficulty": 2,
            "agent_ids": ["agent_1"],
            "world_seed": 0,
            "instruction": (
                "Validate the R5 offline continuous casting contract: "
                "make all three target cells obsidian."
            ),
            "spawn_positions": {"agent_1": [0, 4, 0]},
            "initial_inventories": {
                "agent_1": {
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
    backend = FakeEnvironmentBackend()
    backend.open()
    try:
        driver_result = run_casting_c3_driver(backend, task)
        if driver_result.status != "completed":
            raise RuntimeError(
                "R5 driver must reach the end of the bounded plan"
            )
        per_cell = driver_result.per_cell_relevant_action_steps
        if not per_cell:
            raise RuntimeError(
                "R5 driver must submit at least one relevant action per cell"
            )
        cells: list[ContinuousCastingCellTruth] = []
        target_cells: tuple[tuple[int, int, int], ...] = (
            (2, 4, 3),
            (3, 4, 3),
            (4, 4, 3),
        )
        for cell_index, target_cell in enumerate(target_cells):
            steps = per_cell.get(cell_index, ())
            if not steps:
                raise RuntimeError(
                    f"cell {cell_index} has no relevant action steps"
                )
            last_action = max(steps)
            first_action = min(steps)
            cells.append(
                ContinuousCastingCellTruth(
                    target_cell=target_cell,
                    initial_block="air",
                    current_block="obsidian",
                    water_truth=CastingFluidTruth(
                        present=True, evidence_step=last_action
                    ),
                    lava_truth=CastingFluidTruth(
                        present=True, evidence_step=first_action
                    ),
                    transition_evidence=CastingTransitionEvidence(
                        before_block="air",
                        after_block="obsidian",
                        update_step=last_action,
                    ),
                    relevant_action_steps=tuple(steps),
                )
            )
        success_state = ContinuousCastingEvaluationState(
            episode_id=task.task_id,
            step_id=driver_result.steps_executed,
            agent_id="agent_1",
            cells=tuple(cells),
            episode_terminated=True,
            terminated_step=driver_result.steps_executed,
            terminated_reason="driver_done",
            max_environment_steps=task.limits["max_environment_steps"],
            max_game_time_seconds=task.limits["max_game_time_seconds"],
        )
        backend.set_continuous_casting_evaluation_state(success_state)
        result = ContinuousCastingEvaluator().evaluate(
            backend.get_continuous_casting_evaluation_state()
        )
    finally:
        backend.close()
    if result.outcome != CONTINUOUS_OUTCOME_SUCCESS:
        raise RuntimeError(
            "R5 evaluator must return complete success on the contract check; got "
            f"{result.outcome!r}"
        )
    if result.completed_cells != 3 or result.total_cells != 3:
        raise RuntimeError(
            "R5 contract check requires exactly 3 completed cells; got "
            f"{result.completed_cells}/{result.total_cells}"
        )
    return {
        "c3_task_id": task.task_id,
        "c3_driver_status": driver_result.status,
        "c3_driver_steps_executed": driver_result.steps_executed,
        "c3_driver_recovery_attempts": driver_result.recovery_attempts,
        "c3_driver_per_cell_relevant": {
            int(k): len(v) for k, v in per_cell.items()
        },
        "c3_evaluator_outcome": result.outcome,
        "c3_evaluator_completed_cells": result.completed_cells,
        "c3_evaluator_total_cells": result.total_cells,
        "c3_evaluator_success": result.success,
    }


def _offline_contract_check() -> dict[str, object]:
    # ------------------------------------------------------------------
    # R4 single-cell contract: the legacy single-block driver + R3
    # evaluator are still required to pass.
    # ------------------------------------------------------------------
    r4_task = TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": "casting_c1_contract_check",
            "route": "lava_casting",
            "difficulty": 1,
            "agent_ids": ["agent_1"],
            "world_seed": 0,
            "instruction": "Validate the offline casting task contract.",
            "spawn_positions": {"agent_1": [0, 4, 0]},
            "initial_inventories": {
                "agent_1": {
                    "water_bucket": 1,
                    "lava_bucket": 1,
                    "cobblestone": 8,
                }
            },
            "workflow": "casting_c1_fixed",
            "milestones": [
                "task_reset",
                "liquid_resources_ready",
                "first_obsidian_cast",
            ],
            "limits": {
                "max_environment_steps": 160,
                "max_model_calls": 1,
                "max_game_time_seconds": 120,
            },
            "split": "development",
        }
    )
    backend = FakeEnvironmentBackend()
    backend.open()
    try:
        r4_summary = _run_r4_contract_check(backend, r4_task)
    finally:
        backend.close()
    # ------------------------------------------------------------------
    # R5 multi-cell contract: 3-cell straight line segment with the
    # continuous evaluator.
    # ------------------------------------------------------------------
    r5_summary = _run_r5_contract_check()
    catalog = load_task_catalog(TASK_CATALOG_PATH)
    validate_catalog_references(catalog, ROOT)
    active_entry = catalog.active_entry
    if active_entry.taxonomy is None:  # guarded by TaskCatalog, defensive here
        raise RuntimeError("active task catalog entry must have taxonomy")
    taxonomy = active_entry.taxonomy.as_dict()
    taxonomy["compatibility_task_name"] = active_entry.canonical_name
    return {
        "status": "ok",
        "phase": "r6_c5_live_minerl_backend_wiring_done",
        "active_task": active_entry.compatibility_id,
        "task_taxonomy": taxonomy,
        "task_catalog_version": catalog.catalog_version,
        "task_catalog_entries": len(catalog.entries),
        "live_run_allowed": False,
        "r4": r4_summary,
        "r5": r5_summary,
        "note": (
            "FakeBackend + R4 single-block driver + R3 single-cell evaluator "
            "+ R5 multi-cell driver + R5 continuous casting evaluator offline "
            "contract check + strict task catalog validation only; no real MineRL "
            "task, no real casting driver "
            "execution against Minecraft, and no model API call were made. "
            "The casting evaluators are type-strict, fail-closed, and do not "
            "simulate Minecraft fluid physics. "
            "The R6 Casting-S-C3 frozen-frame evaluator/driver, R6 Casting-S-C4 "
            "ignition evaluator/driver, and R6 Casting-S-C5 Nether-entry "
            "evaluator/driver are implemented and covered by the offline "
            "test suite; this compact check does not execute the 336-step C3, "
            "340-step C4, or 347-step C5 driver. The R6-C5-LIVE-MINERL-BACKEND-WIRING "
            "milestone now wires typed target-block / fluid truth through the "
            "production MineRL backend (capabilities "
            "exposes_target_block_truth / exposes_fluid_truth flipped to True "
            "only after the typed truth surface, the per-step cast credit "
            "history, and the reset / step / close cleanup passed the offline "
            "test suite); live MineRL, Gradle, and model API calls remain "
            "unimplemented, and no MineRL / Gradle / model API call was "
            "performed in this check."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obsidianlink",
        description="ObsidianLink benchmark development utilities.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the offline core contract check without starting MineRL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check:
        build_parser().print_help()
        return 0
    print(
        json.dumps(_offline_contract_check(), ensure_ascii=False, sort_keys=True)
    )
    return 0
