from __future__ import annotations

import argparse
import json
from typing import Sequence

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.drivers.casting_c1 import run_casting_c1_driver
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting import (
    CastingEvaluationState,
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
    OUTCOME_SUCCESS,
    OUTCOME_TRUTH_MISSING,
)
from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator
from obsidianlink.core.types import TaskInstance


def _offline_contract_check() -> dict[str, object]:
    # Use the same ``casting_c1_fixed`` task the rest of the
    # project freezes for R4. The driver expects a support block
    # (``cobblestone``) plus the two buckets, so the offline
    # contract check uses the full inventory.
    task = TaskInstance.from_dict(
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
    parsed = parse_macro_action(
        '{"action_type":"wait","target":null,"duration_ticks":1,"parameters":{}}'
    )

    backend = FakeEnvironmentBackend()
    backend.open()
    try:
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
    finally:
        backend.close()

    return {
        "status": "ok",
        "phase": "reset_4_deterministic_casting_driver",
        "active_task": "casting_c1_fixed",
        "live_run_allowed": False,
        "task_id": task.task_id,
        "agent_ids": sorted(observations),
        "action_parser_accepted": parsed.accepted,
        "backend_step": step.step_id,
        "portal_evaluator_success": portal_result.success,
        "casting_evaluator_outcome": truth_missing_result.outcome,
        "driver_status": driver_result.status,
        "driver_steps_executed": driver_result.steps_executed,
        "driver_relevant_action_steps": len(driver_result.relevant_action_steps),
        "driver_success_outcome": success_result.outcome,
        "note": (
            "FakeBackend + R4 deterministic driver + casting_c1 evaluator "
            "offline contract check only; no real MineRL task, no real "
            "casting driver execution against Minecraft, and no model API "
            "call were made. The casting evaluator is type-strict, "
            "fail-closed, and does not simulate Minecraft fluid physics."
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
