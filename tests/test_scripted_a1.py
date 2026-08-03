"""Tests for the Phase 4 A1 deterministic ``scripted_a1`` driver."""

from __future__ import annotations

import unittest

from obsidianlink.core.types import MacroAction
from obsidianlink.drivers.scripted_a1 import (
    AGENT_EYE,
    AGENT_ID,
    MAX_CAMERA_DELTA,
    MAX_MINING_PLAN_STEPS,
    MiningPlanStep,
    ScriptedA1Result,
    build_mining_action_plan,
    run_scripted_a1,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation import PortalEvaluator
from tests.test_minerl_backend import (
    _BackendFactory,
    _ControlledMineRL_A1Env,
    _a1_task,
)


class ScriptedA1PlanTests(unittest.TestCase):
    """Plan-shape and budget tests for the A1 mining slice plan."""

    def test_plan_is_bounded_and_mines_14_cells(self) -> None:
        plan = build_mining_action_plan(quota=14)
        mines = [
            item
            for item in plan
            if item.action.action_type == "mine_target"
        ]
        walks = [
            item
            for item in plan
            if item.action.action_type == "move"
            and item.action.parameters.get("forward") == 1.0
        ]
        equips = [
            item
            for item in plan
            if item.action.action_type == "equip_item"
        ]
        self.assertEqual(len(mines), 14)
        self.assertEqual(len(walks), 2)
        self.assertEqual(
            [item.action.target for item in equips], ["diamond_pickaxe"]
        )
        self.assertLess(len(plan), MAX_MINING_PLAN_STEPS)
        for item in plan:
            if item.action.action_type == "look":
                self.assertLessEqual(
                    abs(float(item.action.parameters.get("yaw", 0.0))),
                    MAX_CAMERA_DELTA,
                )
                self.assertLessEqual(
                    abs(float(item.action.parameters.get("pitch", 0.0))),
                    MAX_CAMERA_DELTA,
                )

    def test_plan_emits_mining_milestones_in_order(self) -> None:
        plan = build_mining_action_plan(quota=14)
        seen = False
        saw_mine_then_settle = 0
        for item in plan:
            if (
                seen
                and item.action.action_type == "wait"
                and item.label.endswith(".settle")
            ):
                saw_mine_then_settle += 1
            if item.action.action_type == "mine_target":
                seen = True
        self.assertEqual(saw_mine_then_settle, 14)

    def test_plan_invalid_quota_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "quota"):
            build_mining_action_plan(quota=0)
        with self.assertRaisesRegex(ValueError, "walk_forward_steps"):
            build_mining_action_plan(quota=14, walk_forward_steps=-1)

    def test_plan_quota_exceeds_deposit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "deposit"):
            build_mining_action_plan(quota=20)


class ScriptedA1DriverTests(unittest.TestCase):
    """End-to-end tests for ``run_scripted_a1`` against the A1 fixture."""

    def test_driver_records_three_mining_milestones(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                step_timeout_seconds=5.0,
            )
            backend.mark_terminated(
                step_id=result.steps_completed,
                reason="scripted_a1_slice_complete",
            )
            evaluation = PortalEvaluator().evaluate(
                backend.get_evaluation_state()
            )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.obsidian_mined_count, 14)
        self.assertIsNotNone(result.obsidian_quota_collected_step)
        self.assertGreaterEqual(result.obsidian_quota_collected_step, 14)
        self.assertIsNotNone(result.obsidian_source_located_step)
        self.assertIsNotNone(result.first_obsidian_mined_step)
        self.assertEqual(len(result.obsidian_mined_offsets), 14)
        self.assertEqual(result.external_mined_offsets, ())
        # All action records must carry the canonical identity
        # fields used by the rest of the ObsidianLink event log.
        self.assertTrue(
            all(
                event["episode_id"] == "route_a_a1_phase4_seed_0"
                and event["agent_id"] == AGENT_ID
                and type(event["step_id"]) is int
                for event in result.events
            )
        )
        # The slice is not a full Route A1 build. The downstream
        # portal-frame / Nether-entry failures must still apply.
        self.assertFalse(evaluation.success)
        self.assertIsNotNone(evaluation.failure_type)

    def test_driver_terminates_cleanly_when_quota_collected(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                step_timeout_seconds=5.0,
            )
        self.assertIsNotNone(result.obsidian_quota_collected_step)
        self.assertEqual(result.status, "passed")
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(len(result.obsidian_mined_offsets), 14)
        self.assertGreaterEqual(result.steps_completed, 14)

    def test_driver_rejects_non_a1_task(self) -> None:
        from obsidianlink.core.types import TaskInstance

        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            non_a1_task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "fake_non_a1",
                    "route": "obsidian_mining",
                    "difficulty": 1,
                    "agent_ids": ["agent_1"],
                    "world_seed": 0,
                    "instruction": "Build a portal.",
                    "spawn_positions": {"agent_1": [0, 4, 0]},
                    "initial_inventories": {
                        "agent_1": {"obsidian": 14, "flint_and_steel": 1}
                    },
                    "workflow": "route_a_a0",
                    "milestones": [
                        "task_reset",
                        "valid_portal_frame",
                        "portal_activated",
                        "agent_entered_nether",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 40,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            with self.assertRaisesRegex(ValueError, "route_a_a1"):
                run_scripted_a1(backend, non_a1_task, step_timeout_seconds=1.0)

    def test_driver_records_no_progress_on_wrong_target(self) -> None:
        """A driver that issues ``mine_target(stone)`` instead of
        ``mine_target(obsidian)`` must not be credited.

        The driver hard-codes ``mine_target(target="obsidian")``, so
        we exercise the bounded no-progress path by mutating the
        grid so the env refuses to mine (i.e. by clearing the
        deposit so the env's view cone hits no obsidian).
        """
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            # Strip the deposit before the driver runs.
            for offset in env._deposit_offsets:
                index = (
                    offset[1] * 7 * 7
                    + offset[2] * 7
                    + offset[0]
                )
                env.grid[index] = 0  # air
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_no_progress_retries=2,
                max_cell_retry_attempts=1,
                step_timeout_seconds=5.0,
            )
        self.assertNotEqual(result.status, "passed")
        self.assertIsNone(result.obsidian_quota_collected_step)
        self.assertEqual(result.obsidian_mined_count, 0)
        self.assertIsNotNone(result.blocked_reason)


class ScriptedA1BudgetTests(unittest.TestCase):
    """Argument validation and budget guard tests."""

    def test_invalid_retry_budgets_are_rejected(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            with self.assertRaisesRegex(
                ValueError, "max_no_progress_retries"
            ):
                run_scripted_a1(
                    backend,
                    _a1_task(),
                    max_no_progress_retries=-1,
                    step_timeout_seconds=1.0,
                )
            with self.assertRaisesRegex(
                ValueError, "max_cell_retry_attempts"
            ):
                run_scripted_a1(
                    backend,
                    _a1_task(),
                    max_cell_retry_attempts=-1,
                    step_timeout_seconds=1.0,
                )

    def test_invalid_step_timeout_is_rejected(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            with self.assertRaisesRegex(
                ValueError, "step_timeout_seconds"
            ):
                run_scripted_a1(
                    backend,
                    _a1_task(),
                    step_timeout_seconds=0.0,
                )


if __name__ == "__main__":
    unittest.main()
