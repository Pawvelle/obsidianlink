"""Tests for the Phase 4 A1 deterministic ``scripted_a1`` driver."""

from __future__ import annotations

import time
import unittest

from obsidianlink.core.types import MacroAction
from obsidianlink.drivers.scripted_a1 import (
    AGENT_EYE,
    AGENT_ID,
    DEFAULT_MAX_ATTACK_TICKS_PER_CELL,
    DEFAULT_MAX_CELLS,
    DEFAULT_MAX_REAIM_ATTEMPTS_PER_CELL,
    DEFAULT_MAX_NO_PROGRESS_TICKS,
    DEFAULT_STEP_TIMEOUT_SECONDS,
    MAX_CAMERA_DELTA,
    MiningPlanStep,
    ScriptedA1Result,
    deposit_world_cells,
    run_scripted_a1,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation import PortalEvaluator
from tests.test_minerl_backend import (
    PORTAL_GRID_SHAPE,
    _BackendFactory,
    _ControlledMineRL_A1Env,
    _a1_task,
)


class ScriptedA1ConfigTests(unittest.TestCase):
    """Argument validation and budget guard tests."""

    def test_invalid_budgets_are_rejected(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            for kw, value in (
                ("max_cells", 0),
                ("max_attack_ticks_per_cell", 0),
                ("max_reaim_attempts_per_cell", 0),
                ("max_no_progress_ticks", 0),
            ):
                with self.assertRaisesRegex(ValueError, kw):
                    run_scripted_a1(
                        backend,
                        _a1_task(),
                        **{kw: value},
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

    def test_non_a1_task_is_rejected(self) -> None:
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


class ScriptedA1PlanConstantsTests(unittest.TestCase):
    """Public constants and helpers stay frozen."""

    def test_deposit_world_cells_contains_16_cells(self) -> None:
        cells = deposit_world_cells()
        self.assertEqual(len(cells), 16)
        # Row-major in (z, x), z outer.
        first = cells[0]
        self.assertEqual(first, (-2.5, 5.0, 3.5))
        last = cells[-1]
        self.assertEqual(last, (0.5, 5.0, 6.5))

    def test_default_budgets_are_positive(self) -> None:
        for value in (
            DEFAULT_MAX_ATTACK_TICKS_PER_CELL,
            DEFAULT_MAX_CELLS,
            DEFAULT_MAX_REAIM_ATTEMPTS_PER_CELL,
            DEFAULT_MAX_NO_PROGRESS_TICKS,
        ):
            self.assertGreater(value, 0)
        self.assertGreater(DEFAULT_STEP_TIMEOUT_SECONDS, 0)
        self.assertLessEqual(MAX_CAMERA_DELTA, 30.0)


class ScriptedA1SingleBlockCalibrationTests(unittest.TestCase):
    """The canonical Phase 4 A1 single-block calibration path.

    The single-block calibration must:

    * equip the diamond pickaxe and walk to the deposit edge;
    * issue many consecutive ``mine_target(obsidian)`` actions on
      the same cell until the controlled env removes the obsidian
      and adds 1 to the visible ``obsidian`` inventory on the
      same observation boundary;
    * stop after one credited cell, even when the configured
      quota is 14;
    * never claim ``obsidian_source_located`` until the first
      cell is reliably credited.
    """

    def test_single_block_calibration_credits_one_cell(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            started = time.monotonic()
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=1,
                max_attack_ticks_per_cell=20,
                max_reaim_attempts_per_cell=2,
                max_no_progress_ticks=200,
                step_timeout_seconds=5.0,
            )
            backend.mark_terminated(
                step_id=result.steps_completed,
                reason="scripted_a1_single_block_calibration",
            )
            evaluation = PortalEvaluator().evaluate(
                backend.get_evaluation_state()
            )
        elapsed = time.monotonic() - started
        # Single-block calibration explicitly stops at 1 cell,
        # so the driver reports "blocked" (quota not collected)
        # even though the cell itself was credited. This is by
        # design: the calibration run is not claiming that the
        # full A1 quota was satisfied.
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.failure_type, "quota_not_collected")
        self.assertEqual(result.obsidian_mined_count, 1)
        self.assertEqual(result.max_cells, 1)
        self.assertEqual(len(result.cells_attempted), 1)
        cell_summary = result.cells_attempted[0]
        self.assertTrue(cell_summary["credited"])
        # The fixture defaults to 3 attack ticks per cell, so the
        # first cell must take at least 3 attack ticks.
        self.assertGreaterEqual(cell_summary["attack_ticks"], 3)
        self.assertEqual(result.first_attack_step, cell_summary["first_attack_step"])
        self.assertEqual(
            result.block_removed_step, cell_summary["grid_removed_step"]
        )
        self.assertEqual(
            result.inventory_increased_step,
            cell_summary["inventory_step"],
        )
        # grid removed and inventory increased must be on the
        # same step (the dual-evidence contract).
        self.assertEqual(
            cell_summary["grid_removed_step"],
            cell_summary["inventory_step"],
        )
        self.assertEqual(result.obsidian_quota_required, 14)
        # Single-block mode must stop at 1 even though the quota
        # is 14.
        self.assertEqual(result.obsidian_mined_count, 1)
        self.assertIsNone(result.obsidian_quota_collected_step)
        # MineRL side: only the first-credited cell ever latches
        # source-located. ``quota_collected`` stays unset because
        # the calibration explicitly stops at 1 block.
        self.assertIsNotNone(evaluation.failure_type)
        self.assertGreater(result.elapsed_seconds, 0)
        # Sanity guard against accidental test sleep explosion.
        self.assertLess(elapsed, 30.0)

    def test_single_block_calibration_emits_dual_evidence(self) -> None:
        """The calibration run must record the post-step grid
        delta and the inventory delta on the **same** step id.
        """
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=1,
                max_attack_ticks_per_cell=20,
                max_reaim_attempts_per_cell=2,
                step_timeout_seconds=5.0,
            )
        self.assertEqual(result.status, "blocked")
        cell = result.cells_attempted[0]
        self.assertTrue(cell["credited"])
        self.assertIsNotNone(cell["grid_removed_step"])
        self.assertIsNotNone(cell["inventory_step"])
        self.assertEqual(
            cell["grid_removed_step"], cell["inventory_step"]
        )
        # Final visible inventory must show exactly 1 obsidian.
        self.assertEqual(
            result.final_visible_inventory.get("obsidian"), 1
        )

    def test_single_block_calibration_carries_identity_fields(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=1,
                max_attack_ticks_per_cell=20,
                max_reaim_attempts_per_cell=2,
                step_timeout_seconds=5.0,
            )
        # Every emitted event must carry the canonical identity
        # fields and be a Mapping.
        self.assertTrue(
            all(
                isinstance(event, dict)
                and event["episode_id"] == "route_a_a1_phase4_seed_0"
                and event["agent_id"] == AGENT_ID
                and type(event["step_id"]) is int
                for event in result.events
            )
        )
        # At least one mine_target event with the diamond_pickaxe
        # intent must appear in the log.
        mine_events = [
            event
            for event in result.events
            if event.get("action_type") == "mine_target"
            and event.get("target") == "obsidian"
        ]
        self.assertGreaterEqual(len(mine_events), 3)


class ScriptedA1FullSliceTests(unittest.TestCase):
    """The full 14-cell slice uses the same driver at ``max_cells=14``."""

    def test_full_slice_credits_all_14_cells(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=14,
                max_attack_ticks_per_cell=20,
                max_reaim_attempts_per_cell=2,
                step_timeout_seconds=5.0,
            )
            backend.mark_terminated(
                step_id=result.steps_completed,
                reason="scripted_a1_slice_complete",
            )
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.obsidian_mined_count, 14)
        self.assertIsNotNone(result.obsidian_quota_collected_step)
        # Every attempted cell must be credited.
        self.assertEqual(len(result.cells_attempted), 14)
        self.assertTrue(
            all(cell["credited"] for cell in result.cells_attempted)
        )
        # Each cell needs at least 3 attack ticks (fixture
        # threshold).
        for cell in result.cells_attempted:
            self.assertGreaterEqual(cell["attack_ticks"], 3)


class ScriptedA1FailureModesTests(unittest.TestCase):
    """Fail-closed behavior under inconsistent or missing evidence."""

    def test_no_progress_budget_terminates(self) -> None:
        """A driver that runs out of attack budget on the first
        cell must terminate with a per-cell-budget failure.
        """
        env = _ControlledMineRL_A1Env()
        # Strip the deposit before the driver runs.
        for offset in env._deposit_offsets:
            index = (
                offset[1] * 7 * 7
                + offset[2] * 7
                + offset[0]
            )
            env.grid[index] = 0  # air
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=1,
                max_attack_ticks_per_cell=3,
                max_reaim_attempts_per_cell=1,
                max_no_progress_ticks=5,
                step_timeout_seconds=5.0,
            )
        self.assertNotEqual(result.status, "passed")
        self.assertIsNone(result.obsidian_quota_collected_step)
        self.assertEqual(result.obsidian_mined_count, 0)
        self.assertIsNotNone(result.blocked_reason)

    def test_source_located_stays_unlatched_without_credit(self) -> None:
        """If no cell is ever credited, ``obsidian_source_located``
        must stay None (no intent-only latching).
        """
        env = _ControlledMineRL_A1Env()
        for offset in env._deposit_offsets:
            index = (
                offset[1] * 7 * 7
                + offset[2] * 7
                + offset[0]
            )
            env.grid[index] = 0  # air
        with _BackendFactory(env) as backend:
            result = run_scripted_a1(
                backend,
                _a1_task(),
                max_cells=1,
                max_attack_ticks_per_cell=2,
                max_reaim_attempts_per_cell=1,
                max_no_progress_ticks=2,
                step_timeout_seconds=5.0,
            )
            state = backend.get_evaluation_state()
        self.assertNotEqual(result.status, "passed")
        self.assertEqual(result.obsidian_mined_count, 0)
        self.assertIsNone(state.obsidian_source_located_step)
        self.assertIsNone(state.first_obsidian_mined_step)


if __name__ == "__main__":
    unittest.main()
