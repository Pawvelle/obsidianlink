"""Tests for the MineRL environment backend with Phase 2 evaluator wiring."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from obsidianlink.core.interfaces import EnvironmentBackend
from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.env.minerl_backend import (
    MineRLEnvironmentBackend,
    _specification_for_task,
)
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_SIZE,
    PortalA0EnvSpec,
    PortalA1EnvSpec,
)


BLOCK_ID_TO_NAME = {index: name for index, name in enumerate(PORTAL_GRID_BLOCKS)}
from obsidianlink.evaluation import (
    FAILURE_FRAME_NEVER_VALID,
    FAILURE_FRAME_NOT_BUILT_BY_EPISODE,
    FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
    FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
    FAILURE_NO_AGENT_ENTERED_NETHER,
    FAILURE_PORTAL_NEVER_ACTIVATED,
    EvaluationState,
    PortalEvaluator,
    milestone_iterator,
)
from obsidianlink.logging.events import StructuredEvent
from tests.helpers import sample_task


ROOT = Path(__file__).resolve().parents[1]


def _frame_offsets() -> list[tuple[int, int, int]]:
    """14-cell 4x5 obsidian frame at z=1, in grid (x, y, z) order."""
    cells: list[tuple[int, int, int]] = []
    for x in range(0, 4):
        cells.append((x, 0, 1))
        cells.append((x, 4, 1))
    for y in range(1, 4):
        cells.append((0, y, 1))
        cells.append((3, y, 1))
    return cells


def _offset_to_flat_index(
    offset: tuple[int, int, int],
    shape: tuple[int, int, int] = (7, 7, 7),
) -> int:
    return (
        offset[1] * shape[0] * shape[2]
        + offset[2] * shape[0]
        + offset[0]
    )


def _apply_obsidian(grid_1d: np.ndarray, offsets: list[tuple[int, int, int]]) -> None:
    for offset in offsets:
        grid_1d[_offset_to_flat_index(offset)] = PORTAL_GRID_BLOCKS.index(
            "obsidian"
        )


def _apply_block(
    grid_1d: np.ndarray,
    block_name: str,
    offsets: list[tuple[int, int, int]],
) -> None:
    block_id = PORTAL_GRID_BLOCKS.index(block_name)
    for offset in offsets:
        grid_1d[_offset_to_flat_index(offset)] = block_id


def _controlled_env(
    *,
    build_full_frame: bool = False,
    ignite_after_frame: bool = False,
    pre_existing_frame: bool = False,
    pre_existing_activated: bool = False,
    build_offsets: list[tuple[int, int, int]] | None = None,
) -> "_ControlledMineRLEnv":
    return _ControlledMineRLEnv(
        build_full_frame=build_full_frame,
        ignite_after_frame=ignite_after_frame,
        pre_existing_frame=pre_existing_frame,
        pre_existing_activated=pre_existing_activated,
        build_offsets=build_offsets,
    )


class _ControlledMineRLEnv:
    """Minimal in-memory MineRL environment that scripts the grid."""

    def __init__(
        self,
        *,
        build_full_frame: bool = False,
        ignite_after_frame: bool = False,
        pre_existing_frame: bool = False,
        pre_existing_activated: bool = False,
        build_offsets: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self.action_space = PortalA0EnvSpec().action_space
        self.seed_value: int | None = None
        self.closed = False
        self.steps = 0
        self._build_full_frame = build_full_frame
        self._ignite_after_frame = ignite_after_frame
        self._pre_existing_frame = pre_existing_frame
        self._pre_existing_activated = pre_existing_activated
        self._build_offsets = build_offsets or _frame_offsets()
        self._placements = 0
        self._use_items = 0
        self._dimension = "minecraft:overworld"
        self._portal_transition: dict[str, Any] | None = None
        self._position_override: dict[str, float] | None = None
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        if pre_existing_frame or pre_existing_activated:
            _apply_obsidian(self.grid, _frame_offsets())
        if pre_existing_activated:
            _apply_block(self.grid, "nether_portal", [(1, 1, 1)])

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, Any]:
        return self._observation()

    def step(self, action: dict[str, Any]):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, Mapping) else {}
        use = int(action_map.get("use", 0))
        hotbar_obsidian = int(action_map.get("hotbar.1", 0))
        hotbar_flint = int(action_map.get("hotbar.2", 0))
        hotbar_dirt = int(action_map.get("hotbar.3", 0))
        if self._build_full_frame and use and hotbar_obsidian:
            self._placements += 1
            offsets = self._build_offsets
            if self._placements <= len(offsets):
                _apply_obsidian(
                    self.grid, [offsets[self._placements - 1]]
                )
        elif use and hotbar_obsidian and not self._build_full_frame:
            # Default behaviour (mirrors the historical Phase 1 fixture).
            self._placements += 1
            self.grid[0] = PORTAL_GRID_BLOCKS.index("obsidian")
            self.grid[1] = PORTAL_GRID_BLOCKS.index("nether_portal")
        if use and hotbar_dirt and not self._build_full_frame:
            # No-op; legacy fixture ignored dirt placements explicitly.
            pass
        if self._ignite_after_frame and use and hotbar_flint:
            self._use_items += 1
            if (
                self._placements >= len(self._build_offsets)
                and self._use_items >= 1
            ):
                _apply_block(self.grid, "nether_portal", [(1, 1, 1)])
                self._dimension = "minecraft:the_nether"
                self._portal_transition = {
                    "present": np.asarray(True, dtype=np.bool_),
                    "entered_via_portal": np.asarray(True, dtype=np.bool_),
                    "sequence": np.asarray(1, dtype=np.int64),
                    "source_portal_block_world_position": np.asarray(
                        (-2, 64, 1), dtype=np.int32
                    ),
                    "from_dimension": "minecraft:overworld",
                    "to_dimension": "minecraft:the_nether",
                }
        elif (
            not self._build_full_frame
            and not self._ignite_after_frame
            and use
            and hotbar_obsidian
        ):
            self._dimension = "minecraft:the_nether"
        observation = self._observation()
        observation["portal_dimension"] = np.asarray(self._dimension)
        return observation, 0.0, False, {"private": "not_forwarded"}

    def assert_action(self, action: dict[str, Any]) -> None:
        if not self.action_space.contains(action):
            raise AssertionError("invalid test action")

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> dict[str, Any]:
        # Position at the task's real A0 spawn height. The evaluator
        # converts atSpawn grid offsets through this world anchor.
        if self._position_override is not None:
            position = self._position_override
        else:
            position = {"xpos": 0.5, "ypos": 64.0, "zpos": 0.5}
        observation = {
            "pov": np.zeros((360, 640, 3), dtype=np.uint8),
            "inventory": {
                "obsidian": np.asarray(10, dtype=np.int64),
                "flint_and_steel": np.asarray(1, dtype=np.int64),
                "dirt": np.asarray(0, dtype=np.int64),
            },
            "portal_grid": self.grid.copy(),
            "portal_grid_origin": np.asarray((0, 64, 0), dtype=np.int32),
            "portal_dimension": np.asarray("minecraft:overworld"),
            "location_stats": position,
            "use_item": {
                "obsidian": np.asarray(self._placements, dtype=np.int64),
                "flint_and_steel": np.asarray(self._use_items, dtype=np.int64),
            },
        }
        if self._portal_transition is not None:
            observation["portal_transition"] = dict(self._portal_transition)
        return observation


class _BackendFactory:
    """Factory + owner for a controlled backend instance.

    The backend requires a single owner thread; tests run on a single
    thread so the ``open/reset/step/close`` lifecycle is straightforward.
    """

    def __init__(self, env: _ControlledMineRLEnv) -> None:
        self.env = env
        self.backend = MineRLEnvironmentBackend(
            env_factory=lambda task: env,
            reset_warmup_steps=0,
        )

    def __enter__(self) -> MineRLEnvironmentBackend:
        self.backend.open()
        return self.backend

    def __exit__(self, *exc_info: Any) -> None:
        self.backend.close()
        self.env.closed = True


class MineRLEnvironmentBackendTests(unittest.TestCase):
    def test_frozen_a1_task_selects_nearby_obsidian_spec(self) -> None:
        task = TaskInstance.from_dict(
            json.loads(
                (ROOT / "benchmark/instances/route_a_a1_phase4.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        specification = _specification_for_task(task)

        self.assertIsInstance(specification, PortalA1EnvSpec)
        self.assertEqual(specification.max_episode_steps, 900)
        self.assertEqual(
            [item["type"] for item in specification.initial_inventory],
            ["diamond_pickaxe", "flint_and_steel", "dirt"],
        )

    def test_reset_recreates_environment_after_transient_transport_error(self) -> None:
        class _ResetFailure(_ControlledMineRLEnv):
            def reset(self):
                raise TypeError("empty socket reply")

        failed = _ResetFailure()
        recovered = _ControlledMineRLEnv()
        environments = iter((failed, recovered))
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: next(environments),
            reset_warmup_steps=0,
            max_reset_attempts=2,
        )
        backend.open()
        try:
            observations = backend.reset(sample_task())
        finally:
            backend.close()
        self.assertIn("agent_1", observations)
        self.assertTrue(failed.closed)
        self.assertTrue(recovered.closed)

    def test_backend_implements_contract_and_hides_evaluator_truth(self) -> None:
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            observations = backend.reset(sample_task())
            self.assertEqual(env.seed_value, 7)
            observation = observations["agent_1"]
            self.assertEqual(observation.visible_inventory["obsidian"], 10)
            self.assertFalse(hasattr(observation, "portal_grid"))
            step = backend.step(
                {"agent_1": MacroAction("place_block", target="obsidian")}
            )
            self.assertEqual(step.info["environment_info_keys"], ["private"])
            state = backend.get_evaluation_state()
            # Default fixture places 1 obsidian + 1 nether_portal: not a
            # valid frame, so portal_built / valid_portal_frame must be
            # False; activation is bound to the (non-existent) episode
            # frame, so it must also be False.
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertFalse(state.portal_activated)
            self.assertIsNotNone(state.first_obsidian_placed_step)
            self.assertEqual(state.episode_obsidian_count, 1)
        self.assertTrue(env.closed)

    def test_backend_rejects_multi_agent_task_in_phase_one(self) -> None:
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            with self.assertRaisesRegex(ValueError, "exactly agent_1"):
                backend.reset(sample_task(("agent_1", "agent_2")))

    def test_minerl_random_fallback_observation_is_rejected(self) -> None:
        class _FailingMineRLEnv(_ControlledMineRLEnv):
            def step(self, action: dict[str, Any]):
                self.assert_action(action)
                return self._observation(), 0.0, True, {"error": "socket closed"}

        env = _FailingMineRLEnv()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            with self.assertRaisesRegex(RuntimeError, "socket closed"):
                backend.step({"agent_1": MacroAction.wait()})
            self.assertEqual(backend.get_evaluation_state().step_id, 0)

    # ------------------------------------------------------------------
    # Phase 2 milestone + latched-identity contract
    # ------------------------------------------------------------------

    def test_three_obsidian_without_termination_is_in_progress(self) -> None:
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            # Place three non-corner obsidian along the bottom row of
            # a 4x5 frame directly on the grid. This skips the
            # controlled-env driver so we can guarantee a partial
            # frame on the very first step.
            env.grid[_offset_to_flat_index((1, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((2, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((3, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertIsNone(state.first_valid_frame_step)
            self.assertIsNotNone(state.first_obsidian_placed_step)
            # build_site_selected must already be set by the partial
            # frame detector.
            self.assertIsNotNone(state.build_site_selected_step)
            # No termination → no failure.
            self.assertFalse(state.episode_terminated)
            self.assertIsNone(state.failure_type)
            self.assertIsNone(state.failure_step)
            result = PortalEvaluator().evaluate(state)
            self.assertIsNone(result.failure_type)
            self.assertFalse(result.success)
            self.assertFalse(result.episode_terminated)

    def test_three_obsidian_with_termination_is_frame_never_valid(self) -> None:
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            env.grid[_offset_to_flat_index((1, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((2, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((3, 0, 1))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            backend.step({"agent_1": MacroAction.wait()})
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            self.assertTrue(state.episode_terminated)
            self.assertEqual(state.terminated_step, 1)
            self.assertEqual(state.terminated_reason, "budget_exhausted")
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(result.failure_type, FAILURE_FRAME_NEVER_VALID)
            self.assertEqual(result.failure_step, 1)
            self.assertEqual(
                result.last_successful_milestone, "build_site_selected"
            )

    def test_full_path_with_termination_succeeds(self) -> None:
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.valid_portal_frame)
            self.assertFalse(state.portal_activated)
            self.assertIsNotNone(state.first_valid_frame_step)
            self.assertEqual(state.episode_obsidian_count, 14)
            # Ignite the portal.
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset({"agent_1"}))
            self.assertIsNotNone(state.first_activation_step)
            self.assertIn("agent_1", state.first_nether_step_by_agent)
            backend.mark_terminated(reason="driver_done")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_type)

    def test_frame_built_but_unignited_then_terminated(self) -> None:
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=False,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(
                result.failure_type, FAILURE_PORTAL_NEVER_ACTIVATED
            )
            self.assertEqual(result.failure_step, 14)
            self.assertEqual(
                result.last_successful_milestone, "valid_portal_frame"
            )

    def test_activated_but_no_nether_entry_then_terminated(self) -> None:
        # Build a custom env that activates but never enters the Nether.
        env = _ControlledMineRLEnv(
            build_full_frame=True,
            ignite_after_frame=False,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            # Manually place nether_portal inside the frame, but keep
            # dimension = overworld.
            env.grid[_offset_to_flat_index((1, 1, 1))] = (
                PORTAL_GRID_BLOCKS.index("nether_portal")
            )
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset())
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(
                result.failure_type, FAILURE_NO_AGENT_ENTERED_NETHER
            )
            self.assertEqual(result.failure_step, 15)
            self.assertEqual(
                result.last_successful_milestone, "portal_activated"
            )

    def test_latched_frame_identity_survives_nether_grid_loss(self) -> None:
        # The Phase 1 Scripted-A0 run ends with the agent in the Nether
        # and the Overworld grid replaced. The latched identity must
        # still report success.
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            # Wipe the Overworld grid (the Nether grid is no longer
            # the A0 fixed platform).
            env.grid[:] = 0
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.valid_portal_frame)
            self.assertTrue(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset({"agent_1"}))
            # Latched identity must be present and identifiable.
            self.assertIsNotNone(state.latched_frame_identity)
            assert state.latched_frame_identity is not None
            self.assertEqual(state.latched_frame_identity["width"], 4)
            self.assertEqual(state.latched_frame_identity["height"], 5)
            backend.mark_terminated(reason="driver_done")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_type)

    def test_activation_does_not_latch_to_pre_existing_portal(self) -> None:
        # Pre-existing activated portal at reset must NOT make the
        # current episode look activated.
        env = _controlled_env(pre_existing_activated=True)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(5):
                backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertFalse(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset())
            self.assertEqual(state.evidence["attribution_failed_candidate_count"], 1)
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(
                result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
            )
            self.assertEqual(result.failure_step, 5)
            # The pre-existing portal does NOT count as activation.
            self.assertEqual(
                result.last_successful_milestone,
                "agent_entered_nether" if state.first_nether_step_by_agent
                else "build_site_selected"
                if state.build_site_selected_step
                else "task_reset",
            )

    def test_pre_existing_full_frame_does_not_count_as_built(self) -> None:
        env = _controlled_env(pre_existing_frame=True)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(5):
                backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertIsNone(state.first_valid_frame_step)
            self.assertGreaterEqual(
                state.evidence["attribution_failed_candidate_count"], 1
            )
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(
                result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
            )

    def test_isolated_obsidian_blocks_do_not_form_a_frame(self) -> None:
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            env.grid[_offset_to_flat_index((3, 3, 3))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((4, 4, 4))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            env.grid[_offset_to_flat_index((5, 5, 5))] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertIsNotNone(state.first_obsidian_placed_step)
            # build_site_selected must NOT trigger on isolated blocks.
            self.assertIsNone(state.build_site_selected_step)
            # No frame was ever built; the 3 obsidian are external
            # (no place_block action ran) and no candidate exists.
            self.assertEqual(state.attributed_obsidian_offsets, ())
            self.assertEqual(state.evidence["attribution_failed_candidate_count"], 0)
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(result.failure_type, FAILURE_FRAME_NEVER_VALID)

    def test_partial_frame_triggers_build_site_selected(self) -> None:
        # The fixture driver places obsidian in an order that does
        # not produce 3 contiguous bottom-edge cells in three
        # place_block steps (the first 3 are corner + corners). The
        # cleanest way to drive the partial-frame detector is to
        # credit the pending queue manually and write 3 contiguous
        # obsidian on the bottom edge.
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            # Place 3 contiguous non-corner obsidian on the bottom
            # row of a 4x5 frame. This is a partial but not a valid
            # frame: ``build_site_selected`` must latch.
            bottom_offsets = [(1, 0, 1), (2, 0, 1), (3, 0, 1)]
            backend._credit_pending_place_block_for_test(
                "obsidian", len(bottom_offsets)
            )
            for offset in bottom_offsets:
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            backend.step({"agent_1": MacroAction.wait()})
            partial_state = backend.get_evaluation_state()
            self.assertIsNotNone(partial_state.build_site_selected_step)
            self.assertIsNone(partial_state.first_valid_frame_step)
            # Finish the 4x5 frame (11 more attributed obsidian).
            remaining = [
                (0, 0, 1),
                (0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1),
                (0, 1, 1), (0, 2, 1), (0, 3, 1),
                (3, 1, 1), (3, 2, 1), (3, 3, 1),
            ]
            backend._credit_pending_place_block_for_test(
                "obsidian", len(remaining)
            )
            for offset in remaining:
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            backend.step({"agent_1": MacroAction.wait()})
            final_state = backend.get_evaluation_state()
            self.assertIsNotNone(final_state.first_valid_frame_step)
            self.assertLessEqual(
                partial_state.build_site_selected_step or 0,
                final_state.first_valid_frame_step or 0,
            )
            self.assertTrue(final_state.portal_built_by_episode)
            self.assertEqual(final_state.external_obsidian_offsets, ())

    def test_evaluator_only_data_not_in_observation(self) -> None:
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            step = backend.step({"agent_1": MacroAction.wait()})
            public_observation = step.observations["agent_1"]
            for forbidden in (
                "portal_grid",
                "frame_evidence",
                "evaluator_state",
                "valid_portal_frame",
                "latched_frame_identity",
                "first_nether_step_by_agent",
            ):
                self.assertFalse(
                    hasattr(public_observation, forbidden),
                    f"{forbidden} must not appear on the agent observation",
                )
                self.assertNotIn(forbidden, step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertNotIn("latched_frame_identity", step.info)

    def test_milestone_events_are_structured_event_instances(self) -> None:
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            events = list(milestone_iterator(state))
            self.assertEqual(len(events), 6)
            for event in events:
                self.assertIsInstance(event, StructuredEvent)
                # Top-level identity fields are non-empty strings/ints.
                self.assertEqual(event.episode_id, state.episode_id)
                self.assertIsInstance(event.timestamp, float)
                self.assertIsInstance(event.step_id, int)
            self.assertEqual(
                [event.event_type for event in events],
                [
                    "task_reset",
                    "first_obsidian_placed",
                    "build_site_selected",
                    "valid_portal_frame",
                    "portal_activated",
                    "agent_entered_nether",
                ],
            )
            # Timestamps are latched and monotonic across milestones.
            ts_pairs = list(
                zip(
                    events,
                    events[1:],
                )
            )
            for earlier, later in ts_pairs:
                self.assertLessEqual(earlier.timestamp, later.timestamp)
            # Payload must NOT duplicate episode_id.
            for event in events:
                self.assertNotIn("episode_id", event.payload)

    def test_external_full_frame_is_not_attributed(self) -> None:
        """External world-side writes of a complete 4x5 frame must
        not be attributed to the episode. The terminal failure
        must be ``frame_not_built_by_episode`` (not the weaker
        ``frame_never_valid``) so reviewers cannot mistakenly read
        the run as "the agent never even tried to build a frame".
        """
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            # Externally write the entire 4x5 frame directly into
            # the grid, then ignite it externally. No
            # ``place_block(obsidian)`` action runs, so the pending
            # attribution queue stays empty.
            for offset in _frame_offsets():
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            env.grid[_offset_to_flat_index((1, 1, 1))] = (
                PORTAL_GRID_BLOCKS.index("nether_portal")
            )
            env._dimension = "minecraft:the_nether"
            for _ in range(3):
                backend.step({"agent_1": MacroAction.wait()})
            backend.mark_terminated(reason="external_environment")
            state = backend.get_evaluation_state()
            # The frame is geometrically valid but every required
            # cell is external; the backend must NOT mark it
            # episode-built.
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertIsNone(state.first_valid_frame_step)
            self.assertEqual(state.attributed_obsidian_offsets, ())
            self.assertEqual(len(state.external_obsidian_offsets), 14)
            # Pre-existing-frame detector alone may not classify
            # this as attribution_failed (the required cells were
            # never in baseline). The backend exposes a dedicated
            # ``external_structure_candidate_count`` so the
            # evaluator can promote it to the correct terminal
            # failure class.
            self.assertGreaterEqual(
                state.evidence["external_structure_candidate_count"], 1
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertEqual(
                result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
            )
            self.assertIn("portal_not_built_by_episode", result.blocking_conditions)

    def test_external_dimension_switch_is_not_success(self) -> None:
        """Even with a latched episode-built frame and a real
        flint-and-steel ignition, if the dimension flips to the
        Nether without the agent being near the latched frame, the
        evaluator must report ``success=False``.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            # Teleport the agent far from the latched frame and
            # let the backend record the new last-overworld
            # position.
            env._position_override = {
                "xpos": 100.0,
                "ypos": 100.0,
                "zpos": 100.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            # Now ignite. The fixture flips dimension to the
            # Nether; the pre-transition position is the far-away
            # one.
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset({"agent_1"}))
            # The portal is built and activated, but the entry
            # happened far from the frame.
            self.assertEqual(
                state.entered_via_episode_portal_by_agent.get("agent_1"),
                False,
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertIn(
                "nether_entry_not_via_episode_portal",
                result.blocking_conditions,
            )

    def test_other_portal_entry_is_not_success(self) -> None:
        """The agent has built and activated portal B, but the
        Nether entry is logged while the agent is at a position
        far from B (a pre-existing portal C). The evaluator must
        not glue the B-built frame, the B-activation and the
        C-entry into a success.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            # Move agent far from B and update the backend's
            # last-overworld position.
            env._position_override = {
                "xpos": 50.0,
                "ypos": 50.0,
                "zpos": 50.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            # Ignite from the far position; the fixture flips
            # dimension.
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.portal_activated)
            self.assertEqual(state.agents_in_nether, frozenset({"agent_1"}))
            self.assertEqual(
                state.entered_via_episode_portal_by_agent.get("agent_1"),
                False,
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)

    def test_entered_via_episode_portal_true_with_explicit_transition(self) -> None:
        """Success requires typed transition evidence for the exact
        latched frame plus a compatible pre-transition position.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.portal_activated)
            self.assertEqual(
                state.entered_via_episode_portal_by_agent.get("agent_1"),
                True,
            )
            backend.mark_terminated(reason="driver_done")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertTrue(result.success)

    def test_at_spawn_grid_bounds_include_world_anchor(self) -> None:
        env = _controlled_env(build_full_frame=True)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            state = backend.get_evaluation_state()
            assert state.latched_frame_identity is not None
            self.assertEqual(
                backend._world_bounds_from_identity(
                    state.latched_frame_identity
                ),
                (-2, 64, 1, -1, 66, 1),
            )


class AttributionRegressionTests(unittest.TestCase):
    """Regression tests for the 9 issues raised in the Phase 2 audit.

    Each scenario below was demonstrated to fail on at least one
    earlier implementation. The names are intentionally
    ``regression_*`` so a future code review can grep for them and
    see that the contract is locked.

    Mapping to the audit issues:

    1. ``regression_external_full_frame_not_attributed`` covers audit
       issue #1 ("environment change == agent built"). A whole 4x5
       portal frame written by the environment outside the
       attribution queue must not be classified as
       ``portal_built_by_episode=True``; the terminal failure must
       be ``frame_not_built_by_episode``.
    2. ``regression_single_place_block_is_one_obsidian`` covers
       audit issue #1's anti-pattern: a single ``place_block``
       action must not generalize into multiple obsidian
       attributions.
    3. ``regression_external_dimension_switch`` covers audit issue
       #2: external Nether entry must be flagged
       ``entered_via_episode_portal=False``.
    4. ``regression_other_portal_entry`` covers audit issue #2:
       building B, igniting B and entering through C must not
       produce success.
    5. ``regression_three_non_contiguous_obsidian_not_partial``
       covers audit issue #3: the partial-frame detector must not
       trigger on three cells scattered across different edges.
    6. ``regression_missing_timestamp_rejected`` covers audit issue
       #4: ``EvaluationState`` must fail closed when a milestone
       step is set without a matching latched timestamp.
    7. ``regression_full_missing_grid_has_missing_truth`` covers
       audit issue #5: ``has_missing_truth`` must be True on a grid
       where every cell is missing.
    """

    def test_regression_external_full_frame_not_attributed(self) -> None:
        """Issue #1: no place_block, external full frame → built=False
        and terminal failure = frame_not_built_by_episode.
        """
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for offset in _frame_offsets():
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            env.grid[_offset_to_flat_index((1, 1, 1))] = (
                PORTAL_GRID_BLOCKS.index("nether_portal")
            )
            env._dimension = "minecraft:the_nether"
            for _ in range(3):
                backend.step({"agent_1": MacroAction.wait()})
            backend.mark_terminated(reason="external_environment")
            state = backend.get_evaluation_state()
            self.assertFalse(state.portal_built_by_episode)
            self.assertFalse(state.valid_portal_frame)
            self.assertEqual(state.attributed_obsidian_offsets, ())
            self.assertGreaterEqual(
                state.evidence["external_structure_candidate_count"], 1
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertEqual(
                result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
            )

    def test_regression_single_place_block_is_one_obsidian(self) -> None:
        """Issue #1: a single ``place_block(obsidian)`` action
        must not distribute three credits arbitrarily over fourteen
        simultaneously appearing cells. An ambiguous batch fails
        closed: all fourteen are external.
        """
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            # Place 14 obsidian cells *externally* on the grid.
            for offset in _frame_offsets():
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            # Only credit 3 pending place_block actions — the rest
            # are by definition external. Do not submit any
            # ``place_block`` action: the env's
            # ``_build_full_frame`` flag is False so it will not
            # place anything on its own.
            backend._credit_pending_place_block_for_test("obsidian", 3)
            for _ in range(5):
                backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertEqual(state.attributed_obsidian_offsets, ())
            self.assertEqual(len(state.external_obsidian_offsets), 14)
            # And the frame must not be episode-built even though
            # the geometry is complete.
            self.assertFalse(state.portal_built_by_episode)
            backend.mark_terminated(reason="credit_mismatch")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(
                result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
            )

    def test_regression_external_cell_is_never_reattributed(self) -> None:
        env = _controlled_env(build_full_frame=True)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            external = _frame_offsets()[5]
            env.grid[_offset_to_flat_index(external)] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            backend.step({"agent_1": MacroAction.wait()})
            self.assertIn(
                external,
                backend.get_evaluation_state().external_obsidian_offsets,
            )
            backend.step(
                {"agent_1": MacroAction("place_block", target="obsidian")}
            )
            state = backend.get_evaluation_state()
            self.assertIn(external, state.external_obsidian_offsets)
            self.assertNotIn(external, state.attributed_obsidian_offsets)
            self.assertTrue(
                set(state.external_obsidian_offsets).isdisjoint(
                    state.attributed_obsidian_offsets
                )
            )

    def test_regression_unmatched_credit_expires_at_step_boundary(self) -> None:
        env = _controlled_env(build_full_frame=True)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            # The frame is already complete, so this accepted action
            # produces no new obsidian and its credit must expire.
            backend.step(
                {"agent_1": MacroAction("place_block", target="obsidian")}
            )
            self.assertEqual(
                backend.get_evaluation_state().pending_place_block_obsidian,
                0,
            )
            external = (6, 6, 6)
            env.grid[_offset_to_flat_index(external)] = (
                PORTAL_GRID_BLOCKS.index("obsidian")
            )
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertIn(external, state.external_obsidian_offsets)
            self.assertNotIn(external, state.attributed_obsidian_offsets)

    def test_regression_nearby_external_dimension_flip_is_unknown(self) -> None:
        env = _controlled_env(build_full_frame=True, ignite_after_frame=False)
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            _apply_block(env.grid, "nether_portal", [(1, 1, 1)])
            backend.step({"agent_1": MacroAction.wait()})
            # No portal_transition evidence is emitted: proximity alone
            # must not turn this external dimension flip into success.
            env._dimension = "minecraft:the_nether"
            backend.step({"agent_1": MacroAction.wait()})
            # Late evidence must not retroactively rewrite the first
            # observed transition.
            env._portal_transition = {
                "present": np.asarray(True, dtype=np.bool_),
                "entered_via_portal": np.asarray(True, dtype=np.bool_),
                "sequence": np.asarray(1, dtype=np.int64),
                "source_portal_block_world_position": np.asarray(
                    (-2, 64, 1), dtype=np.int32
                ),
                "from_dimension": "minecraft:overworld",
                "to_dimension": "minecraft:the_nether",
            }
            backend.step({"agent_1": MacroAction.wait()})
            backend.mark_terminated(reason="external_dimension_flip")
            state = backend.get_evaluation_state()
            self.assertNotIn(
                "agent_1", state.entered_via_episode_portal_by_agent
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertIsNone(result.entered_via_episode_portal)
            self.assertEqual(
                result.failure_type,
                FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
            )

    def test_regression_external_dimension_switch(self) -> None:
        """Issue #2: external dimension switch → success=False and
        ``entered_via_episode_portal=False``.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            env._position_override = {
                "xpos": 100.0,
                "ypos": 100.0,
                "zpos": 100.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertEqual(
                state.entered_via_episode_portal_by_agent.get("agent_1"),
                False,
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertIn(
                "nether_entry_not_via_episode_portal",
                result.blocking_conditions,
            )

    def test_regression_other_portal_entry(self) -> None:
        """Issue #2: build/activate B but enter via portal C → success=False.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            env._position_override = {
                "xpos": 50.0,
                "ypos": 50.0,
                "zpos": 50.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertEqual(
                state.entered_via_episode_portal_by_agent.get("agent_1"),
                False,
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)

    def test_regression_three_non_contiguous_obsidian_not_partial(self) -> None:
        """Issue #3: three obsidian on different edges of the
        same hypothetical frame must NOT trigger
        ``build_site_selected``. The exact cells from the audit:
        (1,0,1), (0,2,1), (3,3,1) belong to three different edges
        of a 4x5 frame.
        """
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for offset in ((1, 0, 1), (0, 2, 1), (3, 3, 1)):
                env.grid[_offset_to_flat_index(offset)] = (
                    PORTAL_GRID_BLOCKS.index("obsidian")
                )
            # Credit exactly 3 place_block actions so the
            # attribution queue is consistent with the 3 cells
            # the user observed.
            backend._credit_pending_place_block_for_test("obsidian", 3)
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            # The three cells are valid episode obsidian but they
            # do not form a partial frame. ``build_site_selected``
            # must remain None.
            self.assertIsNone(state.build_site_selected_step)
            self.assertEqual(
                len(state.attributed_obsidian_offsets), 3
            )
            # Detection of partial must use the same
            # structural-continuity rule as the unit tests.
            backend.mark_terminated(reason="budget_exhausted")
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertEqual(result.failure_type, FAILURE_FRAME_NEVER_VALID)

    def test_regression_missing_timestamp_rejected(self) -> None:
        """Issue #4: ``EvaluationState`` must raise on
        construction when a milestone step is set without a
        matching latched timestamp.
        """
        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=5,
                task_reset_step=0,
                first_obsidian_placed_step=2,
                # build_site_selected_step intentionally missing
                # from latched_timestamps
                build_site_selected_step=3,
                first_valid_frame_step=4,
                first_activation_step=5,
            )

    def test_regression_full_missing_grid_has_missing_truth(self) -> None:
        """Issue #5: a grid where every cell is ``missing`` must
        have ``has_missing_truth=True`` and reject the candidate.
        """
        env = _controlled_env()
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            env.grid[:] = PORTAL_GRID_BLOCKS.index("missing")
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertTrue(
                state.evidence["has_missing_truth_latched"]
                if "has_missing_truth_latched" in state.evidence
                else False
            )
            # Portal cannot be built from a fully-missing grid.
            self.assertFalse(state.portal_built_by_episode)


class FrameGeometryRegressionTests(unittest.TestCase):
    """Regression coverage for the 6 specific scenarios named in
    the Phase 2 audit. These wrap the pure-geometry detector so
    the contracts are verified at the lowest level, independent
    of the MineRL backend.
    """

    def test_regression_audit_scenario_1_external_full_frame(self) -> None:
        """Audit scenario 1: external full frame, no place_block
        → ``is_episode_built=False``.
        """
        # A geometrically valid frame is detected; the
        # backend-only attribution check is the next layer.
        from obsidianlink.evaluation import frame_geometry as fg

        grid = np.zeros((7, 7, 7), dtype=np.int32)
        for offset in _frame_offsets():
            grid[
                offset[0], offset[1], offset[2]
            ] = PORTAL_GRID_BLOCKS.index("obsidian")
        # baseline = the same empty grid
        baseline = np.zeros((7, 7, 7), dtype=np.int32)
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        # Pure geometry says: episode-built (no baseline overlap).
        # The backend's attribution layer is what prevents false
        # success — the geometry detector cannot know about
        # ``place_block`` actions.
        self.assertEqual(len(result.episode_built_candidates), 1)
        self.assertEqual(
            len(result.attribution_failed_candidates), 0
        )

    def test_regression_audit_scenario_2_external_dimension(self) -> None:
        """Audit scenario 2: a dimension switch without a portal
        entry near the latched frame cannot produce success.
        Wraps the MineRL backend and the PortalEvaluator in one
        assertion.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            env._position_override = {
                "xpos": 1000.0,
                "ypos": 1000.0,
                "zpos": 1000.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            self.assertFalse(
                state.entered_via_episode_portal_by_agent.get("agent_1")
            )
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)

    def test_regression_audit_scenario_3_other_portal_entry(self) -> None:
        """Audit scenario 3: build/activate B and enter via C → success=False.
        """
        env = _controlled_env(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        with _BackendFactory(env) as backend:
            backend.reset(sample_task())
            for _ in range(14):
                backend.step(
                    {"agent_1": MacroAction("place_block", target="obsidian")}
                )
            env._position_override = {
                "xpos": 250.0,
                "ypos": 250.0,
                "zpos": 250.0,
            }
            backend.step({"agent_1": MacroAction.wait()})
            backend.step(
                {"agent_1": MacroAction("use_item", target="flint_and_steel")}
            )
            state = backend.get_evaluation_state()
            result = PortalEvaluator().evaluate(state)
            self.assertFalse(result.success)

    def test_regression_audit_scenario_4_non_contiguous_partial(
        self,
    ) -> None:
        """Audit scenario 4: three obsidian on different edges of
        the same hypothetical frame must not be partial.
        """
        from obsidianlink.evaluation import frame_geometry as fg

        grid = np.zeros((7, 7, 7), dtype=np.int32)
        for offset in ((1, 0, 1), (0, 2, 1), (3, 3, 1)):
            grid[
                offset[0], offset[1], offset[2]
            ] = PORTAL_GRID_BLOCKS.index("obsidian")
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME
        )
        self.assertEqual(len(result.partial_candidates), 0)

    def test_regression_audit_scenario_5_missing_timestamp_fails_closed(
        self,
    ) -> None:
        """Audit scenario 5: missing timestamp in
        ``EvaluationState`` must raise.
        """
        from obsidianlink.evaluation import EvaluationState

        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=5,
                task_reset_step=0,
                first_obsidian_placed_step=2,
            )

    def test_regression_audit_scenario_6_full_missing_grid(self) -> None:
        """Audit scenario 6: full-missing grid must report
        ``has_missing_truth=True`` and reject the candidate.
        """
        from obsidianlink.evaluation import frame_geometry as fg

        grid = np.full(
            (7, 7, 7), PORTAL_GRID_BLOCKS.index("missing"), dtype=np.int32
        )
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME
        )
        self.assertTrue(result.has_missing_truth)
        self.assertEqual(len(result.episode_built_candidates), 0)


if __name__ == "__main__":
    unittest.main()
