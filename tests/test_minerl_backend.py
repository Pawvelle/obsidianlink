"""Tests for the MineRL environment backend with Phase 2 evaluator wiring."""

from __future__ import annotations

import json
import math
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
    PORTAL_GRID_MIN,
    PORTAL_GRID_SHAPE,
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

    def __init__(self, env: Any) -> None:
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


def _a1_task(agent_ids: tuple[str, ...] = ("agent_1",)) -> TaskInstance:
    """Build the frozen A1 task instance for offline tests."""
    return TaskInstance.from_dict(
        json.loads(
            (ROOT / "benchmark/instances/route_a_a1_phase4.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _a1_deposit_grid_offsets() -> list[tuple[int, int, int]]:
    """Return the 16 grid offsets for the A1 deposit at the spec spawn (0, 4, 0).

    The portal grid is anchored to the agent's spawn position with the
    atSpawn bounds ``PORTAL_GRID_MIN..PORTAL_GRID_MAX`` (in grid
    coordinates). For a spawn at world ``(0, 4, 0)`` the grid world
    origin is ``(-3, 3, 0)``, so the deposit cell at world
    ``(-3, 4, 3)`` sits at grid offset ``(0, 1, 3)``. The y_offset
    is therefore ``world_y - (anchor_y + PORTAL_GRID_MIN[1])``, not
    ``world_y - PORTAL_GRID_MIN[1]`` as in a world-only origin.
    """
    anchor = (0, 4, 0)
    offsets: list[tuple[int, int, int]] = []
    for wx in range(-3, 1):
        for wz in range(3, 7):
            offsets.append(
                (
                    wx - (anchor[0] + PORTAL_GRID_MIN[0]),
                    4 - (anchor[1] + PORTAL_GRID_MIN[1]),
                    wz - (anchor[2] + PORTAL_GRID_MIN[2]),
                )
            )
    return offsets


def _world_offset_for_grid(
    grid_offset: tuple[int, int, int],
    anchor: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Convert a grid offset into a world (x, y, z) block coordinate."""
    return (
        anchor[0] + grid_offset[0] + PORTAL_GRID_MIN[0],
        anchor[1] + grid_offset[1] + PORTAL_GRID_MIN[1],
        anchor[2] + grid_offset[2] + PORTAL_GRID_MIN[2],
    )


class _ControlledMineRL_A1Env:
    """Minimal A1 MineRL environment with the fixed obsidian deposit.

    The agent starts at world (0, 4, 0) with the canonical
    MineRL eye offset of +1.62. The deposit zone is a 4x1x4 obsidian
    slab at world y=4 between z=3 and z=6, fully inside the
    atSpawn grid that the A1 EnvSpec uses. Each ``attack=1`` step
    consults the agent's eye + view direction and, if a deposit
    cell lies within a tight cone, sets that cell to ``air`` and
    returns the post-step observation with an updated grid. The
    fixture is intentionally simple — it does not attempt to
    simulate Minecraft pathing, jumping, or chunk loading.
    """

    def __init__(
        self,
        *,
        anchor: tuple[int, int, int] = (0, 4, 0),
        strip_external: bool = False,
    ) -> None:
        self.action_space = PortalA0EnvSpec().action_space
        self.seed_value: int | None = None
        self.closed = False
        self.steps = 0
        self._anchor = anchor
        self._strip_external = strip_external
        self._deposit_offsets = tuple(_a1_deposit_grid_offsets())
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        # Pre-populate the deposit zone with obsidian.
        for offset in self._deposit_offsets:
            flat = (
                offset[1] * PORTAL_GRID_SHAPE[0] * PORTAL_GRID_SHAPE[2]
                + offset[2] * PORTAL_GRID_SHAPE[0]
                + offset[0]
            )
            self.grid[flat] = PORTAL_GRID_BLOCKS.index("obsidian")
        # Agent state (eye, yaw, pitch).
        self._eye = (anchor[0] + 0.5, anchor[1] + 1.62, anchor[2] + 0.5)
        self._yaw = 0.0
        self._pitch = 0.0
        self._moves: list[tuple[float, float]] = [(0.0, 0.0)]
        self._attack_attempts = 0
        self._successful_mines: list[tuple[int, int, int]] = []
        self._mismatched_mines: int = 0
        self._last_action_summary: dict[str, Any] = {}

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, Any]:
        return self._observation()

    def step(self, action: dict[str, Any]):
        self.assert_action(action)
        self.steps += 1
        action_map = dict(action) if isinstance(action, Mapping) else {}
        previous_eye = self._eye
        previous_yaw = self._yaw
        previous_pitch = self._pitch
        camera = action_map.get("camera")
        if camera is not None:
            try:
                pitch_value = float(np.asarray(camera).reshape(-1)[0])
                yaw_value = float(np.asarray(camera).reshape(-1)[1])
            except (TypeError, ValueError, IndexError):
                pitch_value = 0.0
                yaw_value = 0.0
            self._pitch = max(-90.0, min(90.0, self._pitch + pitch_value))
            self._yaw = ((self._yaw + yaw_value + 180.0) % 360.0) - 180.0
        forward = int(action_map.get("forward", 0))
        back = int(action_map.get("back", 0))
        if forward or back:
            yaw_rad = math.radians(self._yaw)
            # The A1 driver plans its look angles assuming 1 block
            # per ``forward=1`` step (it pre-computes ``approach_eye``
            # with the same convention). The fixture uses the same
            # rate so the dot-product cone check inside
            # ``_mine_targeted_cell`` matches the driver's intent.
            distance = 1.0 * (forward - back)
            self._eye = (
                self._eye[0] - math.sin(yaw_rad) * distance,
                self._eye[1],
                self._eye[2] + math.cos(yaw_rad) * distance,
            )
        attack = int(action_map.get("attack", 0))
        mined_cell: tuple[int, int, int] | None = None
        if attack:
            self._attack_attempts += 1
            mined_cell = self._mine_targeted_cell()
            if mined_cell is not None:
                self._successful_mines.append(mined_cell)
            else:
                self._mismatched_mines += 1
        observation = self._observation()
        self._last_action_summary = {
            "previous_eye": previous_eye,
            "previous_yaw": previous_yaw,
            "previous_pitch": previous_pitch,
            "new_eye": self._eye,
            "new_yaw": self._yaw,
            "new_pitch": self._pitch,
            "attack": bool(attack),
            "moved": bool(forward or back),
            "mined_cell": (
                list(mined_cell) if mined_cell is not None else None
            ),
        }
        return observation, 0.0, False, {"private": "not_forwarded"}

    def _mine_targeted_cell(self) -> tuple[int, int, int] | None:
        """Return the grid offset mined by this ``attack`` (or None)."""
        if not self._deposit_offsets:
            return None
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)
        cos_pitch = math.cos(pitch_rad)
        # MineRL/Malmo forward vector: yaw=0 is +z, yaw=90° is -x.
        view_dir = (
            -math.sin(yaw_rad) * cos_pitch,
            -math.sin(pitch_rad),
            math.cos(yaw_rad) * cos_pitch,
        )
        best_offset: tuple[int, int, int] | None = None
        best_dot = -2.0
        for offset in self._deposit_offsets:
            world = _world_offset_for_grid(offset, self._anchor)
            # Aim at the top face of the cell, which is the surface
            # the agent actually faces when looking down at a
            # deposit block from the side. The driver computes
            # look angles to this exact point.
            target = (
                float(world[0]) + 0.5,
                float(world[1]) + 1.0,
                float(world[2]) + 0.5,
            )
            delta = (
                target[0] - self._eye[0],
                target[1] - self._eye[1],
                target[2] - self._eye[2],
            )
            norm = math.sqrt(sum(value * value for value in delta))
            if norm < 0.05:
                continue
            to_cell = tuple(value / norm for value in delta)
            dot = sum(a * b for a, b in zip(view_dir, to_cell))
            if dot > best_dot:
                best_dot = dot
                best_offset = offset
        # The driver's look helper may produce a pitch that is
        # slightly off the cell-top vector (because the agent stands
        # in front of the slab and the top face is offset 0.5 above
        # the cell center). A 0.95 dot-product cone (~18°) is the
        # smallest threshold that still accepts the cell that the
        # driver actually aims at.
        if best_dot < 0.95 or best_offset is None:
            return None
        flat = (
            best_offset[1] * PORTAL_GRID_SHAPE[0] * PORTAL_GRID_SHAPE[2]
            + best_offset[2] * PORTAL_GRID_SHAPE[0]
            + best_offset[0]
        )
        if self.grid[flat] != PORTAL_GRID_BLOCKS.index("obsidian"):
            return None
        self.grid[flat] = PORTAL_GRID_BLOCKS.index("air")
        return best_offset

    def assert_action(self, action: dict[str, Any]) -> None:
        if not self.action_space.contains(action):
            raise AssertionError("invalid test action")

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> dict[str, Any]:
        position = {
            "xpos": float(self._eye[0]),
            "ypos": float(self._eye[1]),
            "zpos": float(self._eye[2]),
            "yaw": float(self._yaw),
            "pitch": float(self._pitch),
        }
        observation: dict[str, Any] = {
            "pov": np.zeros((360, 640, 3), dtype=np.uint8),
            "inventory": {
                "diamond_pickaxe": np.asarray(1, dtype=np.int64),
                "flint_and_steel": np.asarray(1, dtype=np.int64),
                "dirt": np.asarray(2, dtype=np.int64),
            },
            "portal_grid": self.grid.copy(),
            "portal_grid_origin": np.asarray(self._anchor, dtype=np.int32),
            "portal_dimension": np.asarray("minecraft:overworld"),
            "location_stats": position,
            "use_item": {
                "obsidian": np.asarray(0, dtype=np.int64),
                "flint_and_steel": np.asarray(0, dtype=np.int64),
            },
        }
        if self._strip_external:
            observation["__strip_external_after_steps"] = self.steps
        return observation


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


# ----------------------------------------------------------------------
# Phase 4 A1 mining-slice contract
# ----------------------------------------------------------------------


class Phase4A1MiningContractTests(unittest.TestCase):
    """Offline contract tests for the A1 mining-slice backend.

    The tests in this class only validate the *backend* contract. The
    full ``run_scripted_a1`` driver is exercised separately in
    ``tests/test_scripted_a1.py``; here we focus on:

    * the deposit zone is populated and large enough to satisfy the
      14-obsidian quota;
    * the initial inventory has no obsidian;
    * ``equip_item(diamond_pickaxe)`` translates to ``hotbar.1``;
    * ``mine_target(obsidian)`` translates to ``attack=1`` and
      latches ``obsidian_source_located_step``;
    * the mining milestones (``first_obsidian_mined_step`` and
      ``obsidian_quota_collected_step``) only latch on actual grid
      deltas;
    * the agent observation never exposes evaluator-only state;
    * external non-attributed mining deltas never satisfy the quota.
    """

    def test_a1_task_instance_starts_with_no_obsidian_in_inventory(self) -> None:
        task = _a1_task()
        inventory = task.initial_inventories["agent_1"]
        self.assertNotIn("obsidian", inventory)
        self.assertEqual(inventory["diamond_pickaxe"], 1)
        self.assertEqual(inventory["flint_and_steel"], 1)
        self.assertEqual(inventory["dirt"], 2)
        self.assertEqual(
            task.scenario_parameters["obsidian_required"], 14
        )

    def test_a1_deposit_zone_baseline_contains_at_least_14_obsidian(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            state = backend.get_evaluation_state()
            evidence = state.evidence["a1_mining_evidence"]
            self.assertGreaterEqual(
                evidence["baseline_deposit_obsidian_count"], 14
            )
            self.assertGreaterEqual(
                state.obsidian_quota_required, 14
            )
            self.assertEqual(state.obsidian_mined_count, 0)
            self.assertEqual(state.obsidian_source_located_step, None)
            self.assertEqual(state.first_obsidian_mined_step, None)
            self.assertEqual(state.obsidian_quota_collected_step, None)

    def test_a1_equip_diamond_pickaxe_translates(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            step = backend.step(
                {
                    "agent_1": MacroAction(
                        "equip_item", target="diamond_pickaxe"
                    )
                }
            )
            self.assertTrue(step.info["translation_accepted"])
            self.assertIsNone(step.info["translation_error"])

    def test_a1_mine_target_without_grid_change_does_not_credit(self) -> None:
        """A ``mine_target(obsidian)`` aimed away from the deposit
        must NOT latch ``first_obsidian_mined``. ``obsidian_source_located``
        still fires on intent (the agent's intent is recorded), but
        without an actual grid delta the mining counter stays at
        zero and the source-located step is the only mining
        milestone that latches.
        """
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            # Turn the agent 180° to face -z, away from the deposit
            # which lives at z=3..6. The controlled env's view
            # cone then contains no deposit cell.
            backend.step(
                {
                    "agent_1": MacroAction(
                        "look", parameters={"yaw": 180.0, "pitch": 0.0}
                    )
                }
            )
            step = backend.step(
                {
                    "agent_1": MacroAction(
                        "mine_target", target="obsidian"
                    )
                }
            )
            self.assertTrue(step.info["translation_accepted"])
            state = backend.get_evaluation_state()
            # Source-located fires on intent (action accepted); but
            # no grid delta means first_obsidian_mined stays None
            # and the count remains 0.
            self.assertIsNotNone(state.obsidian_source_located_step)
            self.assertIsNone(state.first_obsidian_mined_step)
            self.assertEqual(state.obsidian_mined_count, 0)
            self.assertEqual(state.obsidian_mined_offsets, ())

    def test_a1_first_successful_mine_latches_first_mined_milestone(self) -> None:
        from obsidianlink.drivers.scripted_a1 import _world_to_local_angles, AGENT_EYE

        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            # Move forward 2 ticks (the driver walks to z=2, eye at z=1.5).
            for _ in range(2):
                backend.step(
                    {"agent_1": MacroAction("move", parameters={"forward": 1.0})}
                )
            # Look at the closest deposit cell top (world (0, 4, 3) → y=5).
            # After 2 forward steps (1 block per step) the eye sits at
            # (0.5, 5.62, 2.5), so we aim at the cell top in front.
            approach_eye = (AGENT_EYE[0], AGENT_EYE[1], AGENT_EYE[2] + 2.0)
            yaw, pitch = _world_to_local_angles(
                (0.5, 5.0, 3.5), eye=approach_eye
            )
            backend.step(
                {
                    "agent_1": MacroAction(
                        "look", parameters={"yaw": yaw, "pitch": pitch}
                    )
                }
            )
            backend.step(
                {
                    "agent_1": MacroAction(
                        "mine_target", target="obsidian"
                    )
                }
            )
            state = backend.get_evaluation_state()
            self.assertEqual(state.obsidian_mined_count, 1)
            self.assertIsNotNone(state.first_obsidian_mined_step)
            self.assertIsNotNone(state.obsidian_source_located_step)
            # Source-located must precede or equal first-mined.
            # The two are typically recorded on the same step
            # because the backend increments ``pending_mine_obsidian``
            # and runs ``_refresh_evaluation_milestones`` in the
            # same ``step()`` call.
            self.assertLessEqual(
                state.obsidian_source_located_step,
                state.first_obsidian_mined_step,
            )
            # Quota still unmet.
            self.assertIsNone(state.obsidian_quota_collected_step)

    def test_a1_quota_latches_after_14_attributed_mines(self) -> None:
        from obsidianlink.drivers.scripted_a1 import build_mining_action_plan

        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            plan = build_mining_action_plan(quota=14)
            for item in plan:
                backend.step({"agent_1": item.action})
            state = backend.get_evaluation_state()
            self.assertGreaterEqual(state.obsidian_mined_count, 14)
            self.assertIsNotNone(state.obsidian_quota_collected_step)
            self.assertEqual(state.obsidian_mined_count, 14)
            self.assertEqual(state.external_mined_offsets, ())
            self.assertEqual(state.pending_mine_obsidian, 0)
            self.assertEqual(
                state.evidence["a1_mining_evidence"][
                    "obsidian_mined_count"
                ],
                14,
            )

    def test_a1_external_mining_does_not_credit_quota(self) -> None:
        """Strip a deposit cell via a direct grid mutation (bypassing
        any ``mine_target`` action). The exact-count rule must fail
        closed to ``external_mined_offsets`` and the quota must
        stay unmet.
        """
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            # Mutate the grid directly: the bridge / environment
            # would have to do this for an external mining delta to
            # appear.
            deposit = _a1_deposit_grid_offsets()
            flat = (
                deposit[0][1] * PORTAL_GRID_SHAPE[0] * PORTAL_GRID_SHAPE[2]
                + deposit[0][2] * PORTAL_GRID_SHAPE[0]
                + deposit[0][0]
            )
            env.grid[flat] = PORTAL_GRID_BLOCKS.index("air")
            backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertEqual(state.obsidian_mined_count, 0)
            self.assertEqual(
                tuple(state.external_mined_offsets)[:1],
                (deposit[0],),
            )
            self.assertIsNone(state.obsidian_quota_collected_step)
            self.assertIsNone(state.first_obsidian_mined_step)

    def test_a1_burst_mines_without_credit_do_not_latch_quota(self) -> None:
        """Mine without ever issuing ``mine_target``: the grid never
        changes, so neither first_mined nor quota_collected latches.
        """
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            for _ in range(20):
                backend.step({"agent_1": MacroAction.wait()})
            state = backend.get_evaluation_state()
            self.assertEqual(state.obsidian_mined_count, 0)
            self.assertIsNone(state.obsidian_source_located_step)
            self.assertIsNone(state.first_obsidian_mined_step)
            self.assertIsNone(state.obsidian_quota_collected_step)

    def test_a1_observation_never_exposes_evaluator_truth(self) -> None:
        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            observations = backend.reset(_a1_task())
            observation = observations["agent_1"]
            for forbidden in (
                "portal_grid",
                "obsidian_mined_offsets",
                "obsidian_quota_required",
                "obsidian_source_located_step",
                "first_obsidian_mined_step",
                "obsidian_quota_collected_step",
                "a1_mining_evidence",
            ):
                self.assertFalse(
                    hasattr(observation, forbidden),
                    f"{forbidden} must not appear on the agent observation",
                )
            step = backend.step(
                {
                    "agent_1": MacroAction(
                        "mine_target", target="obsidian"
                    )
                }
            )
            self.assertNotIn("obsidian_mined_count", step.info)
            self.assertNotIn("a1_mining_evidence", step.info)

    def test_a1_milestone_events_emit_three_mining_milestones(self) -> None:
        from obsidianlink.drivers.scripted_a1 import build_mining_action_plan

        env = _ControlledMineRL_A1Env()
        with _BackendFactory(env) as backend:
            backend.reset(_a1_task())
            plan = build_mining_action_plan(quota=14)
            for item in plan:
                backend.step({"agent_1": item.action})
            backend.mark_terminated(
                step_id=backend.get_evaluation_state().step_id,
                reason="scripted_a1_slice_complete",
            )
            state = backend.get_evaluation_state()
            event_types = [event.event_type for event in state.milestone_events()]
            self.assertIn("task_reset", event_types)
            self.assertIn("obsidian_source_located", event_types)
            self.assertIn("first_obsidian_mined", event_types)
            self.assertIn("obsidian_quota_collected", event_types)
            # First mining must precede source-located; quota-collected
            # must follow first-mined and source-located.
            def _index(name: str) -> int:
                return next(
                    i
                    for i, event in enumerate(state.milestone_events())
                    if event.event_type == name
                )

            self.assertLess(
                _index("obsidian_source_located"),
                _index("first_obsidian_mined"),
            )
            self.assertLess(
                _index("first_obsidian_mined"),
                _index("obsidian_quota_collected"),
            )


if __name__ == "__main__":
    unittest.main()
