from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from obsidianlink.actions.minerl_translator import translate_macro_action
from obsidianlink.core.types import BackendStep, MacroAction, Observation
from obsidianlink.env.integration.e7_config import build_e7_compatibility_task
from obsidianlink.env.integration.e8_adapter import (
    MineRLE8BlockTruthAdapter,
    server_truth_snapshot,
)
from obsidianlink.env.integration.e8_config import (
    E8_AGENT_ID,
    E8_PROBE_WORLD_CELLS,
    E8_STIMULUS_BLOCK,
    build_e8_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
)
from obsidianlink.env.validation import E8_SERVER_BLOCK_TRUTH_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.truth import BlockTruthActionExecution, inspect_block_truth
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e8-adapter-episode"
KNOWN_SPAWN = (0, 4, 0)
KNOWN_WORLD = ((0, 4, 1), (1, 4, 1), (-1, 4, 1))
KNOWN_GRID = ((0, 0, 1), (1, 0, 1), (-1, 0, 1))
MISTAKEN_WORLD_AS_GRID = (0, 4, 1)


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    return x + x_size * z + x_size * z_size * y


class _Backend:
    instances = []

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.blocks = ["air", "air", "air"]
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E8_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E8_AGENT_ID,
                step_id=0,
                frame="drop",
                visible_inventory={"dirt": 1},
                selected_item="dirt",
                messages=("drop",),
                workflow_stage="drop",
            )
        }

    def get_server_truth_snapshot(self, cells):
        records = []
        for index, cell in enumerate(cells):
            records.append(
                {
                    "block": self.blocks[index],
                    "grid_cell": list(KNOWN_GRID[index]),
                    "world_cell": list(cell),
                }
            )
        return {
            "agent_id": E8_AGENT_ID,
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "grid_anchor_world": list(KNOWN_SPAWN),
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        self.step_id = 1
        self.blocks = ["dirt", "air", "air"]
        obs = Observation(EPISODE, E8_AGENT_ID, 1, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            1,
            {E8_AGENT_ID: obs},
            {E8_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class _TruthEnv(_ControlledMineRLEnv):
    def __init__(
        self,
        *,
        after_target: str | None = "dirt",
        after_right: str = "air",
        after_left: str = "air",
        grid_origin: tuple[int, int, int] | None = KNOWN_SPAWN,
        drop_grid: bool = False,
        unknown_target: bool = False,
    ):
        super().__init__()
        self.after_target = after_target
        self.after_right = after_right
        self.after_left = after_left
        self._grid_origin = grid_origin
        self.drop_grid = drop_grid
        self.unknown_target = unknown_target
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        for cell in KNOWN_GRID:
            self.grid[_flat_index(cell)] = PORTAL_GRID_BLOCKS.index("air")
        self.grid[_flat_index(MISTAKEN_WORLD_AS_GRID)] = PORTAL_GRID_BLOCKS.index("grass")

    def _observation(self):
        observation = super()._observation()
        observation["location_stats"] = {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}
        observation["portal_dimension"] = np.asarray("minecraft:overworld")
        if self._grid_origin is None:
            observation.pop("portal_grid_origin", None)
        else:
            observation["portal_grid_origin"] = np.asarray(
                self._grid_origin, dtype=np.int32
            )
        if self.drop_grid:
            observation.pop("portal_grid", None)
        return observation

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("use", 0)) and int(action_map.get("hotbar.1", 0)):
            if self.unknown_target:
                self.grid[_flat_index(KNOWN_GRID[0])] = PORTAL_GRID_BLOCKS.index("other")
            elif self.after_target is not None:
                self.grid[_flat_index(KNOWN_GRID[0])] = PORTAL_GRID_BLOCKS.index(
                    self.after_target
                )
            self.grid[_flat_index(KNOWN_GRID[1])] = PORTAL_GRID_BLOCKS.index(
                self.after_right
            )
            self.grid[_flat_index(KNOWN_GRID[2])] = PORTAL_GRID_BLOCKS.index(
                self.after_left
            )
        observation = self._observation()
        info = {
            "location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5},
            "secret": "not-public",
        }
        return observation, 0.0, False, info


class E8MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_and_minimal(self):
        task = build_e8_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E8_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E8_AGENT_ID], {"dirt": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E8")
        self.assertTrue(task.scenario_parameters["not_a_benchmark_task"])
        self.assertTrue(task.scenario_parameters["calibration_only"])

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E8_SERVER_BLOCK_TRUTH_CASE,
            MineRLE8BlockTruthAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        payload = result.as_dict()
        self.assertNotIn("portal_grid", payload)
        self.assertEqual(payload["outcome"], "block_truth_ok")

    def test_adapter_rejects_evaluator_truth_in_backend_info(self):
        class Leaky(_Backend):
            def step(self, actions):
                self.calls.append("step")
                self.step_id = 1
                self.blocks = ["dirt", "air", "air"]
                obs = Observation(EPISODE, E8_AGENT_ID, 1, 0.0, frame="not-used")
                return BackendStep(
                    EPISODE,
                    1,
                    {E8_AGENT_ID: obs},
                    {E8_AGENT_ID: 0.0},
                    False,
                    False,
                    {"translation_accepted": True, "portal_grid": "must-not-leak"},
                )

        result = EnvironmentValidationRunner().run(
            E8_SERVER_BLOCK_TRUTH_CASE,
            MineRLE8BlockTruthAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=Leaky
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "truth_leak")

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE8BlockTruthAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("place_block", target="dirt")
        adapter.execute_truth_stimulus(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_truth_stimulus(action)
        adapter.close()
        adapter = MineRLE8BlockTruthAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_truth_stimulus(MacroAction("move"))
        with self.assertRaises(ValueError):
            adapter.execute_truth_stimulus(MacroAction("place_block", target="obsidian"))
        adapter.close()

    def test_snapshot_rejects_missing_extra_malformed_and_unknown(self):
        base = _Backend().get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
        for value in (
            {k: v for k, v in base.items() if k != "block_truth"},
            {**base, "inventory": {}},
            {**base, "position_world": None},
            {**base, "dimension": "unknown"},
            {**base, "block_truth": [{**base["block_truth"][0], "block": "other"}]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server_truth_snapshot(value)

    def test_snapshot_accepts_python_and_numpy_integer_coordinates(self):
        base = _Backend().get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
        python_ints = server_truth_snapshot(
            {**base, "grid_anchor_world": [0, 4, 0]}
        )
        self.assertEqual(python_ints.grid_anchor_world, (0, 4, 0))
        numpy_anchor = server_truth_snapshot(
            {
                **base,
                "grid_anchor_world": [
                    np.asarray(0, dtype=np.int32),
                    np.asarray(4, dtype=np.int64),
                    np.asarray(0, dtype=np.int32),
                ],
            }
        )
        self.assertEqual(numpy_anchor.grid_anchor_world, (0, 4, 0))
        records = [dict(item) for item in base["block_truth"]]
        records[0]["world_cell"] = [
            np.asarray(0, dtype=np.int32),
            np.asarray(4, dtype=np.int64),
            np.asarray(1, dtype=np.int32),
        ]
        records[0]["grid_cell"] = [
            np.asarray(0, dtype=np.int32),
            np.asarray(0, dtype=np.int64),
            np.asarray(1, dtype=np.int32),
        ]
        numpy_cells = server_truth_snapshot({**base, "block_truth": records})
        self.assertEqual(numpy_cells.block_truth[0].world_cell, (0, 4, 1))
        self.assertEqual(numpy_cells.block_truth[0].grid_cell, (0, 0, 1))

    def test_snapshot_rejects_non_integer_coordinate_coercion(self):
        base = _Backend().get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
        illegal_anchors = (
            [0.0, 4.0, 0.0],
            [0.5, 4.0, 0.0],
            [False, 4, 0],
            ["0", 4, 0],
        )
        for anchor in illegal_anchors:
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                server_truth_snapshot({**base, "grid_anchor_world": anchor})
        illegal_world = (
            [0.0, 4.0, 1.0],
            [0.5, 4.0, 1.0],
            [False, 4, 1],
            ["0", 4, 1],
        )
        for world in illegal_world:
            records = [dict(item) for item in base["block_truth"]]
            records[0]["world_cell"] = world
            with self.subTest(world=world), self.assertRaises(ValueError):
                server_truth_snapshot({**base, "block_truth": records})

    def test_truth_missing_count_must_match_missing_records_exactly(self):
        base = _Backend().get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
        zero = server_truth_snapshot({**base, "truth_missing_count": 0})
        self.assertEqual(zero.truth_missing_count, 0)
        records = [dict(item) for item in base["block_truth"]]
        records[0]["block"] = "missing"
        consistent = server_truth_snapshot(
            {**base, "block_truth": records, "truth_missing_count": 1}
        )
        self.assertEqual(consistent.truth_missing_count, 1)
        self.assertEqual(len(consistent.block_truth), 2)
        after = server_truth_snapshot({**base, "step_id": 1})
        inspection = inspect_block_truth(
            consistent,
            after,
            BlockTruthActionExecution(
                EPISODE, E8_AGENT_ID, 1, "place_block", "dirt", 1, True, 1
            ),
            probe_world_cells=KNOWN_WORLD,
            probe_grid_cells=KNOWN_GRID,
            expected_before_blocks={cell: "air" for cell in KNOWN_WORLD},
            expected_after_blocks={
                KNOWN_WORLD[0]: "dirt",
                KNOWN_WORLD[1]: "air",
                KNOWN_WORLD[2]: "air",
            },
            target_world_cell=KNOWN_WORLD[0],
            control_world_cells=KNOWN_WORLD[1:],
            duration_ticks=1,
            stimulus_target="dirt",
        )
        self.assertNotEqual(inspection.outcome, "block_truth_ok")
        self.assertEqual(inspection.outcome, "truth_block_missing")
        mismatches = (
            ({**base, "truth_missing_count": 1},),
            ({**base, "block_truth": records, "truth_missing_count": 0},),
            ({**base, "block_truth": records, "truth_missing_count": 2},),
        )
        for (payload,) in mismatches:
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "truth_missing_count does not match block_truth records"
            ):
                server_truth_snapshot(payload)

    def test_real_backend_truth_comes_from_grid_not_action_target(self):
        env = _TruthEnv(after_target=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            before = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual([item["block"] for item in before["block_truth"]], ["air", "air", "air"])
            step = backend.step(
                {E8_AGENT_ID: MacroAction("place_block", target=E8_STIMULUS_BLOCK)}
            )
            after = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual([item["block"] for item in after["block_truth"]], ["air", "air", "air"])
            self.assertNotIn("block_truth", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertNotIn("server_truth", step.info)
            self.assertFalse(hasattr(step.observations[E8_AGENT_ID], "block_truth"))
            self.assertFalse(hasattr(step.observations[E8_AGENT_ID], "portal_grid"))
        finally:
            backend.close()

    def test_snapshot_follows_raw_grid_without_copying_action_intent(self):
        env = _TruthEnv(after_target=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            backend._latest_raw["portal_grid"][_flat_index(KNOWN_GRID[0])] = (
                PORTAL_GRID_BLOCKS.index("dirt")
            )
            after = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual([item["block"] for item in after["block_truth"]], ["dirt", "air", "air"])
        finally:
            backend.close()

    def test_real_backend_observes_only_the_target_cell_change(self):
        env = _TruthEnv(after_target="dirt")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            backend.step({E8_AGENT_ID: MacroAction("place_block", target="dirt")})
            after = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual([item["block"] for item in after["block_truth"]], ["dirt", "air", "air"])
            self.assertEqual(after["dimension"], "minecraft:overworld")
            self.assertEqual(after["position_world"], [0.5, 4.0, 0.5])
            self.assertEqual(after["anchor_source"], "portal_grid_origin")
            self.assertEqual(backend._hotbar_mapping["dirt"], "hotbar.1")
        finally:
            backend.close()

    def test_wrong_flat_index_and_xz_swap_are_caught(self):
        env = _TruthEnv(after_target=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            raw_grid = backend._latest_raw["portal_grid"]
            raw_grid[_flat_index(MISTAKEN_WORLD_AS_GRID)] = PORTAL_GRID_BLOCKS.index("dirt")
            after_wrong_index = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual(after_wrong_index["block_truth"][0]["block"], "air")
            raw_grid[_flat_index(KNOWN_GRID[1])] = PORTAL_GRID_BLOCKS.index("dirt")
            swapped = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual(swapped["block_truth"][0]["block"], "air")
            self.assertEqual(swapped["block_truth"][1]["block"], "dirt")
        finally:
            backend.close()

    def test_origin_mismatch_out_of_bounds_empty_and_duplicate_fail_closed(self):
        env = _TruthEnv(grid_origin=(0, 64, 0))
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            with self.assertRaisesRegex(ValueError, "grid anchor differs from spawn"):
                backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
        finally:
            backend.close()
        env = _TruthEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            with self.assertRaisesRegex(ValueError, "outside the evaluator grid"):
                backend.get_server_truth_snapshot(((0, 4, 20),))
            with self.assertRaisesRegex(ValueError, "empty"):
                backend.get_server_truth_snapshot(())
            with self.assertRaisesRegex(ValueError, "duplicate world cell"):
                backend.get_server_truth_snapshot(((0, 4, 1), (0, 4, 1)))
        finally:
            backend.close()

    def test_missing_grid_origin_falls_back_and_missing_grid_is_truth_missing(self):
        env = _TruthEnv(after_target="dirt", grid_origin=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            before = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual(before["anchor_source"], "expected_spawn_fallback")
            backend.step({E8_AGENT_ID: MacroAction("place_block", target="dirt")})
            after = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            self.assertEqual(after["block_truth"][0]["block"], "dirt")
        finally:
            backend.close()
        env = _TruthEnv(drop_grid=True)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            self.assertIsNone(backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS))
        finally:
            backend.close()

    def test_unknown_block_fails_closed_and_e6_e7_compatibility_holds(self):
        env = _TruthEnv(unknown_target=True)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e8_compatibility_task(EPISODE))
            backend.step({E8_AGENT_ID: MacroAction("place_block", target="dirt")})
            payload = backend.get_server_truth_snapshot(E8_PROBE_WORLD_CELLS)
            with self.assertRaisesRegex(ValueError, "unknown block truth"):
                server_truth_snapshot(payload)
            placement = backend.get_block_placement_truth((0, 4, 1))
            self.assertEqual(placement["block"], "other")
            self.assertEqual(
                set(placement),
                {"episode_id", "agent_id", "step_id", "x", "y", "z", "block"},
            )
        finally:
            backend.close()
        env = _TruthEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "water"))
            env.grid[_flat_index(KNOWN_GRID[0])] = PORTAL_GRID_BLOCKS.index("flowing_water")
            backend._latest_raw["portal_grid"][_flat_index(KNOWN_GRID[0])] = (
                PORTAL_GRID_BLOCKS.index("flowing_water")
            )
            fluid = backend.get_bucket_fluid_truth((0, 4, 1))
            self.assertEqual(fluid["fluid"], "water")
            backend._latest_raw["portal_grid"][_flat_index(KNOWN_GRID[0])] = (
                PORTAL_GRID_BLOCKS.index("flowing_lava")
            )
            lava = backend.get_bucket_fluid_truth((0, 4, 1))
            self.assertEqual(lava["fluid"], "lava")
        finally:
            backend.close()

    def test_legacy_a0_hotbar_is_unchanged_for_non_e8(self):
        env = _ControlledMineRLEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(sample_task())
            self.assertEqual(backend._hotbar_mapping["dirt"], "hotbar.3")
        finally:
            backend.close()

    def test_protocol_translator_path_and_import_is_lazy(self):
        env = _ControlledMineRLEnv()
        action = MacroAction("place_block", target="dirt")
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        with patch.object(env.action_space, "contains", return_value=False):
            self.assertFalse(translate_macro_action(action, env.action_space).accepted)
        for name in ("e8_adapter.py", "e8_config.py", "e8_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE8BlockTruthAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE8BlockTruthAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()
