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
from obsidianlink.env.integration.e8_adapter import server_truth_snapshot
from obsidianlink.env.integration.e9_adapter import MineRLE9FluidTruthAdapter
from obsidianlink.env.integration.e9_config import (
    E9_AGENT_ID,
    E9_PROBE_WORLD_CELLS,
    E9_WATER_CALIBRATION,
    build_e9_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
)
from obsidianlink.env.validation import E9_SERVER_FLUID_TRUTH_CASE, EnvironmentValidationRunner
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e9-adapter-episode"
KNOWN_SPAWN = (0, 4, 0)
KNOWN_WORLD = ((0, 4, 1), (0, 5, 1), (0, 5, 0))
KNOWN_GRID = ((0, 0, 1), (0, 1, 1), (0, 1, 0))


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    return x + x_size * z + x_size * z_size * y


def _fluid_record(block: str, world, grid) -> dict[str, object]:
    if block in {"water", "lava"}:
        return {
            "flow_state": "source",
            "fluid_present": True,
            "fluid_type": block,
            "grid_cell": list(grid),
            "observed_block": block,
            "world_cell": list(world),
        }
    if block in {"flowing_water", "flowing_lava"}:
        return {
            "flow_state": "flowing",
            "fluid_present": True,
            "fluid_type": "water" if block == "flowing_water" else "lava",
            "grid_cell": list(grid),
            "observed_block": block,
            "world_cell": list(world),
        }
    return {
        "flow_state": "none",
        "fluid_present": False,
        "fluid_type": "none",
        "grid_cell": list(grid),
        "observed_block": block,
        "world_cell": list(world),
    }


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
            E9_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E9_AGENT_ID,
                step_id=0,
                frame="drop",
                visible_inventory={"water_bucket": 1},
                selected_item="water_bucket",
                messages=("drop",),
                workflow_stage="drop",
            )
        }

    def get_server_truth_snapshot(self, cells):
        records = []
        fluids = []
        for index, cell in enumerate(cells):
            block = self.blocks[index]
            records.append(
                {
                    "block": block,
                    "grid_cell": list(KNOWN_GRID[index]),
                    "world_cell": list(cell),
                }
            )
            fluids.append(_fluid_record(block, cell, KNOWN_GRID[index]))
        return {
            "agent_id": E9_AGENT_ID,
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "fluid_truth": fluids,
            "grid_anchor_world": list(KNOWN_SPAWN),
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        self.step_id = 1
        self.blocks = ["water", "air", "air"]
        obs = Observation(EPISODE, E9_AGENT_ID, 1, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            1,
            {E9_AGENT_ID: obs},
            {E9_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class _FluidEnv(_ControlledMineRLEnv):
    def __init__(
        self,
        *,
        after_target: str | None = "water",
        after_above_target: str = "air",
        after_above_spawn: str = "air",
        grid_origin: tuple[int, int, int] | None = KNOWN_SPAWN,
        drop_grid: bool = False,
        unknown_target: bool = False,
    ):
        super().__init__()
        self.after_target = after_target
        self.after_above_target = after_above_target
        self.after_above_spawn = after_above_spawn
        self._grid_origin = grid_origin
        self.drop_grid = drop_grid
        self.unknown_target = unknown_target
        self.inventory = {"water_bucket": 1}
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        for cell in KNOWN_GRID:
            self.grid[_flat_index(cell)] = PORTAL_GRID_BLOCKS.index("air")

    def _observation(self):
        observation = super()._observation()
        observation["inventory"] = {
            item: np.asarray(quantity, dtype=np.int64)
            for item, quantity in self.inventory.items()
        }
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
                self.after_above_target
            )
            self.grid[_flat_index(KNOWN_GRID[2])] = PORTAL_GRID_BLOCKS.index(
                self.after_above_spawn
            )
        observation = self._observation()
        info = {
            "location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5},
            "secret": "not-public",
        }
        return observation, 0.0, False, info


class E9MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_and_minimal(self):
        task = build_e9_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E9_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E9_AGENT_ID], {"water_bucket": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E9")
        self.assertTrue(task.scenario_parameters["not_a_benchmark_task"])
        self.assertTrue(task.scenario_parameters["calibration_only"])
        self.assertEqual(E9_WATER_CALIBRATION.expected_flow_state, "source")
        lava = build_e9_compatibility_task(EPISODE, "lava")
        self.assertEqual(lava.initial_inventories[E9_AGENT_ID], {"lava_bucket": 1})

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E9_SERVER_FLUID_TRUTH_CASE,
            MineRLE9FluidTruthAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        payload = result.as_dict()
        self.assertNotIn("portal_grid", payload)
        self.assertEqual(payload["outcome"], "fluid_truth_ok")
        self.assertEqual(payload["after_fluid_truth"][0]["flow_state"], "source")

    def test_adapter_rejects_evaluator_truth_in_backend_info(self):
        class Leaky(_Backend):
            def step(self, actions):
                self.calls.append("step")
                self.step_id = 1
                self.blocks = ["water", "air", "air"]
                obs = Observation(EPISODE, E9_AGENT_ID, 1, 0.0, frame="not-used")
                return BackendStep(
                    EPISODE,
                    1,
                    {E9_AGENT_ID: obs},
                    {E9_AGENT_ID: 0.0},
                    False,
                    False,
                    {"translation_accepted": True, "fluid_truth": "must-not-leak"},
                )

        result = EnvironmentValidationRunner().run(
            E9_SERVER_FLUID_TRUTH_CASE,
            MineRLE9FluidTruthAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=Leaky
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "truth_leak")

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE9FluidTruthAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("use_item", target="water_bucket")
        adapter.execute_fluid_stimulus(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_fluid_stimulus(action)
        adapter.close()
        adapter = MineRLE9FluidTruthAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_fluid_stimulus(MacroAction("place_block", target="dirt"))
        with self.assertRaises(ValueError):
            adapter.execute_fluid_stimulus(MacroAction("use_item", target="lava_bucket"))
        adapter.close()

    def test_snapshot_rejects_malformed_and_source_flow_collapse(self):
        base = _Backend().get_server_truth_snapshot(E9_PROBE_WORLD_CELLS)
        collapsed = [dict(item) for item in base["fluid_truth"]]
        collapsed[0]["observed_block"] = "flowing_water"
        collapsed[0]["fluid_type"] = "water"
        collapsed[0]["flow_state"] = "source"
        with self.assertRaisesRegex(ValueError, "malformed fluid state"):
            server_truth_snapshot(
                {**base, "fluid_truth": collapsed},
                expected_cells=E9_PROBE_WORLD_CELLS,
            )
        with self.assertRaises(ValueError):
            server_truth_snapshot({**base, "inventory": {}}, expected_cells=E9_PROBE_WORLD_CELLS)
        missing = [dict(item) for item in base["block_truth"]]
        missing[0]["block"] = "missing"
        fluids = [dict(item) for item in base["fluid_truth"]]
        fluids[0]["observed_block"] = "missing"
        with self.assertRaisesRegex(ValueError, "truth_missing_count does not match"):
            server_truth_snapshot(
                {
                    **base,
                    "block_truth": missing,
                    "fluid_truth": fluids,
                    "truth_missing_count": 0,
                },
                expected_cells=E9_PROBE_WORLD_CELLS,
            )

    def test_real_backend_distinguishes_source_and_flowing(self):
        env = _FluidEnv(after_target="water")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e9_compatibility_task(EPISODE))
            before = backend.get_server_truth_snapshot(E9_PROBE_WORLD_CELLS)
            parsed_before = server_truth_snapshot(before, expected_cells=E9_PROBE_WORLD_CELLS)
            self.assertEqual(parsed_before.fluid_truth[0].fluid_type, "none")
            backend.step({E9_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_server_truth_snapshot(E9_PROBE_WORLD_CELLS)
            parsed = server_truth_snapshot(after, expected_cells=E9_PROBE_WORLD_CELLS)
            self.assertEqual(parsed.fluid_truth[0].observed_block, "water")
            self.assertEqual(parsed.fluid_truth[0].flow_state, "source")
            self.assertEqual(parsed.fluid_truth[1].fluid_type, "none")
            self.assertEqual(backend._hotbar_mapping["water_bucket"], "hotbar.1")
            backend._latest_raw["portal_grid"][_flat_index(KNOWN_GRID[0])] = (
                PORTAL_GRID_BLOCKS.index("flowing_water")
            )
            flowing = backend.get_server_truth_snapshot(E9_PROBE_WORLD_CELLS)
            parsed_flowing = server_truth_snapshot(
                flowing, expected_cells=E9_PROBE_WORLD_CELLS
            )
            self.assertEqual(parsed_flowing.fluid_truth[0].observed_block, "flowing_water")
            self.assertEqual(parsed_flowing.fluid_truth[0].flow_state, "flowing")
            self.assertNotEqual(
                parsed.fluid_truth[0].flow_state, parsed_flowing.fluid_truth[0].flow_state
            )
        finally:
            backend.close()

    def test_real_backend_does_not_copy_action_intent_or_leak(self):
        env = _FluidEnv(after_target=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e9_compatibility_task(EPISODE))
            step = backend.step({E9_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_server_truth_snapshot(E9_PROBE_WORLD_CELLS)
            self.assertEqual([item["observed_block"] for item in after["fluid_truth"]], ["air", "air", "air"])
            self.assertNotIn("fluid_truth", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E9_AGENT_ID], "fluid_truth"))
        finally:
            backend.close()

    def test_protocol_translator_path_and_import_is_lazy(self):
        env = _ControlledMineRLEnv()
        action = MacroAction("use_item", target="water_bucket")
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        for name in ("e9_adapter.py", "e9_config.py", "e9_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE9FluidTruthAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE9FluidTruthAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()
