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
from obsidianlink.env.integration.e10_adapter import MineRLE10ObsidianAdapter
from obsidianlink.env.integration.e10_config import (
    E10_AGENT_ID,
    E10_CALIBRATION,
    E10_PROBE_WORLD_CELLS,
    build_e10_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
)
from obsidianlink.env.validation import E10_OBSIDIAN_CONVERSION_CASE, EnvironmentValidationRunner
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e10-adapter-episode"
KNOWN_SPAWN = (0, 4, 0)
KNOWN_WORLD = ((0, 4, 2), (0, 4, 1), (0, 5, 1), (0, 5, 2))
KNOWN_GRID = ((0, 0, 2), (0, 0, 1), (0, 1, 1), (0, 1, 2))


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
        self.blocks = ["lava", "air", "air", "air"]
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E10_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E10_AGENT_ID,
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
            "agent_id": E10_AGENT_ID,
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
        self.step_id += 1
        action = next(iter(actions.values()))
        if action.action_type == "use_item":
            self.blocks = ["obsidian", "water", "air", "air"]
        obs = Observation(EPISODE, E10_AGENT_ID, self.step_id, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            self.step_id,
            {E10_AGENT_ID: obs},
            {E10_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class _ConversionEnv(_ControlledMineRLEnv):
    def __init__(self, *, convert: bool = True, flowing_before: bool = False, place_water: bool = True):
        super().__init__()
        self.convert = convert
        self.flowing_before = flowing_before
        self.place_water = place_water
        self.inventory = {"water_bucket": 1}
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        for cell in KNOWN_GRID:
            self.grid[_flat_index(cell)] = PORTAL_GRID_BLOCKS.index("air")
        start = "flowing_lava" if flowing_before else "lava"
        self.grid[_flat_index(KNOWN_GRID[0])] = PORTAL_GRID_BLOCKS.index(start)

    def _observation(self):
        observation = super()._observation()
        observation["inventory"] = {
            item: np.asarray(quantity, dtype=np.int64)
            for item, quantity in self.inventory.items()
        }
        observation["location_stats"] = {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}
        observation["portal_dimension"] = np.asarray("minecraft:overworld")
        observation["portal_grid_origin"] = np.asarray(KNOWN_SPAWN, dtype=np.int32)
        return observation

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("use", 0)) and int(action_map.get("hotbar.1", 0)):
            if self.place_water:
                self.grid[_flat_index(KNOWN_GRID[1])] = PORTAL_GRID_BLOCKS.index("water")
            if self.convert:
                self.grid[_flat_index(KNOWN_GRID[0])] = PORTAL_GRID_BLOCKS.index("obsidian")
        observation = self._observation()
        info = {
            "location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5},
            "secret": "not-public",
        }
        return observation, 0.0, False, info


class E10MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_and_minimal(self):
        task = build_e10_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E10_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E10_AGENT_ID], {"water_bucket": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E10")
        self.assertTrue(task.scenario_parameters["not_a_benchmark_task"])
        self.assertTrue(task.scenario_parameters["calibration_only"])
        self.assertIs(task.scenario_parameters["obsidian_preplaced"], False)
        self.assertIs(task.scenario_parameters["runtime_applies_drawing_decorator"], True)
        self.assertIs(task.scenario_parameters["controlled_initial_geometry"], True)
        self.assertIs(task.scenario_parameters["lava_preplaced"], True)
        self.assertEqual(task.scenario_parameters["lava_target_world_cell"], (0, 4, 2))
        self.assertEqual(task.scenario_parameters["expected_target_after"], "obsidian")
        self.assertEqual(
            task.scenario_parameters["expected_water_after"],
            {"block": "water", "fluid_type": "water", "flow_state": "source"},
        )
        self.assertEqual(E10_CALIBRATION.target_world_cell, (0, 4, 2))
        self.assertEqual(E10_CALIBRATION.water_world_cell, (0, 4, 1))
        self.assertEqual(E10_CALIBRATION.probe_grid_cells, KNOWN_GRID)
        self.assertEqual(E10_CALIBRATION.observation_window_ticks, 5)

    def test_e10_envspec_xml_places_lava_source_without_obsidian(self):
        from obsidianlink.env.integration.e10_config import e10_initial_blocks
        from obsidianlink.env.portal_spec import PortalA0EnvSpec, parse_mission_draw_blocks

        xml = PortalA0EnvSpec(
            max_episode_steps=12,
            max_game_time_seconds=30,
            initial_inventory=({"type": "water_bucket", "quantity": 1},),
            initial_position=(0, 4, 0),
            initial_yaw=0.0,
            initial_pitch=60.0,
            initial_blocks=e10_initial_blocks(),
        ).to_xml()
        self.assertEqual(parse_mission_draw_blocks(xml), ((0, 4, 2, "lava"),))
        self.assertFalse(any(block == "obsidian" for _, _, _, block in parse_mission_draw_blocks(xml)))
        default_xml = PortalA0EnvSpec().to_xml()
        self.assertEqual(parse_mission_draw_blocks(default_xml), ())

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E10_OBSIDIAN_CONVERSION_CASE,
            MineRLE10ObsidianAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        payload = result.as_dict()
        self.assertNotIn("portal_grid", payload)
        self.assertEqual(payload["outcome"], "obsidian_conversion_ok")
        self.assertEqual(payload["tested_action_count"], 1)
        self.assertFalse(payload["integration_verified"])

    def test_adapter_rejects_evaluator_truth_in_backend_info(self):
        class Leaky(_Backend):
            def step(self, actions):
                self.calls.append("step")
                self.step_id = 1
                self.blocks = ["obsidian", "water", "air", "air"]
                obs = Observation(EPISODE, E10_AGENT_ID, 1, 0.0, frame="not-used")
                return BackendStep(
                    EPISODE,
                    1,
                    {E10_AGENT_ID: obs},
                    {E10_AGENT_ID: 0.0},
                    False,
                    False,
                    {"translation_accepted": True, "obsidian_present": True},
                )

        result = EnvironmentValidationRunner().run(
            E10_OBSIDIAN_CONVERSION_CASE,
            MineRLE10ObsidianAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=Leaky
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "truth_leak")

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE10ObsidianAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("use_item", target="water_bucket")
        adapter.execute_conversion_stimulus(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_conversion_stimulus(action)
        adapter.close()
        adapter = MineRLE10ObsidianAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_conversion_stimulus(MacroAction("place_block", target="dirt"))
        with self.assertRaises(ValueError):
            adapter.execute_conversion_stimulus(MacroAction("use_item", target="lava_bucket"))
        adapter.close()

    def test_real_backend_reads_lava_then_obsidian_without_intent_copy(self):
        env = _ConversionEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e10_compatibility_task(EPISODE))
            before = backend.get_server_truth_snapshot(E10_PROBE_WORLD_CELLS)
            parsed_before = server_truth_snapshot(before, expected_cells=E10_PROBE_WORLD_CELLS)
            self.assertEqual(parsed_before.block_truth[0].block, "lava")
            self.assertEqual(parsed_before.fluid_truth[0].flow_state, "source")
            self.assertEqual(parsed_before.fluid_truth[0].fluid_type, "lava")
            step = backend.step({E10_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_server_truth_snapshot(E10_PROBE_WORLD_CELLS)
            parsed = server_truth_snapshot(after, expected_cells=E10_PROBE_WORLD_CELLS)
            self.assertEqual(parsed.block_truth[0].block, "obsidian")
            self.assertEqual(parsed.fluid_truth[0].fluid_type, "none")
            self.assertEqual(parsed.fluid_truth[1].observed_block, "water")
            self.assertEqual(parsed.fluid_truth[1].flow_state, "source")
            self.assertEqual(parsed_before.block_truth[1].block, "air")
            self.assertNotIn("obsidian_present", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E10_AGENT_ID], "block_truth"))
            self.assertEqual(backend._hotbar_mapping["water_bucket"], "hotbar.1")
            backend._latest_raw["portal_grid"][_flat_index(KNOWN_GRID[0])] = (
                PORTAL_GRID_BLOCKS.index("flowing_lava")
            )
            flowing = backend.get_server_truth_snapshot(E10_PROBE_WORLD_CELLS)
            parsed_flowing = server_truth_snapshot(
                flowing, expected_cells=E10_PROBE_WORLD_CELLS
            )
            self.assertEqual(parsed_flowing.fluid_truth[0].observed_block, "flowing_lava")
            self.assertEqual(parsed_flowing.fluid_truth[0].flow_state, "flowing")
            self.assertNotEqual(
                parsed_before.fluid_truth[0].flow_state,
                parsed_flowing.fluid_truth[0].flow_state,
            )
        finally:
            backend.close()

    def test_real_backend_does_not_treat_rgb_or_action_intent_as_obsidian(self):
        env = _ConversionEnv(convert=False)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e10_compatibility_task(EPISODE))
            backend.step({E10_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_server_truth_snapshot(E10_PROBE_WORLD_CELLS)
            self.assertEqual(after["block_truth"][0]["block"], "lava")
            self.assertNotEqual(after["block_truth"][0]["block"], "obsidian")
        finally:
            backend.close()

    def test_real_backend_obsidian_without_water_is_not_success(self):
        env = _ConversionEnv(convert=True, place_water=False)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e10_compatibility_task(EPISODE))
            backend.step({E10_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_server_truth_snapshot(E10_PROBE_WORLD_CELLS)
            parsed = server_truth_snapshot(after, expected_cells=E10_PROBE_WORLD_CELLS)
            self.assertEqual(parsed.block_truth[0].block, "obsidian")
            self.assertEqual(parsed.block_truth[1].block, "air")
        finally:
            backend.close()

        class NoWater(_Backend):
            def step(self, actions):
                self.calls.append("step")
                self.step_id += 1
                self.blocks = ["obsidian", "air", "air", "air"]
                obs = Observation(EPISODE, E10_AGENT_ID, self.step_id, 0.0, frame="not-used")
                return BackendStep(
                    EPISODE,
                    self.step_id,
                    {E10_AGENT_ID: obs},
                    {E10_AGENT_ID: 0.0},
                    False,
                    False,
                    {"translation_accepted": True},
                )

        result = EnvironmentValidationRunner().run(
            E10_OBSIDIAN_CONVERSION_CASE,
            MineRLE10ObsidianAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=NoWater
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "water_placement_not_observed")
        self.assertEqual(result.after_target_block, "obsidian")
        self.assertEqual(result.after_water_block, "air")
        self.assertFalse(result.water_placement_observed)

    def test_protocol_translator_path_and_import_is_lazy(self):
        env = _ControlledMineRLEnv()
        action = MacroAction("use_item", target="water_bucket")
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        wait = translate_macro_action(MacroAction.wait(), env.action_space)
        self.assertTrue(wait.accepted)
        for name in ("e10_adapter.py", "e10_config.py", "e10_run.py", "e10_geometry.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE10ObsidianAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE10ObsidianAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()
