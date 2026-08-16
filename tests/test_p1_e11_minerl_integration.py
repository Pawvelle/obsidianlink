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
from obsidianlink.env.integration.e11_adapter import MineRLE11PortalActivationAdapter
from obsidianlink.env.integration.e11_config import (
    E11_AGENT_ID,
    E11_CALIBRATION,
    E11_CONTROL_WORLD_CELLS,
    E11_FRAME_BLOCKS,
    E11_INTERIOR_CELLS,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
    E11_SPAWN_WORLD,
    build_e11_compatibility_task,
    e11_initial_blocks,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
    PortalA0EnvSpec,
    parse_mission_draw_blocks,
)
from obsidianlink.env.validation import E11_PORTAL_ACTIVATION_CASE, EnvironmentValidationRunner
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e11-adapter-episode"


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    return x + x_size * z + x_size * z_size * y


def _world_blocks(interior: str = "air") -> dict[tuple[int, int, int], str]:
    blocks = {cell: "obsidian" for cell in E11_FRAME_BLOCKS}
    blocks.update({cell: interior for cell in E11_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E11_CONTROL_WORLD_CELLS})
    return blocks


class _Backend:
    instances = []

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.blocks = _world_blocks()
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E11_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E11_AGENT_ID,
                step_id=0,
                frame="drop",
                visible_inventory={"flint_and_steel": 1},
                selected_item="flint_and_steel",
                messages=("drop",),
                workflow_stage="drop",
            )
        }

    def get_server_truth_snapshot(self, cells):
        records = []
        for cell, grid in zip(cells, E11_PROBE_GRID_CELLS):
            block = self.blocks[cell]
            records.append(
                {
                    "block": block,
                    "grid_cell": list(grid),
                    "world_cell": list(cell),
                }
            )
        return {
            "agent_id": E11_AGENT_ID,
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "grid_anchor_world": list(E11_SPAWN_WORLD),
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        self.step_id += 1
        action = next(iter(actions.values()))
        if action.action_type == "use_item":
            self.blocks = _world_blocks("nether_portal")
        obs = Observation(EPISODE, E11_AGENT_ID, self.step_id, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            self.step_id,
            {E11_AGENT_ID: obs},
            {E11_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None

    def get_reset_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}


class _ActivationEnv(_ControlledMineRLEnv):
    def __init__(self, *, activate: bool = True):
        super().__init__()
        self.activate = activate
        self.inventory = {"flint_and_steel": 1}
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        for cell in E11_PROBE_GRID_CELLS:
            self.grid[_flat_index(cell)] = PORTAL_GRID_BLOCKS.index("air")
        for cell in E11_FRAME_BLOCKS:
            grid = (cell[0], cell[1] - 4, cell[2])
            self.grid[_flat_index(grid)] = PORTAL_GRID_BLOCKS.index("obsidian")

    def _observation(self):
        observation = super()._observation()
        observation["inventory"] = {
            item: np.asarray(quantity, dtype=np.int64)
            for item, quantity in self.inventory.items()
        }
        observation["location_stats"] = {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}
        observation["portal_dimension"] = np.asarray("minecraft:overworld")
        observation["portal_grid_origin"] = np.asarray(E11_SPAWN_WORLD, dtype=np.int32)
        return observation

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("use", 0)) and int(action_map.get("hotbar.1", 0)) and self.activate:
            for cell in E11_INTERIOR_CELLS:
                grid = (cell[0], cell[1] - 4, cell[2])
                self.grid[_flat_index(grid)] = PORTAL_GRID_BLOCKS.index("nether_portal")
        observation = self._observation()
        return observation, 0.0, False, {"location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}}


class E11MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_fixture_not_agent_construction(self):
        task = build_e11_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E11_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E11_AGENT_ID], {"flint_and_steel": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E11")
        self.assertTrue(task.scenario_parameters["prebuilt_frame_is_calibration_fixture"])
        self.assertIs(task.scenario_parameters["agent_built_portal"], False)
        self.assertIs(task.scenario_parameters["portal_preplaced"], False)
        self.assertIs(task.scenario_parameters["fire_preplaced"], False)
        self.assertIs(task.scenario_parameters["needs_e11_runtime_geometry_authorization"], True)
        self.assertEqual(len(E11_CALIBRATION.frame_blocks), 14)
        self.assertEqual(len(E11_CALIBRATION.interior_cells), 6)
        self.assertEqual(E11_CALIBRATION.probe_world_cells, E11_PROBE_WORLD_CELLS)

    def test_e11_envspec_xml_places_obsidian_frame_without_portal(self):
        xml = PortalA0EnvSpec(
            max_episode_steps=12,
            max_game_time_seconds=30,
            initial_inventory=({"type": "flint_and_steel", "quantity": 1},),
            initial_position=(0, 4, 0),
            initial_yaw=0.0,
            initial_pitch=60.0,
            initial_blocks=e11_initial_blocks(),
            allow_obsidian_frame_fixture=True,
        ).to_xml()
        draw = parse_mission_draw_blocks(xml)
        self.assertEqual(len(draw), 14)
        self.assertTrue(all(block == "obsidian" for _, _, _, block in draw))
        self.assertFalse(any(block in {"portal", "nether_portal", "fire"} for _, _, _, block in draw))
        default_xml = PortalA0EnvSpec().to_xml()
        self.assertEqual(parse_mission_draw_blocks(default_xml), ())

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E11_PORTAL_ACTIVATION_CASE,
            MineRLE11PortalActivationAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "portal_activation_ok")
        self.assertEqual(result.tested_action_count, 1)
        self.assertFalse(result.integration_verified)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE11PortalActivationAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("use_item", target="flint_and_steel")
        adapter.execute_activation_stimulus(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_activation_stimulus(action)
        adapter.close()
        adapter = MineRLE11PortalActivationAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_activation_stimulus(MacroAction("place_block", target="obsidian"))
        with self.assertRaises(ValueError):
            adapter.execute_activation_stimulus(MacroAction("use_item", target="water_bucket"))
        adapter.close()

    def test_real_backend_reads_nether_portal_without_intent_copy(self):
        env = _ActivationEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e11_compatibility_task(EPISODE))
            before = backend.get_server_truth_snapshot(E11_PROBE_WORLD_CELLS)
            parsed_before = server_truth_snapshot(before, expected_cells=E11_PROBE_WORLD_CELLS)
            frame_blocks = {
                item.world_cell: item.block
                for item in parsed_before.block_truth
                if item.world_cell in E11_FRAME_BLOCKS
            }
            self.assertEqual(set(frame_blocks.values()), {"obsidian"})
            interior_before = {
                item.world_cell: item.block
                for item in parsed_before.block_truth
                if item.world_cell in E11_INTERIOR_CELLS
            }
            self.assertEqual(set(interior_before.values()), {"air"})
            step = backend.step({E11_AGENT_ID: MacroAction("use_item", target="flint_and_steel")})
            after = backend.get_server_truth_snapshot(E11_PROBE_WORLD_CELLS)
            parsed = server_truth_snapshot(after, expected_cells=E11_PROBE_WORLD_CELLS)
            interior_after = {
                item.world_cell: item.block
                for item in parsed.block_truth
                if item.world_cell in E11_INTERIOR_CELLS
            }
            self.assertEqual(set(interior_after.values()), {"nether_portal"})
            self.assertNotIn("portal_activated", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E11_AGENT_ID], "block_truth"))
            self.assertEqual(backend._hotbar_mapping["flint_and_steel"], "hotbar.1")
        finally:
            backend.close()

    def test_real_backend_does_not_treat_rgb_or_fire_as_portal(self):
        env = _ActivationEnv(activate=False)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e11_compatibility_task(EPISODE))
            backend.step({E11_AGENT_ID: MacroAction("use_item", target="flint_and_steel")})
            after = backend.get_server_truth_snapshot(E11_PROBE_WORLD_CELLS)
            interior = [
                item["block"]
                for item in after["block_truth"]
                if tuple(item["world_cell"]) in E11_INTERIOR_CELLS
            ]
            self.assertEqual(set(interior), {"air"})
        finally:
            backend.close()

    def test_protocol_translator_path_and_import_is_lazy(self):
        env = _ControlledMineRLEnv()
        action = MacroAction("use_item", target="flint_and_steel")
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        wait = translate_macro_action(MacroAction.wait(), env.action_space)
        self.assertTrue(wait.accepted)
        for name in ("e11_adapter.py", "e11_config.py", "e11_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE11PortalActivationAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE11PortalActivationAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()
