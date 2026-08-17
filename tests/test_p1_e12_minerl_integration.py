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
from obsidianlink.env.integration.e12_adapter import (
    MineRLE12DimensionTransitionAdapter,
    dimension_truth_snapshot,
)
from obsidianlink.env.integration.e12_config import (
    E12_AGENT_ID,
    E12_CONTROL_WORLD_CELLS,
    E12_FRAME_BLOCKS,
    E12_INITIAL_DRAW_BLOCKS,
    E12_INTERIOR_CELLS,
    E12_PROBE_GRID_CELLS,
    E12_PROBE_WORLD_CELLS,
    E12_SPAWN_WORLD,
    build_e12_compatibility_task,
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
from obsidianlink.env.validation import E12_DIMENSION_TRANSITION_CASE, EnvironmentValidationRunner
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e12-adapter-episode"


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    return x + x_size * z + x_size * z_size * y


def _world_blocks(interior: str = "nether_portal") -> dict[tuple[int, int, int], str]:
    blocks = {cell: "obsidian" for cell in E12_FRAME_BLOCKS}
    blocks.update({cell: interior for cell in E12_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E12_CONTROL_WORLD_CELLS})
    return blocks


class _Backend:
    instances = []

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.blocks = _world_blocks()
        self.dimension = "minecraft:overworld"
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E12_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E12_AGENT_ID,
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
        for cell, grid in zip(cells, E12_PROBE_GRID_CELLS):
            records.append(
                {
                    "block": self.blocks[cell],
                    "grid_cell": list(grid),
                    "world_cell": list(cell),
                }
            )
        return {
            "agent_id": E12_AGENT_ID,
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "grid_anchor_world": list(E12_SPAWN_WORLD),
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def get_dimension_truth(self):
        return {
            "agent_id": E12_AGENT_ID,
            "dimension": self.dimension,
            "episode_id": EPISODE,
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
        }

    def step(self, actions):
        self.calls.append("step")
        self.step_id += 1
        action = next(iter(actions.values()))
        if action.action_type == "move":
            self.dimension = "minecraft:the_nether"
        obs = Observation(EPISODE, E12_AGENT_ID, self.step_id, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            self.step_id,
            {E12_AGENT_ID: obs},
            {E12_AGENT_ID: 0.0},
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


class _TransitionEnv(_ControlledMineRLEnv):
    def __init__(self, *, enter: bool = True):
        super().__init__()
        self.enter = enter
        self.inventory = {"dirt": 1}
        self.dimension = "minecraft:overworld"
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        for cell in E12_PROBE_GRID_CELLS:
            self.grid[_flat_index(cell)] = PORTAL_GRID_BLOCKS.index("air")
        for cell in E12_FRAME_BLOCKS:
            grid = (cell[0], cell[1] - 4, cell[2])
            self.grid[_flat_index(grid)] = PORTAL_GRID_BLOCKS.index("obsidian")
        for cell in E12_INTERIOR_CELLS:
            grid = (cell[0], cell[1] - 4, cell[2])
            self.grid[_flat_index(grid)] = PORTAL_GRID_BLOCKS.index("nether_portal")

    def _observation(self):
        observation = super()._observation()
        observation["inventory"] = {
            item: np.asarray(quantity, dtype=np.int64)
            for item, quantity in self.inventory.items()
        }
        observation["location_stats"] = {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}
        observation["portal_dimension"] = np.asarray(self.dimension)
        observation["portal_grid_origin"] = np.asarray(E12_SPAWN_WORLD, dtype=np.int32)
        return observation

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("forward", 0)) and self.enter:
            self.dimension = "minecraft:the_nether"
        observation = self._observation()
        return observation, 0.0, False, {"location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5}}


class E12MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_fixture_not_agent_construction(self):
        task = build_e12_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E12_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E12_AGENT_ID], {"dirt": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E12")
        self.assertTrue(task.scenario_parameters["prebuilt_active_portal_is_calibration_fixture"])
        self.assertIs(task.scenario_parameters["agent_built_portal"], False)
        self.assertIs(task.scenario_parameters["portal_preplaced"], True)
        self.assertIs(task.scenario_parameters["fire_preplaced"], False)
        self.assertIs(task.scenario_parameters["needs_e12_runtime_portal_fixture_authorization"], True)
        self.assertEqual(len(E12_FRAME_BLOCKS), 14)
        self.assertEqual(len(E12_INTERIOR_CELLS), 6)
        self.assertEqual(len(E12_PROBE_WORLD_CELLS), 22)

    def test_e12_envspec_xml_places_active_portal_without_fire(self):
        xml = PortalA0EnvSpec(
            max_episode_steps=130,
            max_game_time_seconds=60,
            initial_inventory=({"type": "dirt", "quantity": 1},),
            initial_position=(0, 4, 0),
            initial_yaw=0.0,
            initial_pitch=0.0,
            initial_blocks=E12_INITIAL_DRAW_BLOCKS,
            allow_active_portal_fixture=True,
        ).to_xml()
        draw = parse_mission_draw_blocks(xml)
        self.assertEqual(len(draw), 20)
        self.assertEqual(sum(block == "obsidian" for _, _, _, block in draw), 14)
        self.assertEqual(sum(block == "portal" for _, _, _, block in draw), 6)
        self.assertFalse(any(block in {"nether_portal", "fire"} for _, _, _, block in draw))
        default_xml = PortalA0EnvSpec().to_xml()
        self.assertEqual(parse_mission_draw_blocks(default_xml), ())
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PortalA0EnvSpec(
                initial_blocks=E12_INITIAL_DRAW_BLOCKS,
                allow_obsidian_frame_fixture=True,
                allow_active_portal_fixture=True,
            )

    def test_adapter_executes_one_move_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E12_DIMENSION_TRANSITION_CASE,
            MineRLE12DimensionTransitionAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "dimension_transition_ok")
        self.assertEqual(result.tested_action_count, 1)
        self.assertFalse(result.integration_verified)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE12DimensionTransitionAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction(
            "move",
            duration_ticks=8,
            parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False},
        )
        adapter.execute_transition_stimulus(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_transition_stimulus(action)
        adapter.close()
        adapter = MineRLE12DimensionTransitionAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_transition_stimulus(MacroAction("use_item", target="flint_and_steel"))
        with self.assertRaises(ValueError):
            adapter.execute_transition_stimulus(MacroAction("move", duration_ticks=1))
        adapter.close()

    def test_real_backend_reads_nether_dimension_without_after_grid(self):
        env = _TransitionEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e12_compatibility_task(EPISODE))
            before = backend.get_dimension_truth()
            parsed_before = dimension_truth_snapshot(before)
            self.assertEqual(parsed_before.dimension, "minecraft:overworld")
            snapshot = backend.get_server_truth_snapshot(E12_PROBE_WORLD_CELLS)
            interior = {
                tuple(item["world_cell"]): item["block"]
                for item in snapshot["block_truth"]
                if tuple(item["world_cell"]) in E12_INTERIOR_CELLS
            }
            self.assertEqual(set(interior.values()), {"nether_portal"})
            step = backend.step(
                {
                    E12_AGENT_ID: MacroAction(
                        "move",
                        duration_ticks=8,
                        parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False},
                    )
                }
            )
            after = backend.get_dimension_truth()
            parsed = dimension_truth_snapshot(after)
            self.assertEqual(parsed.dimension, "minecraft:the_nether")
            self.assertNotIn("dimension_transition_observed", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E12_AGENT_ID], "block_truth"))
        finally:
            backend.close()

    def test_real_backend_does_not_treat_rgb_as_dimension_change(self):
        env = _TransitionEnv(enter=False)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e12_compatibility_task(EPISODE))
            backend.step(
                {
                    E12_AGENT_ID: MacroAction(
                        "move",
                        duration_ticks=8,
                        parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False},
                    )
                }
            )
            after = dimension_truth_snapshot(backend.get_dimension_truth())
            self.assertEqual(after.dimension, "minecraft:overworld")
        finally:
            backend.close()

    def test_protocol_translator_path_and_import_is_lazy(self):
        env = _ControlledMineRLEnv()
        action = MacroAction(
            "move",
            duration_ticks=8,
            parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False},
        )
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["forward"], 1)
        wait = translate_macro_action(MacroAction.wait(), env.action_space)
        self.assertTrue(wait.accepted)
        for name in ("e12_adapter.py", "e12_config.py", "e12_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE12DimensionTransitionAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE12DimensionTransitionAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()
