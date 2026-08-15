"""Independent E8 world ↔ atSpawn-grid coordinate regression.

These literals are the known geometry, not copies of production constants.
"""

from __future__ import annotations

import unittest

from obsidianlink.env.integration.e8_config import (
    E8_EXPECTED_GRID_ANCHOR,
    E8_PROBE_GRID_CELLS,
    E8_PROBE_WORLD_CELLS,
    E8_SPAWN_WORLD,
    build_e8_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import PORTAL_GRID_MAX, PORTAL_GRID_MIN
from obsidianlink.env.validation.placement import spawn_relative_grid_cell
from obsidianlink.env.validation.rgb import inspect_public_rgb
import numpy as np


KNOWN_SPAWN = (0, 4, 0)
KNOWN_WORLD = (
    (0, 4, 1),
    (1, 4, 1),
    (-1, 4, 1),
)
KNOWN_GRID = (
    (0, 0, 1),
    (1, 0, 1),
    (-1, 0, 1),
)
MISTAKEN_WORLD_AS_GRID = (0, 4, 1)


class E8TruthCoordinateTests(unittest.TestCase):
    def test_known_world_region_converts_to_spawn_relative_grid(self):
        converted = tuple(
            spawn_relative_grid_cell(world, KNOWN_SPAWN) for world in KNOWN_WORLD
        )
        self.assertEqual(converted, KNOWN_GRID)
        self.assertNotEqual(KNOWN_WORLD, KNOWN_GRID)
        self.assertEqual(E8_SPAWN_WORLD, KNOWN_SPAWN)
        self.assertEqual(E8_EXPECTED_GRID_ANCHOR, KNOWN_SPAWN)
        self.assertEqual(E8_PROBE_WORLD_CELLS, KNOWN_WORLD)
        self.assertEqual(E8_PROBE_GRID_CELLS, KNOWN_GRID)

    def test_mistaken_world_as_grid_is_a_different_index(self):
        world_as_grid = MineRLEnvironmentBackend._cell_index_in_grid(MISTAKEN_WORLD_AS_GRID)
        spawn_relative = MineRLEnvironmentBackend._cell_index_in_grid(KNOWN_GRID[0])
        self.assertIsNotNone(world_as_grid)
        self.assertIsNotNone(spawn_relative)
        self.assertNotEqual(world_as_grid, spawn_relative)
        self.assertNotEqual(MISTAKEN_WORLD_AS_GRID, KNOWN_GRID[0])

    def test_probe_cells_are_inside_portal_grid_bounds(self):
        for cell in KNOWN_GRID:
            for axis, value in enumerate(cell):
                self.assertGreaterEqual(value, PORTAL_GRID_MIN[axis])
                self.assertLessEqual(value, PORTAL_GRID_MAX[axis])
            self.assertIsNotNone(MineRLEnvironmentBackend._cell_index_in_grid(cell))

    def test_task_records_both_coordinate_namespaces(self):
        task = build_e8_compatibility_task("e8-coordinate-task")
        params = dict(task.scenario_parameters)
        self.assertEqual(
            tuple(tuple(cell) for cell in params["probe_world_cells"]), KNOWN_WORLD
        )
        self.assertEqual(
            tuple(tuple(cell) for cell in params["probe_grid_cells"]), KNOWN_GRID
        )
        self.assertEqual(task.spawn_positions["agent_1"], KNOWN_SPAWN)
        self.assertTrue(params["not_a_benchmark_task"])
        self.assertTrue(params["calibration_only"])

    def test_coordinate_metadata_is_rejected_by_public_rgb(self):
        leaked = {
            "agent_1": {
                "episode_id": "episode",
                "agent_id": "agent_1",
                "step_id": 0,
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "probe_grid_cells": [list(KNOWN_GRID[0])],
            }
        }
        self.assertEqual(
            inspect_public_rgb(leaked, episode_id="episode").outcome, "rgb_leak"
        )


if __name__ == "__main__":
    unittest.main()
