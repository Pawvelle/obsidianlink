"""Independent E6 world ↔ atSpawn-grid coordinate regression.

These literals are the known geometry, not copies of production constants.
Production values are checked against them afterwards.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.integration.e6_config import (
    E6_EXPECTED_GRID_ANCHOR,
    E6_INITIAL_PITCH,
    E6_INITIAL_YAW,
    E6_SPAWN_WORLD,
    E6_TARGET_GRID_CELL,
    E6_TARGET_WORLD_CELL,
    build_e6_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import PORTAL_GRID_MAX, PORTAL_GRID_MIN
from obsidianlink.env.validation.placement import spawn_relative_grid_cell
from obsidianlink.env.validation.rgb import inspect_public_rgb


KNOWN_SPAWN_WORLD = (0, 4, 0)
KNOWN_TARGET_WORLD = (0, 4, 1)
KNOWN_TARGET_GRID = (0, 0, 1)
MISTAKEN_WORLD_AS_GRID = (0, 4, 1)


def _minecraft_look_vector(yaw_degrees: float, pitch_degrees: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    return (
        -math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )


class E6PlacementCoordinateTests(unittest.TestCase):
    def test_known_world_target_converts_to_spawn_relative_grid(self) -> None:
        converted = spawn_relative_grid_cell(KNOWN_TARGET_WORLD, KNOWN_SPAWN_WORLD)
        self.assertEqual(converted, KNOWN_TARGET_GRID)
        self.assertNotEqual(KNOWN_TARGET_WORLD, KNOWN_TARGET_GRID)
        self.assertEqual(E6_SPAWN_WORLD, KNOWN_SPAWN_WORLD)
        self.assertEqual(E6_TARGET_WORLD_CELL, KNOWN_TARGET_WORLD)
        self.assertEqual(E6_EXPECTED_GRID_ANCHOR, KNOWN_SPAWN_WORLD)
        self.assertEqual(E6_TARGET_GRID_CELL, KNOWN_TARGET_GRID)
        self.assertEqual(
            spawn_relative_grid_cell(E6_TARGET_WORLD_CELL, E6_SPAWN_WORLD),
            KNOWN_TARGET_GRID,
        )

    def test_world_and_grid_cells_are_distinct_grid_indices(self) -> None:
        world_as_grid = MineRLEnvironmentBackend._cell_index_in_grid(MISTAKEN_WORLD_AS_GRID)
        spawn_relative = MineRLEnvironmentBackend._cell_index_in_grid(KNOWN_TARGET_GRID)
        self.assertIsNotNone(world_as_grid)
        self.assertIsNotNone(spawn_relative)
        self.assertNotEqual(world_as_grid, spawn_relative)

    def test_relative_target_is_inside_portal_grid_bounds(self) -> None:
        for axis, value in enumerate(KNOWN_TARGET_GRID):
            self.assertGreaterEqual(value, PORTAL_GRID_MIN[axis])
            self.assertLessEqual(value, PORTAL_GRID_MAX[axis])
        self.assertIsNotNone(
            MineRLEnvironmentBackend._cell_index_in_grid(KNOWN_TARGET_GRID)
        )
        self.assertIsNone(
            MineRLEnvironmentBackend._cell_index_in_grid((0, PORTAL_GRID_MIN[1] - 1, 1))
        )

    def test_pitch_60_places_on_known_world_cell(self) -> None:
        eye = (0.5, 4.0 + 1.62, 0.5)
        direction = _minecraft_look_vector(0.0, 60.0)
        t = (eye[1] - 4.0) / -direction[1]
        hit = (eye[0] + direction[0] * t, 4.0, eye[2] + direction[2] * t)
        support = (math.floor(hit[0]), 3, math.floor(hit[2]))
        placed = (support[0], support[1] + 1, support[2])
        self.assertLess(t, 4.5)
        self.assertEqual(support, (0, 3, 1))
        self.assertEqual(placed, KNOWN_TARGET_WORLD)
        self.assertEqual((E6_INITIAL_YAW, E6_INITIAL_PITCH), (0.0, 60.0))
        self.assertEqual(E6_TARGET_WORLD_CELL, placed)

    def test_task_records_both_coordinate_namespaces(self) -> None:
        task = build_e6_compatibility_task("e6-coordinate-task")
        params = dict(task.scenario_parameters)
        self.assertEqual(tuple(params["target_world_cell"]), KNOWN_TARGET_WORLD)
        self.assertEqual(tuple(params["target_grid_cell"]), KNOWN_TARGET_GRID)
        self.assertEqual(task.spawn_positions["agent_1"], KNOWN_SPAWN_WORLD)

    def test_coordinate_metadata_is_rejected_by_public_rgb(self) -> None:
        leaked = {
            "agent_1": {
                "episode_id": "episode",
                "agent_id": "agent_1",
                "step_id": 0,
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "target_grid_cell": list(KNOWN_TARGET_GRID),
            }
        }
        self.assertEqual(
            inspect_public_rgb(leaked, episode_id="episode").outcome, "rgb_leak"
        )


if __name__ == "__main__":
    unittest.main()
