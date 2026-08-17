"""Frozen P1 E12 vanilla Overworld → Nether dimension-transition case."""

from types import MappingProxyType

from obsidianlink.env.validation.contract import P1_VALIDATION_CASES
from obsidianlink.env.validation.placement import spawn_relative_grid_cell


E12_DIMENSION_TRANSITION_CASE = P1_VALIDATION_CASES[12]
E12_SPAWN_WORLD = (0, 4, 0)
E12_INITIAL_YAW = 0.0
E12_INITIAL_PITCH = 0.0
E12_DURATION_TICKS = 8
E12_OBSERVATION_WINDOW_TICKS = 100
E12_POSITION_MIN = (-2.0, 2.0, -2.0)
E12_POSITION_MAX = (3.0, 8.0, 4.0)
E12_MOVE_PARAMETERS = MappingProxyType(
    {
        "forward": 1.0,
        "strafe": 0.0,
        "sprint": False,
        "jump": False,
    }
)
# Same 4x5 obsidian ring and 2x3 interior as E11, frozen here so the
# MineRL-independent runner does not import the E11 integration config.
E12_FRAME_BLOCKS = (
    (-1, 3, 1),
    (-1, 4, 1),
    (-1, 5, 1),
    (-1, 6, 1),
    (-1, 7, 1),
    (0, 3, 1),
    (0, 7, 1),
    (1, 3, 1),
    (1, 7, 1),
    (2, 3, 1),
    (2, 4, 1),
    (2, 5, 1),
    (2, 6, 1),
    (2, 7, 1),
)
E12_INTERIOR_CELLS = (
    (0, 4, 1),
    (1, 4, 1),
    (0, 5, 1),
    (1, 5, 1),
    (0, 6, 1),
    (1, 6, 1),
)
E12_CONTROL_WORLD_CELLS = ((0, 8, 1), (0, 4, 3))
E12_PROBE_WORLD_CELLS = E12_FRAME_BLOCKS + E12_INTERIOR_CELLS + E12_CONTROL_WORLD_CELLS
E12_PROBE_GRID_CELLS = tuple(
    spawn_relative_grid_cell(cell, E12_SPAWN_WORLD) for cell in E12_PROBE_WORLD_CELLS
)
