"""MineRL-independent P1 E8/E9 evaluator-only server-truth contract.

E8 validates the block-truth portion of ServerTruthSnapshot. E9 extends
the same snapshot with typed fluid truth, including the source/flowing
distinction that E7 deliberately collapsed. Neither channel is
Agent-visible, a benchmark task, or E10 water-lava-to-obsidian mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from obsidianlink.env.validation.movement import finite_number
from obsidianlink.env.validation.placement import (
    ALLOWED_BLOCK_NAMES,
    spawn_relative_grid_cell,
    validate_block_name,
    validate_target_cell,
)


BLOCK_TRUTH_OK = "block_truth_ok"
TRUTH_SNAPSHOT_MISSING = "truth_snapshot_missing"
TRUTH_IDENTITY_MISMATCH = "truth_identity_mismatch"
TRUTH_POSITION_MISSING = "truth_position_missing"
TRUTH_POSITION_INVALID = "truth_position_invalid"
TRUTH_DIMENSION_MISSING = "truth_dimension_missing"
TRUTH_DIMENSION_INVALID = "truth_dimension_invalid"
TRUTH_WRONG_DIMENSION = "truth_wrong_dimension"
TRUTH_ANCHOR_MISMATCH = "truth_anchor_mismatch"
TRUTH_REGION_EMPTY = "truth_region_empty"
TRUTH_DUPLICATE_CELL = "truth_duplicate_cell"
TRUTH_CELL_OUT_OF_BOUNDS = "truth_cell_out_of_bounds"
TRUTH_BLOCK_MISSING = "truth_block_missing"
TRUTH_BLOCK_UNKNOWN = "truth_block_unknown"
TRUTH_BEFORE_MISMATCH = "truth_before_mismatch"
TRUTH_AFTER_MISMATCH = "truth_after_mismatch"
TRUTH_CONTROL_CELL_CHANGED = "truth_control_cell_changed"
TRUTH_STIMULUS_REJECTED = "truth_stimulus_rejected"
TRUTH_TEST_ACTION_NOT_EXECUTED = "truth_test_action_not_executed"
TRUTH_MULTIPLE_TEST_ACTIONS = "truth_multiple_test_actions"
TRUTH_LEAK = "truth_leak"
TRUTH_WRONG_ACTION_TYPE = "truth_wrong_action_type"
TRUTH_WRONG_TARGET = "truth_wrong_target"
TRUTH_CALIBRATION_MISMATCH = "truth_calibration_mismatch"
FLUID_TRUTH_OK = "fluid_truth_ok"
TRUTH_FLUID_MISSING = "truth_fluid_missing"
TRUTH_FLUID_UNKNOWN = "truth_fluid_unknown"
TRUTH_SOURCE_FLOWING_MISMATCH = "truth_source_flowing_mismatch"
TRUTH_BEFORE_FLUID_MISMATCH = "truth_before_fluid_mismatch"
TRUTH_AFTER_FLUID_MISMATCH = "truth_after_fluid_mismatch"

BLOCK_TRUTH_OUTCOMES = frozenset(
    {
        BLOCK_TRUTH_OK,
        TRUTH_SNAPSHOT_MISSING,
        TRUTH_IDENTITY_MISMATCH,
        TRUTH_POSITION_MISSING,
        TRUTH_POSITION_INVALID,
        TRUTH_DIMENSION_MISSING,
        TRUTH_DIMENSION_INVALID,
        TRUTH_WRONG_DIMENSION,
        TRUTH_ANCHOR_MISMATCH,
        TRUTH_REGION_EMPTY,
        TRUTH_DUPLICATE_CELL,
        TRUTH_CELL_OUT_OF_BOUNDS,
        TRUTH_BLOCK_MISSING,
        TRUTH_BLOCK_UNKNOWN,
        TRUTH_BEFORE_MISMATCH,
        TRUTH_AFTER_MISMATCH,
        TRUTH_CONTROL_CELL_CHANGED,
        TRUTH_STIMULUS_REJECTED,
        TRUTH_TEST_ACTION_NOT_EXECUTED,
        TRUTH_MULTIPLE_TEST_ACTIONS,
        TRUTH_LEAK,
        TRUTH_WRONG_ACTION_TYPE,
        TRUTH_WRONG_TARGET,
        TRUTH_CALIBRATION_MISMATCH,
    }
)
FLUID_TRUTH_OUTCOMES = frozenset(
    {
        FLUID_TRUTH_OK,
        TRUTH_SNAPSHOT_MISSING,
        TRUTH_IDENTITY_MISMATCH,
        TRUTH_POSITION_MISSING,
        TRUTH_POSITION_INVALID,
        TRUTH_DIMENSION_MISSING,
        TRUTH_DIMENSION_INVALID,
        TRUTH_WRONG_DIMENSION,
        TRUTH_ANCHOR_MISMATCH,
        TRUTH_REGION_EMPTY,
        TRUTH_DUPLICATE_CELL,
        TRUTH_CELL_OUT_OF_BOUNDS,
        TRUTH_FLUID_MISSING,
        TRUTH_FLUID_UNKNOWN,
        TRUTH_SOURCE_FLOWING_MISMATCH,
        TRUTH_BEFORE_FLUID_MISMATCH,
        TRUTH_AFTER_FLUID_MISMATCH,
        TRUTH_CONTROL_CELL_CHANGED,
        TRUTH_STIMULUS_REJECTED,
        TRUTH_TEST_ACTION_NOT_EXECUTED,
        TRUTH_MULTIPLE_TEST_ACTIONS,
        TRUTH_LEAK,
        TRUTH_WRONG_ACTION_TYPE,
        TRUTH_WRONG_TARGET,
        TRUTH_CALIBRATION_MISMATCH,
    }
)

ALLOWED_DIMENSIONS = frozenset(
    {
        "minecraft:overworld",
        "minecraft:the_nether",
        "minecraft:the_end",
    }
)
E8_REQUIRED_DIMENSION = "minecraft:overworld"
E9_REQUIRED_DIMENSION = "minecraft:overworld"
ANCHOR_SOURCE_ORIGIN = "portal_grid_origin"
ANCHOR_SOURCE_SPAWN_FALLBACK = "expected_spawn_fallback"
ALLOWED_ANCHOR_SOURCES = frozenset(
    {ANCHOR_SOURCE_ORIGIN, ANCHOR_SOURCE_SPAWN_FALLBACK}
)
ALLOWED_FLUID_TYPES = frozenset({"none", "water", "lava"})
ALLOWED_FLOW_STATES = frozenset({"none", "source", "flowing"})
NONE_FLUID_BLOCKS = frozenset(
    {
        "air",
        "bedrock",
        "dirt",
        "grass",
        "grass_block",
        "obsidian",
        "fire",
        "portal",
        "nether_portal",
    }
)
SOURCE_FLOW_BY_BLOCK = {
    "water": ("water", "source"),
    "flowing_water": ("water", "flowing"),
    "lava": ("lava", "source"),
    "flowing_lava": ("lava", "flowing"),
}
EVALUATOR_TRUTH_LEAK_KEYS = frozenset(
    {
        "block_truth",
        "evaluator_dimension",
        "flow_state",
        "fluid_present",
        "fluid_truth",
        "grid_anchor",
        "observed_block",
        "portal_grid",
        "portal_grid_origin",
        "server_fluid_truth",
        "server_truth",
        "truth_snapshot",
    }
)

_TRUTH_ERROR_MARKERS = (
    ("duplicate world cell", TRUTH_DUPLICATE_CELL),
    ("duplicate grid cell", TRUTH_DUPLICATE_CELL),
    ("truth region is empty", TRUTH_REGION_EMPTY),
    ("outside the evaluator grid", TRUTH_CELL_OUT_OF_BOUNDS),
    ("grid anchor differs", TRUTH_ANCHOR_MISMATCH),
    ("unknown block truth", TRUTH_BLOCK_UNKNOWN),
    ("block truth is missing", TRUTH_BLOCK_MISSING),
    ("position truth is missing", TRUTH_POSITION_MISSING),
    ("position is invalid", TRUTH_POSITION_INVALID),
    ("dimension is missing", TRUTH_DIMENSION_MISSING),
    ("dimension is invalid", TRUTH_DIMENSION_INVALID),
    ("wrong dimension", TRUTH_WRONG_DIMENSION),
    ("leaked evaluator truth", TRUTH_LEAK),
    ("does not match block_truth records", TRUTH_BLOCK_MISSING),
    ("unknown fluid truth", TRUTH_FLUID_UNKNOWN),
    ("malformed fluid", TRUTH_FLUID_UNKNOWN),
    ("fluid truth is missing", TRUTH_FLUID_MISSING),
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_dimension(value: object, field_name: str = "dimension") -> str:
    name = _identifier(value, field_name)
    if name not in ALLOWED_DIMENSIONS:
        raise ValueError(f"{field_name} is invalid")
    return name


def validate_anchor_source(value: object) -> str:
    name = _identifier(value, "anchor_source")
    if name not in ALLOWED_ANCHOR_SOURCES:
        raise ValueError("anchor_source is invalid")
    return name


def validate_position_world(
    value: object, field_name: str = "position_world"
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be a finite (x, y, z) tuple")
    return (
        finite_number(value[0], f"{field_name}.x"),
        finite_number(value[1], f"{field_name}.y"),
        finite_number(value[2], f"{field_name}.z"),
    )


def validate_world_cells(
    cells: object, field_name: str = "cells"
) -> tuple[tuple[int, int, int], ...]:
    """Fail closed on empty, duplicate, or non-integer world cells."""

    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError(f"{field_name} must be a non-empty sequence of world cells")
    if len(cells) == 0:
        raise ValueError("truth region is empty")
    normalized: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for cell in cells:
        world = validate_target_cell(cell, "world_cell")
        if world in seen:
            raise ValueError("truth region contains a duplicate world cell")
        seen.add(world)
        normalized.append(world)
    return tuple(normalized)


class FluidCalibrationVariant(str, Enum):
    WATER = "water"
    LAVA = "lava"


def validate_fluid_variant(value: object) -> FluidCalibrationVariant:
    if isinstance(value, FluidCalibrationVariant):
        return value
    name = _identifier(value, "variant")
    try:
        return FluidCalibrationVariant(name)
    except ValueError as exc:
        raise ValueError("variant is not a frozen E9 fluid calibration") from exc


def validate_fluid_type(value: object, field_name: str = "fluid_type") -> str:
    name = _identifier(value, field_name)
    if name not in ALLOWED_FLUID_TYPES:
        raise ValueError(f"{field_name} is not an allowed E9 fluid type")
    return name


def validate_flow_state(value: object, field_name: str = "flow_state") -> str:
    name = _identifier(value, field_name)
    if name not in ALLOWED_FLOW_STATES:
        raise ValueError(f"{field_name} is not an allowed E9 flow state")
    return name


def classify_server_fluid(block: object) -> tuple[bool, str, str]:
    """Classify an exact ObservationFromGrid block into E9 fluid truth.

    Source and flowing states stay distinct. ``water`` is source water;
    ``flowing_water`` is flowing water; likewise for lava. Known non-fluid
    blocks become ``none``. ``missing``, ``other``, and unknown names fail
    closed instead of collapsing to ``none`` or to a coarse E7 class.
    """

    name = _identifier(block, "observed_block")
    if name in SOURCE_FLOW_BY_BLOCK:
        fluid_type, flow_state = SOURCE_FLOW_BY_BLOCK[name]
        return True, fluid_type, flow_state
    if name in NONE_FLUID_BLOCKS:
        return False, "none", "none"
    if name == "missing":
        raise ValueError("fluid truth is missing")
    raise ValueError("unknown fluid truth")


def frozen_fluid_bucket_item(variant: object) -> str:
    resolved = validate_fluid_variant(variant)
    if resolved is FluidCalibrationVariant.WATER:
        return "water_bucket"
    return "lava_bucket"


def frozen_expected_fluid_type(variant: object) -> str:
    return validate_fluid_variant(variant).value


def frozen_expected_flow_state(variant: object) -> str:
    validate_fluid_variant(variant)
    return "source"


def truth_error_outcome(exc: BaseException) -> str:
    """Map a fail-closed snapshot error onto the E8 outcome taxonomy."""

    message = str(exc).lower()
    for marker, outcome in _TRUTH_ERROR_MARKERS:
        if marker in message:
            return outcome
    return TRUTH_IDENTITY_MISMATCH


def public_payload_leaks_evaluator_truth(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(key in EVALUATOR_TRUTH_LEAK_KEYS for key in payload)


@dataclass(frozen=True)
class ServerBlockTruth:
    """One evaluator-only world/grid cell with an exact server block name."""

    world_cell: tuple[int, int, int]
    grid_cell: tuple[int, int, int]
    block: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "world_cell", validate_target_cell(self.world_cell, "world_cell")
        )
        object.__setattr__(
            self, "grid_cell", validate_target_cell(self.grid_cell, "grid_cell")
        )
        object.__setattr__(self, "block", validate_block_name(self.block, "block"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "grid_cell": list(self.grid_cell),
            "world_cell": list(self.world_cell),
        }


@dataclass(frozen=True)
class ServerFluidTruth:
    """One evaluator-only world/grid cell with exact source/flowing fluid state.

    ``observed_block`` is the raw ObservationFromGrid name. ``fluid_type``
    and ``flow_state`` must match that name; source and flowing water/lava
    are never collapsed.
    """

    world_cell: tuple[int, int, int]
    grid_cell: tuple[int, int, int]
    observed_block: str
    fluid_present: bool
    fluid_type: str
    flow_state: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "world_cell", validate_target_cell(self.world_cell, "world_cell")
        )
        object.__setattr__(
            self, "grid_cell", validate_target_cell(self.grid_cell, "grid_cell")
        )
        object.__setattr__(
            self, "observed_block", validate_block_name(self.observed_block, "observed_block")
        )
        if type(self.fluid_present) is not bool:
            raise ValueError("fluid_present must be bool")
        present, fluid_type, flow_state = classify_server_fluid(self.observed_block)
        object.__setattr__(self, "fluid_type", validate_fluid_type(self.fluid_type))
        object.__setattr__(self, "flow_state", validate_flow_state(self.flow_state))
        if (
            self.fluid_present != present
            or self.fluid_type != fluid_type
            or self.flow_state != flow_state
        ):
            raise ValueError("fluid truth fields do not match observed_block")

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow_state": self.flow_state,
            "fluid_present": self.fluid_present,
            "fluid_type": self.fluid_type,
            "grid_cell": list(self.grid_cell),
            "observed_block": self.observed_block,
            "world_cell": list(self.world_cell),
        }


@dataclass(frozen=True)
class ServerTruthSnapshot:
    """Evaluator-only identity context plus E8 block and E9 fluid truth.

    ``fluid_truth`` defaults to empty so existing E8 snapshots remain valid.
    E9 inspection requires a populated fluid region.
    """

    episode_id: str
    agent_id: str
    step_id: int
    position_world: tuple[float, float, float]
    dimension: str
    grid_anchor_world: tuple[int, int, int]
    anchor_source: str
    block_truth: tuple[ServerBlockTruth, ...]
    truth_missing_count: int = 0
    fluid_truth: tuple[ServerFluidTruth, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(
            self, "position_world", validate_position_world(self.position_world)
        )
        object.__setattr__(self, "dimension", validate_dimension(self.dimension))
        object.__setattr__(
            self,
            "grid_anchor_world",
            validate_target_cell(self.grid_anchor_world, "grid_anchor_world"),
        )
        object.__setattr__(self, "anchor_source", validate_anchor_source(self.anchor_source))
        if type(self.truth_missing_count) is not int or self.truth_missing_count < 0:
            raise ValueError("truth_missing_count must be a non-negative int")
        if not isinstance(self.block_truth, tuple):
            raise ValueError("block_truth must be a tuple of ServerBlockTruth")
        if not self.block_truth and self.truth_missing_count == 0:
            raise ValueError("block truth region is empty")
        world_cells: set[tuple[int, int, int]] = set()
        grid_cells: set[tuple[int, int, int]] = set()
        normalized: list[ServerBlockTruth] = []
        for item in self.block_truth:
            if not isinstance(item, ServerBlockTruth):
                raise ValueError("block_truth items must be ServerBlockTruth")
            expected_grid = spawn_relative_grid_cell(item.world_cell, self.grid_anchor_world)
            if item.grid_cell != expected_grid:
                raise ValueError("world/grid mapping mismatch")
            if item.world_cell in world_cells:
                raise ValueError("truth region contains a duplicate world cell")
            if item.grid_cell in grid_cells:
                raise ValueError("truth region contains a duplicate grid cell")
            world_cells.add(item.world_cell)
            grid_cells.add(item.grid_cell)
            normalized.append(item)
        object.__setattr__(self, "block_truth", tuple(normalized))
        if not isinstance(self.fluid_truth, tuple):
            raise ValueError("fluid_truth must be a tuple of ServerFluidTruth")
        fluid_world: set[tuple[int, int, int]] = set()
        fluid_grid: set[tuple[int, int, int]] = set()
        fluids: list[ServerFluidTruth] = []
        for item in self.fluid_truth:
            if not isinstance(item, ServerFluidTruth):
                raise ValueError("fluid_truth items must be ServerFluidTruth")
            expected_grid = spawn_relative_grid_cell(item.world_cell, self.grid_anchor_world)
            if item.grid_cell != expected_grid:
                raise ValueError("world/grid mapping mismatch")
            if item.world_cell in fluid_world:
                raise ValueError("truth region contains a duplicate world cell")
            if item.grid_cell in fluid_grid:
                raise ValueError("truth region contains a duplicate grid cell")
            fluid_world.add(item.world_cell)
            fluid_grid.add(item.grid_cell)
            fluids.append(item)
        object.__setattr__(self, "fluid_truth", tuple(fluids))
        if (
            self.block_truth
            and self.fluid_truth
            and self.truth_missing_count == 0
            and tuple(item.world_cell for item in self.block_truth)
            != tuple(item.world_cell for item in self.fluid_truth)
        ):
            raise ValueError("world/grid mapping mismatch")

    def block_at(self, world_cell: tuple[int, int, int]) -> str | None:
        world_cell = validate_target_cell(world_cell, "world_cell")
        matches = [item.block for item in self.block_truth if item.world_cell == world_cell]
        if len(matches) != 1:
            return None
        return matches[0]

    def fluid_at(self, world_cell: tuple[int, int, int]) -> ServerFluidTruth | None:
        world_cell = validate_target_cell(world_cell, "world_cell")
        matches = [item for item in self.fluid_truth if item.world_cell == world_cell]
        if len(matches) != 1:
            return None
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "anchor_source": self.anchor_source,
            "block_truth": [item.as_dict() for item in self.block_truth],
            "dimension": self.dimension,
            "episode_id": self.episode_id,
            "fluid_truth": [item.as_dict() for item in self.fluid_truth],
            "grid_anchor_world": list(self.grid_anchor_world),
            "position_world": list(self.position_world),
            "step_id": self.step_id,
            "truth_missing_count": self.truth_missing_count,
        }


@dataclass(frozen=True)
class BlockTruthActionExecution:
    """Observed backend response to the single bounded E8 dirt stimulus."""

    episode_id: str
    agent_id: str
    step_id: int
    action_type: str
    target: str
    duration_ticks: int
    translated_action_accepted: bool
    tested_action_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "action_type", _identifier(self.action_type, "action_type"))
        object.__setattr__(self, "target", _identifier(self.target, "target"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be a positive int")
        if type(self.translated_action_accepted) is not bool:
            raise ValueError("translated_action_accepted must be bool")
        if type(self.tested_action_count) is not int or self.tested_action_count < 0:
            raise ValueError("tested_action_count must be a non-negative int")


@dataclass(frozen=True)
class BlockTruthInspection:
    outcome: str
    error: str | None
    before_snapshot: ServerTruthSnapshot | None = None
    after_snapshot: ServerTruthSnapshot | None = None
    target_changed: bool | None = None
    target_expected_block_present: bool | None = None
    control_cells_unchanged: bool | None = None
    identity_valid: bool | None = None
    truth_missing_count: int | None = None

    def __post_init__(self) -> None:
        if self.outcome not in BLOCK_TRUTH_OUTCOMES:
            raise ValueError(f"unknown block-truth outcome: {self.outcome!r}")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        for field_name in (
            "target_changed",
            "target_expected_block_present",
            "control_cells_unchanged",
            "identity_valid",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field_name} must be bool or None")
        if self.truth_missing_count is not None and (
            type(self.truth_missing_count) is not int or self.truth_missing_count < 0
        ):
            raise ValueError("truth_missing_count must be a non-negative int or None")
        for field_name in ("before_snapshot", "after_snapshot"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ServerTruthSnapshot):
                raise ValueError(f"{field_name} must be ServerTruthSnapshot or None")

    @property
    def valid(self) -> bool:
        return self.outcome == BLOCK_TRUTH_OK


def inspect_block_truth(
    before: ServerTruthSnapshot,
    after: ServerTruthSnapshot,
    execution: BlockTruthActionExecution,
    *,
    probe_world_cells: Sequence[tuple[int, int, int]],
    probe_grid_cells: Sequence[tuple[int, int, int]],
    expected_before_blocks: Mapping[tuple[int, int, int], str],
    expected_after_blocks: Mapping[tuple[int, int, int], str],
    target_world_cell: tuple[int, int, int],
    control_world_cells: Sequence[tuple[int, int, int]],
    expected_dimension: str = E8_REQUIRED_DIMENSION,
    duration_ticks: int,
    stimulus_target: str,
    position_min: tuple[float, float, float] | None = None,
    position_max: tuple[float, float, float] | None = None,
) -> BlockTruthInspection:
    """Fail closed unless the region snapshot tracks the controlled stimulus."""

    probes = validate_world_cells(probe_world_cells, "probe_world_cells")
    grids = tuple(
        validate_target_cell(cell, "probe_grid_cell") for cell in probe_grid_cells
    )
    if len(grids) != len(probes):
        raise ValueError("probe world/grid regions must have the same length")
    target_world_cell = validate_target_cell(target_world_cell, "target_world_cell")
    controls = tuple(
        validate_target_cell(cell, "control_world_cell") for cell in control_world_cells
    )
    expected_dimension = validate_dimension(expected_dimension, "expected_dimension")
    stimulus_target = validate_block_name(stimulus_target, "stimulus_target")
    if type(duration_ticks) is not int or duration_ticks < 1:
        raise ValueError("duration_ticks must be a positive int")
    expected_before = {
        validate_target_cell(cell, "expected_before_cell"): validate_block_name(
            block, "expected_before_block"
        )
        for cell, block in expected_before_blocks.items()
    }
    expected_after = {
        validate_target_cell(cell, "expected_after_cell"): validate_block_name(
            block, "expected_after_block"
        )
        for cell, block in expected_after_blocks.items()
    }
    if execution.action_type != "place_block":
        return BlockTruthInspection(
            TRUTH_WRONG_ACTION_TYPE, "E8 stimulus must be place_block"
        )
    if execution.target != stimulus_target:
        return BlockTruthInspection(
            TRUTH_WRONG_TARGET, "E8 stimulus target differs from frozen calibration"
        )
    if execution.duration_ticks != duration_ticks:
        return BlockTruthInspection(
            TRUTH_CALIBRATION_MISMATCH,
            "E8 stimulus duration differs from frozen calibration",
        )
    if execution.tested_action_count == 0:
        return BlockTruthInspection(
            TRUTH_TEST_ACTION_NOT_EXECUTED, "E8 stimulus was not executed"
        )
    if execution.tested_action_count != 1:
        return BlockTruthInspection(
            TRUTH_MULTIPLE_TEST_ACTIONS, "exactly one E8 stimulus is required"
        )
    if not execution.translated_action_accepted:
        return BlockTruthInspection(
            TRUTH_STIMULUS_REJECTED, "E8 stimulus translation was rejected"
        )

    identity_valid = (
        before.episode_id == execution.episode_id
        and after.episode_id == execution.episode_id
        and before.agent_id == execution.agent_id
        and after.agent_id == execution.agent_id
        and before.step_id == 0
        and execution.step_id == 1
        and after.step_id == execution.step_id
        and before.grid_anchor_world == after.grid_anchor_world
        and before.anchor_source == after.anchor_source
    )
    missing_count = before.truth_missing_count + after.truth_missing_count
    if not identity_valid:
        return BlockTruthInspection(
            TRUTH_IDENTITY_MISMATCH,
            "block-truth identity, region, or step sequence is invalid",
            identity_valid=False,
            truth_missing_count=missing_count,
        )
    if missing_count != 0:
        return BlockTruthInspection(
            TRUTH_BLOCK_MISSING,
            "E8 success requires truth_missing_count=0",
            identity_valid=True,
            truth_missing_count=missing_count,
        )
    region_valid = (
        tuple(item.world_cell for item in before.block_truth) == probes
        and tuple(item.world_cell for item in after.block_truth) == probes
        and tuple(item.grid_cell for item in before.block_truth) == grids
        and tuple(item.grid_cell for item in after.block_truth) == grids
    )
    if not region_valid:
        return BlockTruthInspection(
            TRUTH_IDENTITY_MISMATCH,
            "block-truth identity, region, or step sequence is invalid",
            identity_valid=False,
            truth_missing_count=0,
        )
    if before.dimension != expected_dimension or after.dimension != expected_dimension:
        return BlockTruthInspection(
            TRUTH_WRONG_DIMENSION,
            "E8 calibration must remain in minecraft:overworld",
            identity_valid=True,
            truth_missing_count=missing_count,
        )
    for snapshot, label in ((before, "before"), (after, "after")):
        if position_min is not None or position_max is not None:
            for index, axis in enumerate(("x", "y", "z")):
                value = snapshot.position_world[index]
                if position_min is not None and value < position_min[index]:
                    return BlockTruthInspection(
                        TRUTH_POSITION_INVALID,
                        f"{label} position {axis} is outside E8 calibration bounds",
                        identity_valid=True,
                        truth_missing_count=missing_count,
                    )
                if position_max is not None and value > position_max[index]:
                    return BlockTruthInspection(
                        TRUTH_POSITION_INVALID,
                        f"{label} position {axis} is outside E8 calibration bounds",
                        identity_valid=True,
                        truth_missing_count=missing_count,
                    )
    def _lookup(snapshot: ServerTruthSnapshot, cell: tuple[int, int, int]) -> str | None:
        return snapshot.block_at(cell)

    for cell, expected in expected_before.items():
        observed = _lookup(before, cell)
        if observed is None or observed not in ALLOWED_BLOCK_NAMES:
            return BlockTruthInspection(
                TRUTH_BLOCK_UNKNOWN if observed is None else TRUTH_BEFORE_MISMATCH,
                "before snapshot is missing an exact probe block",
                identity_valid=True,
                truth_missing_count=0,
            )
        if observed != expected:
            return BlockTruthInspection(
                TRUTH_BEFORE_MISMATCH,
                "before snapshot does not match the frozen E8 region",
                identity_valid=True,
                truth_missing_count=0,
            )

    after_controls_ok = True
    for cell in controls:
        expected = expected_after[cell]
        observed = _lookup(after, cell)
        before_block = _lookup(before, cell)
        if observed is None or observed != expected or before_block != observed:
            after_controls_ok = False
            break
    target_after = _lookup(after, target_world_cell)
    target_before = _lookup(before, target_world_cell)
    target_changed = (
        target_before is not None
        and target_after is not None
        and target_before != target_after
    )
    target_present = target_after == expected_after[target_world_cell]
    evidence = dict(
        identity_valid=True,
        truth_missing_count=0,
        target_changed=target_changed,
        target_expected_block_present=target_present,
        control_cells_unchanged=after_controls_ok,
        before_snapshot=before,
        after_snapshot=after,
    )
    if not after_controls_ok:
        return BlockTruthInspection(
            TRUTH_CONTROL_CELL_CHANGED,
            "a control probe cell changed or is not the expected isolation block",
            **evidence,
        )
    if target_after != expected_after[target_world_cell]:
        return BlockTruthInspection(
            TRUTH_AFTER_MISMATCH,
            "after snapshot does not show the expected target block",
            **evidence,
        )
    if not target_changed or not target_present:
        return BlockTruthInspection(
            TRUTH_AFTER_MISMATCH,
            "target cell did not change to the expected server block",
            **evidence,
        )
    return BlockTruthInspection(BLOCK_TRUTH_OK, None, **evidence)


@dataclass(frozen=True)
class FluidTruthActionExecution:
    """Observed backend response to the single bounded E9 bucket stimulus."""

    episode_id: str
    agent_id: str
    step_id: int
    action_type: str
    target: str
    duration_ticks: int
    translated_action_accepted: bool
    tested_action_count: int
    variant: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "action_type", _identifier(self.action_type, "action_type"))
        object.__setattr__(self, "target", _identifier(self.target, "target"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be a positive int")
        if type(self.translated_action_accepted) is not bool:
            raise ValueError("translated_action_accepted must be bool")
        if type(self.tested_action_count) is not int or self.tested_action_count < 0:
            raise ValueError("tested_action_count must be a non-negative int")
        variant = validate_fluid_variant(self.variant)
        object.__setattr__(self, "variant", variant.value)


@dataclass(frozen=True)
class FluidTruthInspection:
    outcome: str
    error: str | None
    before_snapshot: ServerTruthSnapshot | None = None
    after_snapshot: ServerTruthSnapshot | None = None
    target_changed: bool | None = None
    target_expected_fluid_present: bool | None = None
    control_cells_unchanged: bool | None = None
    identity_valid: bool | None = None
    truth_missing_count: int | None = None
    source_flowing_match: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in FLUID_TRUTH_OUTCOMES:
            raise ValueError(f"unknown fluid-truth outcome: {self.outcome!r}")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        for field_name in (
            "target_changed",
            "target_expected_fluid_present",
            "control_cells_unchanged",
            "identity_valid",
            "source_flowing_match",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field_name} must be bool or None")
        if self.truth_missing_count is not None and (
            type(self.truth_missing_count) is not int or self.truth_missing_count < 0
        ):
            raise ValueError("truth_missing_count must be a non-negative int or None")
        for field_name in ("before_snapshot", "after_snapshot"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ServerTruthSnapshot):
                raise ValueError(f"{field_name} must be ServerTruthSnapshot or None")

    @property
    def valid(self) -> bool:
        return self.outcome == FLUID_TRUTH_OK


def _fluid_expectation(
    mapping: Mapping[tuple[int, int, int], tuple[str, str]],
    field_name: str,
) -> dict[tuple[int, int, int], tuple[str, str]]:
    expected: dict[tuple[int, int, int], tuple[str, str]] = {}
    for cell, value in mapping.items():
        world = validate_target_cell(cell, f"{field_name}_cell")
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(f"{field_name} values must be (fluid_type, flow_state)")
        fluid_type = validate_fluid_type(value[0], f"{field_name}_fluid_type")
        flow_state = validate_flow_state(value[1], f"{field_name}_flow_state")
        if fluid_type == "none" and flow_state != "none":
            raise ValueError("none fluid type requires flow_state none")
        if fluid_type != "none" and flow_state == "none":
            raise ValueError("present fluid type requires a source or flowing state")
        expected[world] = (fluid_type, flow_state)
    return expected


def inspect_fluid_truth(
    before: ServerTruthSnapshot,
    after: ServerTruthSnapshot,
    execution: FluidTruthActionExecution,
    *,
    probe_world_cells: Sequence[tuple[int, int, int]],
    probe_grid_cells: Sequence[tuple[int, int, int]],
    expected_before_fluids: Mapping[tuple[int, int, int], tuple[str, str]],
    expected_after_fluids: Mapping[tuple[int, int, int], tuple[str, str]],
    target_world_cell: tuple[int, int, int],
    control_world_cells: Sequence[tuple[int, int, int]],
    expected_dimension: str = E9_REQUIRED_DIMENSION,
    duration_ticks: int,
    stimulus_target: str,
    variant: object,
    position_min: tuple[float, float, float] | None = None,
    position_max: tuple[float, float, float] | None = None,
) -> FluidTruthInspection:
    """Fail closed unless region fluid truth tracks the controlled stimulus."""

    probes = validate_world_cells(probe_world_cells, "probe_world_cells")
    grids = tuple(
        validate_target_cell(cell, "probe_grid_cell") for cell in probe_grid_cells
    )
    if len(grids) != len(probes):
        raise ValueError("probe world/grid regions must have the same length")
    target_world_cell = validate_target_cell(target_world_cell, "target_world_cell")
    controls = tuple(
        validate_target_cell(cell, "control_world_cell") for cell in control_world_cells
    )
    expected_dimension = validate_dimension(expected_dimension, "expected_dimension")
    resolved_variant = validate_fluid_variant(variant)
    stimulus_target = _identifier(stimulus_target, "stimulus_target")
    if stimulus_target != frozen_fluid_bucket_item(resolved_variant):
        raise ValueError("stimulus_target does not match the frozen E9 variant")
    if type(duration_ticks) is not int or duration_ticks < 1:
        raise ValueError("duration_ticks must be a positive int")
    expected_before = _fluid_expectation(expected_before_fluids, "expected_before")
    expected_after = _fluid_expectation(expected_after_fluids, "expected_after")
    if execution.action_type != "use_item":
        return FluidTruthInspection(
            TRUTH_WRONG_ACTION_TYPE, "E9 stimulus must be use_item"
        )
    if execution.target != stimulus_target:
        return FluidTruthInspection(
            TRUTH_WRONG_TARGET, "E9 stimulus target differs from frozen calibration"
        )
    if execution.variant != resolved_variant.value:
        return FluidTruthInspection(
            TRUTH_CALIBRATION_MISMATCH,
            "E9 stimulus variant differs from frozen calibration",
        )
    if execution.duration_ticks != duration_ticks:
        return FluidTruthInspection(
            TRUTH_CALIBRATION_MISMATCH,
            "E9 stimulus duration differs from frozen calibration",
        )
    if execution.tested_action_count == 0:
        return FluidTruthInspection(
            TRUTH_TEST_ACTION_NOT_EXECUTED, "E9 stimulus was not executed"
        )
    if execution.tested_action_count != 1:
        return FluidTruthInspection(
            TRUTH_MULTIPLE_TEST_ACTIONS, "exactly one E9 stimulus is required"
        )
    if not execution.translated_action_accepted:
        return FluidTruthInspection(
            TRUTH_STIMULUS_REJECTED, "E9 stimulus translation was rejected"
        )

    identity_valid = (
        before.episode_id == execution.episode_id
        and after.episode_id == execution.episode_id
        and before.agent_id == execution.agent_id
        and after.agent_id == execution.agent_id
        and before.step_id == 0
        and execution.step_id == 1
        and after.step_id == execution.step_id
        and before.grid_anchor_world == after.grid_anchor_world
        and before.anchor_source == after.anchor_source
    )
    missing_count = before.truth_missing_count + after.truth_missing_count
    if not identity_valid:
        return FluidTruthInspection(
            TRUTH_IDENTITY_MISMATCH,
            "fluid-truth identity, region, or step sequence is invalid",
            identity_valid=False,
            truth_missing_count=missing_count,
        )
    if missing_count != 0:
        return FluidTruthInspection(
            TRUTH_FLUID_MISSING,
            "E9 success requires truth_missing_count=0",
            identity_valid=True,
            truth_missing_count=missing_count,
        )
    if not before.fluid_truth or not after.fluid_truth:
        return FluidTruthInspection(
            TRUTH_FLUID_MISSING,
            "fluid truth is missing",
            identity_valid=True,
            truth_missing_count=0,
        )
    region_valid = (
        tuple(item.world_cell for item in before.fluid_truth) == probes
        and tuple(item.world_cell for item in after.fluid_truth) == probes
        and tuple(item.grid_cell for item in before.fluid_truth) == grids
        and tuple(item.grid_cell for item in after.fluid_truth) == grids
    )
    if not region_valid:
        return FluidTruthInspection(
            TRUTH_IDENTITY_MISMATCH,
            "fluid-truth identity, region, or step sequence is invalid",
            identity_valid=False,
            truth_missing_count=0,
        )
    if before.dimension != expected_dimension or after.dimension != expected_dimension:
        return FluidTruthInspection(
            TRUTH_WRONG_DIMENSION,
            "E9 calibration must remain in minecraft:overworld",
            identity_valid=True,
            truth_missing_count=missing_count,
        )
    for snapshot, label in ((before, "before"), (after, "after")):
        if position_min is not None or position_max is not None:
            for index, axis in enumerate(("x", "y", "z")):
                value = snapshot.position_world[index]
                if position_min is not None and value < position_min[index]:
                    return FluidTruthInspection(
                        TRUTH_POSITION_INVALID,
                        f"{label} position {axis} is outside E9 calibration bounds",
                        identity_valid=True,
                        truth_missing_count=missing_count,
                    )
                if position_max is not None and value > position_max[index]:
                    return FluidTruthInspection(
                        TRUTH_POSITION_INVALID,
                        f"{label} position {axis} is outside E9 calibration bounds",
                        identity_valid=True,
                        truth_missing_count=missing_count,
                    )

    def _lookup(
        snapshot: ServerTruthSnapshot, cell: tuple[int, int, int]
    ) -> ServerFluidTruth | None:
        return snapshot.fluid_at(cell)

    for cell, expected in expected_before.items():
        observed = _lookup(before, cell)
        if observed is None:
            return FluidTruthInspection(
                TRUTH_FLUID_MISSING,
                "before snapshot is missing an exact probe fluid",
                identity_valid=True,
                truth_missing_count=0,
            )
        if (observed.fluid_type, observed.flow_state) != expected:
            if observed.fluid_type == expected[0] and observed.flow_state != expected[1]:
                return FluidTruthInspection(
                    TRUTH_SOURCE_FLOWING_MISMATCH,
                    "before snapshot source/flowing state differs from frozen E9 region",
                    identity_valid=True,
                    truth_missing_count=0,
                    source_flowing_match=False,
                )
            return FluidTruthInspection(
                TRUTH_BEFORE_FLUID_MISMATCH,
                "before snapshot does not match the frozen E9 fluid region",
                identity_valid=True,
                truth_missing_count=0,
            )

    after_controls_ok = True
    source_flowing_match = True
    for cell in controls:
        expected = expected_after[cell]
        observed = _lookup(after, cell)
        before_fluid = _lookup(before, cell)
        if (
            observed is None
            or (observed.fluid_type, observed.flow_state) != expected
            or before_fluid is None
            or (before_fluid.fluid_type, before_fluid.flow_state)
            != (observed.fluid_type, observed.flow_state)
        ):
            after_controls_ok = False
            if (
                observed is not None
                and observed.fluid_type == expected[0]
                and observed.flow_state != expected[1]
            ):
                source_flowing_match = False
            break
    target_after = _lookup(after, target_world_cell)
    target_before = _lookup(before, target_world_cell)
    expected_target = expected_after[target_world_cell]
    target_changed = (
        target_before is not None
        and target_after is not None
        and (
            target_before.fluid_type != target_after.fluid_type
            or target_before.flow_state != target_after.flow_state
        )
    )
    target_present = (
        target_after is not None
        and (target_after.fluid_type, target_after.flow_state) == expected_target
    )
    if (
        target_after is not None
        and target_after.fluid_type == expected_target[0]
        and target_after.flow_state != expected_target[1]
    ):
        source_flowing_match = False
    evidence = dict(
        identity_valid=True,
        truth_missing_count=0,
        target_changed=target_changed,
        target_expected_fluid_present=target_present,
        control_cells_unchanged=after_controls_ok,
        source_flowing_match=source_flowing_match,
        before_snapshot=before,
        after_snapshot=after,
    )
    if not after_controls_ok:
        if not source_flowing_match:
            return FluidTruthInspection(
                TRUTH_SOURCE_FLOWING_MISMATCH,
                "a control probe cell has the wrong source/flowing state",
                **evidence,
            )
        return FluidTruthInspection(
            TRUTH_CONTROL_CELL_CHANGED,
            "a control probe cell changed or is not the expected isolation fluid",
            **evidence,
        )
    if target_after is None:
        return FluidTruthInspection(
            TRUTH_FLUID_MISSING,
            "after snapshot is missing the target fluid",
            **evidence,
        )
    if target_after.fluid_type == expected_target[0] and target_after.flow_state != expected_target[1]:
        return FluidTruthInspection(
            TRUTH_SOURCE_FLOWING_MISMATCH,
            "target cell source/flowing state differs from frozen E9 calibration",
            **evidence,
        )
    if (target_after.fluid_type, target_after.flow_state) != expected_target:
        return FluidTruthInspection(
            TRUTH_AFTER_FLUID_MISMATCH,
            "after snapshot does not show the expected target fluid",
            **evidence,
        )
    if not target_changed or not target_present:
        return FluidTruthInspection(
            TRUTH_AFTER_FLUID_MISMATCH,
            "target cell did not change to the expected server fluid",
            **evidence,
        )
    return FluidTruthInspection(FLUID_TRUTH_OK, None, **evidence)
