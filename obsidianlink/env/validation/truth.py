"""MineRL-independent P1 E8 server-side block-truth contract.

E8 validates the evaluator block-truth channel itself: given a target
region, can the evaluator read exact server-side block state? This is
not Agent-visible, not a benchmark task, not E6 placement success, and
not the E9 fluid-truth portion of ServerTruthSnapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
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

ALLOWED_DIMENSIONS = frozenset(
    {
        "minecraft:overworld",
        "minecraft:the_nether",
        "minecraft:the_end",
    }
)
E8_REQUIRED_DIMENSION = "minecraft:overworld"
ANCHOR_SOURCE_ORIGIN = "portal_grid_origin"
ANCHOR_SOURCE_SPAWN_FALLBACK = "expected_spawn_fallback"
ALLOWED_ANCHOR_SOURCES = frozenset(
    {ANCHOR_SOURCE_ORIGIN, ANCHOR_SOURCE_SPAWN_FALLBACK}
)
EVALUATOR_TRUTH_LEAK_KEYS = frozenset(
    {
        "block_truth",
        "evaluator_dimension",
        "grid_anchor",
        "portal_grid",
        "portal_grid_origin",
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
class ServerTruthSnapshot:
    """Evaluator-only E8/E9 identity context plus the E8 block-truth portion.

    Fluid truth is an E9 concern and is intentionally absent.
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

    def block_at(self, world_cell: tuple[int, int, int]) -> str | None:
        world_cell = validate_target_cell(world_cell, "world_cell")
        matches = [item.block for item in self.block_truth if item.world_cell == world_cell]
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
