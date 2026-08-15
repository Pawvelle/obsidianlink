"""MineRL-independent P1 E6 block-placement calibration contract.

The single-cell block values here are a temporary P1 evaluator-only
surface. They are not Agent-visible, not the future v2 canonical
Observation, not E8 generalized server truth, and not the legacy
:class:`obsidianlink.core.types.Observation`.
"""

from __future__ import annotations

from dataclasses import dataclass


PLACEMENT_OK = "placement_ok"
BLOCK_BEFORE_MISSING = "block_before_missing"
BLOCK_AFTER_MISSING = "block_after_missing"
BLOCK_TRUTH_INVALID = "block_truth_invalid"
PLACEMENT_WRONG_ACTION_TYPE = "placement_wrong_action_type"
PLACEMENT_WRONG_TARGET = "placement_wrong_target"
PLACEMENT_CALIBRATION_MISMATCH = "placement_calibration_mismatch"
PLACEMENT_ACTION_REJECTED = "placement_action_rejected"
PLACEMENT_TEST_ACTION_NOT_EXECUTED = "placement_test_action_not_executed"
PLACEMENT_MULTIPLE_TEST_ACTIONS = "placement_multiple_test_actions"
PLACEMENT_STEP_IDENTITY_MISMATCH = "placement_step_identity_mismatch"
PLACEMENT_TARGET_PREEXISTING = "placement_target_preexisting"
PLACEMENT_NO_WORLD_EFFECT = "placement_no_world_effect"
PLACEMENT_WRONG_WORLD_EFFECT = "placement_wrong_world_effect"
PLACEMENT_TRUTH_LEAK = "placement_truth_leak"

PLACEMENT_OUTCOMES = frozenset(
    {
        PLACEMENT_OK,
        BLOCK_BEFORE_MISSING,
        BLOCK_AFTER_MISSING,
        BLOCK_TRUTH_INVALID,
        PLACEMENT_WRONG_ACTION_TYPE,
        PLACEMENT_WRONG_TARGET,
        PLACEMENT_CALIBRATION_MISMATCH,
        PLACEMENT_ACTION_REJECTED,
        PLACEMENT_TEST_ACTION_NOT_EXECUTED,
        PLACEMENT_MULTIPLE_TEST_ACTIONS,
        PLACEMENT_STEP_IDENTITY_MISMATCH,
        PLACEMENT_TARGET_PREEXISTING,
        PLACEMENT_NO_WORLD_EFFECT,
        PLACEMENT_WRONG_WORLD_EFFECT,
        PLACEMENT_TRUTH_LEAK,
    }
)

# Closed observed-block names copied from the portal grid vocabulary
# without importing MineRL or the future E8 truth API. ``other`` and
# ``missing`` are rejected so unknown/absent cells cannot pass.
ALLOWED_BLOCK_NAMES = frozenset(
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
        "water",
        "flowing_water",
        "lava",
        "flowing_lava",
    }
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_block_name(value: object, field_name: str) -> str:
    """Return a closed block name; unknown or missing names fail closed."""

    name = _identifier(value, field_name)
    if name not in ALLOWED_BLOCK_NAMES:
        raise ValueError(f"{field_name} is not an allowed E6 block name")
    return name


def validate_cell_coordinate(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an int")
    return value


def validate_target_cell(
    value: object, field_name: str = "target_cell"
) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be an int (x, y, z) tuple")
    return (
        validate_cell_coordinate(value[0], f"{field_name}.x"),
        validate_cell_coordinate(value[1], f"{field_name}.y"),
        validate_cell_coordinate(value[2], f"{field_name}.z"),
    )


def spawn_relative_grid_cell(
    world_cell: tuple[int, int, int],
    spawn_world: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Convert a Minecraft world cell to ObservationFromGrid(atSpawn=true) space.

    ``_cell_index_in_grid`` indexes the evaluator array in this spawn-relative
    namespace, not in world coordinates. The two are equal only when spawn is
    ``(0, 0, 0)``.
    """

    world_cell = validate_target_cell(world_cell, "world_cell")
    spawn_world = validate_target_cell(spawn_world, "spawn_world")
    return (
        world_cell[0] - spawn_world[0],
        world_cell[1] - spawn_world[1],
        world_cell[2] - spawn_world[2],
    )


@dataclass(frozen=True)
class BlockPlacementTruthSnapshot:
    """Temporary evaluator-only single-cell block truth."""

    episode_id: str
    agent_id: str
    step_id: int
    x: int
    y: int
    z: int
    block: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "x", validate_cell_coordinate(self.x, "x"))
        object.__setattr__(self, "y", validate_cell_coordinate(self.y, "y"))
        object.__setattr__(self, "z", validate_cell_coordinate(self.z, "z"))
        object.__setattr__(self, "block", validate_block_name(self.block, "block"))

    @property
    def cell(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class PlacementActionExecution:
    """Observed backend response to the single bounded E6 place_block action."""

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
class PlacementInspection:
    outcome: str
    error: str | None
    before_block: str | None = None
    after_block: str | None = None
    world_changed: bool | None = None
    intended_block_present: bool | None = None
    identity_valid: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in PLACEMENT_OUTCOMES:
            raise ValueError(f"unknown placement outcome: {self.outcome!r}")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        for field_name in ("world_changed", "intended_block_present", "identity_valid"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field_name} must be bool or None")
        for field_name in ("before_block", "after_block"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, validate_block_name(value, field_name)
                )

    @property
    def valid(self) -> bool:
        return self.outcome == PLACEMENT_OK


def inspect_block_placement(
    before: BlockPlacementTruthSnapshot,
    after: BlockPlacementTruthSnapshot,
    execution: PlacementActionExecution,
    *,
    calibration_block: str,
    expected_before_block: str,
    target_cell: tuple[int, int, int],
    duration_ticks: int,
) -> PlacementInspection:
    """Fail closed unless one accepted place_block changes the frozen cell."""

    calibration_block = validate_block_name(calibration_block, "calibration_block")
    expected_before_block = validate_block_name(
        expected_before_block, "expected_before_block"
    )
    target_cell = validate_target_cell(target_cell)
    if type(duration_ticks) is not int or duration_ticks < 1:
        raise ValueError("duration_ticks must be a positive int")
    if execution.action_type != "place_block":
        return PlacementInspection(
            PLACEMENT_WRONG_ACTION_TYPE, "tested action must be place_block"
        )
    if execution.target != calibration_block:
        return PlacementInspection(
            PLACEMENT_WRONG_TARGET, "tested place_block target differs from frozen E6 calibration"
        )
    if execution.duration_ticks != duration_ticks:
        return PlacementInspection(
            PLACEMENT_CALIBRATION_MISMATCH,
            "tested place_block duration differs from frozen E6 calibration",
        )
    if execution.tested_action_count == 0:
        return PlacementInspection(
            PLACEMENT_TEST_ACTION_NOT_EXECUTED, "placement test action was not executed"
        )
    if execution.tested_action_count != 1:
        return PlacementInspection(
            PLACEMENT_MULTIPLE_TEST_ACTIONS, "exactly one placement test action is required"
        )
    if not execution.translated_action_accepted:
        return PlacementInspection(
            PLACEMENT_ACTION_REJECTED, "placement action translation was rejected"
        )
    identity_valid = (
        before.episode_id == execution.episode_id
        and after.episode_id == execution.episode_id
        and before.agent_id == execution.agent_id
        and after.agent_id == execution.agent_id
        and before.step_id == 0
        and execution.step_id == 1
        and after.step_id == execution.step_id
        and before.cell == target_cell
        and after.cell == target_cell
    )
    if not identity_valid:
        return PlacementInspection(
            PLACEMENT_STEP_IDENTITY_MISMATCH,
            "placement identity, cell, or step sequence is invalid",
            identity_valid=False,
        )
    world_changed = before.block != after.block
    intended_present = after.block == calibration_block
    evidence = dict(
        before_block=before.block,
        after_block=after.block,
        world_changed=world_changed,
        intended_block_present=intended_present,
        identity_valid=True,
    )
    if before.block == calibration_block:
        return PlacementInspection(
            PLACEMENT_TARGET_PREEXISTING,
            "target cell already contained the intended calibration block",
            **evidence,
        )
    if before.block != expected_before_block:
        return PlacementInspection(
            PLACEMENT_CALIBRATION_MISMATCH,
            "target cell before-state differs from the frozen replaceable block",
            **evidence,
        )
    if not world_changed:
        return PlacementInspection(
            PLACEMENT_NO_WORLD_EFFECT,
            "placement produced no server-side block change",
            **evidence,
        )
    if not intended_present:
        return PlacementInspection(
            PLACEMENT_WRONG_WORLD_EFFECT,
            "placement changed the cell to a block other than the calibration target",
            **evidence,
        )
    return PlacementInspection(PLACEMENT_OK, None, **evidence)
