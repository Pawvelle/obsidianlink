"""MineRL-independent P1 E7 bucket-usage calibration contract.

E7 verifies one bounded ``use_item`` on a filled water or lava bucket.
Success requires both a public inventory transition and an evaluator-only
fluid world effect. This module is not Agent-visible, not the future v2
canonical Observation, not E8/E9 generalized truth, and not the legacy
casting evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from obsidianlink.env.validation.inventory import inspect_inventory
from obsidianlink.env.validation.placement import (
    spawn_relative_grid_cell,
    validate_cell_coordinate,
    validate_target_cell,
)


BUCKET_OK = "bucket_ok"
INVENTORY_BEFORE_MISSING = "inventory_before_missing"
INVENTORY_AFTER_MISSING = "inventory_after_missing"
INVENTORY_INVALID = "inventory_invalid"
BUCKET_WRONG_ACTION_TYPE = "bucket_wrong_action_type"
BUCKET_WRONG_TARGET = "bucket_wrong_target"
BUCKET_CALIBRATION_MISMATCH = "bucket_calibration_mismatch"
BUCKET_ACTION_REJECTED = "bucket_action_rejected"
BUCKET_TEST_ACTION_NOT_EXECUTED = "bucket_test_action_not_executed"
BUCKET_MULTIPLE_TEST_ACTIONS = "bucket_multiple_test_actions"
BUCKET_STEP_IDENTITY_MISMATCH = "bucket_step_identity_mismatch"
BUCKET_INVENTORY_PRECONDITION_INVALID = "bucket_inventory_precondition_invalid"
BUCKET_INVENTORY_NO_CHANGE = "bucket_inventory_no_change"
BUCKET_INVENTORY_WRONG_CHANGE = "bucket_inventory_wrong_change"
FLUID_BEFORE_MISSING = "fluid_before_missing"
FLUID_AFTER_MISSING = "fluid_after_missing"
FLUID_TRUTH_INVALID = "fluid_truth_invalid"
BUCKET_FLUID_PREEXISTING = "bucket_fluid_preexisting"
BUCKET_NO_WORLD_EFFECT = "bucket_no_world_effect"
BUCKET_WRONG_FLUID_EFFECT = "bucket_wrong_fluid_effect"
BUCKET_TRUTH_LEAK = "bucket_truth_leak"

BUCKET_OUTCOMES = frozenset(
    {
        BUCKET_OK,
        INVENTORY_BEFORE_MISSING,
        INVENTORY_AFTER_MISSING,
        INVENTORY_INVALID,
        BUCKET_WRONG_ACTION_TYPE,
        BUCKET_WRONG_TARGET,
        BUCKET_CALIBRATION_MISMATCH,
        BUCKET_ACTION_REJECTED,
        BUCKET_TEST_ACTION_NOT_EXECUTED,
        BUCKET_MULTIPLE_TEST_ACTIONS,
        BUCKET_STEP_IDENTITY_MISMATCH,
        BUCKET_INVENTORY_PRECONDITION_INVALID,
        BUCKET_INVENTORY_NO_CHANGE,
        BUCKET_INVENTORY_WRONG_CHANGE,
        FLUID_BEFORE_MISSING,
        FLUID_AFTER_MISSING,
        FLUID_TRUTH_INVALID,
        BUCKET_FLUID_PREEXISTING,
        BUCKET_NO_WORLD_EFFECT,
        BUCKET_WRONG_FLUID_EFFECT,
        BUCKET_TRUTH_LEAK,
    }
)

ALLOWED_FLUID_CLASSES = frozenset({"none", "water", "lava"})
EMPTY_BUCKET_ITEM = "bucket"
WATER_BLOCKS = frozenset({"water", "flowing_water"})
LAVA_BLOCKS = frozenset({"lava", "flowing_lava"})
NONE_BLOCKS = frozenset({"air", "dirt", "grass", "grass_block"})


class BucketCalibrationVariant(str, Enum):
    WATER = "water"
    LAVA = "lava"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_fluid_class(value: object, field_name: str) -> str:
    name = _identifier(value, field_name)
    if name not in ALLOWED_FLUID_CLASSES:
        raise ValueError(f"{field_name} is not an allowed E7 fluid class")
    return name


def validate_bucket_variant(value: object) -> BucketCalibrationVariant:
    if isinstance(value, BucketCalibrationVariant):
        return value
    name = _identifier(value, "variant")
    try:
        return BucketCalibrationVariant(name)
    except ValueError as exc:
        raise ValueError("variant is not a frozen E7 bucket calibration") from exc


def classify_bucket_fluid(block: object) -> str:
    """Classify a portal-grid block name into the closed E7 fluid set.

    ``water`` / ``flowing_water`` become ``water``. ``lava`` /
    ``flowing_lava`` become ``lava``. Ordinary non-fluid replaceable
    states become ``none``. Unknown, missing, and portal/obsidian blocks
    fail closed instead of collapsing to ``none``. Source versus flowing
    distinction is an E9 question and is not answered here.
    """

    name = _identifier(block, "block")
    if name in WATER_BLOCKS:
        return "water"
    if name in LAVA_BLOCKS:
        return "lava"
    if name in NONE_BLOCKS:
        return "none"
    raise ValueError("block cannot be classified as an E7 fluid class")


def frozen_bucket_item(variant: object) -> str:
    resolved = validate_bucket_variant(variant)
    if resolved is BucketCalibrationVariant.WATER:
        return "water_bucket"
    return "lava_bucket"


def frozen_expected_fluid(variant: object) -> str:
    return validate_bucket_variant(variant).value


def frozen_before_inventory(variant: object) -> dict[str, int]:
    return {frozen_bucket_item(variant): 1}


def frozen_after_inventory(variant: object) -> dict[str, int]:
    return {EMPTY_BUCKET_ITEM: 1}


def inventory_quantity(inventory: Mapping[str, int], item: str) -> int:
    """Return a missing inventory key as quantity 0.

    The mapping itself must already be a valid public inventory. A missing
    entire inventory is not represented here and must fail closed before
    this helper is called. Explicit quantity ``0`` is malformed public
    inventory, not a missing key.
    """

    if not isinstance(item, str) or not item.strip():
        raise ValueError("inventory item must be a non-empty string")
    inspection = inspect_inventory(inventory)
    if not inspection.valid or inspection.inventory is None:
        raise ValueError(inspection.error or "inventory is invalid")
    return int(inspection.inventory.get(item, 0))


def _validated_public_inventory(
    inventory: object, field_name: str
) -> dict[str, int]:
    inspection = inspect_inventory(inventory)
    if not inspection.valid or inspection.inventory is None:
        detail = inspection.error or "invalid inventory"
        raise ValueError(f"{field_name} must be a valid public inventory: {detail}")
    return dict(inspection.inventory)


@dataclass(frozen=True)
class BucketInventorySnapshot:
    """Public inventory projection reused from the E2 adapter path."""

    episode_id: str
    agent_id: str
    step_id: int
    inventory: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(
            self, "inventory", _validated_public_inventory(self.inventory, "inventory")
        )

    def quantity(self, item: str) -> int:
        return inventory_quantity(self.inventory, item)


@dataclass(frozen=True)
class BucketFluidTruthSnapshot:
    """Temporary evaluator-only single-cell E7 fluid truth."""

    episode_id: str
    agent_id: str
    step_id: int
    world_x: int
    world_y: int
    world_z: int
    grid_x: int
    grid_y: int
    grid_z: int
    fluid: str
    fluid_present: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "world_x", validate_cell_coordinate(self.world_x, "world_x"))
        object.__setattr__(self, "world_y", validate_cell_coordinate(self.world_y, "world_y"))
        object.__setattr__(self, "world_z", validate_cell_coordinate(self.world_z, "world_z"))
        object.__setattr__(self, "grid_x", validate_cell_coordinate(self.grid_x, "grid_x"))
        object.__setattr__(self, "grid_y", validate_cell_coordinate(self.grid_y, "grid_y"))
        object.__setattr__(self, "grid_z", validate_cell_coordinate(self.grid_z, "grid_z"))
        object.__setattr__(self, "fluid", validate_fluid_class(self.fluid, "fluid"))
        if type(self.fluid_present) is not bool:
            raise ValueError("fluid_present must be bool")
        if self.fluid_present != (self.fluid != "none"):
            raise ValueError("fluid_present must match the classified fluid")

    @property
    def target_world_cell(self) -> tuple[int, int, int]:
        return (self.world_x, self.world_y, self.world_z)

    @property
    def target_grid_cell(self) -> tuple[int, int, int]:
        return (self.grid_x, self.grid_y, self.grid_z)


@dataclass(frozen=True)
class BucketActionExecution:
    """Observed backend response to the single bounded E7 use_item action."""

    episode_id: str
    agent_id: str
    step_id: int
    action_type: str
    target: str
    duration_ticks: int
    translated_action_accepted: bool
    tested_action_count: int
    variant: str
    expected_fluid: str

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
        variant = validate_bucket_variant(self.variant)
        object.__setattr__(self, "variant", variant.value)
        object.__setattr__(
            self, "expected_fluid", validate_fluid_class(self.expected_fluid, "expected_fluid")
        )
        if self.expected_fluid != frozen_expected_fluid(variant):
            raise ValueError("expected_fluid does not match the frozen E7 variant")


@dataclass(frozen=True)
class BucketUsageInspection:
    outcome: str
    error: str | None
    inventory_changed: bool | None = None
    bucket_consumed: bool | None = None
    empty_bucket_produced: bool | None = None
    before_fluid: str | None = None
    after_fluid: str | None = None
    fluid_changed: bool | None = None
    intended_fluid_present: bool | None = None
    identity_valid: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in BUCKET_OUTCOMES:
            raise ValueError(f"unknown bucket outcome: {self.outcome!r}")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        for field_name in (
            "inventory_changed",
            "bucket_consumed",
            "empty_bucket_produced",
            "fluid_changed",
            "intended_fluid_present",
            "identity_valid",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field_name} must be bool or None")
        for field_name in ("before_fluid", "after_fluid"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, validate_fluid_class(value, field_name)
                )

    @property
    def valid(self) -> bool:
        return self.outcome == BUCKET_OK


def inspect_bucket_usage(
    before_inventory: BucketInventorySnapshot,
    after_inventory: BucketInventorySnapshot,
    before_fluid: BucketFluidTruthSnapshot,
    after_fluid: BucketFluidTruthSnapshot,
    execution: BucketActionExecution,
    *,
    variant: object,
    bucket_item: str,
    expected_fluid: str,
    expected_before_inventory: Mapping[str, int],
    expected_after_inventory: Mapping[str, int],
    target_world_cell: tuple[int, int, int],
    target_grid_cell: tuple[int, int, int],
    duration_ticks: int,
) -> BucketUsageInspection:
    """Fail closed unless one accepted use_item changes inventory and fluid."""

    resolved_variant = validate_bucket_variant(variant)
    bucket_item = _identifier(bucket_item, "bucket_item")
    expected_fluid = validate_fluid_class(expected_fluid, "expected_fluid")
    expected_before = _validated_public_inventory(
        expected_before_inventory, "expected_before_inventory"
    )
    expected_after = _validated_public_inventory(
        expected_after_inventory, "expected_after_inventory"
    )
    target_world_cell = validate_target_cell(target_world_cell, "target_world_cell")
    target_grid_cell = validate_target_cell(target_grid_cell, "target_grid_cell")
    if type(duration_ticks) is not int or duration_ticks < 1:
        raise ValueError("duration_ticks must be a positive int")
    if bucket_item != frozen_bucket_item(resolved_variant):
        raise ValueError("bucket_item does not match the frozen E7 variant")
    if expected_fluid != frozen_expected_fluid(resolved_variant):
        raise ValueError("expected_fluid does not match the frozen E7 variant")
    if expected_before != frozen_before_inventory(resolved_variant):
        raise ValueError("expected_before_inventory does not match the frozen E7 variant")
    if expected_after != frozen_after_inventory(resolved_variant):
        raise ValueError("expected_after_inventory does not match the frozen E7 variant")
    if execution.action_type != "use_item":
        return BucketUsageInspection(
            BUCKET_WRONG_ACTION_TYPE, "tested action must be use_item"
        )
    if execution.target != bucket_item:
        return BucketUsageInspection(
            BUCKET_WRONG_TARGET,
            "tested use_item target differs from frozen E7 calibration",
        )
    if (
        execution.variant != resolved_variant.value
        or execution.expected_fluid != expected_fluid
        or execution.duration_ticks != duration_ticks
    ):
        return BucketUsageInspection(
            BUCKET_CALIBRATION_MISMATCH,
            "tested bucket action differs from frozen E7 calibration",
        )
    if execution.tested_action_count == 0:
        return BucketUsageInspection(
            BUCKET_TEST_ACTION_NOT_EXECUTED, "bucket test action was not executed"
        )
    if execution.tested_action_count != 1:
        return BucketUsageInspection(
            BUCKET_MULTIPLE_TEST_ACTIONS, "exactly one bucket test action is required"
        )
    if not execution.translated_action_accepted:
        return BucketUsageInspection(
            BUCKET_ACTION_REJECTED, "bucket action translation was rejected"
        )
    identity_valid = (
        before_inventory.episode_id == execution.episode_id
        and after_inventory.episode_id == execution.episode_id
        and before_fluid.episode_id == execution.episode_id
        and after_fluid.episode_id == execution.episode_id
        and before_inventory.agent_id == execution.agent_id
        and after_inventory.agent_id == execution.agent_id
        and before_fluid.agent_id == execution.agent_id
        and after_fluid.agent_id == execution.agent_id
        and before_inventory.step_id == 0
        and before_fluid.step_id == 0
        and execution.step_id == 1
        and after_inventory.step_id == execution.step_id
        and after_fluid.step_id == execution.step_id
        and before_fluid.target_world_cell == target_world_cell
        and after_fluid.target_world_cell == target_world_cell
        and before_fluid.target_grid_cell == target_grid_cell
        and after_fluid.target_grid_cell == target_grid_cell
        and spawn_relative_grid_cell(target_world_cell, (0, 4, 0)) == target_grid_cell
    )
    if not identity_valid:
        return BucketUsageInspection(
            BUCKET_STEP_IDENTITY_MISMATCH,
            "bucket identity, cell, or step sequence is invalid",
            identity_valid=False,
        )
    filled = bucket_item
    empty = EMPTY_BUCKET_ITEM
    inventory_changed = dict(before_inventory.inventory) != dict(after_inventory.inventory)
    bucket_consumed = (
        before_inventory.quantity(filled) == 1 and after_inventory.quantity(filled) == 0
    )
    empty_bucket_produced = (
        before_inventory.quantity(empty) == 0 and after_inventory.quantity(empty) == 1
    )
    fluid_changed = before_fluid.fluid != after_fluid.fluid
    intended_present = after_fluid.fluid == expected_fluid
    evidence = dict(
        inventory_changed=inventory_changed,
        bucket_consumed=bucket_consumed,
        empty_bucket_produced=empty_bucket_produced,
        before_fluid=before_fluid.fluid,
        after_fluid=after_fluid.fluid,
        fluid_changed=fluid_changed,
        intended_fluid_present=intended_present,
        identity_valid=True,
    )
    if dict(before_inventory.inventory) != expected_before:
        return BucketUsageInspection(
            BUCKET_INVENTORY_PRECONDITION_INVALID,
            "before inventory is not the frozen filled-bucket state",
            **evidence,
        )
    if dict(after_inventory.inventory) == expected_before:
        return BucketUsageInspection(
            BUCKET_INVENTORY_NO_CHANGE,
            "bucket inventory did not change",
            **evidence,
        )
    if dict(after_inventory.inventory) != expected_after:
        return BucketUsageInspection(
            BUCKET_INVENTORY_WRONG_CHANGE,
            "bucket inventory changed to a state other than empty bucket",
            **evidence,
        )
    if before_fluid.fluid != "none":
        return BucketUsageInspection(
            BUCKET_FLUID_PREEXISTING,
            "target cell already contained fluid before the bucket action",
            **evidence,
        )
    if not fluid_changed:
        return BucketUsageInspection(
            BUCKET_NO_WORLD_EFFECT,
            "bucket use produced no server-side fluid change",
            **evidence,
        )
    if not intended_present:
        return BucketUsageInspection(
            BUCKET_WRONG_FLUID_EFFECT,
            "bucket use changed the cell to a fluid other than the calibration target",
            **evidence,
        )
    return BucketUsageInspection(BUCKET_OK, None, **evidence)
