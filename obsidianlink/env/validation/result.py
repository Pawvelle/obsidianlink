"""P1 environment-validation result type.

This type is independent from benchmark evaluation. It must not subclass
or reuse ``EvaluatorVerdict``. Offline results cannot claim
``integration_verified``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.camera import CAMERA_OK, CAMERA_OUTCOMES, finite_angle
from obsidianlink.env.validation.movement import MOVEMENT_OK, MOVEMENT_OUTCOMES, finite_number
from obsidianlink.env.validation.bucket import (
    BUCKET_OK,
    BUCKET_OUTCOMES,
    EMPTY_BUCKET_ITEM,
    frozen_after_inventory,
    frozen_before_inventory,
    frozen_bucket_item,
    frozen_expected_fluid,
    validate_bucket_variant,
    validate_fluid_class,
)
from obsidianlink.env.validation.placement import (
    PLACEMENT_OK,
    PLACEMENT_OUTCOMES,
    validate_block_name,
    validate_cell_coordinate,
)
from obsidianlink.env.validation.truth import (
    BLOCK_TRUTH_OK,
    BLOCK_TRUTH_OUTCOMES,
    E10_EXPECTED_AFTER_BLOCK,
    E10_EXPECTED_AFTER_WATER_BLOCK,
    E10_EXPECTED_AFTER_WATER_FLOW_STATE,
    E10_EXPECTED_AFTER_WATER_FLUID_TYPE,
    E10_EXPECTED_BEFORE_BLOCK,
    E10_STIMULUS_ITEM,
    FLUID_TRUTH_OK,
    FLUID_TRUTH_OUTCOMES,
    OBSIDIAN_CONVERSION_OK,
    OBSIDIAN_CONVERSION_OUTCOMES,
    PORTAL_ACTIVATION_OK,
    PORTAL_ACTIVATION_OUTCOMES,
    DIMENSION_TRANSITION_OK,
    DIMENSION_TRANSITION_OUTCOMES,
    E12_STIMULUS_ACTION_TYPE,
    frozen_expected_flow_state,
    frozen_expected_fluid_type,
    frozen_fluid_bucket_item,
    validate_anchor_source,
    validate_dimension,
    validate_flow_state,
    validate_fluid_type,
    validate_fluid_variant,
)
from obsidianlink.env.validation.inventory import (
    INVENTORY_OK,
    INVENTORY_OUTCOMES,
    inspect_inventory,
)
from obsidianlink.env.validation.selected_item import (
    SELECTED_ITEM_OK,
    SELECTED_ITEM_OUTCOMES,
    validate_selected_item,
)


UNIT_VERIFIED = "unit_verified"

INVENTORY_MISMATCH = "inventory_mismatch"
SELECTED_ITEM_MISMATCH = "selected_item_mismatch"

VALIDATION_OUTCOMES = frozenset(
    {
        "lifecycle_ok",
        "rgb_ok",
        "rgb_missing",
        "rgb_none",
        "rgb_shape_invalid",
        "rgb_dtype_invalid",
        "rgb_leak",
        "create_failed",
        "reset_failed",
        "initial_state_missing",
        "close_failed",
        "runtime_error",
    }
) | INVENTORY_OUTCOMES | SELECTED_ITEM_OUTCOMES | frozenset(
    {INVENTORY_MISMATCH, SELECTED_ITEM_MISMATCH}
) | CAMERA_OUTCOMES | MOVEMENT_OUTCOMES | PLACEMENT_OUTCOMES | BUCKET_OUTCOMES | BLOCK_TRUTH_OUTCOMES | FLUID_TRUTH_OUTCOMES | OBSIDIAN_CONVERSION_OUTCOMES | PORTAL_ACTIVATION_OUTCOMES | DIMENSION_TRANSITION_OUTCOMES | frozenset({"cleanup_failed", "action_failed"})
E0_SUCCESS_OUTCOME = "lifecycle_ok"
E1_SUCCESS_OUTCOME = "rgb_ok"
E2_SUCCESS_OUTCOME = INVENTORY_OK
E3_SUCCESS_OUTCOME = SELECTED_ITEM_OK
E4_SUCCESS_OUTCOME = CAMERA_OK
E5_SUCCESS_OUTCOME = MOVEMENT_OK
E6_SUCCESS_OUTCOME = PLACEMENT_OK
E7_SUCCESS_OUTCOME = BUCKET_OK
E8_SUCCESS_OUTCOME = BLOCK_TRUTH_OK
E9_SUCCESS_OUTCOME = FLUID_TRUTH_OK
E10_SUCCESS_OUTCOME = OBSIDIAN_CONVERSION_OK
E11_SUCCESS_OUTCOME = PORTAL_ACTIVATION_OK
E12_SUCCESS_OUTCOME = DIMENSION_TRANSITION_OK


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _optional_error(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be None or a non-empty string")
    return value.strip()


def _inventory_snapshot(
    value: object, field_name: str
) -> dict[str, int]:
    inspection = inspect_inventory(value)
    if not inspection.valid or inspection.inventory is None:
        detail = inspection.error or "invalid inventory"
        raise ValueError(f"{field_name} must be a valid inventory Mapping: {detail}")
    return dict(inspection.inventory)


def _probe_cell_region(
    value: object, field_name: str
) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple of cells")
    cells: list[tuple[int, int, int]] = []
    for cell in value:
        if not isinstance(cell, tuple) or len(cell) != 3:
            raise ValueError(f"{field_name} must contain int (x, y, z) tuples")
        cells.append(
            (
                validate_cell_coordinate(cell[0], f"{field_name}.x"),
                validate_cell_coordinate(cell[1], f"{field_name}.y"),
                validate_cell_coordinate(cell[2], f"{field_name}.z"),
            )
        )
    return tuple(cells)


def _block_truth_records(
    value: object, field_name: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple of block-truth records")
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} items must be mappings")
        required = {"block", "grid_cell", "world_cell"}
        if set(item) != required:
            raise ValueError(f"{field_name} fields are missing or unknown")
        world = item["world_cell"]
        grid = item["grid_cell"]
        if not isinstance(world, (list, tuple)) or len(world) != 3:
            raise ValueError(f"{field_name}.world_cell must be an int triple")
        if not isinstance(grid, (list, tuple)) or len(grid) != 3:
            raise ValueError(f"{field_name}.grid_cell must be an int triple")
        records.append(
            {
                "block": validate_block_name(item["block"], f"{field_name}.block"),
                "grid_cell": [
                    validate_cell_coordinate(grid[0], f"{field_name}.grid_x"),
                    validate_cell_coordinate(grid[1], f"{field_name}.grid_y"),
                    validate_cell_coordinate(grid[2], f"{field_name}.grid_z"),
                ],
                "world_cell": [
                    validate_cell_coordinate(world[0], f"{field_name}.world_x"),
                    validate_cell_coordinate(world[1], f"{field_name}.world_y"),
                    validate_cell_coordinate(world[2], f"{field_name}.world_z"),
                ],
            }
        )
    return tuple(records)


def _fluid_truth_records(
    value: object, field_name: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple of fluid-truth records")
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} items must be mappings")
        required = {
            "flow_state",
            "fluid_present",
            "fluid_type",
            "grid_cell",
            "observed_block",
            "world_cell",
        }
        if set(item) != required:
            raise ValueError(f"{field_name} fields are missing or unknown")
        world = item["world_cell"]
        grid = item["grid_cell"]
        if not isinstance(world, (list, tuple)) or len(world) != 3:
            raise ValueError(f"{field_name}.world_cell must be an int triple")
        if not isinstance(grid, (list, tuple)) or len(grid) != 3:
            raise ValueError(f"{field_name}.grid_cell must be an int triple")
        if type(item["fluid_present"]) is not bool:
            raise ValueError(f"{field_name}.fluid_present must be bool")
        records.append(
            {
                "flow_state": validate_flow_state(item["flow_state"], f"{field_name}.flow_state"),
                "fluid_present": item["fluid_present"],
                "fluid_type": validate_fluid_type(item["fluid_type"], f"{field_name}.fluid_type"),
                "grid_cell": [
                    validate_cell_coordinate(grid[0], f"{field_name}.grid_x"),
                    validate_cell_coordinate(grid[1], f"{field_name}.grid_y"),
                    validate_cell_coordinate(grid[2], f"{field_name}.grid_z"),
                ],
                "observed_block": validate_block_name(
                    item["observed_block"], f"{field_name}.observed_block"
                ),
                "world_cell": [
                    validate_cell_coordinate(world[0], f"{field_name}.world_x"),
                    validate_cell_coordinate(world[1], f"{field_name}.world_y"),
                    validate_cell_coordinate(world[2], f"{field_name}.world_z"),
                ],
            }
        )
    return tuple(records)


@dataclass(frozen=True)
class EnvironmentValidationResult:
    """Deterministic, serializable P1 validation result.

    Success is fail-closed: missing or invalid observations/truth, calibration
    mismatch, lifecycle failure, cleanup failure, or any runtime exception
    cannot be recorded as a clean result. This runtime always
    emits ``unit_verified`` and never sets ``integration_verified`` or
    ``real_execution_performed``.
    """

    check_id: EnvironmentValidationId
    name: str
    episode_id: str
    step_id: int
    success: bool
    outcome: str
    created: bool
    reset_completed: bool
    initial_state_present: bool
    closed: bool
    error: str | None = None
    close_error: str | None = None
    failure_stage: str | None = None
    original_exception_type: str | None = None
    reset_attempt_count: int | None = None
    environment_launch_count: int | None = None
    exception_traceback: str | None = None
    verification_level: str = UNIT_VERIFIED
    real_execution_performed: bool = False
    integration_verified: bool = False
    calibration_only: bool = True
    rgb_present: bool | None = None
    rgb_height: int | None = None
    rgb_width: int | None = None
    rgb_channels: int | None = None
    rgb_dtype: str | None = None
    inventory_present: bool | None = None
    observed_inventory: Mapping[str, int] | None = None
    expected_inventory: Mapping[str, int] | None = None
    inventory_matches_expected: bool | None = None
    selected_item_present: bool | None = None
    observed_selected_item: str | None = None
    expected_selected_item: str | None = None
    selected_item_matches_expected: bool | None = None
    agent_id: str | None = None
    tested_step_id: int | None = None
    action_type: str | None = None
    requested_yaw: float | None = None
    requested_pitch: float | None = None
    translated_action_accepted: bool | None = None
    tested_action_count: int | None = None
    before_yaw: float | None = None
    before_pitch: float | None = None
    after_yaw: float | None = None
    after_pitch: float | None = None
    normalized_yaw_delta: float | None = None
    pitch_delta: float | None = None
    direction_match: bool | None = None
    magnitude_match: bool | None = None
    requested_forward: float | None = None
    requested_strafe: float | None = None
    requested_sprint: bool | None = None
    requested_jump: bool | None = None
    requested_duration_ticks: int | None = None
    before_x: float | None = None
    before_y: float | None = None
    before_z: float | None = None
    movement_before_yaw: float | None = None
    after_x: float | None = None
    after_y: float | None = None
    after_z: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    delta_z: float | None = None
    horizontal_distance: float | None = None
    total_distance: float | None = None
    forward_projection: float | None = None
    lateral_projection: float | None = None
    minimum_horizontal_distance: float | None = None
    minimum_forward_projection: float | None = None
    maximum_lateral_drift: float | None = None
    maximum_horizontal_distance: float | None = None
    maximum_vertical_drift: float | None = None
    moved: bool | None = None
    movement_direction_match: bool | None = None
    lateral_drift_ok: bool | None = None
    teleport_guard_ok: bool | None = None
    vertical_drift_ok: bool | None = None
    requested_target: str | None = None
    calibration_block: str | None = None
    expected_before_block: str | None = None
    target_x: int | None = None
    target_y: int | None = None
    target_z: int | None = None
    target_grid_x: int | None = None
    target_grid_y: int | None = None
    target_grid_z: int | None = None
    before_block: str | None = None
    after_block: str | None = None
    world_changed: bool | None = None
    intended_block_present: bool | None = None
    bucket_variant: str | None = None
    bucket_item: str | None = None
    expected_fluid: str | None = None
    before_inventory: Mapping[str, int] | None = None
    after_inventory: Mapping[str, int] | None = None
    inventory_changed: bool | None = None
    bucket_consumed: bool | None = None
    empty_bucket_produced: bool | None = None
    before_fluid: str | None = None
    after_fluid: str | None = None
    fluid_changed: bool | None = None
    intended_fluid_present: bool | None = None
    before_selected_item: str | None = None
    after_selected_item: str | None = None
    before_step_id: int | None = None
    after_step_id: int | None = None
    before_position_x: float | None = None
    before_position_y: float | None = None
    before_position_z: float | None = None
    after_position_x: float | None = None
    after_position_y: float | None = None
    after_position_z: float | None = None
    before_dimension: str | None = None
    after_dimension: str | None = None
    grid_anchor_x: int | None = None
    grid_anchor_y: int | None = None
    grid_anchor_z: int | None = None
    anchor_source: str | None = None
    probe_world_cells: tuple[tuple[int, int, int], ...] | None = None
    probe_grid_cells: tuple[tuple[int, int, int], ...] | None = None
    before_block_truth: tuple[Mapping[str, object], ...] | None = None
    after_block_truth: tuple[Mapping[str, object], ...] | None = None
    truth_missing_count: int | None = None
    stimulus_target: str | None = None
    target_changed: bool | None = None
    target_expected_block_present: bool | None = None
    control_cells_unchanged: bool | None = None
    before_server_fluid_truth: tuple[Mapping[str, object], ...] | None = None
    after_server_fluid_truth: tuple[Mapping[str, object], ...] | None = None
    fluid_variant: str | None = None
    expected_target_fluid_type: str | None = None
    expected_target_flow_state: str | None = None
    target_expected_fluid_present: bool | None = None
    source_flowing_match: bool | None = None
    before_target_block: str | None = None
    after_target_block: str | None = None
    before_water_block: str | None = None
    after_water_block: str | None = None
    before_water_fluid_type: str | None = None
    before_water_flow_state: str | None = None
    after_water_fluid_type: str | None = None
    after_water_flow_state: str | None = None
    water_placement_observed: bool | None = None
    obsidian_present: bool | None = None
    conversion_observed: bool | None = None
    conversion_observed_at_step: int | None = None
    observation_window_ticks: int | None = None
    observation_wait_count: int | None = None
    frame_valid_before: bool | None = None
    frame_block_count: int | None = None
    expected_frame_cells: tuple[tuple[int, int, int], ...] | None = None
    observed_frame_cells: tuple[tuple[int, int, int], ...] | None = None
    interior_cells: tuple[tuple[int, int, int], ...] | None = None
    before_portal_block_count: int | None = None
    after_portal_block_count: int | None = None
    ignition_effect_observed: bool | None = None
    portal_activation_observed: bool | None = None
    portal_activation_observed_at_step: int | None = None
    portal_activated: bool | None = None
    active_portal_before: bool | None = None
    dimension_transition_observed: bool | None = None
    dimension_transition_observed_at_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, EnvironmentValidationId):
            raise ValueError("check_id must be EnvironmentValidationId")
        _require_identifier(self.name, "name")
        _require_identifier(self.episode_id, "episode_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        for field_name in (
            "success",
            "created",
            "reset_completed",
            "initial_state_present",
            "closed",
            "real_execution_performed",
            "integration_verified",
            "calibration_only",
        ):
            _require_bool(getattr(self, field_name), field_name)
        if self.outcome not in VALIDATION_OUTCOMES:
            raise ValueError(f"unknown validation outcome: {self.outcome!r}")
        object.__setattr__(self, "error", _optional_error(self.error, "error"))
        object.__setattr__(
            self, "close_error", _optional_error(self.close_error, "close_error")
        )
        object.__setattr__(
            self,
            "exception_traceback",
            _optional_error(self.exception_traceback, "exception_traceback"),
        )
        if self.failure_stage is not None:
            object.__setattr__(
                self,
                "failure_stage",
                _require_identifier(self.failure_stage, "failure_stage"),
            )
        if self.original_exception_type is not None:
            object.__setattr__(
                self,
                "original_exception_type",
                _require_identifier(
                    self.original_exception_type, "original_exception_type"
                ),
            )
        for field_name in ("reset_attempt_count", "environment_launch_count"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative int or None")
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.real_execution_performed:
            raise ValueError("this runtime cannot claim real execution")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("P1 validation results must remain calibration-only")
        for field_name in (
            "rgb_height",
            "rgb_width",
            "rgb_channels",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative int or None")
        if self.rgb_present is not None:
            _require_bool(self.rgb_present, "rgb_present")
        if self.rgb_dtype is not None:
            _require_identifier(self.rgb_dtype, "rgb_dtype")
        if self.inventory_present is not None:
            _require_bool(self.inventory_present, "inventory_present")
        if self.inventory_matches_expected is not None:
            _require_bool(
                self.inventory_matches_expected,
                "inventory_matches_expected",
            )
        if self.selected_item_present is not None:
            _require_bool(self.selected_item_present, "selected_item_present")
        if self.selected_item_matches_expected is not None:
            _require_bool(
                self.selected_item_matches_expected,
                "selected_item_matches_expected",
            )
        for field_name in (
            "translated_action_accepted",
            "direction_match",
            "magnitude_match",
            "requested_sprint", "requested_jump", "moved",
            "movement_direction_match", "lateral_drift_ok",
            "teleport_guard_ok", "vertical_drift_ok",
            "world_changed", "intended_block_present",
            "inventory_changed", "bucket_consumed", "empty_bucket_produced",
            "fluid_changed", "intended_fluid_present",
            "target_changed", "target_expected_block_present",
            "control_cells_unchanged",
            "target_expected_fluid_present", "source_flowing_match",
            "obsidian_present", "conversion_observed",
            "water_placement_observed",
            "frame_valid_before", "ignition_effect_observed",
            "portal_activation_observed", "portal_activated",
            "active_portal_before", "dimension_transition_observed",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_bool(value, field_name)
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", _require_identifier(self.agent_id, "agent_id"))
        if self.action_type is not None:
            object.__setattr__(self, "action_type", _require_identifier(self.action_type, "action_type"))
        for field_name in ("tested_step_id", "tested_action_count"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative int or None")
        if self.requested_duration_ticks is not None and (
            type(self.requested_duration_ticks) is not int
            or self.requested_duration_ticks < 1
        ):
            raise ValueError("requested_duration_ticks must be a positive int or None")
        for field_name in (
            "requested_yaw", "requested_pitch", "before_yaw", "before_pitch",
            "after_yaw", "after_pitch", "normalized_yaw_delta", "pitch_delta",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, finite_angle(value, field_name))
        for field_name in (
            "requested_forward", "requested_strafe", "before_x", "before_y",
            "before_z", "movement_before_yaw", "after_x", "after_y", "after_z",
            "delta_x", "delta_y", "delta_z", "horizontal_distance",
            "total_distance", "forward_projection", "lateral_projection",
            "minimum_horizontal_distance", "minimum_forward_projection",
            "maximum_lateral_drift", "maximum_horizontal_distance",
            "maximum_vertical_drift",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, finite_number(value, field_name))
        if self.observed_selected_item is not None:
            object.__setattr__(
                self,
                "observed_selected_item",
                validate_selected_item(
                    self.observed_selected_item, "observed_selected_item"
                ),
            )
        if self.expected_selected_item is not None:
            object.__setattr__(
                self,
                "expected_selected_item",
                validate_selected_item(
                    self.expected_selected_item, "expected_selected_item"
                ),
            )
        if self.observed_inventory is not None:
            object.__setattr__(
                self,
                "observed_inventory",
                _inventory_snapshot(self.observed_inventory, "observed_inventory"),
            )
        if self.expected_inventory is not None:
            object.__setattr__(
                self,
                "expected_inventory",
                _inventory_snapshot(self.expected_inventory, "expected_inventory"),
            )
        if self.check_id is not EnvironmentValidationId.E2 and any(
            value is not None
            for value in (
                self.inventory_present,
                self.observed_inventory,
                self.expected_inventory,
                self.inventory_matches_expected,
            )
        ):
            raise ValueError("inventory metadata is only valid for E2 results")
        if self.check_id is not EnvironmentValidationId.E3 and any(
            value is not None
            for value in (
                self.selected_item_present,
                self.observed_selected_item,
                self.expected_selected_item,
                self.selected_item_matches_expected,
            )
        ):
            raise ValueError("selected-item metadata is only valid for E3 results")
        action_fields = (
            self.agent_id, self.tested_step_id, self.action_type,
            self.translated_action_accepted, self.tested_action_count,
        )
        if self.check_id not in (
            EnvironmentValidationId.E4,
            EnvironmentValidationId.E5,
            EnvironmentValidationId.E6,
            EnvironmentValidationId.E7,
            EnvironmentValidationId.E8,
            EnvironmentValidationId.E9,
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(
            value is not None for value in action_fields
        ):
            raise ValueError("action metadata is only valid for E4/E5/E6/E7/E8/E9/E10/E11/E12 results")
        camera_fields = (
            self.requested_yaw, self.requested_pitch,
            self.before_yaw, self.before_pitch, self.after_yaw, self.after_pitch,
            self.normalized_yaw_delta, self.pitch_delta,
            self.direction_match, self.magnitude_match,
        )
        if self.check_id is not EnvironmentValidationId.E4 and any(
            value is not None for value in camera_fields
        ):
            raise ValueError("camera metadata is only valid for E4 results")
        movement_fields = (
            self.requested_forward, self.requested_strafe, self.requested_sprint,
            self.requested_jump,
            self.before_x, self.before_y, self.before_z, self.movement_before_yaw,
            self.after_x, self.after_y, self.after_z,
            self.delta_x, self.delta_y, self.delta_z, self.horizontal_distance,
            self.total_distance, self.forward_projection, self.lateral_projection,
            self.minimum_horizontal_distance, self.minimum_forward_projection,
            self.maximum_lateral_drift, self.maximum_horizontal_distance,
            self.maximum_vertical_drift, self.moved, self.movement_direction_match,
            self.lateral_drift_ok, self.teleport_guard_ok, self.vertical_drift_ok,
        )
        if self.check_id is not EnvironmentValidationId.E5 and any(
            value is not None for value in movement_fields
        ):
            raise ValueError("movement metadata is only valid for E5 results")
        if self.requested_duration_ticks is not None and self.check_id not in (
            EnvironmentValidationId.E5,
            EnvironmentValidationId.E6,
            EnvironmentValidationId.E7,
            EnvironmentValidationId.E8,
            EnvironmentValidationId.E9,
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ):
            raise ValueError("requested_duration_ticks is only valid for E5/E6/E7/E8/E9/E10/E11/E12 results")
        if self.check_id not in (
            EnvironmentValidationId.E6,
            EnvironmentValidationId.E7,
        ) and any(
            value is not None for value in (
                self.target_x, self.target_y, self.target_z,
                self.target_grid_x, self.target_grid_y, self.target_grid_z,
            )
        ):
            raise ValueError("target-cell metadata is only valid for E6/E7 results")
        placement_fields = (
            self.requested_target, self.calibration_block, self.expected_before_block,
            self.before_block, self.after_block,
            self.world_changed, self.intended_block_present,
        )
        if self.check_id is not EnvironmentValidationId.E6 and any(
            value is not None for value in placement_fields
        ):
            raise ValueError("placement metadata is only valid for E6 results")
        for field_name in (
            "target_x", "target_y", "target_z",
            "target_grid_x", "target_grid_y", "target_grid_z",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, validate_cell_coordinate(value, field_name))
        for field_name in (
            "requested_target", "calibration_block", "expected_before_block",
            "before_block", "after_block",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, validate_block_name(value, field_name))
        bucket_fields = (
            self.bucket_variant, self.bucket_item, self.expected_fluid,
            self.before_inventory, self.after_inventory,
            self.inventory_changed, self.bucket_consumed, self.empty_bucket_produced,
            self.before_fluid, self.after_fluid, self.fluid_changed,
            self.intended_fluid_present, self.before_selected_item,
            self.after_selected_item,
        )
        if self.check_id is not EnvironmentValidationId.E7 and any(
            value is not None for value in bucket_fields
        ):
            raise ValueError("bucket metadata is only valid for E7 results")
        shared_truth_fields = (
            self.before_step_id, self.after_step_id,
            self.before_position_x, self.before_position_y, self.before_position_z,
            self.after_position_x, self.after_position_y, self.after_position_z,
            self.before_dimension, self.after_dimension,
            self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z,
            self.anchor_source, self.probe_world_cells, self.probe_grid_cells,
            self.truth_missing_count, self.stimulus_target,
            self.target_changed, self.control_cells_unchanged,
        )
        e8_only_fields = (
            self.before_block_truth, self.after_block_truth,
            self.target_expected_block_present,
        )
        e9_only_fields = (
            self.before_server_fluid_truth, self.after_server_fluid_truth,
            self.fluid_variant, self.expected_target_fluid_type,
            self.expected_target_flow_state, self.target_expected_fluid_present,
            self.source_flowing_match,
        )
        e10_only_fields = (
            self.before_target_block, self.after_target_block,
            self.before_water_block, self.after_water_block,
            self.before_water_fluid_type, self.before_water_flow_state,
            self.after_water_fluid_type, self.after_water_flow_state,
            self.water_placement_observed,
            self.obsidian_present, self.conversion_observed,
            self.conversion_observed_at_step,
        )
        e11_only_fields = (
            self.after_portal_block_count,
            self.ignition_effect_observed, self.portal_activation_observed,
            self.portal_activation_observed_at_step, self.portal_activated,
        )
        e11_e12_frame_fields = (
            self.frame_valid_before, self.frame_block_count,
            self.expected_frame_cells, self.observed_frame_cells,
            self.interior_cells, self.before_portal_block_count,
        )
        e12_only_fields = (
            self.active_portal_before,
            self.dimension_transition_observed,
            self.dimension_transition_observed_at_step,
        )
        e10_e11_window_fields = (
            self.observation_window_ticks, self.observation_wait_count,
        )
        if self.check_id not in (
            EnvironmentValidationId.E8,
            EnvironmentValidationId.E9,
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(value is not None for value in shared_truth_fields):
            raise ValueError("server-truth metadata is only valid for E8/E9/E10/E11/E12 results")
        if self.check_id not in (
            EnvironmentValidationId.E8,
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(
            value is not None for value in e8_only_fields
        ):
            raise ValueError("block-truth metadata is only valid for E8/E10/E11/E12 results")
        if self.check_id not in (
            EnvironmentValidationId.E9,
            EnvironmentValidationId.E10,
        ) and any(
            value is not None for value in e9_only_fields
        ):
            raise ValueError("fluid-truth metadata is only valid for E9/E10 results")
        if self.check_id is not EnvironmentValidationId.E10 and any(
            value is not None for value in e10_only_fields
        ):
            raise ValueError("obsidian-conversion metadata is only valid for E10 results")
        if self.check_id not in (
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(value is not None for value in e10_e11_window_fields):
            raise ValueError("observation-window metadata is only valid for E10/E11/E12 results")
        if self.check_id not in (
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(value is not None for value in e11_e12_frame_fields):
            raise ValueError("portal-frame metadata is only valid for E11/E12 results")
        if self.check_id is not EnvironmentValidationId.E11 and any(
            value is not None for value in e11_only_fields
        ):
            raise ValueError("portal-activation metadata is only valid for E11 results")
        if self.check_id is not EnvironmentValidationId.E12 and any(
            value is not None for value in e12_only_fields
        ):
            raise ValueError("dimension-transition metadata is only valid for E12 results")
        if self.bucket_variant is not None:
            object.__setattr__(
                self,
                "bucket_variant",
                validate_bucket_variant(self.bucket_variant).value,
            )
        if self.bucket_item is not None:
            object.__setattr__(
                self, "bucket_item", _require_identifier(self.bucket_item, "bucket_item")
            )
        if self.expected_fluid is not None:
            object.__setattr__(
                self,
                "expected_fluid",
                validate_fluid_class(self.expected_fluid, "expected_fluid"),
            )
        for field_name in ("before_fluid", "after_fluid"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, validate_fluid_class(value, field_name)
                )
        for field_name in ("before_inventory", "after_inventory"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _inventory_snapshot(value, field_name)
                )
        for field_name in ("before_selected_item", "after_selected_item"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    validate_selected_item(value, field_name),
                )
        if self.check_id in (EnvironmentValidationId.E8, EnvironmentValidationId.E9, EnvironmentValidationId.E10, EnvironmentValidationId.E11, EnvironmentValidationId.E12):
            for field_name in ("before_step_id", "after_step_id"):
                value = getattr(self, field_name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{field_name} must be a non-negative int or None")
            for field_name in (
                "before_position_x", "before_position_y", "before_position_z",
                "after_position_x", "after_position_y", "after_position_z",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(self, field_name, finite_number(value, field_name))
            for field_name in ("before_dimension", "after_dimension"):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(self, field_name, validate_dimension(value, field_name))
            for field_name in ("grid_anchor_x", "grid_anchor_y", "grid_anchor_z"):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(self, field_name, validate_cell_coordinate(value, field_name))
            if self.anchor_source is not None:
                object.__setattr__(self, "anchor_source", validate_anchor_source(self.anchor_source))
            if self.probe_world_cells is not None:
                object.__setattr__(
                    self, "probe_world_cells", _probe_cell_region(self.probe_world_cells, "probe_world_cells")
                )
            if self.probe_grid_cells is not None:
                object.__setattr__(
                    self, "probe_grid_cells", _probe_cell_region(self.probe_grid_cells, "probe_grid_cells")
                )
            if self.truth_missing_count is not None and (
                type(self.truth_missing_count) is not int or self.truth_missing_count < 0
            ):
                raise ValueError("truth_missing_count must be a non-negative int or None")
            if self.stimulus_target is not None:
                if self.check_id is EnvironmentValidationId.E8:
                    object.__setattr__(
                        self, "stimulus_target", validate_block_name(self.stimulus_target, "stimulus_target")
                    )
                else:
                    object.__setattr__(
                        self, "stimulus_target", _require_identifier(self.stimulus_target, "stimulus_target")
                    )
        if self.check_id in (EnvironmentValidationId.E8, EnvironmentValidationId.E10, EnvironmentValidationId.E11, EnvironmentValidationId.E12):
            if self.before_block_truth is not None:
                object.__setattr__(
                    self, "before_block_truth", _block_truth_records(self.before_block_truth, "before_block_truth")
                )
            if self.after_block_truth is not None:
                object.__setattr__(
                    self, "after_block_truth", _block_truth_records(self.after_block_truth, "after_block_truth")
                )
        if self.check_id in (EnvironmentValidationId.E9, EnvironmentValidationId.E10):
            if self.fluid_variant is not None:
                object.__setattr__(
                    self, "fluid_variant", validate_fluid_variant(self.fluid_variant).value
                )
            if self.expected_target_fluid_type is not None:
                object.__setattr__(
                    self,
                    "expected_target_fluid_type",
                    validate_fluid_type(self.expected_target_fluid_type, "expected_target_fluid_type"),
                )
            if self.expected_target_flow_state is not None:
                object.__setattr__(
                    self,
                    "expected_target_flow_state",
                    validate_flow_state(self.expected_target_flow_state, "expected_target_flow_state"),
                )
            if self.before_server_fluid_truth is not None:
                object.__setattr__(
                    self,
                    "before_server_fluid_truth",
                    _fluid_truth_records(self.before_server_fluid_truth, "before_server_fluid_truth"),
                )
            if self.after_server_fluid_truth is not None:
                object.__setattr__(
                    self,
                    "after_server_fluid_truth",
                    _fluid_truth_records(self.after_server_fluid_truth, "after_server_fluid_truth"),
                )
        if self.check_id is EnvironmentValidationId.E10:
            for field_name in (
                "before_target_block",
                "after_target_block",
                "before_water_block",
                "after_water_block",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(
                        self, field_name, validate_block_name(value, field_name)
                    )
            for field_name in ("before_water_fluid_type", "after_water_fluid_type"):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(
                        self, field_name, validate_fluid_type(value, field_name)
                    )
            for field_name in ("before_water_flow_state", "after_water_flow_state"):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(
                        self, field_name, validate_flow_state(value, field_name)
                    )
            for field_name in (
                "conversion_observed_at_step",
                "observation_window_ticks",
                "observation_wait_count",
            ):
                value = getattr(self, field_name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{field_name} must be a non-negative int or None")
            if self.observation_window_ticks is not None and self.observation_window_ticks < 1:
                raise ValueError("observation_window_ticks must be a positive int or None")
        if self.check_id is EnvironmentValidationId.E11:
            for field_name in (
                "expected_frame_cells",
                "observed_frame_cells",
                "interior_cells",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(
                        self, field_name, _probe_cell_region(value, field_name)
                    )
            for field_name in (
                "frame_block_count",
                "before_portal_block_count",
                "after_portal_block_count",
                "portal_activation_observed_at_step",
                "observation_window_ticks",
                "observation_wait_count",
            ):
                value = getattr(self, field_name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{field_name} must be a non-negative int or None")
            if self.observation_window_ticks is not None and self.observation_window_ticks < 1:
                raise ValueError("observation_window_ticks must be a positive int or None")
        if self.check_id is EnvironmentValidationId.E12:
            for field_name in (
                "expected_frame_cells",
                "observed_frame_cells",
                "interior_cells",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    object.__setattr__(
                        self, field_name, _probe_cell_region(value, field_name)
                    )
            for field_name in (
                "frame_block_count",
                "before_portal_block_count",
                "dimension_transition_observed_at_step",
                "observation_window_ticks",
                "observation_wait_count",
            ):
                value = getattr(self, field_name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{field_name} must be a non-negative int or None")
            if self.observation_window_ticks is not None and self.observation_window_ticks < 1:
                raise ValueError("observation_window_ticks must be a positive int or None")
        reset_failure_fields = (
            self.failure_stage,
            self.original_exception_type,
            self.reset_attempt_count,
            self.environment_launch_count,
            self.exception_traceback,
        )
        if self.check_id not in (
            EnvironmentValidationId.E5,
            EnvironmentValidationId.E6,
            EnvironmentValidationId.E7,
            EnvironmentValidationId.E8,
            EnvironmentValidationId.E9,
            EnvironmentValidationId.E10,
            EnvironmentValidationId.E11,
            EnvironmentValidationId.E12,
        ) and any(
            value is not None for value in reset_failure_fields
        ):
            raise ValueError("reset-failure audit metadata is only valid for E5/E6/E7/E8/E9/E10/E11/E12 results")
        if self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E5:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_x,
                        self.before_y,
                        self.before_z,
                        self.movement_before_yaw,
                        self.after_x,
                        self.after_y,
                        self.after_z,
                        self.delta_x,
                        self.delta_y,
                        self.delta_z,
                        self.horizontal_distance,
                        self.total_distance,
                        self.forward_projection,
                        self.lateral_projection,
                        self.moved,
                        self.movement_direction_match,
                        self.lateral_drift_ok,
                        self.teleport_guard_ok,
                        self.vertical_drift_ok,
                    )
                )
            ):
                raise ValueError(
                    "E5 reset_failed requires complete reset audit and zero action evidence"
                )
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E6:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_block,
                        self.after_block,
                        self.world_changed,
                        self.intended_block_present,
                    )
                )
            ):
                raise ValueError(
                    "E6 reset_failed requires complete reset audit and zero placement evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E6:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E6 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E7:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_inventory,
                        self.after_inventory,
                        self.inventory_changed,
                        self.bucket_consumed,
                        self.empty_bucket_produced,
                        self.before_fluid,
                        self.after_fluid,
                        self.fluid_changed,
                        self.intended_fluid_present,
                    )
                )
            ):
                raise ValueError(
                    "E7 reset_failed requires complete reset audit and zero bucket evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E7:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E7 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E8:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_block_truth,
                        self.after_block_truth,
                        self.target_changed,
                        self.target_expected_block_present,
                        self.control_cells_unchanged,
                    )
                )
            ):
                raise ValueError(
                    "E8 reset_failed requires complete reset audit and zero block-truth evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E8:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E8 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E9:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_server_fluid_truth,
                        self.after_server_fluid_truth,
                        self.target_changed,
                        self.target_expected_fluid_present,
                        self.control_cells_unchanged,
                        self.source_flowing_match,
                    )
                )
            ):
                raise ValueError(
                    "E9 reset_failed requires complete reset audit and zero fluid-truth evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E9:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E9 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E10:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_block_truth,
                        self.after_block_truth,
                        self.before_server_fluid_truth,
                        self.after_server_fluid_truth,
                        self.target_changed,
                        self.obsidian_present,
                        self.conversion_observed,
                        self.control_cells_unchanged,
                        self.before_water_block,
                        self.after_water_block,
                        self.water_placement_observed,
                    )
                )
            ):
                raise ValueError(
                    "E10 reset_failed requires complete reset audit and zero conversion evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E10:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E10 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E11:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_block_truth,
                        self.after_block_truth,
                        self.portal_activated,
                        self.portal_activation_observed,
                        self.ignition_effect_observed,
                        self.control_cells_unchanged,
                    )
                )
            ):
                raise ValueError(
                    "E11 reset_failed requires complete reset audit and zero activation evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E11:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E11 action_failed requires complete action audit")
        elif self.outcome == "reset_failed" and self.check_id is EnvironmentValidationId.E12:
            if not (
                self.failure_stage == "reset"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
                and self.tested_action_count == 0
                and self.translated_action_accepted is None
                and all(
                    value is None
                    for value in (
                        self.before_block_truth,
                        self.dimension_transition_observed,
                        self.active_portal_before,
                        self.control_cells_unchanged,
                    )
                )
            ):
                raise ValueError(
                    "E12 reset_failed requires complete reset audit and zero transition evidence"
                )
        elif self.outcome == "action_failed" and self.check_id is EnvironmentValidationId.E12:
            if not (
                self.failure_stage == "action"
                and self.original_exception_type is not None
                and self.exception_traceback is not None
            ):
                raise ValueError("E12 action_failed requires complete action audit")
        elif any(value is not None for value in reset_failure_fields):
            raise ValueError(
                "reset-failure audit metadata requires E5/E6/E7/E8/E9/E10/E11/E12 reset_failed or E6/E7/E8/E9/E10/E11/E12 action_failed"
            )
        movement_thresholds = (
            self.minimum_horizontal_distance,
            self.minimum_forward_projection,
            self.maximum_lateral_drift,
            self.maximum_horizontal_distance,
            self.maximum_vertical_drift,
        )
        if any(value is not None and value < 0 for value in movement_thresholds):
            raise ValueError("movement thresholds must be non-negative")
        if (
            self.minimum_horizontal_distance is not None
            and self.maximum_horizontal_distance is not None
            and self.maximum_horizontal_distance < self.minimum_horizontal_distance
        ):
            raise ValueError("maximum horizontal distance must cover the minimum")
        if (
            self.check_id is EnvironmentValidationId.E2
            and self.observed_inventory is not None
            and self.expected_inventory is not None
            and self.inventory_matches_expected is not None
            and self.inventory_matches_expected
            != (self.observed_inventory == self.expected_inventory)
        ):
            raise ValueError("inventory_matches_expected contradicts inventory mappings")
        if (
            self.check_id is EnvironmentValidationId.E3
            and self.observed_selected_item is not None
            and self.expected_selected_item is not None
            and self.selected_item_matches_expected is not None
            and self.selected_item_matches_expected
            != (self.observed_selected_item == self.expected_selected_item)
        ):
            raise ValueError(
                "selected_item_matches_expected contradicts selected-item values"
            )
        lifecycle_clean = (
            self.created
            and self.reset_completed
            and self.initial_state_present
            and self.closed
            and self.error is None
            and self.close_error is None
        )
        if self.check_id is EnvironmentValidationId.E0:
            success_outcome = E0_SUCCESS_OUTCOME
            success_error = "success requires a clean E0 lifecycle"
        elif self.check_id is EnvironmentValidationId.E1:
            success_outcome = E1_SUCCESS_OUTCOME
            success_error = "success requires a clean E1 RGB observation"
        elif self.check_id is EnvironmentValidationId.E2:
            success_outcome = E2_SUCCESS_OUTCOME
            success_error = "success requires a clean matching E2 inventory"
        elif self.check_id is EnvironmentValidationId.E3:
            success_outcome = E3_SUCCESS_OUTCOME
            success_error = "success requires a clean matching E3 selected item"
        elif self.check_id is EnvironmentValidationId.E4:
            success_outcome = E4_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E4 camera change"
        elif self.check_id is EnvironmentValidationId.E5:
            success_outcome = E5_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E5 movement"
        elif self.check_id is EnvironmentValidationId.E6:
            success_outcome = E6_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E6 placement"
        elif self.check_id is EnvironmentValidationId.E7:
            success_outcome = E7_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E7 bucket use"
        elif self.check_id is EnvironmentValidationId.E8:
            success_outcome = E8_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E8 block-truth region"
        elif self.check_id is EnvironmentValidationId.E9:
            success_outcome = E9_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E9 fluid-truth region"
        elif self.check_id is EnvironmentValidationId.E10:
            success_outcome = E10_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E10 obsidian conversion"
        elif self.check_id is EnvironmentValidationId.E11:
            success_outcome = E11_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E11 portal activation"
        elif self.check_id is EnvironmentValidationId.E12:
            success_outcome = E12_SUCCESS_OUTCOME
            success_error = "success requires a clean independently observed E12 dimension transition"
        else:
            success_outcome = None
            success_error = "success is not defined for this validation case"
        if self.success:
            if success_outcome is None or self.outcome != success_outcome or not lifecycle_clean:
                raise ValueError(success_error)
            if self.check_id is EnvironmentValidationId.E1:
                if (
                    self.rgb_present is not True
                    or type(self.rgb_height) is not int
                    or type(self.rgb_width) is not int
                    or self.rgb_channels != 3
                    or self.rgb_dtype != "uint8"
                    or self.rgb_height < 1
                    or self.rgb_width < 1
                ):
                    raise ValueError("E1 success requires HxWx3 uint8 RGB metadata")
            elif self.check_id is EnvironmentValidationId.E2:
                if (
                    self.inventory_present is not True
                    or self.observed_inventory is None
                    or self.expected_inventory is None
                    or not self.expected_inventory
                    or self.inventory_matches_expected is not True
                    or self.observed_inventory != self.expected_inventory
                ):
                    raise ValueError(
                        "E2 success requires a non-empty expected inventory and "
                        "exact observed/expected inventory equality"
                    )
            elif self.check_id is EnvironmentValidationId.E3:
                if (
                    self.selected_item_present is not True
                    or self.observed_selected_item is None
                    or self.expected_selected_item is None
                    or self.selected_item_matches_expected is not True
                    or self.observed_selected_item != self.expected_selected_item
                ):
                    raise ValueError(
                        "E3 success requires exact observed/expected selected-item equality"
                    )
            elif self.check_id is EnvironmentValidationId.E4:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "look"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.before_yaw is not None
                    and self.before_pitch is not None
                    and self.after_yaw is not None
                    and self.after_pitch is not None
                    and self.normalized_yaw_delta is not None
                    and self.pitch_delta is not None
                    and self.direction_match is True
                    and self.magnitude_match is True
                ):
                    raise ValueError("E4 success requires one accepted look and complete orientation evidence")
            elif self.check_id is EnvironmentValidationId.E5:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "move"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.requested_forward == 1.0
                    and self.requested_strafe == 0.0
                    and self.requested_sprint is False
                    and self.requested_jump is False
                    and self.requested_duration_ticks == 1
                    and all(value is not None for value in (
                        self.before_x, self.before_y, self.before_z,
                        self.movement_before_yaw, self.after_x, self.after_y,
                        self.after_z, self.delta_x, self.delta_y, self.delta_z,
                        self.horizontal_distance, self.total_distance,
                        self.forward_projection, self.lateral_projection,
                        self.minimum_horizontal_distance,
                        self.minimum_forward_projection,
                        self.maximum_lateral_drift,
                        self.maximum_horizontal_distance,
                        self.maximum_vertical_drift,
                    ))
                    and self.moved is True
                    and self.movement_direction_match is True
                    and self.lateral_drift_ok is True
                    and self.teleport_guard_ok is True
                    and self.vertical_drift_ok is True
                ):
                    raise ValueError("E5 success requires one accepted move and complete position evidence")
            elif self.check_id is EnvironmentValidationId.E6:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "place_block"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.requested_target == self.calibration_block
                    and self.calibration_block is not None
                    and self.expected_before_block is not None
                    and self.requested_duration_ticks == 1
                    and self.target_x is not None
                    and self.target_y is not None
                    and self.target_z is not None
                    and self.target_grid_x is not None
                    and self.target_grid_y is not None
                    and self.target_grid_z is not None
                    and self.before_block == self.expected_before_block
                    and self.after_block == self.calibration_block
                    and self.world_changed is True
                    and self.intended_block_present is True
                ):
                    raise ValueError("E6 success requires one accepted place_block and complete block-truth evidence")
            elif self.check_id is EnvironmentValidationId.E7:
                try:
                    variant = validate_bucket_variant(self.bucket_variant)
                except ValueError as exc:
                    raise ValueError(
                        "E7 success requires a frozen water/lava calibration variant"
                    ) from exc
                expected_item = frozen_bucket_item(variant)
                expected_fluid = frozen_expected_fluid(variant)
                expected_before = frozen_before_inventory(variant)
                expected_after = frozen_after_inventory(variant)
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "use_item"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.bucket_item == expected_item
                    and self.expected_fluid == expected_fluid
                    and self.requested_duration_ticks == 1
                    and self.target_x is not None
                    and self.target_y is not None
                    and self.target_z is not None
                    and self.target_grid_x is not None
                    and self.target_grid_y is not None
                    and self.target_grid_z is not None
                    and self.before_inventory == expected_before
                    and self.after_inventory == expected_after
                    and self.inventory_changed is True
                    and self.bucket_consumed is True
                    and self.empty_bucket_produced is True
                    and self.before_fluid == "none"
                    and self.after_fluid == expected_fluid
                    and self.fluid_changed is True
                    and self.intended_fluid_present is True
                    and EMPTY_BUCKET_ITEM in expected_after
                ):
                    raise ValueError(
                        "E7 success requires one accepted use_item plus matching inventory and fluid evidence"
                    )
            elif self.check_id is EnvironmentValidationId.E8:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "place_block"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.stimulus_target is not None
                    and self.requested_duration_ticks == 1
                    and self.before_step_id == 0
                    and self.after_step_id == 1
                    and self.before_dimension == "minecraft:overworld"
                    and self.after_dimension == "minecraft:overworld"
                    and all(
                        value is not None
                        for value in (
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                            self.grid_anchor_x,
                            self.grid_anchor_y,
                            self.grid_anchor_z,
                            self.anchor_source,
                            self.probe_world_cells,
                            self.probe_grid_cells,
                            self.before_block_truth,
                            self.after_block_truth,
                        )
                    )
                    and self.truth_missing_count == 0
                    and self.target_changed is True
                    and self.target_expected_block_present is True
                    and self.control_cells_unchanged is True
                ):
                    raise ValueError(
                        "E8 success requires one accepted stimulus plus complete region block-truth evidence"
                    )
            elif self.check_id is EnvironmentValidationId.E9:
                try:
                    variant = validate_fluid_variant(self.fluid_variant)
                except ValueError as exc:
                    raise ValueError(
                        "E9 success requires a frozen water/lava calibration variant"
                    ) from exc
                expected_item = frozen_fluid_bucket_item(variant)
                expected_type = frozen_expected_fluid_type(variant)
                expected_flow = frozen_expected_flow_state(variant)
                if not (
                    self.agent_id is not None
                    and self.tested_step_id == 1
                    and self.action_type == "use_item"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.stimulus_target == expected_item
                    and self.expected_target_fluid_type == expected_type
                    and self.expected_target_flow_state == expected_flow
                    and self.requested_duration_ticks == 1
                    and self.before_step_id == 0
                    and self.after_step_id == 1
                    and self.before_dimension == "minecraft:overworld"
                    and self.after_dimension == "minecraft:overworld"
                    and all(
                        value is not None
                        for value in (
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                            self.grid_anchor_x,
                            self.grid_anchor_y,
                            self.grid_anchor_z,
                            self.anchor_source,
                            self.probe_world_cells,
                            self.probe_grid_cells,
                            self.before_server_fluid_truth,
                            self.after_server_fluid_truth,
                        )
                    )
                    and self.truth_missing_count == 0
                    and self.target_changed is True
                    and self.target_expected_fluid_present is True
                    and self.control_cells_unchanged is True
                    and self.source_flowing_match is True
                ):
                    raise ValueError(
                        "E9 success requires one accepted stimulus plus complete region fluid-truth evidence"
                    )
            elif self.check_id is EnvironmentValidationId.E10:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id is not None
                    and self.tested_step_id >= 1
                    and self.action_type == "use_item"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.stimulus_target == E10_STIMULUS_ITEM
                    and self.requested_duration_ticks == 1
                    and self.observation_window_ticks == 5
                    and self.observation_wait_count is not None
                    and self.observation_wait_count <= 5
                    and self.before_step_id == 0
                    and self.after_step_id is not None
                    and self.after_step_id >= 1
                    and self.before_dimension == "minecraft:overworld"
                    and self.after_dimension == "minecraft:overworld"
                    and all(
                        value is not None
                        for value in (
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                            self.grid_anchor_x,
                            self.grid_anchor_y,
                            self.grid_anchor_z,
                            self.anchor_source,
                            self.probe_world_cells,
                            self.probe_grid_cells,
                            self.before_block_truth,
                            self.after_block_truth,
                            self.before_server_fluid_truth,
                            self.after_server_fluid_truth,
                            self.before_target_block,
                            self.after_target_block,
                            self.before_water_block,
                            self.after_water_block,
                            self.before_water_fluid_type,
                            self.before_water_flow_state,
                            self.after_water_fluid_type,
                            self.after_water_flow_state,
                            self.conversion_observed_at_step,
                        )
                    )
                    and self.before_target_block == E10_EXPECTED_BEFORE_BLOCK
                    and self.before_target_block != E10_EXPECTED_AFTER_BLOCK
                    and self.after_target_block == E10_EXPECTED_AFTER_BLOCK
                    and self.before_water_block == "air"
                    and self.after_water_block == E10_EXPECTED_AFTER_WATER_BLOCK
                    and self.before_water_fluid_type == "none"
                    and self.before_water_flow_state == "none"
                    and self.after_water_fluid_type == E10_EXPECTED_AFTER_WATER_FLUID_TYPE
                    and self.after_water_flow_state == E10_EXPECTED_AFTER_WATER_FLOW_STATE
                    and self.water_placement_observed is True
                    and self.truth_missing_count == 0
                    and self.target_changed is True
                    and self.obsidian_present is True
                    and self.conversion_observed is True
                    and self.control_cells_unchanged is True
                    and self.conversion_observed_at_step >= 1
                ):
                    raise ValueError(
                        "E10 success requires accepted water placement plus lava-source conversion"
                    )
            elif self.check_id is EnvironmentValidationId.E11:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id is not None
                    and self.tested_step_id >= 1
                    and self.action_type == "use_item"
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.stimulus_target == "flint_and_steel"
                    and self.requested_duration_ticks == 1
                    and self.observation_window_ticks == 3
                    and self.observation_wait_count is not None
                    and self.observation_wait_count <= 3
                    and self.before_step_id == 0
                    and self.after_step_id is not None
                    and self.after_step_id >= 1
                    and self.before_dimension == "minecraft:overworld"
                    and self.after_dimension == "minecraft:overworld"
                    and self.frame_valid_before is True
                    and self.frame_block_count == 14
                    and self.before_portal_block_count == 0
                    and self.after_portal_block_count == 6
                    and self.ignition_effect_observed is True
                    and self.portal_activation_observed is True
                    and self.portal_activated is True
                    and self.target_changed is True
                    and self.control_cells_unchanged is True
                    and self.truth_missing_count == 0
                    and self.portal_activation_observed_at_step is not None
                    and self.portal_activation_observed_at_step >= 1
                    and self.expected_frame_cells is not None
                    and self.observed_frame_cells is not None
                    and self.interior_cells is not None
                    and len(self.interior_cells) == 6
                    and self.before_block_truth is not None
                    and self.after_block_truth is not None
                ):
                    raise ValueError(
                        "E11 success requires one accepted flint_and_steel plus complete portal interior"
                    )
            elif self.check_id is EnvironmentValidationId.E12:
                if not (
                    self.agent_id is not None
                    and self.tested_step_id is not None
                    and self.tested_step_id >= 1
                    and self.action_type == E12_STIMULUS_ACTION_TYPE
                    and self.translated_action_accepted is True
                    and self.tested_action_count == 1
                    and self.requested_duration_ticks == 8
                    and self.observation_window_ticks == 100
                    and self.observation_wait_count is not None
                    and self.observation_wait_count <= 100
                    and self.before_step_id == 0
                    and self.after_step_id is not None
                    and self.after_step_id >= 1
                    and self.before_dimension == "minecraft:overworld"
                    and self.after_dimension == "minecraft:the_nether"
                    and self.frame_valid_before is True
                    and self.frame_block_count == 14
                    and self.before_portal_block_count == 6
                    and self.active_portal_before is True
                    and self.dimension_transition_observed is True
                    and self.dimension_transition_observed_at_step is not None
                    and self.dimension_transition_observed_at_step >= 1
                    and self.truth_missing_count == 0
                    and self.expected_frame_cells is not None
                    and self.observed_frame_cells is not None
                    and self.interior_cells is not None
                    and len(self.interior_cells) == 6
                    and self.before_block_truth is not None
                ):
                    raise ValueError(
                        "E12 success requires one accepted move plus Overworld to Nether server truth"
                    )
        elif self.outcome in {
            E0_SUCCESS_OUTCOME,
            E1_SUCCESS_OUTCOME,
            E2_SUCCESS_OUTCOME,
            E3_SUCCESS_OUTCOME,
            E4_SUCCESS_OUTCOME,
            E5_SUCCESS_OUTCOME,
            E6_SUCCESS_OUTCOME,
            E7_SUCCESS_OUTCOME,
            E8_SUCCESS_OUTCOME,
            E9_SUCCESS_OUTCOME,
            E10_SUCCESS_OUTCOME,
            E11_SUCCESS_OUTCOME,
            E12_SUCCESS_OUTCOME,
        }:
            raise ValueError(f"{self.outcome} requires success=True")
        if self.outcome == INVENTORY_MISMATCH:
            if (
                self.check_id is not EnvironmentValidationId.E2
                or self.inventory_present is not True
                or self.observed_inventory is None
                or self.expected_inventory is None
                or self.inventory_matches_expected is not False
                or self.observed_inventory == self.expected_inventory
            ):
                raise ValueError(
                    "inventory_mismatch requires unequal valid E2 inventories"
                )
        if self.outcome == SELECTED_ITEM_MISMATCH:
            if (
                self.check_id is not EnvironmentValidationId.E3
                or self.selected_item_present is not True
                or self.observed_selected_item is None
                or self.expected_selected_item is None
                or self.selected_item_matches_expected is not False
                or self.observed_selected_item == self.expected_selected_item
            ):
                raise ValueError(
                    "selected_item_mismatch requires unequal valid E3 items"
                )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""

        payload: dict[str, Any] = {
            "calibration_only": self.calibration_only,
            "check_id": self.check_id.value,
            "close_error": self.close_error,
            "closed": self.closed,
            "created": self.created,
            "episode_id": self.episode_id,
            "error": self.error,
            "initial_state_present": self.initial_state_present,
            "integration_verified": False,
            "name": self.name,
            "outcome": self.outcome,
            "real_execution_performed": False,
            "reset_completed": self.reset_completed,
            "step_id": self.step_id,
            "success": self.success,
            "verification_level": UNIT_VERIFIED,
        }
        if self.check_id is EnvironmentValidationId.E1:
            payload.update(
                {
                    "rgb_channels": self.rgb_channels,
                    "rgb_dtype": self.rgb_dtype,
                    "rgb_height": self.rgb_height,
                    "rgb_present": self.rgb_present,
                    "rgb_width": self.rgb_width,
                }
            )
        elif self.check_id is EnvironmentValidationId.E2:
            payload.update(
                {
                    "expected_inventory": (
                        None
                        if self.expected_inventory is None
                        else dict(self.expected_inventory)
                    ),
                    "inventory_matches_expected": self.inventory_matches_expected,
                    "inventory_present": self.inventory_present,
                    "observed_inventory": (
                        None
                        if self.observed_inventory is None
                        else dict(self.observed_inventory)
                    ),
                }
            )
        elif self.check_id is EnvironmentValidationId.E3:
            payload.update(
                {
                    "expected_selected_item": self.expected_selected_item,
                    "observed_selected_item": self.observed_selected_item,
                    "selected_item_matches_expected": (
                        self.selected_item_matches_expected
                    ),
                    "selected_item_present": self.selected_item_present,
                }
            )
        elif self.check_id is EnvironmentValidationId.E4:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_pitch": self.after_pitch,
                    "after_yaw": self.after_yaw,
                    "agent_id": self.agent_id,
                    "before_pitch": self.before_pitch,
                    "before_yaw": self.before_yaw,
                    "direction_match": self.direction_match,
                    "magnitude_match": self.magnitude_match,
                    "normalized_yaw_delta": self.normalized_yaw_delta,
                    "pitch_delta": self.pitch_delta,
                    "requested_pitch": self.requested_pitch,
                    "requested_yaw": self.requested_yaw,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                }
            )
        elif self.check_id is EnvironmentValidationId.E5:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_x": self.after_x,
                    "after_y": self.after_y,
                    "after_z": self.after_z,
                    "agent_id": self.agent_id,
                    "before_x": self.before_x,
                    "before_y": self.before_y,
                    "before_z": self.before_z,
                    "before_yaw": self.movement_before_yaw,
                    "delta_x": self.delta_x,
                    "delta_y": self.delta_y,
                    "delta_z": self.delta_z,
                    "forward_projection": self.forward_projection,
                    "failure_stage": self.failure_stage,
                    "original_exception_type": self.original_exception_type,
                    "reset_attempt_count": self.reset_attempt_count,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "horizontal_distance": self.horizontal_distance,
                    "lateral_drift_ok": self.lateral_drift_ok,
                    "lateral_projection": self.lateral_projection,
                    "maximum_horizontal_distance": self.maximum_horizontal_distance,
                    "maximum_lateral_drift": self.maximum_lateral_drift,
                    "maximum_vertical_drift": self.maximum_vertical_drift,
                    "minimum_forward_projection": self.minimum_forward_projection,
                    "minimum_horizontal_distance": self.minimum_horizontal_distance,
                    "moved": self.moved,
                    "movement_direction_match": self.movement_direction_match,
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "requested_forward": self.requested_forward,
                    "requested_jump": self.requested_jump,
                    "requested_sprint": self.requested_sprint,
                    "requested_strafe": self.requested_strafe,
                    "teleport_guard_ok": self.teleport_guard_ok,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "total_distance": self.total_distance,
                    "translated_action_accepted": self.translated_action_accepted,
                    "vertical_drift_ok": self.vertical_drift_ok,
                }
            )
        elif self.check_id is EnvironmentValidationId.E6:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_block": self.after_block,
                    "agent_id": self.agent_id,
                    "before_block": self.before_block,
                    "calibration_block": self.calibration_block,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "expected_before_block": self.expected_before_block,
                    "failure_stage": self.failure_stage,
                    "intended_block_present": self.intended_block_present,
                    "original_exception_type": self.original_exception_type,
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "requested_target": self.requested_target,
                    "reset_attempt_count": self.reset_attempt_count,
                    "target_x": self.target_x,
                    "target_y": self.target_y,
                    "target_z": self.target_z,
                    "target_world_cell": (
                        None
                        if self.target_x is None
                        or self.target_y is None
                        or self.target_z is None
                        else [self.target_x, self.target_y, self.target_z]
                    ),
                    "target_grid_cell": (
                        None
                        if self.target_grid_x is None
                        or self.target_grid_y is None
                        or self.target_grid_z is None
                        else [self.target_grid_x, self.target_grid_y, self.target_grid_z]
                    ),
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "world_changed": self.world_changed,
                }
            )
        elif self.check_id is EnvironmentValidationId.E7:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_fluid": self.after_fluid,
                    "after_inventory": (
                        None
                        if self.after_inventory is None
                        else dict(self.after_inventory)
                    ),
                    "after_selected_item": self.after_selected_item,
                    "agent_id": self.agent_id,
                    "before_fluid": self.before_fluid,
                    "before_inventory": (
                        None
                        if self.before_inventory is None
                        else dict(self.before_inventory)
                    ),
                    "before_selected_item": self.before_selected_item,
                    "bucket_consumed": self.bucket_consumed,
                    "bucket_item": self.bucket_item,
                    "bucket_variant": self.bucket_variant,
                    "empty_bucket_produced": self.empty_bucket_produced,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "expected_fluid": self.expected_fluid,
                    "failure_stage": self.failure_stage,
                    "fluid_changed": self.fluid_changed,
                    "intended_fluid_present": self.intended_fluid_present,
                    "inventory_changed": self.inventory_changed,
                    "original_exception_type": self.original_exception_type,
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "target_grid_cell": (
                        None
                        if self.target_grid_x is None
                        or self.target_grid_y is None
                        or self.target_grid_z is None
                        else [self.target_grid_x, self.target_grid_y, self.target_grid_z]
                    ),
                    "target_world_cell": (
                        None
                        if self.target_x is None
                        or self.target_y is None
                        or self.target_z is None
                        else [self.target_x, self.target_y, self.target_z]
                    ),
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                }
            )
        elif self.check_id is EnvironmentValidationId.E8:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_block_truth": (
                        None
                        if self.after_block_truth is None
                        else [dict(item) for item in self.after_block_truth]
                    ),
                    "after_dimension": self.after_dimension,
                    "after_position": (
                        None
                        if self.after_position_x is None
                        or self.after_position_y is None
                        or self.after_position_z is None
                        else [
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                        ]
                    ),
                    "after_step_id": self.after_step_id,
                    "agent_id": self.agent_id,
                    "anchor_source": self.anchor_source,
                    "before_block_truth": (
                        None
                        if self.before_block_truth is None
                        else [dict(item) for item in self.before_block_truth]
                    ),
                    "before_dimension": self.before_dimension,
                    "before_position": (
                        None
                        if self.before_position_x is None
                        or self.before_position_y is None
                        or self.before_position_z is None
                        else [
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                        ]
                    ),
                    "before_step_id": self.before_step_id,
                    "control_cells_unchanged": self.control_cells_unchanged,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "failure_stage": self.failure_stage,
                    "grid_anchor_world": (
                        None
                        if self.grid_anchor_x is None
                        or self.grid_anchor_y is None
                        or self.grid_anchor_z is None
                        else [self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z]
                    ),
                    "original_exception_type": self.original_exception_type,
                    "probe_grid_cells": (
                        None
                        if self.probe_grid_cells is None
                        else [list(cell) for cell in self.probe_grid_cells]
                    ),
                    "probe_world_cells": (
                        None
                        if self.probe_world_cells is None
                        else [list(cell) for cell in self.probe_world_cells]
                    ),
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "stimulus_action": (
                        None
                        if self.action_type is None
                        else {
                            "action_type": self.action_type,
                            "duration_ticks": self.requested_duration_ticks,
                            "target": self.stimulus_target,
                        }
                    ),
                    "target_changed": self.target_changed,
                    "target_expected_block_present": self.target_expected_block_present,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "truth_missing_count": self.truth_missing_count,
                }
            )
        elif self.check_id is EnvironmentValidationId.E9:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_dimension": self.after_dimension,
                    "after_fluid_truth": (
                        None
                        if self.after_server_fluid_truth is None
                        else [dict(item) for item in self.after_server_fluid_truth]
                    ),
                    "after_position": (
                        None
                        if self.after_position_x is None
                        or self.after_position_y is None
                        or self.after_position_z is None
                        else [
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                        ]
                    ),
                    "after_step_id": self.after_step_id,
                    "agent_id": self.agent_id,
                    "anchor_source": self.anchor_source,
                    "before_dimension": self.before_dimension,
                    "before_fluid_truth": (
                        None
                        if self.before_server_fluid_truth is None
                        else [dict(item) for item in self.before_server_fluid_truth]
                    ),
                    "before_position": (
                        None
                        if self.before_position_x is None
                        or self.before_position_y is None
                        or self.before_position_z is None
                        else [
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                        ]
                    ),
                    "before_step_id": self.before_step_id,
                    "control_cells_unchanged": self.control_cells_unchanged,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "expected_target_flow_state": self.expected_target_flow_state,
                    "expected_target_fluid_type": self.expected_target_fluid_type,
                    "failure_stage": self.failure_stage,
                    "fluid_variant": self.fluid_variant,
                    "grid_anchor_world": (
                        None
                        if self.grid_anchor_x is None
                        or self.grid_anchor_y is None
                        or self.grid_anchor_z is None
                        else [self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z]
                    ),
                    "original_exception_type": self.original_exception_type,
                    "probe_grid_cells": (
                        None
                        if self.probe_grid_cells is None
                        else [list(cell) for cell in self.probe_grid_cells]
                    ),
                    "probe_world_cells": (
                        None
                        if self.probe_world_cells is None
                        else [list(cell) for cell in self.probe_world_cells]
                    ),
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "source_flowing_match": self.source_flowing_match,
                    "stimulus_action": (
                        None
                        if self.action_type is None
                        else {
                            "action_type": self.action_type,
                            "duration_ticks": self.requested_duration_ticks,
                            "target": self.stimulus_target,
                        }
                    ),
                    "target_changed": self.target_changed,
                    "target_expected_fluid_present": self.target_expected_fluid_present,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "truth_missing_count": self.truth_missing_count,
                }
            )
        elif self.check_id is EnvironmentValidationId.E10:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_block_truth": (
                        None
                        if self.after_block_truth is None
                        else [dict(item) for item in self.after_block_truth]
                    ),
                    "after_dimension": self.after_dimension,
                    "after_fluid_truth": (
                        None
                        if self.after_server_fluid_truth is None
                        else [dict(item) for item in self.after_server_fluid_truth]
                    ),
                    "after_position": (
                        None
                        if self.after_position_x is None
                        or self.after_position_y is None
                        or self.after_position_z is None
                        else [
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                        ]
                    ),
                    "after_step_id": self.after_step_id,
                    "after_target_block": self.after_target_block,
                    "after_water_block": self.after_water_block,
                    "after_water_flow_state": self.after_water_flow_state,
                    "after_water_fluid_type": self.after_water_fluid_type,
                    "agent_id": self.agent_id,
                    "anchor_source": self.anchor_source,
                    "before_block_truth": (
                        None
                        if self.before_block_truth is None
                        else [dict(item) for item in self.before_block_truth]
                    ),
                    "before_dimension": self.before_dimension,
                    "before_fluid_truth": (
                        None
                        if self.before_server_fluid_truth is None
                        else [dict(item) for item in self.before_server_fluid_truth]
                    ),
                    "before_position": (
                        None
                        if self.before_position_x is None
                        or self.before_position_y is None
                        or self.before_position_z is None
                        else [
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                        ]
                    ),
                    "before_step_id": self.before_step_id,
                    "before_target_block": self.before_target_block,
                    "before_water_block": self.before_water_block,
                    "before_water_flow_state": self.before_water_flow_state,
                    "before_water_fluid_type": self.before_water_fluid_type,
                    "control_cells_unchanged": self.control_cells_unchanged,
                    "conversion_observed": self.conversion_observed,
                    "conversion_observed_at_step": self.conversion_observed_at_step,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "failure_stage": self.failure_stage,
                    "grid_anchor_world": (
                        None
                        if self.grid_anchor_x is None
                        or self.grid_anchor_y is None
                        or self.grid_anchor_z is None
                        else [self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z]
                    ),
                    "observation_wait_count": self.observation_wait_count,
                    "observation_window_ticks": self.observation_window_ticks,
                    "obsidian_present": self.obsidian_present,
                    "original_exception_type": self.original_exception_type,
                    "probe_grid_cells": (
                        None
                        if self.probe_grid_cells is None
                        else [list(cell) for cell in self.probe_grid_cells]
                    ),
                    "probe_world_cells": (
                        None
                        if self.probe_world_cells is None
                        else [list(cell) for cell in self.probe_world_cells]
                    ),
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "source_flowing_match": self.source_flowing_match,
                    "stimulus_action": (
                        None
                        if self.action_type is None
                        else {
                            "action_type": self.action_type,
                            "duration_ticks": self.requested_duration_ticks,
                            "target": self.stimulus_target,
                        }
                    ),
                    "target_changed": self.target_changed,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "truth_missing_count": self.truth_missing_count,
                    "water_placement_observed": self.water_placement_observed,
                }
            )
        elif self.check_id is EnvironmentValidationId.E11:
            payload.update(
                {
                    "action_type": self.action_type,
                    "after_block_truth": (
                        None
                        if self.after_block_truth is None
                        else [dict(item) for item in self.after_block_truth]
                    ),
                    "after_dimension": self.after_dimension,
                    "after_portal_block_count": self.after_portal_block_count,
                    "after_position": (
                        None
                        if self.after_position_x is None
                        or self.after_position_y is None
                        or self.after_position_z is None
                        else [
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                        ]
                    ),
                    "after_step_id": self.after_step_id,
                    "agent_id": self.agent_id,
                    "anchor_source": self.anchor_source,
                    "before_block_truth": (
                        None
                        if self.before_block_truth is None
                        else [dict(item) for item in self.before_block_truth]
                    ),
                    "before_dimension": self.before_dimension,
                    "before_portal_block_count": self.before_portal_block_count,
                    "before_position": (
                        None
                        if self.before_position_x is None
                        or self.before_position_y is None
                        or self.before_position_z is None
                        else [
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                        ]
                    ),
                    "before_step_id": self.before_step_id,
                    "control_cells_unchanged": self.control_cells_unchanged,
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "expected_frame_cells": (
                        None
                        if self.expected_frame_cells is None
                        else [list(cell) for cell in self.expected_frame_cells]
                    ),
                    "failure_stage": self.failure_stage,
                    "frame_block_count": self.frame_block_count,
                    "frame_valid_before": self.frame_valid_before,
                    "grid_anchor_world": (
                        None
                        if self.grid_anchor_x is None
                        or self.grid_anchor_y is None
                        or self.grid_anchor_z is None
                        else [self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z]
                    ),
                    "ignition_effect_observed": self.ignition_effect_observed,
                    "interior_cells": (
                        None
                        if self.interior_cells is None
                        else [list(cell) for cell in self.interior_cells]
                    ),
                    "observation_wait_count": self.observation_wait_count,
                    "observation_window_ticks": self.observation_window_ticks,
                    "observed_frame_cells": (
                        None
                        if self.observed_frame_cells is None
                        else [list(cell) for cell in self.observed_frame_cells]
                    ),
                    "original_exception_type": self.original_exception_type,
                    "portal_activated": self.portal_activated,
                    "portal_activation_observed": self.portal_activation_observed,
                    "portal_activation_observed_at_step": self.portal_activation_observed_at_step,
                    "probe_grid_cells": (
                        None
                        if self.probe_grid_cells is None
                        else [list(cell) for cell in self.probe_grid_cells]
                    ),
                    "probe_world_cells": (
                        None
                        if self.probe_world_cells is None
                        else [list(cell) for cell in self.probe_world_cells]
                    ),
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "stimulus_action": (
                        None
                        if self.action_type is None
                        else {
                            "action_type": self.action_type,
                            "duration_ticks": self.requested_duration_ticks,
                            "target": self.stimulus_target,
                        }
                    ),
                    "target_changed": self.target_changed,
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "truth_missing_count": self.truth_missing_count,
                }
            )
        elif self.check_id is EnvironmentValidationId.E12:
            payload.update(
                {
                    "action_type": self.action_type,
                    "active_portal_before": self.active_portal_before,
                    "after_dimension": self.after_dimension,
                    "after_position": (
                        None
                        if self.after_position_x is None
                        or self.after_position_y is None
                        or self.after_position_z is None
                        else [
                            self.after_position_x,
                            self.after_position_y,
                            self.after_position_z,
                        ]
                    ),
                    "after_step_id": self.after_step_id,
                    "agent_id": self.agent_id,
                    "anchor_source": self.anchor_source,
                    "before_block_truth": (
                        None
                        if self.before_block_truth is None
                        else [dict(item) for item in self.before_block_truth]
                    ),
                    "before_dimension": self.before_dimension,
                    "before_portal_block_count": self.before_portal_block_count,
                    "before_position": (
                        None
                        if self.before_position_x is None
                        or self.before_position_y is None
                        or self.before_position_z is None
                        else [
                            self.before_position_x,
                            self.before_position_y,
                            self.before_position_z,
                        ]
                    ),
                    "before_step_id": self.before_step_id,
                    "dimension_transition_observed": self.dimension_transition_observed,
                    "dimension_transition_observed_at_step": (
                        self.dimension_transition_observed_at_step
                    ),
                    "environment_launch_count": self.environment_launch_count,
                    "exception_traceback": self.exception_traceback,
                    "expected_frame_cells": (
                        None
                        if self.expected_frame_cells is None
                        else [list(cell) for cell in self.expected_frame_cells]
                    ),
                    "failure_stage": self.failure_stage,
                    "frame_block_count": self.frame_block_count,
                    "frame_valid_before": self.frame_valid_before,
                    "grid_anchor_world": (
                        None
                        if self.grid_anchor_x is None
                        or self.grid_anchor_y is None
                        or self.grid_anchor_z is None
                        else [self.grid_anchor_x, self.grid_anchor_y, self.grid_anchor_z]
                    ),
                    "interior_cells": (
                        None
                        if self.interior_cells is None
                        else [list(cell) for cell in self.interior_cells]
                    ),
                    "observation_wait_count": self.observation_wait_count,
                    "observation_window_ticks": self.observation_window_ticks,
                    "observed_frame_cells": (
                        None
                        if self.observed_frame_cells is None
                        else [list(cell) for cell in self.observed_frame_cells]
                    ),
                    "original_exception_type": self.original_exception_type,
                    "probe_grid_cells": (
                        None
                        if self.probe_grid_cells is None
                        else [list(cell) for cell in self.probe_grid_cells]
                    ),
                    "probe_world_cells": (
                        None
                        if self.probe_world_cells is None
                        else [list(cell) for cell in self.probe_world_cells]
                    ),
                    "requested_duration_ticks": self.requested_duration_ticks,
                    "reset_attempt_count": self.reset_attempt_count,
                    "stimulus_action": (
                        None
                        if self.action_type is None
                        else {
                            "action_type": self.action_type,
                            "duration_ticks": self.requested_duration_ticks,
                        }
                    ),
                    "tested_action_count": self.tested_action_count,
                    "tested_step_id": self.tested_step_id,
                    "translated_action_accepted": self.translated_action_accepted,
                    "truth_missing_count": self.truth_missing_count,
                }
            )
        return payload
