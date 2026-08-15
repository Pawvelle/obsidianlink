"""Minimal P1 validation runner.

Executes one validation case in a controlled lifecycle. This phase
implements E0 lifecycle, E1 RGB, E2 inventory, E3 selected item, E4 camera,
E5 movement, E6 block placement, E7 bucket usage, E8 server-side block
truth, and E9 server-side fluid truth. E4/E5/E6/E7/E8/E9 consume only
narrow evaluator truth from their integration adapters; the validation
core remains MineRL-independent.
The runner never uses benchmark evaluator success semantics.
"""

from __future__ import annotations

import json
import traceback
from typing import Callable, Mapping, Protocol, runtime_checkable

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.env.validation.camera import (
    ORIENTATION_AFTER_MISSING,
    ORIENTATION_BEFORE_MISSING,
    ORIENTATION_INVALID,
    CameraActionExecution,
    CameraInspection,
    CameraOrientationSnapshot,
    inspect_camera_change,
)

from obsidianlink.env.validation.cases.lifecycle import initial_state_exists
from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
)
from obsidianlink.env.validation.inventory import (
    InventoryInspection,
    inspect_inventory,
    inspect_public_inventory,
)
from obsidianlink.env.validation.movement import (
    MOVEMENT_ORIENTATION_INVALID,
    MOVEMENT_ORIENTATION_MISSING,
    POSITION_AFTER_MISSING,
    POSITION_BEFORE_MISSING,
    POSITION_INVALID,
    MovementActionExecution,
    MovementInspection,
    MovementOrientationSnapshot,
    PlayerPositionSnapshot,
    inspect_movement,
)
from obsidianlink.env.validation.bucket import (
    FLUID_AFTER_MISSING,
    FLUID_BEFORE_MISSING,
    FLUID_TRUTH_INVALID,
    INVENTORY_AFTER_MISSING,
    INVENTORY_BEFORE_MISSING,
    INVENTORY_INVALID,
    BucketActionExecution,
    BucketCalibrationVariant,
    BucketFluidTruthSnapshot,
    BucketInventorySnapshot,
    BucketUsageInspection,
    frozen_after_inventory,
    frozen_before_inventory,
    frozen_bucket_item,
    frozen_expected_fluid,
    inspect_bucket_usage,
    validate_bucket_variant,
    validate_fluid_class,
)
from obsidianlink.env.validation.placement import (
    BLOCK_AFTER_MISSING,
    BLOCK_BEFORE_MISSING,
    BLOCK_TRUTH_INVALID,
    BlockPlacementTruthSnapshot,
    PlacementActionExecution,
    PlacementInspection,
    inspect_block_placement,
    spawn_relative_grid_cell,
    validate_block_name,
    validate_target_cell,
)
from obsidianlink.env.validation.result import (
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
    INVENTORY_MISMATCH,
    SELECTED_ITEM_MISMATCH,
    EnvironmentValidationResult,
)
from obsidianlink.env.validation.rgb import RGBInspection, inspect_public_rgb
from obsidianlink.env.validation.selected_item import (
    SelectedItemInspection,
    inspect_public_selected_item,
    validate_selected_item,
)
from obsidianlink.env.validation.truth import (
    TRUTH_SNAPSHOT_MISSING,
    BlockTruthActionExecution,
    BlockTruthInspection,
    FluidCalibrationVariant,
    FluidTruthActionExecution,
    FluidTruthInspection,
    ServerTruthSnapshot,
    frozen_expected_flow_state,
    frozen_expected_fluid_type,
    frozen_fluid_bucket_item,
    inspect_block_truth,
    inspect_fluid_truth,
    truth_error_outcome,
    validate_fluid_variant,
)


@runtime_checkable
class LifecycleBackend(Protocol):
    """Smallest common backend surface required by E0--E9.

    ``reset`` must return an initial state mapping. ``close`` must be
    safe to call after both successful and failed execution. Later P1
    cases may require additional methods; they are not defined here.
    """

    def reset(self) -> object:
        ...

    def close(self) -> None:
        ...


BackendFactory = Callable[[], object]


def _format_error(exc: BaseException) -> str:
    message = str(exc).strip()
    name = type(exc).__name__
    if not message:
        return name
    return f"{name}: {message}"


def _root_exception(exc: BaseException) -> BaseException:
    """Return the deepest explicit cause/context without looping."""

    current = exc
    seen = {id(current)}
    while True:
        candidate = current.__cause__
        if candidate is None and not current.__suppress_context__:
            candidate = current.__context__
        if candidate is None or id(candidate) in seen:
            return current
        seen.add(id(candidate))
        current = candidate


def _result(
    *,
    case: EnvironmentValidationCase,
    episode_id: str,
    success: bool,
    outcome: str,
    created: bool,
    reset_completed: bool,
    initial_state_present: bool,
    closed: bool,
    error: str | None = None,
    close_error: str | None = None,
    rgb: RGBInspection | None = None,
    inventory: InventoryInspection | None = None,
    expected_inventory: Mapping[str, int] | None = None,
    inventory_matches_expected: bool | None = None,
    selected_item: SelectedItemInspection | None = None,
    expected_selected_item: str | None = None,
    selected_item_matches_expected: bool | None = None,
    before_orientation: CameraOrientationSnapshot | None = None,
    after_orientation: CameraOrientationSnapshot | None = None,
    camera_execution: CameraActionExecution | None = None,
    camera: CameraInspection | None = None,
    camera_agent_id: str | None = None,
    camera_requested_yaw: float | None = None,
    camera_requested_pitch: float | None = None,
    before_position: PlayerPositionSnapshot | None = None,
    after_position: PlayerPositionSnapshot | None = None,
    movement_orientation: MovementOrientationSnapshot | None = None,
    movement_execution: MovementActionExecution | None = None,
    movement: MovementInspection | None = None,
    movement_agent_id: str | None = None,
    requested_forward: float | None = None,
    requested_strafe: float | None = None,
    requested_sprint: bool | None = None,
    requested_jump: bool | None = None,
    requested_duration_ticks: int | None = None,
    minimum_horizontal_distance: float | None = None,
    minimum_forward_projection: float | None = None,
    maximum_lateral_drift: float | None = None,
    maximum_horizontal_distance: float | None = None,
    maximum_vertical_drift: float | None = None,
    before_block_truth: BlockPlacementTruthSnapshot | None = None,
    after_block_truth: BlockPlacementTruthSnapshot | None = None,
    placement_execution: PlacementActionExecution | None = None,
    placement: PlacementInspection | None = None,
    placement_agent_id: str | None = None,
    calibration_block: str | None = None,
    expected_before_block: str | None = None,
    target_cell: tuple[int, int, int] | None = None,
    target_grid_cell: tuple[int, int, int] | None = None,
    before_inventory_snapshot: BucketInventorySnapshot | None = None,
    after_inventory_snapshot: BucketInventorySnapshot | None = None,
    before_fluid_truth: BucketFluidTruthSnapshot | None = None,
    after_fluid_truth: BucketFluidTruthSnapshot | None = None,
    bucket_execution: BucketActionExecution | None = None,
    bucket: BucketUsageInspection | None = None,
    bucket_agent_id: str | None = None,
    bucket_variant: str | None = None,
    bucket_item: str | None = None,
    expected_fluid: str | None = None,
    before_selected_item: str | None = None,
    after_selected_item: str | None = None,
    before_truth_snapshot: ServerTruthSnapshot | None = None,
    after_truth_snapshot: ServerTruthSnapshot | None = None,
    truth_execution: BlockTruthActionExecution | None = None,
    block_truth: BlockTruthInspection | None = None,
    truth_agent_id: str | None = None,
    stimulus_target: str | None = None,
    probe_world_cells: tuple[tuple[int, int, int], ...] | None = None,
    probe_grid_cells: tuple[tuple[int, int, int], ...] | None = None,
    fluid_execution: FluidTruthActionExecution | None = None,
    fluid_inspection: FluidTruthInspection | None = None,
    fluid_variant_name: str | None = None,
    expected_target_fluid_type: str | None = None,
    expected_target_flow_state: str | None = None,
    failure_stage: str | None = None,
    original_exception_type: str | None = None,
    reset_attempt_count: int | None = None,
    environment_launch_count: int | None = None,
    exception_traceback: str | None = None,
) -> EnvironmentValidationResult:
    return EnvironmentValidationResult(
        check_id=case.check_id,
        name=case.name,
        episode_id=episode_id,
        step_id=0,
        success=success,
        outcome=outcome,
        created=created,
        reset_completed=reset_completed,
        initial_state_present=initial_state_present,
        closed=closed,
        error=error,
        close_error=close_error,
        failure_stage=failure_stage,
        original_exception_type=original_exception_type,
        reset_attempt_count=reset_attempt_count,
        environment_launch_count=environment_launch_count,
        exception_traceback=exception_traceback,
        rgb_present=None if rgb is None else rgb.present,
        rgb_height=None if rgb is None else rgb.height,
        rgb_width=None if rgb is None else rgb.width,
        rgb_channels=None if rgb is None else rgb.channels,
        rgb_dtype=None if rgb is None else rgb.dtype,
        inventory_present=None if inventory is None else inventory.present,
        observed_inventory=None if inventory is None else inventory.inventory,
        expected_inventory=expected_inventory,
        inventory_matches_expected=inventory_matches_expected,
        selected_item_present=(
            None if selected_item is None else selected_item.present
        ),
        observed_selected_item=(
            None if selected_item is None else selected_item.selected_item
        ),
        expected_selected_item=expected_selected_item,
        selected_item_matches_expected=selected_item_matches_expected,
        agent_id=(
            fluid_execution.agent_id if fluid_execution is not None
            else truth_execution.agent_id if truth_execution is not None
            else truth_agent_id if truth_agent_id is not None
            else bucket_execution.agent_id if bucket_execution is not None
            else bucket_agent_id if bucket_agent_id is not None
            else placement_execution.agent_id if placement_execution is not None
            else placement_agent_id if placement_agent_id is not None
            else movement_execution.agent_id if movement_execution is not None
            else movement_agent_id if movement_agent_id is not None
            else camera_execution.agent_id if camera_execution is not None
            else camera_agent_id
        ),
        tested_step_id=(
            fluid_execution.step_id if fluid_execution is not None
            else truth_execution.step_id if truth_execution is not None
            else bucket_execution.step_id if bucket_execution is not None
            else placement_execution.step_id if placement_execution is not None
            else movement_execution.step_id if movement_execution is not None
            else None if camera_execution is None else camera_execution.step_id
        ),
        action_type=(
            fluid_execution.action_type if fluid_execution is not None
            else "use_item" if fluid_variant_name is not None
            else truth_execution.action_type if truth_execution is not None
            else "place_block" if stimulus_target is not None
            else bucket_execution.action_type if bucket_execution is not None
            else "use_item" if bucket_item is not None
            else placement_execution.action_type if placement_execution is not None
            else "place_block" if calibration_block is not None
            else movement_execution.action_type if movement_execution is not None
            else "move" if requested_forward is not None
            else "look" if camera_requested_yaw is not None and camera_execution is None
            else None if camera_execution is None else camera_execution.action_type
        ),
        requested_yaw=(camera_requested_yaw if camera_execution is None else camera_execution.requested_yaw),
        requested_pitch=(camera_requested_pitch if camera_execution is None else camera_execution.requested_pitch),
        translated_action_accepted=(
            fluid_execution.translated_action_accepted if fluid_execution is not None
            else truth_execution.translated_action_accepted if truth_execution is not None
            else bucket_execution.translated_action_accepted if bucket_execution is not None
            else placement_execution.translated_action_accepted if placement_execution is not None
            else movement_execution.translated_action_accepted if movement_execution is not None
            else None if camera_execution is None else camera_execution.translated_action_accepted
        ),
        tested_action_count=(
            fluid_execution.tested_action_count if fluid_execution is not None
            else 0 if fluid_variant_name is not None
            else truth_execution.tested_action_count if truth_execution is not None
            else 0 if stimulus_target is not None
            else bucket_execution.tested_action_count if bucket_execution is not None
            else 0 if bucket_item is not None
            else placement_execution.tested_action_count if placement_execution is not None
            else 0 if calibration_block is not None
            else movement_execution.tested_action_count if movement_execution is not None
            else 0 if requested_forward is not None
            else 0 if camera_requested_yaw is not None and camera_execution is None
            else None if camera_execution is None else camera_execution.tested_action_count
        ),
        before_yaw=None if before_orientation is None else before_orientation.yaw,
        before_pitch=None if before_orientation is None else before_orientation.pitch,
        after_yaw=None if after_orientation is None else after_orientation.yaw,
        after_pitch=None if after_orientation is None else after_orientation.pitch,
        normalized_yaw_delta=None if camera is None else camera.normalized_yaw_delta,
        pitch_delta=None if camera is None else camera.pitch_delta,
        direction_match=None if camera is None else camera.direction_match,
        magnitude_match=None if camera is None else camera.magnitude_match,
        requested_forward=(requested_forward if movement_execution is None else movement_execution.forward),
        requested_strafe=(requested_strafe if movement_execution is None else movement_execution.strafe),
        requested_sprint=(requested_sprint if movement_execution is None else movement_execution.sprint),
        requested_jump=(requested_jump if movement_execution is None else movement_execution.jump),
        requested_duration_ticks=(
            fluid_execution.duration_ticks if fluid_execution is not None
            else truth_execution.duration_ticks if truth_execution is not None
            else bucket_execution.duration_ticks if bucket_execution is not None
            else placement_execution.duration_ticks if placement_execution is not None
            else requested_duration_ticks if movement_execution is None
            else movement_execution.duration_ticks
        ),
        before_x=None if before_position is None else before_position.x,
        before_y=None if before_position is None else before_position.y,
        before_z=None if before_position is None else before_position.z,
        movement_before_yaw=None if movement_orientation is None else movement_orientation.yaw,
        after_x=None if after_position is None else after_position.x,
        after_y=None if after_position is None else after_position.y,
        after_z=None if after_position is None else after_position.z,
        delta_x=None if movement is None else movement.delta_x,
        delta_y=None if movement is None else movement.delta_y,
        delta_z=None if movement is None else movement.delta_z,
        horizontal_distance=None if movement is None else movement.horizontal_distance,
        total_distance=None if movement is None else movement.total_distance,
        forward_projection=None if movement is None else movement.forward_projection,
        lateral_projection=None if movement is None else movement.lateral_projection,
        minimum_horizontal_distance=minimum_horizontal_distance,
        minimum_forward_projection=minimum_forward_projection,
        maximum_lateral_drift=maximum_lateral_drift,
        maximum_horizontal_distance=maximum_horizontal_distance,
        maximum_vertical_drift=maximum_vertical_drift,
        moved=None if movement is None else movement.moved,
        movement_direction_match=None if movement is None else movement.direction_match,
        lateral_drift_ok=None if movement is None else movement.lateral_drift_ok,
        teleport_guard_ok=None if movement is None else movement.teleport_guard_ok,
        vertical_drift_ok=None if movement is None else movement.vertical_drift_ok,
        requested_target=(
            placement_execution.target if placement_execution is not None else calibration_block
        ),
        calibration_block=calibration_block,
        expected_before_block=expected_before_block,
        target_x=None if target_cell is None else target_cell[0],
        target_y=None if target_cell is None else target_cell[1],
        target_z=None if target_cell is None else target_cell[2],
        target_grid_x=None if target_grid_cell is None else target_grid_cell[0],
        target_grid_y=None if target_grid_cell is None else target_grid_cell[1],
        target_grid_z=None if target_grid_cell is None else target_grid_cell[2],
        before_block=None if before_block_truth is None else before_block_truth.block,
        after_block=None if after_block_truth is None else after_block_truth.block,
        world_changed=None if placement is None else placement.world_changed,
        intended_block_present=None if placement is None else placement.intended_block_present,
        bucket_variant=bucket_variant,
        bucket_item=(
            bucket_execution.target if bucket_execution is not None else bucket_item
        ),
        expected_fluid=expected_fluid,
        before_inventory=(
            None if before_inventory_snapshot is None else before_inventory_snapshot.inventory
        ),
        after_inventory=(
            None if after_inventory_snapshot is None else after_inventory_snapshot.inventory
        ),
        inventory_changed=None if bucket is None else bucket.inventory_changed,
        bucket_consumed=None if bucket is None else bucket.bucket_consumed,
        empty_bucket_produced=None if bucket is None else bucket.empty_bucket_produced,
        before_fluid=None if before_fluid_truth is None else before_fluid_truth.fluid,
        after_fluid=None if after_fluid_truth is None else after_fluid_truth.fluid,
        fluid_changed=None if bucket is None else bucket.fluid_changed,
        intended_fluid_present=None if bucket is None else bucket.intended_fluid_present,
        before_selected_item=before_selected_item,
        after_selected_item=after_selected_item,
        before_step_id=None if before_truth_snapshot is None else before_truth_snapshot.step_id,
        after_step_id=None if after_truth_snapshot is None else after_truth_snapshot.step_id,
        before_position_x=None if before_truth_snapshot is None else before_truth_snapshot.position_world[0],
        before_position_y=None if before_truth_snapshot is None else before_truth_snapshot.position_world[1],
        before_position_z=None if before_truth_snapshot is None else before_truth_snapshot.position_world[2],
        after_position_x=None if after_truth_snapshot is None else after_truth_snapshot.position_world[0],
        after_position_y=None if after_truth_snapshot is None else after_truth_snapshot.position_world[1],
        after_position_z=None if after_truth_snapshot is None else after_truth_snapshot.position_world[2],
        before_dimension=None if before_truth_snapshot is None else before_truth_snapshot.dimension,
        after_dimension=None if after_truth_snapshot is None else after_truth_snapshot.dimension,
        grid_anchor_x=(
            None if after_truth_snapshot is None and before_truth_snapshot is None
            else (after_truth_snapshot or before_truth_snapshot).grid_anchor_world[0]
        ),
        grid_anchor_y=(
            None if after_truth_snapshot is None and before_truth_snapshot is None
            else (after_truth_snapshot or before_truth_snapshot).grid_anchor_world[1]
        ),
        grid_anchor_z=(
            None if after_truth_snapshot is None and before_truth_snapshot is None
            else (after_truth_snapshot or before_truth_snapshot).grid_anchor_world[2]
        ),
        anchor_source=(
            None if after_truth_snapshot is None and before_truth_snapshot is None
            else (after_truth_snapshot or before_truth_snapshot).anchor_source
        ),
        probe_world_cells=probe_world_cells,
        probe_grid_cells=probe_grid_cells,
        before_block_truth=(
            None if block_truth is None or before_truth_snapshot is None
            else tuple(item.as_dict() for item in before_truth_snapshot.block_truth)
        ),
        after_block_truth=(
            None if block_truth is None or after_truth_snapshot is None
            else tuple(item.as_dict() for item in after_truth_snapshot.block_truth)
        ),
        truth_missing_count=(
            fluid_inspection.truth_missing_count if fluid_inspection is not None
            else block_truth.truth_missing_count if block_truth is not None
            else (
                None if before_truth_snapshot is None or after_truth_snapshot is None
                else before_truth_snapshot.truth_missing_count
                + after_truth_snapshot.truth_missing_count
            )
        ),
        stimulus_target=(
            fluid_execution.target if fluid_execution is not None
            else truth_execution.target if truth_execution is not None
            else stimulus_target
        ),
        target_changed=(
            None if fluid_inspection is None and block_truth is None
            else fluid_inspection.target_changed if fluid_inspection is not None
            else block_truth.target_changed
        ),
        target_expected_block_present=(
            None if block_truth is None else block_truth.target_expected_block_present
        ),
        control_cells_unchanged=(
            None if fluid_inspection is None and block_truth is None
            else fluid_inspection.control_cells_unchanged if fluid_inspection is not None
            else block_truth.control_cells_unchanged
        ),
        before_server_fluid_truth=(
            None
            if fluid_inspection is None
            or before_truth_snapshot is None
            or not before_truth_snapshot.fluid_truth
            else tuple(item.as_dict() for item in before_truth_snapshot.fluid_truth)
        ),
        after_server_fluid_truth=(
            None
            if fluid_inspection is None
            or after_truth_snapshot is None
            or not after_truth_snapshot.fluid_truth
            else tuple(item.as_dict() for item in after_truth_snapshot.fluid_truth)
        ),
        fluid_variant=fluid_variant_name,
        expected_target_fluid_type=expected_target_fluid_type,
        expected_target_flow_state=expected_target_flow_state,
        target_expected_fluid_present=(
            None if fluid_inspection is None else fluid_inspection.target_expected_fluid_present
        ),
        source_flowing_match=(
            None if fluid_inspection is None else fluid_inspection.source_flowing_match
        ),
    )


def _close_backend(backend: object) -> tuple[bool, str | None]:
    close = getattr(backend, "close", None)
    if not callable(close):
        return False, "close is not callable"
    try:
        close()
    except Exception as exc:
        return False, _format_error(exc)
    return True, None


def _success_outcome(case: EnvironmentValidationCase) -> str | None:
    if case.check_id is EnvironmentValidationId.E0:
        return E0_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E1:
        return E1_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E2:
        return E2_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E3:
        return E3_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E4:
        return E4_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E5:
        return E5_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E6:
        return E6_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E7:
        return E7_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E8:
        return E8_SUCCESS_OUTCOME
    if case.check_id is EnvironmentValidationId.E9:
        return E9_SUCCESS_OUTCOME
    return None


class EnvironmentValidationRunner:
    """Run one P1 validation case without starting MineRL."""

    def run(
        self,
        case: EnvironmentValidationCase,
        backend_factory: BackendFactory,
        *,
        episode_id: str,
        expected_inventory: Mapping[str, int] | None = None,
        expected_selected_item: str | None = None,
        requested_yaw: float = 20.0,
        requested_pitch: float = 0.0,
        yaw_tolerance: float = 1.0,
        pitch_tolerance: float = 1.0,
        requested_forward: float = 1.0,
        requested_strafe: float = 0.0,
        requested_sprint: bool = False,
        requested_jump: bool = False,
        requested_duration_ticks: int = 1,
        minimum_horizontal_distance: float = 0.02,
        minimum_forward_projection: float = 0.02,
        maximum_lateral_drift: float = 0.02,
        maximum_horizontal_distance: float = 0.5,
        maximum_vertical_drift: float = 0.25,
        calibration_block: str = "dirt",
        expected_before_block: str = "air",
        target_cell: tuple[int, int, int] = (0, 4, 1),
        target_grid_cell: tuple[int, int, int] | None = None,
        bucket_variant: str = "water",
        fluid_variant: str = "water",
    ) -> EnvironmentValidationResult:
        if not isinstance(case, EnvironmentValidationCase):
            raise ValueError("case must be EnvironmentValidationCase")
        if not callable(backend_factory):
            raise ValueError("backend_factory must be callable")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        episode_id = episode_id.strip()

        expected_success = _success_outcome(case)
        if expected_success is None:
            return _result(
                case=case,
                episode_id=episode_id,
                success=False,
                outcome="runtime_error",
                created=False,
                reset_completed=False,
                initial_state_present=False,
                closed=False,
                error=f"unimplemented validation case: {case.check_id.value}",
            )

        expected_inventory_snapshot: dict[str, int] | None = None
        if case.check_id is EnvironmentValidationId.E2:
            try:
                expected_inspection = inspect_inventory(expected_inventory)
            except Exception as exc:
                expected_error = _format_error(exc)
            else:
                expected_error = expected_inspection.error
                if expected_inspection.valid:
                    assert expected_inspection.inventory is not None
                    if expected_inspection.inventory:
                        expected_inventory_snapshot = dict(
                            expected_inspection.inventory
                        )
                    else:
                        expected_error = "expected_inventory must be non-empty"
            if expected_inventory_snapshot is None:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid expected_inventory: "
                    + (expected_error or "expected_inventory is required"),
                )

        expected_selected_item_snapshot: str | None = None
        if case.check_id is EnvironmentValidationId.E3:
            try:
                expected_selected_item_snapshot = validate_selected_item(
                    expected_selected_item, "expected_selected_item"
                )
            except (TypeError, ValueError) as exc:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid expected_selected_item: " + _format_error(exc),
                )

        placement_calibration_block: str | None = None
        placement_expected_before: str | None = None
        placement_target_cell: tuple[int, int, int] | None = None
        placement_target_grid_cell: tuple[int, int, int] | None = None
        if case.check_id is EnvironmentValidationId.E6:
            try:
                placement_calibration_block = validate_block_name(
                    calibration_block, "calibration_block"
                )
                placement_expected_before = validate_block_name(
                    expected_before_block, "expected_before_block"
                )
                placement_target_cell = validate_target_cell(target_cell, "target_world_cell")
                if target_grid_cell is None:
                    # Frozen E6 spawn world (0, 4, 0); atSpawn grid = world - spawn.
                    placement_target_grid_cell = spawn_relative_grid_cell(
                        placement_target_cell, (0, 4, 0)
                    )
                else:
                    placement_target_grid_cell = validate_target_cell(
                        target_grid_cell, "target_grid_cell"
                    )
                if type(requested_duration_ticks) is not int or requested_duration_ticks < 1:
                    raise ValueError("requested_duration_ticks must be a positive int")
            except (TypeError, ValueError) as exc:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid E6 calibration: " + _format_error(exc),
                )

        e7_variant: BucketCalibrationVariant | None = None
        e7_bucket_item: str | None = None
        e7_expected_fluid: str | None = None
        e7_before_inventory: dict[str, int] | None = None
        e7_after_inventory: dict[str, int] | None = None
        e7_target_world: tuple[int, int, int] | None = None
        e7_target_grid: tuple[int, int, int] | None = None
        if case.check_id is EnvironmentValidationId.E7:
            try:
                e7_variant = validate_bucket_variant(bucket_variant)
                e7_bucket_item = frozen_bucket_item(e7_variant)
                e7_expected_fluid = frozen_expected_fluid(e7_variant)
                e7_before_inventory = frozen_before_inventory(e7_variant)
                e7_after_inventory = frozen_after_inventory(e7_variant)
                e7_target_world = validate_target_cell(target_cell, "target_world_cell")
                if target_grid_cell is None:
                    e7_target_grid = spawn_relative_grid_cell(
                        e7_target_world, (0, 4, 0)
                    )
                else:
                    e7_target_grid = validate_target_cell(
                        target_grid_cell, "target_grid_cell"
                    )
                if spawn_relative_grid_cell(e7_target_world, (0, 4, 0)) != e7_target_grid:
                    raise ValueError(
                        "E7 target_grid_cell does not match spawn-relative world conversion"
                    )
                if type(requested_duration_ticks) is not int or requested_duration_ticks < 1:
                    raise ValueError("requested_duration_ticks must be a positive int")
                validate_fluid_class(e7_expected_fluid, "expected_fluid")
            except (TypeError, ValueError) as exc:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid E7 calibration: " + _format_error(exc),
                )

        e8_probe_world: tuple[tuple[int, int, int], ...] | None = None
        e8_probe_grid: tuple[tuple[int, int, int], ...] | None = None
        e8_expected_before: dict[tuple[int, int, int], str] | None = None
        e8_expected_after: dict[tuple[int, int, int], str] | None = None
        e8_target_world: tuple[int, int, int] | None = None
        e8_controls: tuple[tuple[int, int, int], ...] | None = None
        e8_stimulus_target: str | None = None
        e8_position_min: tuple[float, float, float] | None = None
        e8_position_max: tuple[float, float, float] | None = None
        if case.check_id is EnvironmentValidationId.E8:
            try:
                e8_stimulus_target = validate_block_name("dirt", "stimulus_target")
                e8_target_world = validate_target_cell((0, 4, 1), "target_world_cell")
                e8_probe_world = (
                    e8_target_world,
                    validate_target_cell((1, 4, 1), "control_right_world"),
                    validate_target_cell((-1, 4, 1), "control_left_world"),
                )
                e8_probe_grid = (
                    spawn_relative_grid_cell(e8_probe_world[0], (0, 4, 0)),
                    spawn_relative_grid_cell(e8_probe_world[1], (0, 4, 0)),
                    spawn_relative_grid_cell(e8_probe_world[2], (0, 4, 0)),
                )
                e8_controls = e8_probe_world[1:]
                e8_expected_before = {cell: "air" for cell in e8_probe_world}
                e8_expected_after = {
                    e8_probe_world[0]: "dirt",
                    e8_probe_world[1]: "air",
                    e8_probe_world[2]: "air",
                }
                e8_position_min = (-2.0, 2.0, -2.0)
                e8_position_max = (3.0, 6.0, 3.0)
                if type(requested_duration_ticks) is not int or requested_duration_ticks < 1:
                    raise ValueError("requested_duration_ticks must be a positive int")
                if e8_probe_grid != ((0, 0, 1), (1, 0, 1), (-1, 0, 1)):
                    raise ValueError("E8 probe grid conversion is not the frozen atSpawn mapping")
            except (TypeError, ValueError) as exc:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid E8 calibration: " + _format_error(exc),
                )

        e9_variant: FluidCalibrationVariant | None = None
        e9_probe_world: tuple[tuple[int, int, int], ...] | None = None
        e9_probe_grid: tuple[tuple[int, int, int], ...] | None = None
        e9_expected_before: dict[tuple[int, int, int], tuple[str, str]] | None = None
        e9_expected_after: dict[tuple[int, int, int], tuple[str, str]] | None = None
        e9_target_world: tuple[int, int, int] | None = None
        e9_controls: tuple[tuple[int, int, int], ...] | None = None
        e9_stimulus_target: str | None = None
        e9_expected_type: str | None = None
        e9_expected_flow: str | None = None
        e9_position_min: tuple[float, float, float] | None = None
        e9_position_max: tuple[float, float, float] | None = None
        if case.check_id is EnvironmentValidationId.E9:
            try:
                e9_variant = validate_fluid_variant(fluid_variant)
                e9_stimulus_target = frozen_fluid_bucket_item(e9_variant)
                e9_expected_type = frozen_expected_fluid_type(e9_variant)
                e9_expected_flow = frozen_expected_flow_state(e9_variant)
                e9_target_world = validate_target_cell((0, 4, 1), "target_world_cell")
                e9_probe_world = (
                    e9_target_world,
                    validate_target_cell((0, 5, 1), "control_above_target"),
                    validate_target_cell((0, 5, 0), "control_above_spawn"),
                )
                e9_probe_grid = (
                    spawn_relative_grid_cell(e9_probe_world[0], (0, 4, 0)),
                    spawn_relative_grid_cell(e9_probe_world[1], (0, 4, 0)),
                    spawn_relative_grid_cell(e9_probe_world[2], (0, 4, 0)),
                )
                e9_controls = e9_probe_world[1:]
                e9_expected_before = {cell: ("none", "none") for cell in e9_probe_world}
                e9_expected_after = {
                    e9_probe_world[0]: (e9_expected_type, e9_expected_flow),
                    e9_probe_world[1]: ("none", "none"),
                    e9_probe_world[2]: ("none", "none"),
                }
                e9_position_min = (-2.0, 2.0, -2.0)
                e9_position_max = (3.0, 7.0, 3.0)
                if type(requested_duration_ticks) is not int or requested_duration_ticks < 1:
                    raise ValueError("requested_duration_ticks must be a positive int")
                if e9_probe_grid != ((0, 0, 1), (0, 1, 1), (0, 1, 0)):
                    raise ValueError("E9 probe grid conversion is not the frozen atSpawn mapping")
            except (TypeError, ValueError) as exc:
                return _result(
                    case=case,
                    episode_id=episode_id,
                    success=False,
                    outcome="runtime_error",
                    created=False,
                    reset_completed=False,
                    initial_state_present=False,
                    closed=False,
                    error="invalid E9 calibration: " + _format_error(exc),
                )

        created = False
        reset_completed = False
        initial_state_present = False
        closed = False
        error: str | None = None
        close_error: str | None = None
        outcome = "runtime_error"
        backend: object | None = None
        rgb: RGBInspection | None = None
        inventory: InventoryInspection | None = None
        inventory_matches_expected: bool | None = None
        selected_item: SelectedItemInspection | None = None
        selected_item_matches_expected: bool | None = None
        before_orientation: CameraOrientationSnapshot | None = None
        after_orientation: CameraOrientationSnapshot | None = None
        camera_execution: CameraActionExecution | None = None
        camera: CameraInspection | None = None
        camera_agent_id: str | None = None
        before_position: PlayerPositionSnapshot | None = None
        after_position: PlayerPositionSnapshot | None = None
        movement_orientation: MovementOrientationSnapshot | None = None
        movement_execution: MovementActionExecution | None = None
        movement: MovementInspection | None = None
        movement_agent_id: str | None = None
        before_block_truth: BlockPlacementTruthSnapshot | None = None
        after_block_truth: BlockPlacementTruthSnapshot | None = None
        placement_execution: PlacementActionExecution | None = None
        placement: PlacementInspection | None = None
        placement_agent_id: str | None = None
        before_inventory_snapshot: BucketInventorySnapshot | None = None
        after_inventory_snapshot: BucketInventorySnapshot | None = None
        before_fluid_truth: BucketFluidTruthSnapshot | None = None
        after_fluid_truth: BucketFluidTruthSnapshot | None = None
        bucket_execution: BucketActionExecution | None = None
        bucket: BucketUsageInspection | None = None
        bucket_agent_id: str | None = None
        before_selected_item: str | None = None
        after_selected_item: str | None = None
        before_truth_snapshot: ServerTruthSnapshot | None = None
        after_truth_snapshot: ServerTruthSnapshot | None = None
        truth_execution: BlockTruthActionExecution | None = None
        block_truth: BlockTruthInspection | None = None
        truth_agent_id: str | None = None
        fluid_execution: FluidTruthActionExecution | None = None
        fluid_inspection: FluidTruthInspection | None = None
        failure_stage: str | None = None
        original_exception_type: str | None = None
        reset_attempt_count: int | None = None
        environment_launch_count: int | None = None
        exception_traceback: str | None = None

        try:
            backend = backend_factory()
            if backend is None:
                outcome = "create_failed"
                error = "backend factory returned None"
            else:
                created = True
                reset = getattr(backend, "reset", None)
                if not callable(reset):
                    outcome = "runtime_error"
                    error = "backend reset is not callable"
                else:
                    reset_result = reset()
                    reset_completed = True
                    if initial_state_exists(reset_result, episode_id=episode_id):
                        initial_state_present = True
                        if case.check_id is EnvironmentValidationId.E1:
                            rgb = inspect_public_rgb(
                                reset_result, episode_id=episode_id
                            )
                            outcome = rgb.outcome
                            error = rgb.error
                        elif case.check_id is EnvironmentValidationId.E2:
                            inventory = inspect_public_inventory(
                                reset_result, episode_id=episode_id
                            )
                            outcome = inventory.outcome
                            error = inventory.error
                            if inventory.valid:
                                assert inventory.inventory is not None
                                assert expected_inventory_snapshot is not None
                                inventory_matches_expected = (
                                    inventory.inventory
                                    == expected_inventory_snapshot
                                )
                                if inventory_matches_expected:
                                    outcome = E2_SUCCESS_OUTCOME
                                    error = None
                                else:
                                    outcome = INVENTORY_MISMATCH
                                    error = (
                                        "observed inventory does not exactly match "
                                        "expected_inventory"
                                    )
                        elif case.check_id is EnvironmentValidationId.E3:
                            selected_item = inspect_public_selected_item(
                                reset_result, episode_id=episode_id
                            )
                            outcome = selected_item.outcome
                            error = selected_item.error
                            if selected_item.valid:
                                assert selected_item.selected_item is not None
                                assert expected_selected_item_snapshot is not None
                                selected_item_matches_expected = (
                                    selected_item.selected_item
                                    == expected_selected_item_snapshot
                                )
                                if selected_item_matches_expected:
                                    outcome = E3_SUCCESS_OUTCOME
                                    error = None
                                else:
                                    outcome = SELECTED_ITEM_MISMATCH
                                    error = (
                                        "observed selected item does not exactly match "
                                        "expected_selected_item"
                                    )
                        elif case.check_id is EnvironmentValidationId.E4:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    camera_agent_id = first_agent.strip()
                            truth = getattr(backend, "camera_orientation_truth", None)
                            execute = getattr(backend, "execute_camera_action", None)
                            if not callable(truth) or not callable(execute):
                                outcome = "runtime_error"
                                error = "E4 backend camera truth/action surface is not callable"
                            else:
                                try:
                                    candidate = truth()
                                except (TypeError, ValueError) as exc:
                                    outcome = ORIENTATION_INVALID
                                    error = _format_error(exc)
                                else:
                                    if candidate is None:
                                        outcome = ORIENTATION_BEFORE_MISSING
                                        error = "orientation truth is missing before action"
                                    elif not isinstance(candidate, CameraOrientationSnapshot):
                                        outcome = ORIENTATION_INVALID
                                        error = "before orientation has the wrong type"
                                    else:
                                        before_orientation = candidate
                                if before_orientation is not None:
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "look",
                                                "duration_ticks": 1,
                                                "parameters": {
                                                    "pitch": requested_pitch,
                                                    "yaw": requested_yaw,
                                                },
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "action_rejected"
                                        error = "camera action protocol rejected: " + (parsed.error or "unknown error")
                                    else:
                                        candidate_execution = execute(parsed.action)
                                        if not isinstance(candidate_execution, CameraActionExecution):
                                            raise TypeError("execute_camera_action must return CameraActionExecution")
                                        camera_execution = candidate_execution
                                        try:
                                            candidate_after = truth()
                                        except (TypeError, ValueError) as exc:
                                            outcome = ORIENTATION_INVALID
                                            error = _format_error(exc)
                                        else:
                                            if candidate_after is None:
                                                outcome = ORIENTATION_AFTER_MISSING
                                                error = "orientation truth is missing after action"
                                            elif not isinstance(candidate_after, CameraOrientationSnapshot):
                                                outcome = ORIENTATION_INVALID
                                                error = "after orientation has the wrong type"
                                            else:
                                                after_orientation = candidate_after
                                                camera = inspect_camera_change(
                                                    before_orientation,
                                                    after_orientation,
                                                    camera_execution,
                                                    yaw_tolerance=yaw_tolerance,
                                                    pitch_tolerance=pitch_tolerance,
                                                )
                                                outcome = camera.outcome
                                                error = camera.error
                        elif case.check_id is EnvironmentValidationId.E5:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    movement_agent_id = first_agent.strip()
                            position_truth = getattr(backend, "player_position_truth", None)
                            orientation_truth = getattr(backend, "movement_orientation_truth", None)
                            execute = getattr(backend, "execute_movement_action", None)
                            if not all(callable(value) for value in (position_truth, orientation_truth, execute)):
                                outcome = "runtime_error"
                                error = "E5 backend position/orientation/action surface is not callable"
                            else:
                                try:
                                    candidate_before = position_truth()
                                except (TypeError, ValueError) as exc:
                                    outcome = POSITION_INVALID
                                    error = _format_error(exc)
                                else:
                                    if candidate_before is None:
                                        outcome = POSITION_BEFORE_MISSING
                                        error = "position truth is missing before action"
                                    elif not isinstance(candidate_before, PlayerPositionSnapshot):
                                        outcome = POSITION_INVALID
                                        error = "before position has the wrong type"
                                    else:
                                        before_position = candidate_before
                                if before_position is not None:
                                    try:
                                        candidate_orientation = orientation_truth()
                                    except (TypeError, ValueError) as exc:
                                        outcome = MOVEMENT_ORIENTATION_INVALID
                                        error = _format_error(exc)
                                    else:
                                        if candidate_orientation is None:
                                            outcome = MOVEMENT_ORIENTATION_MISSING
                                            error = "reset yaw truth is missing"
                                        elif not isinstance(candidate_orientation, MovementOrientationSnapshot):
                                            outcome = MOVEMENT_ORIENTATION_INVALID
                                            error = "reset yaw truth has the wrong type"
                                        else:
                                            movement_orientation = candidate_orientation
                                if before_position is not None and movement_orientation is not None:
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "move",
                                                "duration_ticks": requested_duration_ticks,
                                                "parameters": {
                                                    "forward": requested_forward,
                                                    "strafe": requested_strafe,
                                                    "sprint": requested_sprint,
                                                    "jump": requested_jump,
                                                },
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "movement_action_rejected"
                                        error = "movement action protocol rejected: " + (parsed.error or "unknown error")
                                    else:
                                        candidate_execution = execute(parsed.action)
                                        if not isinstance(candidate_execution, MovementActionExecution):
                                            raise TypeError("execute_movement_action must return MovementActionExecution")
                                        movement_execution = candidate_execution
                                        try:
                                            candidate_after = position_truth()
                                        except (TypeError, ValueError) as exc:
                                            outcome = POSITION_INVALID
                                            error = _format_error(exc)
                                        else:
                                            if candidate_after is None:
                                                outcome = POSITION_AFTER_MISSING
                                                error = "position truth is missing after action"
                                            elif not isinstance(candidate_after, PlayerPositionSnapshot):
                                                outcome = POSITION_INVALID
                                                error = "after position has the wrong type"
                                            else:
                                                after_position = candidate_after
                                                movement = inspect_movement(
                                                    before_position,
                                                    after_position,
                                                    movement_orientation,
                                                    movement_execution,
                                                    minimum_horizontal_distance=minimum_horizontal_distance,
                                                    minimum_forward_projection=minimum_forward_projection,
                                                    maximum_lateral_drift=maximum_lateral_drift,
                                                    maximum_horizontal_distance=maximum_horizontal_distance,
                                                    maximum_vertical_drift=maximum_vertical_drift,
                                                )
                                                outcome = movement.outcome
                                                error = movement.error
                        elif case.check_id is EnvironmentValidationId.E6:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    placement_agent_id = first_agent.strip()
                            truth = getattr(backend, "block_placement_truth", None)
                            execute = getattr(backend, "execute_placement_action", None)
                            if not callable(truth) or not callable(execute):
                                outcome = "runtime_error"
                                error = "E6 backend block-truth/action surface is not callable"
                            else:
                                try:
                                    candidate_before = truth()
                                except (TypeError, ValueError) as exc:
                                    outcome = BLOCK_TRUTH_INVALID
                                    error = _format_error(exc)
                                else:
                                    if candidate_before is None:
                                        outcome = BLOCK_BEFORE_MISSING
                                        error = "block truth is missing before action"
                                    elif not isinstance(candidate_before, BlockPlacementTruthSnapshot):
                                        outcome = BLOCK_TRUTH_INVALID
                                        error = "before block truth has the wrong type"
                                    else:
                                        before_block_truth = candidate_before
                                if before_block_truth is not None:
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "place_block",
                                                "target": placement_calibration_block,
                                                "duration_ticks": requested_duration_ticks,
                                                "parameters": {},
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "placement_action_rejected"
                                        error = "placement action protocol rejected: " + (parsed.error or "unknown error")
                                    else:
                                        candidate_execution = execute(parsed.action)
                                        if not isinstance(candidate_execution, PlacementActionExecution):
                                            raise TypeError("execute_placement_action must return PlacementActionExecution")
                                        placement_execution = candidate_execution
                                        try:
                                            candidate_after = truth()
                                        except (TypeError, ValueError) as exc:
                                            outcome = BLOCK_TRUTH_INVALID
                                            error = _format_error(exc)
                                        else:
                                            if candidate_after is None:
                                                outcome = BLOCK_AFTER_MISSING
                                                error = "block truth is missing after action"
                                            elif not isinstance(candidate_after, BlockPlacementTruthSnapshot):
                                                outcome = BLOCK_TRUTH_INVALID
                                                error = "after block truth has the wrong type"
                                            else:
                                                after_block_truth = candidate_after
                                                assert placement_calibration_block is not None
                                                assert placement_expected_before is not None
                                                assert placement_target_cell is not None
                                                placement = inspect_block_placement(
                                                    before_block_truth,
                                                    after_block_truth,
                                                    placement_execution,
                                                    calibration_block=placement_calibration_block,
                                                    expected_before_block=placement_expected_before,
                                                    target_cell=placement_target_cell,
                                                    duration_ticks=requested_duration_ticks,
                                                )
                                                outcome = placement.outcome
                                                error = placement.error
                        elif case.check_id is EnvironmentValidationId.E7:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    bucket_agent_id = first_agent.strip()
                            inventory_truth = getattr(backend, "public_bucket_inventory", None)
                            fluid_truth = getattr(backend, "bucket_fluid_truth", None)
                            execute = getattr(backend, "execute_bucket_action", None)
                            selected = getattr(backend, "public_bucket_selected_item", None)
                            if not callable(inventory_truth) or not callable(fluid_truth) or not callable(execute):
                                outcome = "runtime_error"
                                error = "E7 backend inventory/fluid/action surface is not callable"
                            else:
                                try:
                                    candidate_before_inventory = inventory_truth()
                                    candidate_before_fluid = fluid_truth()
                                except (TypeError, ValueError) as exc:
                                    message = _format_error(exc)
                                    if "inventory" in message.lower():
                                        outcome = INVENTORY_INVALID
                                    else:
                                        outcome = FLUID_TRUTH_INVALID
                                    error = message
                                else:
                                    if candidate_before_inventory is None:
                                        outcome = INVENTORY_BEFORE_MISSING
                                        error = "public inventory is missing before action"
                                    elif not isinstance(candidate_before_inventory, BucketInventorySnapshot):
                                        outcome = INVENTORY_INVALID
                                        error = "before inventory has the wrong type"
                                    else:
                                        before_inventory_snapshot = candidate_before_inventory
                                    if before_inventory_snapshot is not None:
                                        if candidate_before_fluid is None:
                                            outcome = FLUID_BEFORE_MISSING
                                            error = "fluid truth is missing before action"
                                        elif not isinstance(candidate_before_fluid, BucketFluidTruthSnapshot):
                                            outcome = FLUID_TRUTH_INVALID
                                            error = "before fluid truth has the wrong type"
                                        else:
                                            before_fluid_truth = candidate_before_fluid
                                if before_inventory_snapshot is not None and before_fluid_truth is not None:
                                    if callable(selected):
                                        try:
                                            candidate_selected = selected()
                                        except (TypeError, ValueError):
                                            candidate_selected = None
                                        if isinstance(candidate_selected, str) and candidate_selected.strip():
                                            before_selected_item = candidate_selected.strip()
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "use_item",
                                                "target": e7_bucket_item,
                                                "duration_ticks": requested_duration_ticks,
                                                "parameters": {},
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "bucket_action_rejected"
                                        error = "bucket action protocol rejected: " + (parsed.error or "unknown error")
                                    else:
                                        candidate_execution = execute(parsed.action)
                                        if not isinstance(candidate_execution, BucketActionExecution):
                                            raise TypeError("execute_bucket_action must return BucketActionExecution")
                                        bucket_execution = candidate_execution
                                        try:
                                            candidate_after_inventory = inventory_truth()
                                            candidate_after_fluid = fluid_truth()
                                        except (TypeError, ValueError) as exc:
                                            message = _format_error(exc)
                                            if "inventory" in message.lower():
                                                outcome = INVENTORY_INVALID
                                            else:
                                                outcome = FLUID_TRUTH_INVALID
                                            error = message
                                        else:
                                            if candidate_after_inventory is None:
                                                outcome = INVENTORY_AFTER_MISSING
                                                error = "public inventory is missing after action"
                                            elif not isinstance(candidate_after_inventory, BucketInventorySnapshot):
                                                outcome = INVENTORY_INVALID
                                                error = "after inventory has the wrong type"
                                            else:
                                                after_inventory_snapshot = candidate_after_inventory
                                            if after_inventory_snapshot is not None:
                                                if candidate_after_fluid is None:
                                                    outcome = FLUID_AFTER_MISSING
                                                    error = "fluid truth is missing after action"
                                                elif not isinstance(candidate_after_fluid, BucketFluidTruthSnapshot):
                                                    outcome = FLUID_TRUTH_INVALID
                                                    error = "after fluid truth has the wrong type"
                                                else:
                                                    after_fluid_truth = candidate_after_fluid
                                                    if callable(selected):
                                                        try:
                                                            candidate_selected = selected()
                                                        except (TypeError, ValueError):
                                                            candidate_selected = None
                                                        if isinstance(candidate_selected, str) and candidate_selected.strip():
                                                            after_selected_item = candidate_selected.strip()
                                                    assert e7_variant is not None
                                                    assert e7_bucket_item is not None
                                                    assert e7_expected_fluid is not None
                                                    assert e7_before_inventory is not None
                                                    assert e7_after_inventory is not None
                                                    assert e7_target_world is not None
                                                    assert e7_target_grid is not None
                                                    bucket = inspect_bucket_usage(
                                                        before_inventory_snapshot,
                                                        after_inventory_snapshot,
                                                        before_fluid_truth,
                                                        after_fluid_truth,
                                                        bucket_execution,
                                                        variant=e7_variant,
                                                        bucket_item=e7_bucket_item,
                                                        expected_fluid=e7_expected_fluid,
                                                        expected_before_inventory=e7_before_inventory,
                                                        expected_after_inventory=e7_after_inventory,
                                                        target_world_cell=e7_target_world,
                                                        target_grid_cell=e7_target_grid,
                                                        duration_ticks=requested_duration_ticks,
                                                    )
                                                    outcome = bucket.outcome
                                                    error = bucket.error
                        elif case.check_id is EnvironmentValidationId.E8:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    truth_agent_id = first_agent.strip()
                            snapshot = getattr(backend, "server_truth_snapshot", None)
                            execute = getattr(backend, "execute_truth_stimulus", None)
                            if not callable(snapshot) or not callable(execute):
                                outcome = "runtime_error"
                                error = "E8 backend snapshot/stimulus surface is not callable"
                            else:
                                try:
                                    candidate_before = snapshot()
                                except (TypeError, ValueError) as exc:
                                    outcome = truth_error_outcome(exc)
                                    error = _format_error(exc)
                                else:
                                    if candidate_before is None:
                                        outcome = TRUTH_SNAPSHOT_MISSING
                                        error = "server truth snapshot is missing before stimulus"
                                    elif not isinstance(candidate_before, ServerTruthSnapshot):
                                        outcome = truth_error_outcome(
                                            TypeError("before snapshot has the wrong type")
                                        )
                                        error = "before snapshot has the wrong type"
                                    else:
                                        before_truth_snapshot = candidate_before
                                if before_truth_snapshot is not None:
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "place_block",
                                                "target": e8_stimulus_target,
                                                "duration_ticks": requested_duration_ticks,
                                                "parameters": {},
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "truth_stimulus_rejected"
                                        error = "E8 stimulus protocol rejected: " + (
                                            parsed.error or "unknown error"
                                        )
                                    else:
                                        try:
                                            candidate_execution = execute(parsed.action)
                                        except (TypeError, ValueError) as exc:
                                            outcome = truth_error_outcome(exc)
                                            error = _format_error(exc)
                                        else:
                                            if not isinstance(candidate_execution, BlockTruthActionExecution):
                                                raise TypeError(
                                                    "execute_truth_stimulus must return BlockTruthActionExecution"
                                                )
                                            truth_execution = candidate_execution
                                            try:
                                                candidate_after = snapshot()
                                            except (TypeError, ValueError) as exc:
                                                outcome = truth_error_outcome(exc)
                                                error = _format_error(exc)
                                            else:
                                                if candidate_after is None:
                                                    outcome = TRUTH_SNAPSHOT_MISSING
                                                    error = "server truth snapshot is missing after stimulus"
                                                elif not isinstance(candidate_after, ServerTruthSnapshot):
                                                    outcome = truth_error_outcome(
                                                        TypeError("after snapshot has the wrong type")
                                                    )
                                                    error = "after snapshot has the wrong type"
                                                else:
                                                    after_truth_snapshot = candidate_after
                                                    assert e8_probe_world is not None
                                                    assert e8_probe_grid is not None
                                                    assert e8_expected_before is not None
                                                    assert e8_expected_after is not None
                                                    assert e8_target_world is not None
                                                    assert e8_controls is not None
                                                    assert e8_stimulus_target is not None
                                                    block_truth = inspect_block_truth(
                                                        before_truth_snapshot,
                                                        after_truth_snapshot,
                                                        truth_execution,
                                                        probe_world_cells=e8_probe_world,
                                                        probe_grid_cells=e8_probe_grid,
                                                        expected_before_blocks=e8_expected_before,
                                                        expected_after_blocks=e8_expected_after,
                                                        target_world_cell=e8_target_world,
                                                        control_world_cells=e8_controls,
                                                        duration_ticks=requested_duration_ticks,
                                                        stimulus_target=e8_stimulus_target,
                                                        position_min=e8_position_min,
                                                        position_max=e8_position_max,
                                                    )
                                                    outcome = block_truth.outcome
                                                    error = block_truth.error
                        elif case.check_id is EnvironmentValidationId.E9:
                            if isinstance(reset_result, Mapping) and reset_result:
                                first_agent = next(iter(reset_result))
                                if isinstance(first_agent, str) and first_agent.strip():
                                    truth_agent_id = first_agent.strip()
                            snapshot = getattr(backend, "server_truth_snapshot", None)
                            execute = getattr(backend, "execute_fluid_stimulus", None)
                            if not callable(snapshot) or not callable(execute):
                                outcome = "runtime_error"
                                error = "E9 backend snapshot/stimulus surface is not callable"
                            else:
                                try:
                                    candidate_before = snapshot()
                                except (TypeError, ValueError) as exc:
                                    outcome = truth_error_outcome(exc)
                                    error = _format_error(exc)
                                else:
                                    if candidate_before is None:
                                        outcome = TRUTH_SNAPSHOT_MISSING
                                        error = "server truth snapshot is missing before stimulus"
                                    elif not isinstance(candidate_before, ServerTruthSnapshot):
                                        outcome = truth_error_outcome(
                                            TypeError("before snapshot has the wrong type")
                                        )
                                        error = "before snapshot has the wrong type"
                                    else:
                                        before_truth_snapshot = candidate_before
                                if before_truth_snapshot is not None:
                                    parsed = parse_macro_action(
                                        json.dumps(
                                            {
                                                "action_type": "use_item",
                                                "target": e9_stimulus_target,
                                                "duration_ticks": requested_duration_ticks,
                                                "parameters": {},
                                            },
                                            allow_nan=False,
                                            sort_keys=True,
                                        )
                                    )
                                    if not parsed.accepted:
                                        outcome = "truth_stimulus_rejected"
                                        error = "E9 stimulus protocol rejected: " + (
                                            parsed.error or "unknown error"
                                        )
                                    else:
                                        try:
                                            candidate_execution = execute(parsed.action)
                                        except (TypeError, ValueError) as exc:
                                            outcome = truth_error_outcome(exc)
                                            error = _format_error(exc)
                                        else:
                                            if not isinstance(candidate_execution, FluidTruthActionExecution):
                                                raise TypeError(
                                                    "execute_fluid_stimulus must return FluidTruthActionExecution"
                                                )
                                            fluid_execution = candidate_execution
                                            try:
                                                candidate_after = snapshot()
                                            except (TypeError, ValueError) as exc:
                                                outcome = truth_error_outcome(exc)
                                                error = _format_error(exc)
                                            else:
                                                if candidate_after is None:
                                                    outcome = TRUTH_SNAPSHOT_MISSING
                                                    error = "server truth snapshot is missing after stimulus"
                                                elif not isinstance(candidate_after, ServerTruthSnapshot):
                                                    outcome = truth_error_outcome(
                                                        TypeError("after snapshot has the wrong type")
                                                    )
                                                    error = "after snapshot has the wrong type"
                                                else:
                                                    after_truth_snapshot = candidate_after
                                                    assert e9_variant is not None
                                                    assert e9_probe_world is not None
                                                    assert e9_probe_grid is not None
                                                    assert e9_expected_before is not None
                                                    assert e9_expected_after is not None
                                                    assert e9_target_world is not None
                                                    assert e9_controls is not None
                                                    assert e9_stimulus_target is not None
                                                    fluid_inspection = inspect_fluid_truth(
                                                        before_truth_snapshot,
                                                        after_truth_snapshot,
                                                        fluid_execution,
                                                        probe_world_cells=e9_probe_world,
                                                        probe_grid_cells=e9_probe_grid,
                                                        expected_before_fluids=e9_expected_before,
                                                        expected_after_fluids=e9_expected_after,
                                                        target_world_cell=e9_target_world,
                                                        control_world_cells=e9_controls,
                                                        duration_ticks=requested_duration_ticks,
                                                        stimulus_target=e9_stimulus_target,
                                                        variant=e9_variant,
                                                        position_min=e9_position_min,
                                                        position_max=e9_position_max,
                                                    )
                                                    outcome = fluid_inspection.outcome
                                                    error = fluid_inspection.error
                        else:
                            outcome = E0_SUCCESS_OUTCOME
                    else:
                        outcome = "initial_state_missing"
                        error = "reset did not return a usable initial state"
        except Exception as exc:
            error = _format_error(exc)
            if not created:
                outcome = "create_failed"
            elif not reset_completed:
                outcome = "reset_failed"
                if case.check_id in (
                    EnvironmentValidationId.E5,
                    EnvironmentValidationId.E6,
                    EnvironmentValidationId.E7,
                    EnvironmentValidationId.E8,
                    EnvironmentValidationId.E9,
                ):
                    failure_stage = "reset"
                    original_exception_type = type(_root_exception(exc)).__name__
                    exception_traceback = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    audit = getattr(backend, "reset_failure_audit", None)
                    if callable(audit):
                        try:
                            audit_value = audit()
                        except Exception as audit_exc:
                            exception_traceback += (
                                "\nReset audit unavailable: "
                                + _format_error(audit_exc)
                                + "\n"
                            )
                        else:
                            if isinstance(audit_value, Mapping):
                                reset_attempt_count = audit_value.get(
                                    "reset_attempt_count"
                                )
                                environment_launch_count = audit_value.get(
                                    "environment_launch_count"
                                )
                            else:
                                exception_traceback += (
                                    "\nReset audit unavailable: wrong return type\n"
                                )
                    else:
                        exception_traceback += (
                            "\nReset audit unavailable: surface is not callable\n"
                        )
            elif case.check_id is EnvironmentValidationId.E6 and before_block_truth is not None:
                outcome = "action_failed"
                failure_stage = "action"
                original_exception_type = type(_root_exception(exc)).__name__
                exception_traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            elif (
                case.check_id is EnvironmentValidationId.E7
                and before_inventory_snapshot is not None
                and before_fluid_truth is not None
            ):
                outcome = "action_failed"
                failure_stage = "action"
                original_exception_type = type(_root_exception(exc)).__name__
                exception_traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            elif case.check_id is EnvironmentValidationId.E8 and before_truth_snapshot is not None:
                outcome = "action_failed"
                failure_stage = "action"
                original_exception_type = type(_root_exception(exc)).__name__
                exception_traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            elif case.check_id is EnvironmentValidationId.E9 and before_truth_snapshot is not None:
                outcome = "action_failed"
                failure_stage = "action"
                original_exception_type = type(_root_exception(exc)).__name__
                exception_traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            else:
                outcome = "runtime_error"
        finally:
            if created and backend is not None:
                closed, close_error = _close_backend(backend)

        if close_error is not None:
            if outcome in {
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
            }:
                outcome = "close_failed"
            error = error or close_error

        success = (
            outcome == expected_success
            and created
            and reset_completed
            and initial_state_present
            and closed
            and error is None
            and close_error is None
        )
        if not success and outcome in {
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
        }:
            outcome = "close_failed" if close_error is not None else "runtime_error"

        return _result(
            case=case,
            episode_id=episode_id,
            success=success,
            outcome=outcome,
            created=created,
            reset_completed=reset_completed,
            initial_state_present=initial_state_present,
            closed=closed,
            error=error,
            close_error=close_error,
            rgb=rgb if case.check_id is EnvironmentValidationId.E1 else None,
            inventory=(
                inventory if case.check_id is EnvironmentValidationId.E2 else None
            ),
            expected_inventory=(
                expected_inventory_snapshot
                if case.check_id is EnvironmentValidationId.E2
                else None
            ),
            inventory_matches_expected=(
                inventory_matches_expected
                if case.check_id is EnvironmentValidationId.E2
                else None
            ),
            selected_item=(
                selected_item if case.check_id is EnvironmentValidationId.E3 else None
            ),
            expected_selected_item=(
                expected_selected_item_snapshot
                if case.check_id is EnvironmentValidationId.E3
                else None
            ),
            selected_item_matches_expected=(
                selected_item_matches_expected
                if case.check_id is EnvironmentValidationId.E3
                else None
            ),
            before_orientation=before_orientation if case.check_id is EnvironmentValidationId.E4 else None,
            after_orientation=after_orientation if case.check_id is EnvironmentValidationId.E4 else None,
            camera_execution=camera_execution if case.check_id is EnvironmentValidationId.E4 else None,
            camera=camera if case.check_id is EnvironmentValidationId.E4 else None,
            camera_agent_id=camera_agent_id if case.check_id is EnvironmentValidationId.E4 else None,
            camera_requested_yaw=requested_yaw if case.check_id is EnvironmentValidationId.E4 else None,
            camera_requested_pitch=requested_pitch if case.check_id is EnvironmentValidationId.E4 else None,
            before_position=before_position if case.check_id is EnvironmentValidationId.E5 else None,
            after_position=after_position if case.check_id is EnvironmentValidationId.E5 else None,
            movement_orientation=movement_orientation if case.check_id is EnvironmentValidationId.E5 else None,
            movement_execution=movement_execution if case.check_id is EnvironmentValidationId.E5 else None,
            movement=movement if case.check_id is EnvironmentValidationId.E5 else None,
            movement_agent_id=movement_agent_id if case.check_id is EnvironmentValidationId.E5 else None,
            requested_forward=requested_forward if case.check_id is EnvironmentValidationId.E5 else None,
            requested_strafe=requested_strafe if case.check_id is EnvironmentValidationId.E5 else None,
            requested_sprint=requested_sprint if case.check_id is EnvironmentValidationId.E5 else None,
            requested_jump=requested_jump if case.check_id is EnvironmentValidationId.E5 else None,
            requested_duration_ticks=requested_duration_ticks if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            minimum_horizontal_distance=minimum_horizontal_distance if case.check_id is EnvironmentValidationId.E5 else None,
            minimum_forward_projection=minimum_forward_projection if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_lateral_drift=maximum_lateral_drift if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_horizontal_distance=maximum_horizontal_distance if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_vertical_drift=maximum_vertical_drift if case.check_id is EnvironmentValidationId.E5 else None,
            before_block_truth=before_block_truth if case.check_id is EnvironmentValidationId.E6 else None,
            after_block_truth=after_block_truth if case.check_id is EnvironmentValidationId.E6 else None,
            placement_execution=placement_execution if case.check_id is EnvironmentValidationId.E6 else None,
            placement=placement if case.check_id is EnvironmentValidationId.E6 else None,
            placement_agent_id=placement_agent_id if case.check_id is EnvironmentValidationId.E6 else None,
            calibration_block=placement_calibration_block if case.check_id is EnvironmentValidationId.E6 else None,
            expected_before_block=placement_expected_before if case.check_id is EnvironmentValidationId.E6 else None,
            target_cell=(
                placement_target_cell if case.check_id is EnvironmentValidationId.E6
                else e7_target_world if case.check_id is EnvironmentValidationId.E7
                else None
            ),
            target_grid_cell=(
                placement_target_grid_cell if case.check_id is EnvironmentValidationId.E6
                else e7_target_grid if case.check_id is EnvironmentValidationId.E7
                else None
            ),
            before_inventory_snapshot=before_inventory_snapshot if case.check_id is EnvironmentValidationId.E7 else None,
            after_inventory_snapshot=after_inventory_snapshot if case.check_id is EnvironmentValidationId.E7 else None,
            before_fluid_truth=before_fluid_truth if case.check_id is EnvironmentValidationId.E7 else None,
            after_fluid_truth=after_fluid_truth if case.check_id is EnvironmentValidationId.E7 else None,
            bucket_execution=bucket_execution if case.check_id is EnvironmentValidationId.E7 else None,
            bucket=bucket if case.check_id is EnvironmentValidationId.E7 else None,
            bucket_agent_id=bucket_agent_id if case.check_id is EnvironmentValidationId.E7 else None,
            bucket_variant=None if e7_variant is None or case.check_id is not EnvironmentValidationId.E7 else e7_variant.value,
            bucket_item=e7_bucket_item if case.check_id is EnvironmentValidationId.E7 else None,
            expected_fluid=e7_expected_fluid if case.check_id is EnvironmentValidationId.E7 else None,
            before_selected_item=before_selected_item if case.check_id is EnvironmentValidationId.E7 else None,
            after_selected_item=after_selected_item if case.check_id is EnvironmentValidationId.E7 else None,
            before_truth_snapshot=before_truth_snapshot if case.check_id in (EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            after_truth_snapshot=after_truth_snapshot if case.check_id in (EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            truth_execution=truth_execution if case.check_id is EnvironmentValidationId.E8 else None,
            block_truth=block_truth if case.check_id is EnvironmentValidationId.E8 else None,
            truth_agent_id=truth_agent_id if case.check_id in (EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            stimulus_target=(
                e9_stimulus_target if case.check_id is EnvironmentValidationId.E9
                else e8_stimulus_target if case.check_id is EnvironmentValidationId.E8
                else None
            ),
            probe_world_cells=(
                e9_probe_world if case.check_id is EnvironmentValidationId.E9
                else e8_probe_world if case.check_id is EnvironmentValidationId.E8
                else None
            ),
            probe_grid_cells=(
                e9_probe_grid if case.check_id is EnvironmentValidationId.E9
                else e8_probe_grid if case.check_id is EnvironmentValidationId.E8
                else None
            ),
            fluid_execution=fluid_execution if case.check_id is EnvironmentValidationId.E9 else None,
            fluid_inspection=fluid_inspection if case.check_id is EnvironmentValidationId.E9 else None,
            fluid_variant_name=None if e9_variant is None or case.check_id is not EnvironmentValidationId.E9 else e9_variant.value,
            expected_target_fluid_type=e9_expected_type if case.check_id is EnvironmentValidationId.E9 else None,
            expected_target_flow_state=e9_expected_flow if case.check_id is EnvironmentValidationId.E9 else None,
            failure_stage=failure_stage if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            original_exception_type=original_exception_type if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            reset_attempt_count=reset_attempt_count if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            environment_launch_count=environment_launch_count if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
            exception_traceback=exception_traceback if case.check_id in (EnvironmentValidationId.E5, EnvironmentValidationId.E6, EnvironmentValidationId.E7, EnvironmentValidationId.E8, EnvironmentValidationId.E9) else None,
        )
