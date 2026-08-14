"""Minimal P1 validation runner.

Executes one validation case in a controlled lifecycle. This phase
implements E0 lifecycle, E1 RGB, E2 inventory, E3 selected item, E4 camera,
and E5 movement. E4/E5 consume only narrow evaluator truth from their
integration adapters; the validation core remains MineRL-independent.
The runner never uses benchmark evaluator success semantics.
"""

from __future__ import annotations

import json
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
from obsidianlink.env.validation.result import (
    E0_SUCCESS_OUTCOME,
    E1_SUCCESS_OUTCOME,
    E2_SUCCESS_OUTCOME,
    E3_SUCCESS_OUTCOME,
    E4_SUCCESS_OUTCOME,
    E5_SUCCESS_OUTCOME,
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


@runtime_checkable
class LifecycleBackend(Protocol):
    """Smallest common backend surface required by E0--E5.

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
        agent_id=(movement_execution.agent_id if movement_execution is not None else movement_agent_id if movement_agent_id is not None else camera_execution.agent_id if camera_execution is not None else camera_agent_id),
        tested_step_id=(movement_execution.step_id if movement_execution is not None else None if camera_execution is None else camera_execution.step_id),
        action_type=(
            movement_execution.action_type if movement_execution is not None
            else "move" if requested_forward is not None
            else "look" if camera_requested_yaw is not None and camera_execution is None
            else None if camera_execution is None else camera_execution.action_type
        ),
        requested_yaw=(camera_requested_yaw if camera_execution is None else camera_execution.requested_yaw),
        requested_pitch=(camera_requested_pitch if camera_execution is None else camera_execution.requested_pitch),
        translated_action_accepted=(movement_execution.translated_action_accepted if movement_execution is not None else None if camera_execution is None else camera_execution.translated_action_accepted),
        tested_action_count=(movement_execution.tested_action_count if movement_execution is not None else 0 if requested_forward is not None else 0 if camera_requested_yaw is not None and camera_execution is None else None if camera_execution is None else camera_execution.tested_action_count),
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
        requested_duration_ticks=(requested_duration_ticks if movement_execution is None else movement_execution.duration_ticks),
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
            requested_duration_ticks=requested_duration_ticks if case.check_id is EnvironmentValidationId.E5 else None,
            minimum_horizontal_distance=minimum_horizontal_distance if case.check_id is EnvironmentValidationId.E5 else None,
            minimum_forward_projection=minimum_forward_projection if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_lateral_drift=maximum_lateral_drift if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_horizontal_distance=maximum_horizontal_distance if case.check_id is EnvironmentValidationId.E5 else None,
            maximum_vertical_drift=maximum_vertical_drift if case.check_id is EnvironmentValidationId.E5 else None,
        )
