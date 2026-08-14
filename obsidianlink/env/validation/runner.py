"""Minimal P1 validation runner.

Executes one validation case in a controlled lifecycle. This phase
implements E0 (create, reset, require an initial state, close), E1 (the same
lifecycle plus public RGB inspection), and E2 (reset-time public inventory
inspection plus exact comparison with an explicit calibration expectation),
and E3 (backend-public selected-item inspection and independent comparison).
The runner never uses benchmark evaluator success semantics.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, runtime_checkable

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
from obsidianlink.env.validation.result import (
    E0_SUCCESS_OUTCOME,
    E1_SUCCESS_OUTCOME,
    E2_SUCCESS_OUTCOME,
    E3_SUCCESS_OUTCOME,
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
    """Smallest backend surface required by E0--E3.

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
        )
