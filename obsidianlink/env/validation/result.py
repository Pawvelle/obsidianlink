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
) | CAMERA_OUTCOMES | frozenset({"cleanup_failed"})
E0_SUCCESS_OUTCOME = "lifecycle_ok"
E1_SUCCESS_OUTCOME = "rgb_ok"
E2_SUCCESS_OUTCOME = INVENTORY_OK
E3_SUCCESS_OUTCOME = SELECTED_ITEM_OK
E4_SUCCESS_OUTCOME = CAMERA_OK


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


@dataclass(frozen=True)
class EnvironmentValidationResult:
    """Deterministic, serializable P1 validation result.

    Success is fail-closed: missing initial state, invalid RGB/inventory,
    inventory mismatch, reset failure, close failure, or any runtime
    exception cannot be recorded as a clean result. This runtime always
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
        for field_name in (
            "requested_yaw", "requested_pitch", "before_yaw", "before_pitch",
            "after_yaw", "after_pitch", "normalized_yaw_delta", "pitch_delta",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, finite_angle(value, field_name))
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
        camera_fields = (
            self.agent_id, self.tested_step_id, self.action_type,
            self.requested_yaw, self.requested_pitch,
            self.translated_action_accepted, self.tested_action_count,
            self.before_yaw, self.before_pitch, self.after_yaw, self.after_pitch,
            self.normalized_yaw_delta, self.pitch_delta,
            self.direction_match, self.magnitude_match,
        )
        if self.check_id is not EnvironmentValidationId.E4 and any(
            value is not None for value in camera_fields
        ):
            raise ValueError("camera metadata is only valid for E4 results")
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
        elif self.outcome in {
            E0_SUCCESS_OUTCOME,
            E1_SUCCESS_OUTCOME,
            E2_SUCCESS_OUTCOME,
            E3_SUCCESS_OUTCOME,
            E4_SUCCESS_OUTCOME,
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
        return payload
