"""P1 environment-validation result type.

This type is independent from benchmark evaluation. It must not subclass
or reuse ``EvaluatorVerdict``. Offline results cannot claim
``integration_verified``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.inventory import (
    INVENTORY_OK,
    INVENTORY_OUTCOMES,
    inspect_inventory,
)


UNIT_VERIFIED = "unit_verified"

INVENTORY_MISMATCH = "inventory_mismatch"

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
) | INVENTORY_OUTCOMES | frozenset({INVENTORY_MISMATCH})
E0_SUCCESS_OUTCOME = "lifecycle_ok"
E1_SUCCESS_OUTCOME = "rgb_ok"
E2_SUCCESS_OUTCOME = INVENTORY_OK


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
        if (
            self.check_id is EnvironmentValidationId.E2
            and self.observed_inventory is not None
            and self.expected_inventory is not None
            and self.inventory_matches_expected is not None
            and self.inventory_matches_expected
            != (self.observed_inventory == self.expected_inventory)
        ):
            raise ValueError("inventory_matches_expected contradicts inventory mappings")
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
                    or self.inventory_matches_expected is not True
                    or self.observed_inventory != self.expected_inventory
                ):
                    raise ValueError(
                        "E2 success requires exact observed/expected inventory equality"
                    )
        elif self.outcome in {
            E0_SUCCESS_OUTCOME,
            E1_SUCCESS_OUTCOME,
            E2_SUCCESS_OUTCOME,
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
        return payload
