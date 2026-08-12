"""Frozen E0--E12 checklist for P1.

The cases are definitions only. Their status is deliberately ``not_run``
until a separately authorized real MineRL/Minecraft execution produces
integration evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EnvironmentValidationId(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"
    E7 = "E7"
    E8 = "E8"
    E9 = "E9"
    E10 = "E10"
    E11 = "E11"
    E12 = "E12"


@dataclass(frozen=True)
class EnvironmentValidationCase:
    check_id: EnvironmentValidationId
    name: str
    requires_server_truth: bool
    calibration_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, EnvironmentValidationId):
            raise ValueError("check_id must be EnvironmentValidationId")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if type(self.requires_server_truth) is not bool:
            raise ValueError("requires_server_truth must be bool")
        if type(self.calibration_only) is not bool:
            raise ValueError("calibration_only must be bool")
        if not self.calibration_only:
            raise ValueError("P1 validation cases must remain calibration-only")


P1_VALIDATION_CASES = (
    EnvironmentValidationCase(EnvironmentValidationId.E0, "reset_close", False),
    EnvironmentValidationCase(EnvironmentValidationId.E1, "rgb_observation", False),
    EnvironmentValidationCase(EnvironmentValidationId.E2, "inventory_observation", False),
    EnvironmentValidationCase(EnvironmentValidationId.E3, "selected_item", False),
    EnvironmentValidationCase(EnvironmentValidationId.E4, "camera_control", True),
    EnvironmentValidationCase(EnvironmentValidationId.E5, "movement", True),
    EnvironmentValidationCase(EnvironmentValidationId.E6, "block_placement", True),
    EnvironmentValidationCase(EnvironmentValidationId.E7, "bucket_usage", True),
    EnvironmentValidationCase(EnvironmentValidationId.E8, "server_side_block_truth", True),
    EnvironmentValidationCase(EnvironmentValidationId.E9, "water_lava_fluid_truth", True),
    EnvironmentValidationCase(
        EnvironmentValidationId.E10,
        "vanilla_water_lava_to_obsidian",
        True,
    ),
    EnvironmentValidationCase(EnvironmentValidationId.E11, "portal_activation", True),
    EnvironmentValidationCase(EnvironmentValidationId.E12, "dimension_transition", True),
)


def p1_validation_manifest() -> tuple[dict[str, object], ...]:
    """Return a detached offline manifest; no validation is executed."""

    return tuple(
        {
            "check_id": case.check_id.value,
            "name": case.name,
            "requires_server_truth": case.requires_server_truth,
            "calibration_only": case.calibration_only,
            "status": "not_run",
        }
        for case in P1_VALIDATION_CASES
    )
