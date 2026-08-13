"""E2 inventory observation case.

Verifies reset-time public inventory structure and exact equality with an
explicit calibration expectation. It does not inspect selected item, perform
actions, query server truth, or implement later P1 cases.
"""

from __future__ import annotations

from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
    P1_VALIDATION_CASES,
)


def _e2_case() -> EnvironmentValidationCase:
    for case in P1_VALIDATION_CASES:
        if case.check_id is EnvironmentValidationId.E2:
            if case.name != "inventory_observation":
                raise ValueError("E2 manifest name must be inventory_observation")
            return case
    raise ValueError("E2 is missing from the P1 validation contract")


E2_INVENTORY_CASE = _e2_case()
