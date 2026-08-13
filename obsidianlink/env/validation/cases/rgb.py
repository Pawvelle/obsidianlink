"""E1 RGB observation case.

Verifies that reset produced a public HxWx3 uint8 RGB/POV image. It does
not inspect inventory, selected item, camera, movement, placement,
bucket, block/fluid truth, obsidian, portal, or Nether semantics.
"""

from __future__ import annotations

from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
    P1_VALIDATION_CASES,
)


def _e1_case() -> EnvironmentValidationCase:
    for case in P1_VALIDATION_CASES:
        if case.check_id is EnvironmentValidationId.E1:
            if case.name != "rgb_observation":
                raise ValueError("E1 manifest name must be rgb_observation")
            return case
    raise ValueError("E1 is missing from the P1 validation contract")


E1_RGB_CASE = _e1_case()
