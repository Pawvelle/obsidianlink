"""P1 real-environment validation contracts.

Importing this package never starts MineRL or Minecraft.
"""

from obsidianlink.env.validation.contract import (
    P1_VALIDATION_CASES,
    EnvironmentValidationCase,
    EnvironmentValidationId,
    p1_validation_manifest,
)

__all__ = [
    "EnvironmentValidationCase",
    "EnvironmentValidationId",
    "P1_VALIDATION_CASES",
    "p1_validation_manifest",
]
