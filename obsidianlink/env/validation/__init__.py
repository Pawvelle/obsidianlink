"""P1 real-environment validation contracts.

Importing this package never starts MineRL or Minecraft.
"""

from obsidianlink.env.validation.cases.lifecycle import E0_LIFECYCLE_CASE
from obsidianlink.env.validation.cases.rgb import E1_RGB_CASE
from obsidianlink.env.validation.contract import (
    P1_VALIDATION_CASES,
    EnvironmentValidationCase,
    EnvironmentValidationId,
    p1_validation_manifest,
)
from obsidianlink.env.validation.recorder import EnvironmentValidationRecorder
from obsidianlink.env.validation.result import EnvironmentValidationResult
from obsidianlink.env.validation.runner import EnvironmentValidationRunner
from obsidianlink.env.validation.rgb import (
    PublicRGBObservation,
    RGBInspection,
    inspect_public_rgb,
    inspect_rgb_array,
)

__all__ = [
    "E0_LIFECYCLE_CASE",
    "E1_RGB_CASE",
    "EnvironmentValidationCase",
    "EnvironmentValidationId",
    "EnvironmentValidationRecorder",
    "EnvironmentValidationResult",
    "EnvironmentValidationRunner",
    "P1_VALIDATION_CASES",
    "PublicRGBObservation",
    "RGBInspection",
    "inspect_public_rgb",
    "inspect_rgb_array",
    "p1_validation_manifest",
]
