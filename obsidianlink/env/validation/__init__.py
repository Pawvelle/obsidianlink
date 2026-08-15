"""P1 real-environment validation contracts.

Importing this package never starts MineRL or Minecraft.
"""

from obsidianlink.env.validation.cases.inventory import E2_INVENTORY_CASE
from obsidianlink.env.validation.cases.lifecycle import E0_LIFECYCLE_CASE
from obsidianlink.env.validation.cases.rgb import E1_RGB_CASE
from obsidianlink.env.validation.cases.selected_item import E3_SELECTED_ITEM_CASE
from obsidianlink.env.validation.cases.camera import E4_CAMERA_CASE
from obsidianlink.env.validation.cases.movement import E5_MOVEMENT_CASE
from obsidianlink.env.validation.cases.placement import E6_PLACEMENT_CASE
from obsidianlink.env.validation.camera import (
    CameraActionExecution,
    CameraInspection,
    CameraOrientationSnapshot,
    inspect_camera_change,
    normalized_angular_delta,
)
from obsidianlink.env.validation.contract import (
    P1_VALIDATION_CASES,
    EnvironmentValidationCase,
    EnvironmentValidationId,
    p1_validation_manifest,
)
from obsidianlink.env.validation.inventory import (
    InventoryInspection,
    PublicInventoryObservation,
    inspect_inventory,
    inspect_public_inventory,
)
from obsidianlink.env.validation.movement import (
    MovementActionExecution,
    MovementInspection,
    MovementOrientationSnapshot,
    PlayerPositionSnapshot,
    inspect_movement,
)
from obsidianlink.env.validation.placement import (
    BlockPlacementTruthSnapshot,
    PlacementActionExecution,
    PlacementInspection,
    inspect_block_placement,
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
from obsidianlink.env.validation.selected_item import (
    PublicSelectedItemObservation,
    SelectedItemInspection,
    inspect_public_selected_item,
    inspect_selected_item,
)

__all__ = [
    "E0_LIFECYCLE_CASE",
    "E1_RGB_CASE",
    "E2_INVENTORY_CASE",
    "E3_SELECTED_ITEM_CASE",
    "E4_CAMERA_CASE",
    "E5_MOVEMENT_CASE",
    "E6_PLACEMENT_CASE",
    "EnvironmentValidationCase",
    "EnvironmentValidationId",
    "EnvironmentValidationRecorder",
    "EnvironmentValidationResult",
    "EnvironmentValidationRunner",
    "CameraActionExecution",
    "CameraInspection",
    "CameraOrientationSnapshot",
    "MovementActionExecution",
    "MovementInspection",
    "MovementOrientationSnapshot",
    "PlayerPositionSnapshot",
    "BlockPlacementTruthSnapshot",
    "PlacementActionExecution",
    "PlacementInspection",
    "P1_VALIDATION_CASES",
    "InventoryInspection",
    "PublicInventoryObservation",
    "PublicRGBObservation",
    "PublicSelectedItemObservation",
    "RGBInspection",
    "SelectedItemInspection",
    "inspect_inventory",
    "inspect_public_inventory",
    "inspect_public_rgb",
    "inspect_rgb_array",
    "inspect_public_selected_item",
    "inspect_selected_item",
    "inspect_camera_change",
    "inspect_movement",
    "inspect_block_placement",
    "normalized_angular_delta",
    "p1_validation_manifest",
]
