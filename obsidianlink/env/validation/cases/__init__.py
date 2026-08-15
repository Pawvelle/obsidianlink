"""P1 validation case constants; imports never execute a validation."""

from obsidianlink.env.validation.cases.inventory import E2_INVENTORY_CASE
from obsidianlink.env.validation.cases.lifecycle import E0_LIFECYCLE_CASE
from obsidianlink.env.validation.cases.rgb import E1_RGB_CASE
from obsidianlink.env.validation.cases.selected_item import E3_SELECTED_ITEM_CASE
from obsidianlink.env.validation.cases.camera import E4_CAMERA_CASE
from obsidianlink.env.validation.cases.movement import E5_MOVEMENT_CASE
from obsidianlink.env.validation.cases.placement import E6_PLACEMENT_CASE

__all__ = ["E0_LIFECYCLE_CASE", "E1_RGB_CASE", "E2_INVENTORY_CASE", "E3_SELECTED_ITEM_CASE", "E4_CAMERA_CASE", "E5_MOVEMENT_CASE", "E6_PLACEMENT_CASE"]
