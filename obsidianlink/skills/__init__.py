"""Primitive Minecraft capability registry plus legacy opt-in workflows."""

from obsidianlink.skills.base import SkillLibrary, SkillResult
from obsidianlink.skills.building import BuildStructureSkill
from obsidianlink.skills.crafting import CraftItemSkill
from obsidianlink.skills.mining import CollectWoodSkill, MineBlockSkill
from obsidianlink.skills.movement import ExploreAreaSkill, MoveForwardSkill
from obsidianlink.skills.primitive import (
    AttackSkill,
    CraftSkill,
    EquipItemSkill,
    InspectInventorySkill,
    InteractSkill,
    LookSkill,
    MoveSkill,
    PlaceBlockSkill,
    SmeltSkill,
    WaitSkill,
)


def default_skill_library() -> SkillLibrary:
    return SkillLibrary(
        [
            MoveSkill(),
            LookSkill(),
            AttackSkill(),
            InteractSkill(),
            EquipItemSkill(),
            InspectInventorySkill(),
            PlaceBlockSkill(),
            CraftSkill(),
            SmeltSkill(),
            WaitSkill(),
        ]
    )


def legacy_workflow_skill_library() -> SkillLibrary:
    """Opt-in compatibility library for pre-GeneralAgent prototypes.

    These task workflows are deliberately excluded from the GeneralAgent
    default planner surface.
    """
    return SkillLibrary(
        [
            CollectWoodSkill(),
            MoveForwardSkill(),
            MineBlockSkill(),
            CraftItemSkill(),
            ExploreAreaSkill(),
            BuildStructureSkill(),
        ]
    )


__all__ = [
    "AttackSkill",
    "BuildStructureSkill",
    "CollectWoodSkill",
    "CraftItemSkill",
    "CraftSkill",
    "EquipItemSkill",
    "ExploreAreaSkill",
    "InspectInventorySkill",
    "InteractSkill",
    "LookSkill",
    "MineBlockSkill",
    "MoveForwardSkill",
    "MoveSkill",
    "PlaceBlockSkill",
    "SmeltSkill",
    "SkillLibrary",
    "SkillResult",
    "WaitSkill",
    "default_skill_library",
    "legacy_workflow_skill_library",
]
