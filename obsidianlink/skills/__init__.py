"""Default high-level Minecraft skill library."""

from obsidianlink.skills.base import SkillLibrary, SkillResult
from obsidianlink.skills.building import BuildStructureSkill
from obsidianlink.skills.crafting import CraftItemSkill
from obsidianlink.skills.mining import CollectWoodSkill, MineBlockSkill
from obsidianlink.skills.movement import ExploreAreaSkill, MoveForwardSkill


def default_skill_library() -> SkillLibrary:
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
    "BuildStructureSkill",
    "CollectWoodSkill",
    "CraftItemSkill",
    "ExploreAreaSkill",
    "MineBlockSkill",
    "MoveForwardSkill",
    "SkillLibrary",
    "SkillResult",
    "default_skill_library",
]
