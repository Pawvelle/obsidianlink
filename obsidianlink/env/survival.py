"""Empty-inventory survival world for live GeneralAgent playtests."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment

SURVIVAL_IRON_SWORD_ENV_ID = "MineRLObsidianLinkSurvivalIronSword-v0"
RESOLUTION = (640, 360)
_REGISTERED: dict[str, Any] = {}
_DENSE_FOREST_BIOME = 29

# Items the agent must be able to see for an iron-sword run. MineRL only
# reports listed names through FlatInventoryObservation.
SURVIVAL_INVENTORY_ITEMS = [
    "acacia_log",
    "birch_log",
    "dark_oak_log",
    "jungle_log",
    "oak_log",
    "spruce_log",
    "acacia_planks",
    "birch_planks",
    "dark_oak_planks",
    "jungle_planks",
    "oak_planks",
    "spruce_planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "wooden_axe",
    "wooden_sword",
    "wooden_shovel",
    "cobblestone",
    "stone",
    "furnace",
    "stone_pickaxe",
    "stone_axe",
    "stone_sword",
    "stone_shovel",
    "coal",
    "charcoal",
    "coal_ore",
    "iron_ore",
    "iron_ingot",
    "iron_sword",
    "iron_pickaxe",
    "iron_axe",
    "dirt",
    "grass_block",
    "sand",
    "gravel",
    "flint",
    "torch",
    "apple",
]


def iron_sword_count(inventory: dict[str, int] | None) -> int:
    items = inventory or {}
    total = 0
    for name, qty in items.items():
        key = str(name).strip().lower().split(":", 1)[-1]
        if key == "iron_sword":
            try:
                total += int(qty)
            except (TypeError, ValueError):
                continue
    return total


def register_survival_iron_sword_spec(
    *, name: str = SURVIVAL_IRON_SWORD_ENV_ID
) -> str:
    """Register a natural forest survival spec without starting Minecraft."""
    import gym  # type: ignore[import-untyped]
    from minerl.herobraine.env_specs.treechop_specs import (
        TREECHOP_WORLD_GENERATOR_OPTIONS,
        Treechop,
    )
    from minerl.herobraine.hero import handlers
    from minerl.herobraine.hero.handler import Handler
    from minerl.herobraine.hero.mc import INVERSE_KEYMAP

    if name in _REGISTERED and name in gym.envs.registry.env_specs:
        return name

    class SurvivalIronSwordSpec(Treechop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_observables(self) -> list[Handler]:
            return [
                handlers.POVObservation(self.resolution),
                handlers.FlatInventoryObservation(list(SURVIVAL_INVENTORY_ITEMS)),
                handlers.EquippedItemObservation(
                    list(SURVIVAL_INVENTORY_ITEMS), mainhand=True
                ),
            ]

        def create_agent_start(self) -> list[Handler]:
            # Empty inventory: no iron_axe, no starter tools.
            return [
                handlers.GuiScale(1.0),
                handlers.GammaSetting(2.0),
                handlers.FOVSetting(70.0),
                handlers.FakeCursorSize(16),
            ]

        def create_agent_handlers(self) -> list[Handler]:
            return []

        def create_rewardables(self) -> list[Handler]:
            return []

        def create_server_world_generators(self) -> list[Handler]:
            options = TREECHOP_WORLD_GENERATOR_OPTIONS.replace(
                '"fixedBiome":4', f'"fixedBiome":{_DENSE_FOREST_BIOME}'
            ).replace('"useCaves":false', '"useCaves":true')
            return [
                handlers.DefaultWorldGenerator(
                    force_reset=True, generator_options=options
                )
            ]

        def create_server_quit_producers(self) -> list[Handler]:
            return [handlers.ServerQuitWhenAnyAgentFinishes()]

        def create_server_initial_conditions(self) -> list[Handler]:
            return [
                handlers.TimeInitialCondition(
                    allow_passage_of_time=False,
                    start_time=6000,
                ),
                handlers.SpawningInitialCondition(allow_spawning=False),
                handlers.WeatherInitialCondition(weather="clear"),
            ]

        def create_actionables(self) -> list[Handler]:
            acts = super().create_actionables()
            names = {action.to_string() for action in acts}
            if "use" not in names:
                acts.append(handlers.KeybasedCommandAction("use", INVERSE_KEYMAP["use"]))
            if "inventory" not in names:
                acts.append(
                    handlers.KeybasedCommandAction(
                        "inventory", INVERSE_KEYMAP["inventory"]
                    )
                )
            for slot in range(1, 10):
                key = f"hotbar.{slot}"
                if key not in names:
                    acts.append(handlers.KeybasedCommandAction(key, str(slot)))
            return [
                action
                for action in acts
                if action.to_string() not in {"equip", "place", "craft", "nearbyCraft", "nearbySmelt"}
            ]

    spec = SurvivalIronSwordSpec()
    _REGISTERED[name] = spec
    if name not in gym.envs.registry.env_specs:
        spec.register()
    return name


class SurvivalIronSwordEnv(Environment):
    """Natural forest, empty inventory, human-level GUI controls."""

    def __init__(self) -> None:
        register_survival_iron_sword_spec()
        self.env_id = SURVIVAL_IRON_SWORD_ENV_ID
        self._env = MineRLEnvironment(self.env_id)

    def reset(self) -> Observation:
        return self._env.reset()

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Any) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = [
    "SURVIVAL_INVENTORY_ITEMS",
    "SURVIVAL_IRON_SWORD_ENV_ID",
    "SurvivalIronSwordEnv",
    "iron_sword_count",
    "register_survival_iron_sword_spec",
]
