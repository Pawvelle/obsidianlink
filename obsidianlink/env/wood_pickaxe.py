"""Natural MineRL task with human-level crafting controls."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment

WOOD_PICKAXE_ENV_ID = "MineRLObsidianLinkWoodPickaxe-v0"
RESOLUTION = (640, 360)

LOGS = [
    "acacia_log",
    "birch_log",
    "dark_oak_log",
    "jungle_log",
    "oak_log",
    "spruce_log",
]
PLANKS = [name.replace("_log", "_planks") for name in LOGS]
INVENTORY_ITEMS = LOGS + PLANKS + [
    "iron_axe",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
]
_REGISTERED: dict[str, Any] = {}
_DENSE_FOREST_BIOME = 29


def register_wood_pickaxe_spec(*, name: str = WOOD_PICKAXE_ENV_ID) -> str:
    """Register lazily without starting Minecraft or the JVM server."""
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

    class WoodPickaxeSpec(Treechop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_observables(self) -> list[Handler]:
            return [
                handlers.POVObservation(self.resolution),
                handlers.FlatInventoryObservation(list(INVENTORY_ITEMS)),
                handlers.EquippedItemObservation(list(INVENTORY_ITEMS), mainhand=True),
            ]

        def create_server_world_generators(self) -> list[Handler]:
            # MineRL 1.0.2's nominal forest id (4) produced sparse plains on
            # the installed Minecraft 1.16.5 stack. Dark forest (29) retains
            # natural terrain while increasing nearby resource density.
            options = TREECHOP_WORLD_GENERATOR_OPTIONS.replace(
                '"fixedBiome":4', f'"fixedBiome":{_DENSE_FOREST_BIOME}'
            )
            return [
                handlers.DefaultWorldGenerator(
                    force_reset=True, generator_options=options
                )
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
            return acts

    spec = WoodPickaxeSpec()
    _REGISTERED[name] = spec
    if name not in gym.envs.registry.env_specs:
        spec.register()
    return name


class WoodPickaxeEnv(Environment):
    def __init__(self) -> None:
        register_wood_pickaxe_spec()
        self.env_id = WOOD_PICKAXE_ENV_ID
        self._env = MineRLEnvironment(self.env_id)

    def reset(self) -> Observation:
        return self._env.reset()

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Any) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = ["WOOD_PICKAXE_ENV_ID", "WoodPickaxeEnv", "register_wood_pickaxe_spec"]
