"""Controlled real-Minecraft environment for the first GeneralAgent smoke."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment
from obsidianlink.env.scene import FLAT_WORLD, RESOLUTION

GENERAL_BLOCK_SMOKE_ENV_ID = "MineRLObsidianLinkGeneralBlockSmoke-v0"
_REGISTERED: dict[str, Any] = {}
_PLAYER_X = 0.5
_PLAYER_Y = 4.0
_PLAYER_Z = 0.5
_BLOCK_X = 0
_BLOCK_Y = 4
_BLOCK_Z = 2
_SMOKE_ITEMS = ["air", "diamond_pickaxe", "obsidian"]


def smoke_block_xml() -> str:
    return (
        f'<DrawBlock x="{_BLOCK_X}" y="{_BLOCK_Y}" '
        f'z="{_BLOCK_Z}" type="obsidian" />'
    )


def register_general_block_smoke_spec(
    *, name: str = GENERAL_BLOCK_SMOKE_ENV_ID,
) -> str:
    """Register a flat world with one legal, mineable obsidian target."""
    import gym  # type: ignore[import-untyped]
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import handlers
    from minerl.herobraine.hero.handler import Handler

    if name in _REGISTERED and name in gym.envs.registry.env_specs:
        return name

    class _SafeDrawingDecorator(handlers.DrawingDecorator):
        def xml_template(self) -> str:
            return """<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"""

    class GeneralBlockSmokeSpec(Treechop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_server_world_generators(self) -> list[Handler]:
            return [
                handlers.FlatWorldGenerator(
                    force_reset=True,
                    generatorString=FLAT_WORLD,
                )
            ]

        def create_server_decorators(self) -> list[Handler]:
            return [_SafeDrawingDecorator(smoke_block_xml())]

        def create_server_initial_conditions(self) -> list[Handler]:
            return [
                handlers.TimeInitialCondition(
                    allow_passage_of_time=False,
                    start_time=6000,
                ),
                handlers.SpawningInitialCondition(allow_spawning=False),
                handlers.WeatherInitialCondition(weather="clear"),
            ]

        def create_agent_start(self) -> list[Handler]:
            return [
                handlers.GuiScale(1.0),
                handlers.GammaSetting(2.0),
                handlers.FOVSetting(70.0),
                handlers.FakeCursorSize(0),
                handlers.AgentStartPlacement(
                    x=_PLAYER_X,
                    y=_PLAYER_Y,
                    z=_PLAYER_Z,
                    yaw=0.0,
                    pitch=25.0,
                ),
                handlers.InventoryAgentStart(
                    {0: {"type": "diamond_pickaxe", "quantity": 1}}
                ),
            ]

        def create_observables(self) -> list[Handler]:
            return [
                handlers.POVObservation(self.resolution),
                handlers.FlatInventoryObservation(list(_SMOKE_ITEMS)),
                handlers.EquippedItemObservation(
                    list(_SMOKE_ITEMS),
                    mainhand=True,
                ),
            ]

        def create_actionables(self) -> list[Handler]:
            return [
                action
                for action in super().create_actionables()
                if action.to_string() not in {"equip", "place"}
            ]

    spec = GeneralBlockSmokeSpec()
    _REGISTERED[name] = spec
    if name not in gym.envs.registry.env_specs:
        spec.register()
    return name


class GeneralBlockSmokeEnv(Environment):
    """Fixed survival-mode target using only MCP-Reborn-allowed drawing."""

    def __init__(self, *, warmup_steps: int = 8) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        register_general_block_smoke_spec()
        self.env_id = GENERAL_BLOCK_SMOKE_ENV_ID
        self.warmup_steps = int(warmup_steps)
        self._env = MineRLEnvironment(self.env_id)

    def reset(self) -> Observation:
        observation = self._env.reset()
        for _ in range(self.warmup_steps):
            observation = self._env.step(Action(ActionType.WAIT))
        return observation

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Action) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = [
    "GENERAL_BLOCK_SMOKE_ENV_ID",
    "GeneralBlockSmokeEnv",
    "register_general_block_smoke_spec",
    "smoke_block_xml",
]
