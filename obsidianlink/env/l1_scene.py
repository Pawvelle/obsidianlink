"""Formal L1 controlled construction environment v0.1.

Fixed Overworld grass superflat: spawn, construction floor, and a 4×4
lava source pool. No pre-built portal frame. Inventory is given with
``InventoryAgentStart``. Item select is ``hotbar.1-9`` only.

Malmo 0.37.0 DrawingDecorator can only ``DrawBlock`` ``lava`` /
``obsidian``. An obsidian floor would let an Agent treat the ground as
portal material, so L1 does **not** draw a floor or walls. The walking
surface is the already-verified superflat grass
(``FLAT_WORLD = 3;7,2*3,2;1;``: bedrock, dirt, grass). DrawBlock only
places the lava pool into that grass.

This module is the environment. It is not a Scripted Oracle, L1
evaluator, or ReactiveAgent task.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment
from obsidianlink.env.scene import (
    FLAT_WORLD,
    PLAYER_PITCH,
    PLAYER_X,
    PLAYER_YAW,
    PLAYER_Z,
    RESOLUTION,
    WARMUP_STEPS,
)

L1_ENV_ID = "MineRLL1Controlled-v0"

# Superflat grass is y=3; feet stand at y=4. Do not reuse D1's y=101 platform.
PLAYER_Y = 4.0
FLOOR_Y = 3
FLOOR_SURFACE = "grass"

# 4×4 lava source pool in front of spawn, replacing grass blocks.
LAVA_X1, LAVA_X2 = -1, 2
LAVA_Y = FLOOR_Y
LAVA_Z1, LAVA_Z2 = 5, 8
LAVA_SOURCE_COUNT = 16

# Open grass between spawn and the pool. Not a portal frame.
CONSTRUCTION_X1, CONSTRUCTION_X2 = -4, 4
CONSTRUCTION_Y = FLOOR_Y
CONSTRUCTION_Z1, CONSTRUCTION_Z2 = 1, 4

# Hotbar 1–5. No lava_bucket. Slot numbers are Minecraft/Malmo 0-based.
L1_INVENTORY: dict[int, dict[str, Any]] = {
    0: {"type": "water_bucket", "quantity": 1},
    1: {"type": "bucket", "quantity": 1},
    2: {"type": "cobblestone", "quantity": 64},
    3: {"type": "iron_pickaxe", "quantity": 1},
    4: {"type": "flint_and_steel", "quantity": 1},
}

L1_INV_ITEMS = (
    "air",
    "bucket",
    "cobblestone",
    "flint_and_steel",
    "iron_pickaxe",
    "lava_bucket",
    "obsidian",
    "water_bucket",
)
L1_EQUIP_ITEMS = (
    "none",
    "water_bucket",
    "bucket",
    "lava_bucket",
    "cobblestone",
    "iron_pickaxe",
    "flint_and_steel",
    "other",
)

L1_LAYOUT: dict[str, Any] = {
    "spawn": {
        "x": PLAYER_X,
        "y": PLAYER_Y,
        "z": PLAYER_Z,
        "yaw": PLAYER_YAW,
        "pitch": PLAYER_PITCH,
    },
    "lava_pool": {
        "x1": LAVA_X1,
        "x2": LAVA_X2,
        "y": LAVA_Y,
        "z1": LAVA_Z1,
        "z2": LAVA_Z2,
        "size": "4x4",
        "source_count": LAVA_SOURCE_COUNT,
    },
    "construction_area": {
        "x1": CONSTRUCTION_X1,
        "x2": CONSTRUCTION_X2,
        "y": CONSTRUCTION_Y,
        "z1": CONSTRUCTION_Z1,
        "z2": CONSTRUCTION_Z2,
    },
    "prebuilt_portal": False,
    "floor_surface": FLOOR_SURFACE,
}


def _draw_block_xml(x: int, y: int, z: int, block_type: str) -> str:
    return f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block_type}" />'


def _draw_filled(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block_type: str
) -> str:
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    za, zb = (z1, z2) if z1 <= z2 else (z2, z1)
    parts: list[str] = []
    for x in range(xa, xb + 1):
        for y in range(ya, yb + 1):
            for z in range(za, zb + 1):
                parts.append(_draw_block_xml(x, y, z, block_type))
    return "".join(parts)


def lava_pool_coords() -> list[tuple[int, int, int]]:
    coords: list[tuple[int, int, int]] = []
    for x in range(LAVA_X1, LAVA_X2 + 1):
        for z in range(LAVA_Z1, LAVA_Z2 + 1):
            coords.append((x, LAVA_Y, z))
    return coords


def l1_scene_xml() -> str:
    """4×4 lava pool only. Floor is superflat grass; no obsidian."""
    return _draw_filled(LAVA_X1, LAVA_Y, LAVA_Z1, LAVA_X2, LAVA_Y, LAVA_Z2, "lava")


_REGISTERED_SPECS: dict[str, Any] = {}


def register_l1_spec(*, name: str = L1_ENV_ID) -> str:
    """Register the L1 env id. Idempotent. Imports MineRL lazily."""
    import gym  # type: ignore[import-untyped]
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import handlers
    from minerl.herobraine.hero.handler import Handler
    from minerl.herobraine.hero.mc import INVERSE_KEYMAP

    cached = _REGISTERED_SPECS.get(name)
    if cached is not None and name in gym.envs.registry.env_specs:
        return name

    class _SafeDrawingDecorator(handlers.DrawingDecorator):
        def xml_template(self) -> str:
            return """<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"""

    class L1ControlledSpec(Treechop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_server_world_generators(self) -> list[Handler]:
            return [
                handlers.FlatWorldGenerator(
                    force_reset=True, generatorString=FLAT_WORLD
                )
            ]

        def create_server_decorators(self) -> list[Handler]:
            return [_SafeDrawingDecorator(l1_scene_xml())]

        def create_server_initial_conditions(self) -> list[Handler]:
            return [
                handlers.TimeInitialCondition(
                    allow_passage_of_time=False, start_time=6000
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
                    x=PLAYER_X,
                    y=PLAYER_Y,
                    z=PLAYER_Z,
                    yaw=PLAYER_YAW,
                    pitch=PLAYER_PITCH,
                ),
                handlers.InventoryAgentStart(dict(L1_INVENTORY)),
            ]

        def create_observables(self) -> list[Handler]:
            # No ObservationFromGrid. RGB + inventory + main-hand only.
            return [
                handlers.POVObservation(self.resolution),
                handlers.FlatInventoryObservation(list(L1_INV_ITEMS)),
                handlers.EquippedItemObservation(
                    list(L1_EQUIP_ITEMS), mainhand=True
                ),
            ]

        def create_actionables(self) -> list[Handler]:
            # EquipAction is unusable on this MineRL 1.0.2 / MCP-Reborn stack.
            acts = super().create_actionables()
            names = {a.to_string() for a in acts}
            if "use" not in names:
                acts.append(
                    handlers.KeybasedCommandAction("use", INVERSE_KEYMAP["use"])
                )
            for i in range(1, 10):
                key = f"hotbar.{i}"
                if key not in names:
                    acts.append(handlers.KeybasedCommandAction(key, str(i)))
            return [a for a in acts if a.to_string() not in {"equip", "place"}]

        def create_monitors(self) -> list[Handler]:
            return [handlers.ObservationFromCurrentLocation()]

        def create_rewardables(self) -> list[Handler]:
            # Evaluator-only reward channel. ``reward`` is gym step() return
            # value, never copied onto Observation (see MineRLEnvironment).
            # A per-tick reward while touching ``nether_portal`` is
            # portal-activation evidence: the block only exists if ignition
            # created a real, functioning portal.
            return [
                handlers.RewardForTouchingBlockType(
                    [
                        {
                            "type": "nether_portal",
                            "behaviour": "onceOnly",
                            "reward": 1.0,
                        }
                    ]
                )
            ]

        def create_agent_handlers(self) -> list[Handler]:
            return []

        def create_server_quit_producers(self) -> list[Handler]:
            return [
                handlers.ServerQuitFromTimeUp(400000),
                handlers.ServerQuitWhenAnyAgentFinishes(),
            ]

    spec = L1ControlledSpec()
    _REGISTERED_SPECS[spec.name] = spec
    if spec.name not in gym.envs.registry.env_specs:
        spec.register()
    return spec.name


class L1ControlledEnv(Environment):
    """Fixed L1 grass superflat. Hidden layout is evaluator-only."""

    def __init__(self, warmup_steps: int = WARMUP_STEPS) -> None:
        register_l1_spec()
        self.env_id = L1_ENV_ID
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self._env = MineRLEnvironment(env_id=L1_ENV_ID)

    @property
    def hidden_state(self) -> dict[str, Any]:
        hidden = dict(self._env.hidden_state)
        hidden["l1_layout"] = dict(L1_LAYOUT)
        hidden["target_truths"] = {
            "lava": True,
            "prebuilt_portal": False,
        }
        return hidden

    @property
    def last_info(self) -> dict[str, Any]:
        return self._env.last_info

    @property
    def action_space_keys(self) -> tuple[str, ...] | None:
        return self._env.action_space_keys

    def reset(self) -> Observation:
        observation = self._env.reset()
        if self.warmup_steps:
            wait = Action(type=ActionType.WAIT)
            for _ in range(self.warmup_steps):
                observation = self._env.step(wait)
        return observation

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Action) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = [
    "CONSTRUCTION_X1",
    "CONSTRUCTION_X2",
    "CONSTRUCTION_Y",
    "CONSTRUCTION_Z1",
    "CONSTRUCTION_Z2",
    "FLOOR_SURFACE",
    "FLOOR_Y",
    "L1_ENV_ID",
    "L1_EQUIP_ITEMS",
    "L1_INVENTORY",
    "L1_INV_ITEMS",
    "L1_LAYOUT",
    "LAVA_SOURCE_COUNT",
    "LAVA_X1",
    "LAVA_X2",
    "LAVA_Y",
    "LAVA_Z1",
    "LAVA_Z2",
    "PLAYER_Y",
    "L1ControlledEnv",
    "l1_scene_xml",
    "lava_pool_coords",
    "register_l1_spec",
]
