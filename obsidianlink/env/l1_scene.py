"""Formal L1 controlled construction environment on MineDojo.

Fixed Overworld grass superflat: spawn, construction floor, and a 4×4
lava source pool. No pre-built portal frame. Starting tools are given
through MineDojo ``initial_inventory``. Item select is ``equip`` by name.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minedojo import MineDojoEnvironment
from obsidianlink.env.scene import (
    FLAT_WORLD,
    PLAYER_PITCH,
    PLAYER_X,
    PLAYER_YAW,
    PLAYER_Z,
    RESOLUTION,
    WARMUP_STEPS,
)

L1_ENV_ID = "minedojo_l1_portal"
PLAYER_Y = 4.0
FLOOR_Y = 3
FLOOR_SURFACE = "grass"

LAVA_X1, LAVA_X2 = -1, 2
LAVA_Y = FLOOR_Y
LAVA_Z1, LAVA_Z2 = 5, 8
LAVA_SOURCE_COUNT = 16

CONSTRUCTION_X1, CONSTRUCTION_X2 = -4, 4
CONSTRUCTION_Y = FLOOR_Y
CONSTRUCTION_Z1, CONSTRUCTION_Z2 = 1, 4

L1_INVENTORY: dict[int, dict[str, Any]] = {
    0: {"type": "water_bucket", "quantity": 1},
    1: {"type": "bucket", "quantity": 1},
    2: {"type": "cobblestone", "quantity": 64},
    3: {"type": "iron_pickaxe", "quantity": 1},
    4: {"type": "flint_and_steel", "quantity": 1},
}

L1_SLOT_ITEMS: dict[str, str] = {
    "1": "water_bucket",
    "2": "bucket",
    "3": "cobblestone",
    "4": "iron_pickaxe",
    "5": "flint_and_steel",
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


def l1_equip_target(slot_or_item: str, inventory: dict[str, int] | None = None) -> str:
    """Map a legacy L1 slot or item name onto a MineDojo equip target."""
    raw = str(slot_or_item).strip()
    if raw.startswith("hotbar."):
        raw = raw.split(".", 1)[1]
    if raw in L1_SLOT_ITEMS:
        if raw == "2" and inventory and int(inventory.get("lava_bucket") or 0) >= 1:
            return "lava_bucket"
        return L1_SLOT_ITEMS[raw]
    return raw.replace(" ", "_")


def _l1_inventory_items() -> list[dict[str, Any]]:
    return [
        {"slot": slot, "name": item["type"], "quantity": int(item["quantity"])}
        for slot, item in L1_INVENTORY.items()
    ]


class L1ControlledEnv(Environment):
    """Fixed L1 grass superflat on MineDojo. Hidden layout is evaluator-only."""

    def __init__(self, warmup_steps: int = WARMUP_STEPS) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.env_id = L1_ENV_ID
        self.task_id = L1_ENV_ID
        self.warmup_steps = int(warmup_steps)
        self._env = MineDojoEnvironment(
            "open-ended",
            image_size=RESOLUTION,
            generate_world_type="flat",
            flat_world_seed_string=FLAT_WORLD,
            drawing_str=l1_scene_xml(),
            initial_inventory=_l1_inventory_items(),
            start_position={
                "x": PLAYER_X,
                "y": PLAYER_Y,
                "z": PLAYER_Z,
                "yaw": PLAYER_YAW,
                "pitch": PLAYER_PITCH,
            },
            allow_time_passage=False,
            allow_mob_spawn=False,
            initial_weather="clear",
            start_time=6000,
        )

    @property
    def hidden_state(self) -> dict[str, Any]:
        hidden = dict(self._env.hidden_state)
        hidden["l1_layout"] = dict(L1_LAYOUT)
        hidden["target_truths"] = {"lava": True, "prebuilt_portal": False}
        return hidden

    @property
    def last_info(self) -> dict[str, Any]:
        return self._env.last_info

    @property
    def action_space_keys(self) -> tuple[str, ...] | None:
        raw = getattr(self._env, "_env", None)
        space = getattr(raw, "action_space", None)
        if space is None or not hasattr(space, "no_op"):
            return None
        no_op = space.no_op()
        if isinstance(no_op, dict):
            return tuple(no_op)
        return None

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
        if action.type is ActionType.HOTBAR:
            action = Action(
                ActionType.EQUIP,
                target=l1_equip_target(action.target, self.observe().inventory),
            )
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
    "L1_SLOT_ITEMS",
    "LAVA_SOURCE_COUNT",
    "LAVA_X1",
    "LAVA_X2",
    "LAVA_Y",
    "LAVA_Z1",
    "LAVA_Z2",
    "PLAYER_Y",
    "L1ControlledEnv",
    "l1_equip_target",
    "l1_scene_xml",
    "lava_pool_coords",
]
