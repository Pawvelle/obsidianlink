"""Deterministic MineRL-free world for Agent reasoning tests.

Primitive skills still emit the same ``Action`` objects. This environment only
updates agent-visible inventory and selected_item. Distance, facing, GUI, and
remaining resources stay internal and never appear on ``Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

_PLACEABLE = frozenset(
    {
        "oak_log",
        "oak_planks",
        "cobblestone",
        "dirt",
        "stone",
        "crafting_table",
        "furnace",
    }
)
_INVENTORY_RECIPES: tuple[tuple[str, dict[str, int], dict[str, int]], ...] = (
    ("oak_planks", {"oak_log": 1}, {"oak_planks": 4}),
    ("stick", {"oak_planks": 2}, {"stick": 4}),
    ("crafting_table", {"oak_planks": 4}, {"crafting_table": 1}),
)
_TABLE_RECIPES: tuple[tuple[str, dict[str, int], dict[str, int]], ...] = _INVENTORY_RECIPES + (
    ("wooden_pickaxe", {"oak_planks": 3, "stick": 2}, {"wooden_pickaxe": 1}),
    ("furnace", {"cobblestone": 8}, {"furnace": 1}),
)
_DROPS = {
    "stone": "cobblestone",
    "oak_log": "oak_log",
    "dirt": "dirt",
    "cobblestone": "cobblestone",
}


@dataclass(frozen=True)
class FakeWorldConfig:
    inventory: dict[str, int]
    hotbar: tuple[str | None, ...]
    target: str
    distance: int
    mine_ticks: int
    remaining: int
    selected_slot: int


class FakeMinecraftEnv(Environment):
    """Simulated inventory, resource, and crafting state. No Minecraft process."""

    def __init__(
        self,
        *,
        inventory: dict[str, int] | None = None,
        hotbar: list[str | None] | tuple[str | None, ...] | None = None,
        target: str = "stone",
        distance: int = 0,
        mine_ticks: int = 2,
        remaining: int = 8,
        selected_slot: int = 1,
    ) -> None:
        start_inventory = {str(k): int(v) for k, v in dict(inventory or {}).items() if int(v) > 0}
        slots = _hotbar_slots(start_inventory, hotbar)
        self._initial = FakeWorldConfig(
            inventory=dict(start_inventory),
            hotbar=tuple(slots),
            target=str(target).strip() or "stone",
            distance=max(0, int(distance)),
            mine_ticks=max(1, int(mine_ticks)),
            remaining=max(0, int(remaining)),
            selected_slot=min(9, max(1, int(selected_slot))),
        )
        self.inventory: dict[str, int] = {}
        self.hotbar: list[str | None] = [None] * 9
        self.selected_slot = 1
        self.target = self._initial.target
        self.distance = 0
        self.mine_ticks = 1
        self.remaining = 0
        self.facing_target = True
        self.mine_progress = 0
        self.gui_open = False
        self.table_placed = False
        self._last = Observation(inventory={})
        self.reset()

    def reset(self) -> Observation:
        cfg = self._initial
        self.inventory = dict(cfg.inventory)
        self.hotbar = list(cfg.hotbar)
        self.selected_slot = cfg.selected_slot
        self.target = cfg.target
        self.distance = cfg.distance
        self.mine_ticks = cfg.mine_ticks
        self.remaining = cfg.remaining
        self.facing_target = True
        self.mine_progress = 0
        self.gui_open = False
        self.table_placed = False
        self._last = self._observation()
        return self._last

    def observe(self) -> Observation:
        return self._last

    def step(self, action: Action) -> Observation:
        if not isinstance(action, Action):
            raise TypeError("FakeMinecraftEnv.step requires an Action")
        if action.type is ActionType.MOVE:
            self._move(action)
        elif action.type is ActionType.CAMERA:
            self._look(action)
        elif action.type is ActionType.ATTACK:
            if self.gui_open:
                self._craft()
            else:
                self._mine()
        elif action.type is ActionType.USE:
            if self.gui_open:
                self._craft()
            else:
                self._use()
        elif action.type is ActionType.PLACE:
            self._place_selected()
        elif action.type is ActionType.HOTBAR:
            self._select_hotbar(action.target)
        elif action.type is ActionType.INVENTORY:
            self.gui_open = not self.gui_open
        elif action.type is ActionType.WAIT:
            pass
        self._last = self._observation()
        return self._last

    def close(self) -> None:
        return None

    @property
    def debug_state(self) -> dict[str, object]:
        """Test-only internals. Never copied onto Observation."""
        return {
            "target": self.target,
            "distance": self.distance,
            "facing_target": self.facing_target,
            "remaining": self.remaining,
            "gui_open": self.gui_open,
            "table_placed": self.table_placed,
            "selected_slot": self.selected_slot,
            "hotbar": list(self.hotbar),
        }

    def local_view(self) -> dict[str, object]:
        """Coarse agent-visible surroundings. No world coordinates or remaining count."""
        nearby: list[dict[str, object]] = []
        resources: list[str] = []
        if self.remaining > 0:
            nearby.append(
                {
                    "name": self.target,
                    "reachable": self.distance <= 0,
                    "facing": self.facing_target,
                }
            )
            resources.append(self.target)
        if self.distance <= 0:
            relative = "adjacent"
        elif self.distance <= 2:
            relative = "near"
        else:
            relative = "far"
        return {
            "position": {
                "relative": relative,
                "facing_target": self.facing_target,
            },
            "nearby_blocks": nearby,
            "visible_resources": resources,
            "nearby_entities": [],
            "equipment": {"mainhand": self._selected_item()},
        }

    def _observation(self) -> Observation:
        return Observation(
            frame=None,
            inventory=dict(self.inventory),
            selected_item=self._selected_item(),
        )

    def _selected_item(self) -> str | None:
        name = self.hotbar[self.selected_slot - 1]
        if not name or int(self.inventory.get(name, 0) or 0) <= 0:
            return None
        return name

    def _move(self, action: Action) -> None:
        if action.dx > 0 and self.facing_target:
            self.distance = max(0, self.distance - 1)
        elif action.dx < 0:
            self.distance += 1

    def _look(self, action: Action) -> None:
        if self.gui_open:
            return
        if abs(float(action.yaw)) >= 90.0 or abs(float(action.pitch)) >= 60.0:
            self.facing_target = False
        else:
            self.facing_target = True

    def _mine(self) -> None:
        if not self.facing_target or self.distance > 0 or self.remaining <= 0:
            return
        self.mine_progress += 1
        if self.mine_progress < self.mine_ticks:
            return
        self.mine_progress = 0
        self.remaining -= 1
        self._add(_drop_for(self.target), 1)

    def _use(self) -> None:
        selected = self._selected_item()
        if selected == "crafting_table":
            if self._consume("crafting_table", 1):
                self.table_placed = True
            return
        if self.table_placed and selected is None:
            self.gui_open = True
            return
        self._place_selected()

    def _place_selected(self) -> None:
        selected = self._selected_item()
        if selected in _PLACEABLE:
            self._consume(selected, 1)

    def _select_hotbar(self, target: str) -> None:
        raw = str(target).strip().lower()
        if raw.startswith("hotbar."):
            raw = raw.split(".", 1)[1]
        try:
            slot = int(raw)
        except (TypeError, ValueError):
            return
        if 1 <= slot <= 9:
            self.selected_slot = slot

    def _craft(self) -> None:
        recipes = _TABLE_RECIPES if self.table_placed else _INVENTORY_RECIPES
        for _name, inputs, outputs in recipes:
            if all(int(self.inventory.get(item, 0) or 0) >= count for item, count in inputs.items()):
                for item, count in inputs.items():
                    self._consume(item, count)
                for item, count in outputs.items():
                    self._add(item, count)
                return

    def _add(self, name: str, count: int) -> None:
        if count <= 0:
            return
        self.inventory[name] = int(self.inventory.get(name, 0) or 0) + count
        if name not in self.hotbar:
            for index, slot in enumerate(self.hotbar):
                if slot is None:
                    self.hotbar[index] = name
                    break

    def _consume(self, name: str, count: int) -> bool:
        have = int(self.inventory.get(name, 0) or 0)
        if have < count:
            return False
        left = have - count
        if left:
            self.inventory[name] = left
        else:
            self.inventory.pop(name, None)
            self.hotbar = [slot if slot != name else None for slot in self.hotbar]
        return True


def _hotbar_slots(
    inventory: dict[str, int],
    hotbar: list[str | None] | tuple[str | None, ...] | None,
) -> list[str | None]:
    if hotbar is not None:
        slots = list(hotbar)[:9]
        while len(slots) < 9:
            slots.append(None)
        return slots
    slots: list[str | None] = [None] * 9
    for index, name in enumerate(inventory):
        if index >= 9:
            break
        slots[index] = name
    return slots


def _drop_for(target: str) -> str:
    return _DROPS.get(target, target)


__all__ = ["FakeMinecraftEnv"]
