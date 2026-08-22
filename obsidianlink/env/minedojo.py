"""MineDojo adapter for the active ObsidianLink development platform.

MineDojo is imported only from :meth:`reset`, so offline tests and legacy
MineRL evidence remain runnable without starting a Minecraft backend.  The
adapter uses MineDojo's native event-level action API, which preserves the
GeneralAgent's primitive movement, camera, attack, use, equip, and placement
decisions without adding Voyager or Mineflayer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

_DEFAULT_TASK_ID = "harvest_1_log"
_ARCHIVED_ACTIONS = frozenset({ActionType.HOTBAR, ActionType.INVENTORY})
_TABLE_CRAFT_SUFFIXES = (
    "_sword",
    "_pickaxe",
    "_axe",
    "_shovel",
    "_hoe",
    "_helmet",
    "_chestplate",
    "_leggings",
    "_boots",
)
_TABLE_CRAFT_ITEMS = frozenset(
    {
        "furnace",
        "chest",
        "boat",
        "bucket",
        "bowl",
        "bed",
        "door",
        "fence",
        "gate",
        "sign",
    }
)


class MineDojoEnvironment(Environment):
    """Map one MineDojo task onto ObsidianLink's platform-neutral contract.

    The agent receives POV RGB, inventory, selected item, and world pose.
    MineDojo life, voxel, reward, and task-completion signals stay outside
    ``Observation``.
    """

    def __init__(
        self,
        task_id: str = _DEFAULT_TASK_ID,
        *,
        image_size: tuple[int, int] = (360, 640),
        **task_kwargs: Any,
    ) -> None:
        task_id = task_id.strip()
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must be a positive (height, width) pair")
        self._task_id = task_id
        self._image_size = tuple(int(value) for value in image_size)
        self._task_kwargs = dict(task_kwargs)
        self._env: Any = None
        self._last_observation: Observation | None = None
        self._last_info: dict[str, Any] = {}
        self._last_hidden: dict[str, Any] = {}

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def last_info(self) -> dict[str, Any]:
        """Latest MineDojo info for evaluators/diagnostics, never the planner."""
        return dict(self._last_info)

    @property
    def hidden_state(self) -> dict[str, Any]:
        """Reward and termination state, kept out of agent-visible Observation."""
        return dict(self._last_hidden)

    def reset(self) -> Observation:
        import minedojo  # type: ignore[import-untyped]
        from minedojo.tasks import _meta_task_make, _specific_task_make
        from obsidianlink.env.minedojo_runtime import prepare_minedojo_runtime

        if self._env is not None:
            self.close()
        package_file = getattr(minedojo, "__file__", None)
        if package_file:
            prepare_minedojo_runtime(Path(package_file).resolve().parent)
        self._last_info = {}
        self._last_hidden = {}
        kwargs = _prepare_task_kwargs(self._task_kwargs)
        # MineDojo 0.1's public ``make`` unconditionally applies an ARNN
        # wrapper that is broken on its own documented event-level control
        # profile.  Named tasks and open-ended worlds both bypass that wrapper.
        if self._task_id in {"open-ended", "open_ended"}:
            self._env = _meta_task_make(
                "open-ended",
                image_size=self._image_size,
                event_level_control=True,
                **kwargs,
            )
        else:
            self._env = _specific_task_make(
                self._task_id,
                image_size=self._image_size,
                event_level_control=True,
                **kwargs,
            )
        raw = self._env.reset()
        self._last_hidden = _hidden_from_raw(raw, reward=None, done=False)
        self._last_observation = self._convert(raw)
        return self._last_observation

    def observe(self) -> Observation:
        if self._last_observation is None:
            raise RuntimeError("observe() called before reset()")
        return self._last_observation

    def step(self, action: Action) -> Observation:
        if self._env is None:
            raise RuntimeError("step() called before reset()")
        if action.type in _ARCHIVED_ACTIONS:
            raise ValueError(
                f"{action.type.value} is archived MineRL-only; "
                "use equip/craft/place item names on MineDojo"
            )
        raw_action = self._to_minedojo_action(action, self._env.action_space.no_op())
        raw, reward, done, info = self._env.step(raw_action)
        self._last_info = dict(info) if isinstance(info, Mapping) else {}
        self._last_hidden = _hidden_from_raw(raw, reward=reward, done=done)
        self._last_observation = self._convert(raw)
        return self._last_observation

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._last_observation = None

    @staticmethod
    def _to_minedojo_action(
        action: Action,
        out: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate an ObsidianLink primitive into MineDojo's native action."""
        if not isinstance(action, Action):
            raise TypeError("MineDojoEnvironment.step requires an Action")
        translated = dict(out)

        def set_if_present(key: str, value: Any) -> None:
            if key in translated:
                translated[key] = value

        set_if_present("forward", int(action.dx > 0))
        set_if_present("back", int(action.dx < 0))
        set_if_present("left", int(action.dz < 0))
        set_if_present("right", int(action.dz > 0))
        set_if_present("jump", int(action.jump))
        set_if_present("sneak", int(action.sneak))
        if action.type is ActionType.CAMERA:
            translated["camera"] = [float(action.pitch), float(action.yaw)]
        elif action.type is ActionType.ATTACK:
            set_if_present("attack", 1)
        elif action.type is ActionType.USE:
            set_if_present("use", 1)
        elif action.type is ActionType.DROP:
            set_if_present("drop", 1)
        elif action.type is ActionType.EQUIP:
            translated["equip"] = _item_name(action.target, verb="equip")
        elif action.type is ActionType.PLACE:
            translated["place"] = _item_name(action.target, verb="place")
        elif action.type is ActionType.CRAFT:
            item, use_table = _craft_target(action.target)
            translated["craft_with_table" if use_table else "craft"] = item
        elif action.type is ActionType.SMELT:
            translated["smelt"] = _item_name(action.target, verb="smelt")
        elif action.type not in {ActionType.MOVE, ActionType.WAIT}:
            raise ValueError(f"unsupported MineDojo action: {action.type.value}")
        return translated

    @staticmethod
    def _convert(raw: Mapping[str, Any]) -> Observation:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"MineDojo observation must be a mapping, got {type(raw).__name__}"
            )
        inventory = _summarize_inventory(raw.get("inventory"))
        pose = _pose_from_raw(raw)
        return Observation(
            frame=_rgb_frame(raw.get("rgb")),
            inventory=inventory,
            selected_item=_selected_equipment(raw.get("equipment"))
            or next(iter(inventory), None),
            **pose,
        )


def _prepare_task_kwargs(task_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(task_kwargs)
    inventory = kwargs.get("initial_inventory")
    if inventory:
        from minedojo.sim.inventory import InventoryItem  # type: ignore[import-untyped]

        kwargs["initial_inventory"] = [
            item
            if hasattr(item, "name")
            else InventoryItem(
                slot=item.get("slot", 0),
                name=str(item["name"]),
                quantity=int(item.get("quantity", 1) or 1),
            )
            for item in inventory
        ]
    return kwargs


def _pose_from_raw(raw: Any) -> dict[str, float]:
    loc = raw.get("location_stats") if isinstance(raw, Mapping) else None
    if not isinstance(loc, Mapping):
        return {}
    mapping = (
        ("xpos", "x"),
        ("ypos", "y"),
        ("zpos", "z"),
        ("yaw", "yaw"),
        ("pitch", "pitch"),
    )
    pose: dict[str, float] = {}
    for src, dest in mapping:
        if src not in loc:
            continue
        value = _scalar(loc.get(src))
        if value is not None:
            pose[dest] = value
    return pose


def _hidden_from_raw(raw: Any, *, reward: Any, done: Any) -> dict[str, Any]:
    hidden: dict[str, Any] = {"reward": _scalar(reward), "done": bool(done)}
    loc = raw.get("location_stats") if isinstance(raw, Mapping) else None
    if isinstance(loc, Mapping):
        for key in (
            "xpos",
            "ypos",
            "zpos",
            "yaw",
            "pitch",
            "biome_id",
            "biome_temperature",
            "can_see_sky",
            "light_level",
        ):
            if key in loc:
                value = _scalar(loc.get(key))
                if value is not None:
                    hidden[key] = value
    return hidden


def _item_name(raw: str, *, verb: str) -> str:
    item = str(raw).strip().lower().replace(" ", "_").split(":", 1)[-1]
    if not item or item in {"none", "air"}:
        raise ValueError(f"MineDojo {verb} requires a concrete inventory item")
    return item


def _craft_target(raw: str) -> tuple[str, bool]:
    text = str(raw).strip().lower().replace(" ", "_")
    use_table = False
    if text.startswith("table:") or text.startswith("nearby:"):
        use_table = True
        text = text.split(":", 1)[1]
    item = _item_name(text, verb="craft")
    if not use_table:
        use_table = item in _TABLE_CRAFT_ITEMS or item.endswith(_TABLE_CRAFT_SUFFIXES)
    return item, use_table


def _summarize_inventory(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    names, quantities = raw.get("name"), raw.get("quantity")
    if names is None or quantities is None:
        return {}
    out: dict[str, int] = {}
    for name, quantity in zip(names, quantities):
        item = str(name).strip().replace(" ", "_")
        try:
            count = int(quantity)
        except (TypeError, ValueError):
            continue
        if item and item not in {"air", "none"} and count > 0:
            out[item] = out.get(item, 0) + count
    return out


def _selected_equipment(raw: Any) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    names, quantities = raw.get("name"), raw.get("quantity")
    if names is None or quantities is None:
        return None
    for name, quantity in zip(names, quantities):
        try:
            if int(quantity) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        item = str(name).strip().replace(" ", "_")
        if item and item not in {"air", "none"}:
            return item
    return None


def _rgb_frame(raw: Any) -> Any:
    """MineDojo emits BGR CHW; expose the project-standard RGB HWC frame."""
    try:
        import numpy as np

        frame = np.asarray(raw)
        if frame.ndim == 3 and frame.shape[0] == 3:
            return frame.transpose(1, 2, 0)[:, :, ::-1]
    except (TypeError, ValueError, AttributeError):
        pass
    return raw


def _scalar(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["MineDojoEnvironment"]
