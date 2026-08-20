"""MineRL adapter. ``gym.make`` happens in ``reset()``, never at import."""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

_DEFAULT_ENV_ID = "MineRLTreechop-v0"


class MineRLEnvironment(Environment):
    """Gym/MineRL backend with a strict agent/evaluator split.

    Agent-visible: ``Observation`` (frame, inventory, selected_item).
    Evaluator-only: ``hidden_state`` / ``last_info`` (pose monitors).
    """

    def __init__(self, env_id: str = _DEFAULT_ENV_ID) -> None:
        self._env_id = env_id
        self._env: Any = None
        self._action_keys: tuple[str, ...] | None = None
        self._last_observation: Observation | None = None
        self._last_info: dict[str, Any] = {}
        self._last_hidden: dict[str, Any] = {}

    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def action_space_keys(self) -> tuple[str, ...] | None:
        return self._action_keys

    @property
    def last_info(self) -> dict[str, Any]:
        """Copy of gym ``info``. Never copied onto ``Observation``."""
        return dict(self._last_info)

    @property
    def hidden_state(self) -> dict[str, Any]:
        """Evaluator-only pose snapshot. Never copied onto ``Observation``."""
        return dict(self._last_hidden)

    def reset(self) -> Observation:
        import gym  # type: ignore[import-untyped]
        import minerl  # type: ignore[import-untyped]  # noqa: F401

        if self._env is not None:
            self.close()
        self._action_keys = None
        self._last_info = {}
        self._last_hidden = {}
        self._env = gym.make(self._env_id)
        raw = self._env.reset()
        self._last_observation = self._convert(raw, info={})
        return self._last_observation

    def observe(self) -> Observation:
        if self._last_observation is None:
            raise RuntimeError("observe() called before reset()")
        return self._last_observation

    def step(self, action: Action) -> Observation:
        if self._env is None:
            raise RuntimeError("step() called before reset()")
        if self._action_keys is None:
            self._action_keys = tuple(self._env.action_space.spaces.keys())
        minerl_action = self._to_minerl_action(action, self._action_keys)
        raw, reward, done, info = self._env.step(minerl_action)
        self._last_info = info if isinstance(info, dict) else {}
        self._last_observation = self._convert(raw, info=self._last_info)
        # Evaluator-only: gym reward/done are never copied onto Observation.
        self._last_hidden["reward"] = _scalar(reward)
        self._last_hidden["done"] = bool(done)
        return self._last_observation

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._last_observation = None

    def _convert(self, raw: Mapping[str, Any], *, info: Mapping[str, Any]) -> Observation:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"MineRL observation must be a mapping, got {type(raw).__name__}"
            )
        inventory = _summarize_inventory(raw.get("inventory"))
        self._last_hidden = _hidden_from_raw_and_info(raw, info)
        selected = _equipped_item_name(raw) or _selected_hotbar_item(inventory)
        return Observation(
            frame=raw.get("pov"),
            inventory=inventory,
            selected_item=selected,
        )

    @staticmethod
    def _to_minerl_action(action: Action, keys: tuple[str, ...]) -> dict[str, Any]:
        keyset = set(keys)
        out: dict[str, Any] = {}
        if "forward" in keyset:
            out["forward"] = 1 if action.dx > 0 else 0
        if "back" in keyset:
            out["back"] = 1 if action.dx < 0 else 0
        if "left" in keyset:
            out["left"] = 1 if action.dz < 0 else 0
        if "right" in keyset:
            out["right"] = 1 if action.dz > 0 else 0
        if "camera" in keyset:
            # MineRL CameraAction is ``[delta_pitch, delta_yaw]``.
            out["camera"] = [float(action.pitch), float(action.yaw)]
        if "use" in keyset:
            out["use"] = 1 if action.type is ActionType.USE else 0
        if "jump" in keyset:
            if "use" in keyset:
                out["jump"] = 0
            else:
                out["jump"] = 1 if action.type is ActionType.USE else 0
        if "sneak" in keyset:
            out["sneak"] = 1 if action.sneak else 0
        if "sprint" in keyset:
            out["sprint"] = 0
        if "attack" in keyset:
            out["attack"] = 1 if action.type is ActionType.ATTACK else 0
        if "place" in keyset:
            if action.type is ActionType.PLACE:
                out["place"] = action.target or "none"
            else:
                out["place"] = "none"
        if action.type is ActionType.EQUIP and "equip" in keyset and action.target:
            # L1 specs omit EquipAction. Sending ``equip none`` crashes
            # MineRL 1.0.2 MCP-Reborn. Prefer ActionType.HOTBAR.
            out["equip"] = action.target
        if action.type is ActionType.HOTBAR:
            slot = _hotbar_slot(action.target)
            if slot is not None:
                key = f"hotbar.{slot}"
                if key in keyset:
                    out[key] = 1
        return out


def _hotbar_slot(target: str) -> int | None:
    """Parse ``"3"`` or ``"hotbar.3"`` into 1–9. Invalid → None."""
    raw = str(target).strip().lower()
    if raw.startswith("hotbar."):
        raw = raw.split(".", 1)[1]
    try:
        slot = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= slot <= 9:
        return slot
    return None


def _equipped_item_name(raw: Mapping[str, Any]) -> str | None:
    """Main-hand type from EquippedItemObservation, if present."""
    eq = raw.get("equipped_items")
    if not isinstance(eq, Mapping):
        return None
    hand = eq.get("mainhand", eq)
    if not isinstance(hand, Mapping):
        return None
    item = hand.get("type")
    if item is None:
        return None
    if hasattr(item, "item"):
        try:
            item = item.item()
        except (TypeError, ValueError, AttributeError):
            pass
    name = str(item).strip()
    if name in {"", "none", "air", "None"}:
        return None
    return name


def _summarize_inventory(inventory: Any) -> dict[str, int]:
    if not inventory:
        return {}
    try:
        items = inventory.items()
    except AttributeError:
        return {}
    out: dict[str, int] = {}
    for name, info in items:
        qty = info.get("quantity", 0) if isinstance(info, dict) else info
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            continue
        if qty_int > 0:
            out[str(name)] = qty_int
    return out


def _selected_hotbar_item(inventory: Mapping[str, int]) -> str | None:
    if not inventory:
        return None
    return next(iter(inventory.keys()), None)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        size = getattr(value, "size", None)
        if size == 1:
            return float(value.reshape(-1)[0])
        return float(value)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None


def _pose_from_mapping(loc: Mapping[str, Any]) -> dict[str, float]:
    """Evaluator-only pose + ``ObservationFromCurrentLocation`` fields.

    ``biome_id`` / ``can_see_sky`` / ``light_level`` are dimension-transition
    truth candidates for the L1 Evaluator. Never copied onto Observation.
    """
    pose = {
        "yaw": _scalar(loc.get("yaw")),
        "pitch": _scalar(loc.get("pitch")),
        "xpos": _scalar(loc.get("xpos")),
        "ypos": _scalar(loc.get("ypos")),
        "zpos": _scalar(loc.get("zpos")),
        "biome_id": _scalar(loc.get("biome_id")),
        "biome_temperature": _scalar(loc.get("biome_temperature")),
        "can_see_sky": _scalar(loc.get("can_see_sky")),
        "light_level": _scalar(loc.get("light_level")),
        "sky_light_level": _scalar(loc.get("sky_light_level")),
        "sea_level": _scalar(loc.get("sea_level")),
    }
    return {k: v for k, v in pose.items() if v is not None}


def _hidden_from_raw_and_info(
    raw: Mapping[str, Any], info: Mapping[str, Any]
) -> dict[str, Any]:
    """Pull evaluator-only pose. Never returned as ``Observation`` fields."""
    hidden: dict[str, Any] = {}
    for source in (info, raw):
        if not isinstance(source, Mapping):
            continue
        loc = source.get("location_stats")
        if isinstance(loc, Mapping):
            hidden.update(_pose_from_mapping(loc))
        else:
            hidden.update(_pose_from_mapping(source))
        if hidden:
            break
    return hidden


__all__ = ["MineRLEnvironment"]
