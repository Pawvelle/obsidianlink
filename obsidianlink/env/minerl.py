"""MineRL environment adapter (Phase 1).

Scope:

* ``reset()`` launches the configured MineRL environment and returns the
  first agent-visible ``Observation`` (RGB frame + inventory snapshot).
* ``step(action)`` translates an :class:`Action` into the MineRL
  Dict action space and forwards it.
* ``close()`` shuts the underlying MineRL / Malmo instance down cleanly.

The adapter introspects the MineRL ``action_space.spaces`` on the first
``step()`` call so it only emits keys the live env actually understands
(``MineRLTreechop-v0`` has no ``place``; ``MineRLNavigate-v0`` does).
This keeps the same code path correct across the two missions without
branching on env id.

Out of scope (deferred):

* Vision / inventory wiring beyond what MineRL hands us.
* Benchmark tasks, evaluators, planners, multi-agent.

Importing this module must NOT start MineRL. The actual ``gym.make`` call
happens inside ``reset()`` so the rest of the package stays cheap to
import and unit-testable without Java.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

_DEFAULT_ENV_ID = "MineRLTreechop-v0"


class MineRLEnvironment(Environment):
    """Adapter around a single MineRL ``gym`` environment."""

    def __init__(self, env_id: str = _DEFAULT_ENV_ID) -> None:
        self._env_id = env_id
        self._env: Any = None
        self._action_keys: tuple[str, ...] | None = None
        self._last_pov: Any = None
        self._last_inventory: dict[str, int] = {}
        self._last_compass: Any = None

    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def action_space_keys(self) -> tuple[str, ...] | None:
        """Action space keys cached after the first ``step()`` call.

        ``None`` until the env has been reset at least once.
        """
        return self._action_keys

    def reset(self) -> Observation:
        # Local imports: keep MineRL / Java / gym out of the module-level
        # import graph so that ``import obsidianlink.env.minerl`` itself
        # never triggers a JVM. ``import minerl`` is required because
        # MineRL 1.0.2 registers its env ids as an import side-effect;
        # ``import gym`` alone will not see ``MineRLNavigate-v0``.
        import gym  # type: ignore[import-untyped]
        import minerl  # type: ignore[import-untyped]  # noqa: F401

        if self._env is not None:
            # Idempotent reset semantics: close any previous instance
            # first so we never leak a JVM.
            self.close()
        # Invalidate the action-space cache; the new env may differ.
        self._action_keys = None
        self._env = gym.make(self._env_id)
        raw = self._env.reset()
        return self._convert(raw)

    def step(self, action: Action) -> Observation:
        if self._env is None:
            raise RuntimeError(
                "MineRLEnvironment.step called before reset(); "
                "call reset() first."
            )
        if self._action_keys is None:
            self._action_keys = tuple(self._env.action_space.spaces.keys())
        minerl_action = self._to_minerl_action(action, self._action_keys)
        raw, _reward, _done, _info = self._env.step(minerl_action)
        return self._convert(raw)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    # ------------------------------------------------------------------ helpers

    def _convert(self, raw: Mapping[str, Any]) -> Observation:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"MineRL observation must be a mapping, got {type(raw).__name__}"
            )
        self._last_pov = raw.get("pov")
        self._last_inventory = _summarize_inventory(raw.get("inventory"))
        self._last_compass = raw.get("compass")
        return Observation(
            frame=self._last_pov,
            inventory=self._last_inventory,
            selected_item=_selected_hotbar_item(self._last_inventory),
        )

    @staticmethod
    def _to_minerl_action(
        action: Action, keys: tuple[str, ...]
    ) -> dict[str, Any]:
        """Translate an :class:`Action` into a MineRL Dict action.

        Only keys present in the env's ``action_space`` are emitted, so
        the same code path works for ``MineRLTreechop-v0`` (no
        ``place``) and ``MineRLNavigate-v0`` (with ``place``).
        """
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
            out["camera"] = [float(action.yaw), float(action.pitch)]
        if "jump" in keyset:
            out["jump"] = 1 if action.type is ActionType.USE else 0
        if "sneak" in keyset:
            out["sneak"] = 0
        if "sprint" in keyset:
            out["sprint"] = 0
        if "attack" in keyset:
            out["attack"] = 1 if action.type is ActionType.ATTACK else 0
        if "place" in keyset:
            # PLACE / USE: place the named block if given, else "none".
            if action.type in (ActionType.PLACE, ActionType.USE):
                out["place"] = action.target or "dirt"
            else:
                out["place"] = "none"
        return out


def _summarize_inventory(inventory: Any) -> dict[str, int]:
    """Reduce a MineRL inventory mapping to ``{item_name: count}``."""
    if not inventory:
        return {}
    out: dict[str, int] = {}
    try:
        items = inventory.items()
    except AttributeError:
        return {}
    for name, info in items:
        if isinstance(info, dict):
            qty = info.get("quantity", 0)
        else:
            qty = info
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            continue
        if qty_int > 0:
            out[str(name)] = qty_int
    return out


def _selected_hotbar_item(inventory: Mapping[str, int]) -> str | None:
    """Pick a single ``selected_item`` for the observation.

    MineRL does not expose a hotbar cursor in a portable way across all
    missions. For Phase 1 we surface the first non-empty inventory
    entry as a stable, agent-visible hint.
    """
    if not inventory:
        return None
    return next(iter(inventory.keys()), None)


__all__ = ["MineRLEnvironment"]
