"""MineRL environment adapter (Phase 1 / Step 1).

Scope of this module (intentionally small):

* ``reset()`` launches the configured MineRL environment and returns the
  first agent-visible ``Observation`` (RGB frame + inventory snapshot).
* ``step(action)`` forwards one bounded action and returns the next
  ``Observation``.
* ``close()`` shuts the underlying MineRL / Malmo instance down cleanly.

Out of scope (deferred to later Phase 1 sub-steps):

* The full bounded action set mapping (MOVE / CAMERA / ATTACK / USE / PLACE).
  For now every non-WAIT action is forwarded as a MineRL no-op so the
  env-observation loop can be exercised end-to-end without lying about
  MineRL. ``ActionType.WAIT`` is the only fully-specified type here.
* Benchmark tasks, agents, evaluators, planners, multi-agent.

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
    """Adapter around a single MineRL ``gym`` environment.

    Parameters
    ----------
    env_id:
        The MineRL environment id. Defaults to ``MineRLTreechop-v0``,
        which exposes a minimal ``pov`` observation and is the lightest
        mission spec in MineRL 1.0.2 — chosen to avoid the
        ``NavigationDecorator`` / ``RewardForTouchingBlockType``
        NPE that ``MineRLNavigate-v0`` currently triggers in the
        bundled Malmo 0.37.0 server. Once that server-side bug is
        resolved, ``MineRLNavigate-v0`` should become the default
        again because it exposes the richest agent-visible state.
    """

    def __init__(self, env_id: str = _DEFAULT_ENV_ID) -> None:
        self._env_id = env_id
        self._env: Any = None
        self._last_pov: Any = None
        self._last_inventory: dict[str, int] = {}
        self._last_compass: Any = None

    @property
    def env_id(self) -> str:
        return self._env_id

    def reset(self) -> Observation:
        # Local imports: keep MineRL / Java / gym out of the module-level
        # import graph so that ``import obsidianlink.env.minerl`` itself
        # never triggers a JVM. ``import minerl`` is required because
        # MineRL 1.0.2 registers its env ids as an import side-effect;
        # ``import gym`` alone will not see ``MineRLNavigate-v0``.
        import gym  # type: ignore[import-untyped]
        import minerl  # type: ignore[import-untyped]  # noqa: F401

        if self._env is not None:
            # Idempotent reset semantics: close any previous instance first
            # so we never leak a JVM.
            self.close()
        self._env = gym.make(self._env_id)
        raw = self._env.reset()
        return self._convert(raw)

    def step(self, action: Action) -> Observation:
        if self._env is None:
            raise RuntimeError(
                "MineRLEnvironment.step called before reset(); "
                "call reset() first."
            )
        minerl_action = self._to_minerl_action(action)
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

    def _to_minerl_action(self, action: Action) -> dict[str, Any]:
        # MineRL's Dict action space: see MineRLNavigate-v0 spec.
        # Step 1 keeps the mapping deliberately minimal: WAIT is the
        # only fully-specified type. Other types become no-ops, not
        # because their semantics are decided, but because the bounded
        # action set is deferred to a later Phase 1 sub-step.
        noop: dict[str, Any] = {
            "attack": 0,
            "back": 0,
            "camera": [0.0, 0.0],
            "forward": 0,
            "jump": 0,
            "left": 0,
            "place": "none",
            "right": 0,
            "sneak": 0,
            "sprint": 0,
        }
        if action.type is not ActionType.WAIT:
            # Defer the bounded action set to Phase 1 / Step 3.
            # We do not raise here so the env loop is still exercisable.
            return noop
        return noop


def _summarize_inventory(inventory: Any) -> dict[str, int]:
    """Reduce a MineRL inventory mapping to ``{item_name: count}``.

    MineRL inventories are ``{name: {'type': ..., 'quantity': N}}`` after
    the 1.0.2 refactor, but earlier shapes appear in some missions. We
    accept both ``{name: count}`` and ``{name: {'quantity': N}}`` forms
    and return a flat ``{name: int}`` view.
    """
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
    missions. For Phase 1 we surface the first non-empty inventory entry
    as a stable, agent-visible hint and let the real hotbar wiring land
    with the bounded action set.
    """
    if not inventory:
        return None
    return next(iter(inventory.keys()), None)


__all__ = ["MineRLEnvironment"]
