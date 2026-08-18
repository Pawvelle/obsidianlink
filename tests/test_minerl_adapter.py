"""Offline tests for the Phase 1 MineRL adapter.

These tests must NOT start MineRL or Java. They cover:

* importability of the adapter module without triggering ``gym.make``;
* the action translation layer for the bounded action set across
  Treechop-shaped and Navigate-shaped action spaces;
* the inventory summarizer accepting both legacy ``{name: count}`` and
  current ``{name: {'quantity': N}}`` MineRL inventory shapes.

A live ``reset -> step -> close`` smoke is the responsibility of
``obsidianlink.main`` and is intentionally not exercised here.
"""

from __future__ import annotations

from typing import Any

import pytest

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minerl import (
    MineRLEnvironment,
    _selected_hotbar_item,
    _summarize_inventory,
)


# Treechop-shaped action space, used by the offline tests for the
# action translation layer. The MineRL adapter introspects the live
# env's action space on the first ``step()`` call, so passing the
# same shape here keeps the unit test honest.
_TREECHOP_KEYS = (
    "attack", "back", "camera", "forward", "jump", "left",
    "right", "sneak", "sprint",
)


def test_minerl_environment_is_subclass_of_environment() -> None:
    from obsidianlink.env.environment import Environment

    assert issubclass(MineRLEnvironment, Environment)


def test_minerl_environment_instantiation_does_not_start_jvm() -> None:
    """Constructing the adapter must be side-effect free."""
    env = MineRLEnvironment(env_id="MineRLTreechop-v0")
    assert env.env_id == "MineRLTreechop-v0"
    # Internal gym handle stays None until reset() is called.
    assert env._env is None  # noqa: SLF001 - intentional probe
    # Action-space keys are also unset until first step().
    assert env.action_space_keys is None


def test_wait_action_translates_to_minerl_noop() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.WAIT), _TREECHOP_KEYS
    )
    assert set(translated.keys()) == set(_TREECHOP_KEYS)
    assert translated["forward"] == 0
    assert translated["back"] == 0
    assert translated["camera"] == [0.0, 0.0]
    assert translated["attack"] == 0


@pytest.mark.parametrize(
    "non_wait_type",
    [ActionType.MOVE, ActionType.CAMERA, ActionType.ATTACK, ActionType.USE, ActionType.PLACE],
)
def test_non_wait_actions_do_not_crash_translation(non_wait_type: ActionType) -> None:
    """The adapter must accept any action type and emit a valid MineRL
    Dict action that the live env can step with, rather than raise."""
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=non_wait_type), _TREECHOP_KEYS
    )
    assert set(translated.keys()) == set(_TREECHOP_KEYS)
    assert translated["camera"] == [0.0, 0.0]


def test_summarize_inventory_handles_modern_shape() -> None:
    inv: Any = {
        "dirt": {"type": "item", "quantity": 4},
        "oak_log": {"type": "block", "quantity": 2},
    }
    assert _summarize_inventory(inv) == {"dirt": 4, "oak_log": 2}


def test_summarize_inventory_handles_legacy_shape() -> None:
    inv: Any = {"dirt": 4, "oak_log": 2}
    assert _summarize_inventory(inv) == {"dirt": 4, "oak_log": 2}


def test_summarize_inventory_drops_zero_quantity() -> None:
    inv: Any = {
        "dirt": {"type": "item", "quantity": 0},
        "oak_log": {"type": "block", "quantity": 1},
    }
    assert _summarize_inventory(inv) == {"oak_log": 1}


def test_summarize_inventory_handles_empty_and_none() -> None:
    assert _summarize_inventory(None) == {}
    assert _summarize_inventory({}) == {}


def test_summarize_inventory_handles_non_mapping() -> None:
    # MineRL occasionally returns a non-mapping when the player has no
    # inventory slot accessible; the summarizer must not crash.
    assert _summarize_inventory(0) == {}  # type: ignore[arg-type]


def test_selected_hotbar_item_returns_first_nonempty() -> None:
    assert _selected_hotbar_item({}) is None
    assert _selected_hotbar_item({"oak_log": 1}) == "oak_log"
    assert _selected_hotbar_item({"dirt": 3, "oak_log": 1}) == "dirt"
