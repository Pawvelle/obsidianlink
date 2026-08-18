"""Tests for the Phase 1 extended :class:`Action` payload."""

from __future__ import annotations

import dataclasses

import pytest

from obsidianlink.env.actions import Action, ActionType


def test_wait_action_constructs_with_all_defaults() -> None:
    action = Action(type=ActionType.WAIT)
    assert action.type is ActionType.WAIT
    assert action.dx == 0
    assert action.dz == 0
    assert action.yaw == 0.0
    assert action.pitch == 0.0
    assert action.target == ""
    assert action.slot == 0


def test_move_action_carries_dx_dz() -> None:
    action = Action(type=ActionType.MOVE, dx=1, dz=-1)
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == -1


def test_camera_action_carries_yaw_pitch() -> None:
    action = Action(type=ActionType.CAMERA, yaw=15.0, pitch=-5.0)
    assert action.yaw == 15.0
    assert action.pitch == -5.0


def test_place_action_carries_target_and_slot() -> None:
    action = Action(type=ActionType.PLACE, target="dirt", slot=3)
    assert action.target == "dirt"
    assert action.slot == 3


def test_action_is_frozen() -> None:
    action = Action(type=ActionType.WAIT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.dx = 1  # type: ignore[misc]


def test_all_action_types_present() -> None:
    names = {member.name for member in ActionType}
    assert names == {"MOVE", "CAMERA", "ATTACK", "USE", "PLACE", "WAIT"}


def test_action_values_are_lowercase_strings() -> None:
    for member in ActionType:
        assert member.value == member.name.lower()
