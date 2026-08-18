"""Tests for the Phase 1 bounded action translation in MineRLEnvironment.

The MineRL Dict action space differs across missions
(``MineRLTreechop-v0`` has no ``place``; ``MineRLNavigate-v0`` has it).
The translation must introspect the live action space and only emit
keys the env actually understands.
"""

from __future__ import annotations

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minerl import MineRLEnvironment


_TREECHOP_KEYS = (
    "attack", "back", "camera", "forward", "jump", "left",
    "right", "sneak", "sprint",
)
_NAVIGATE_KEYS = (
    "attack", "back", "camera", "forward", "jump", "left",
    "place", "right", "sneak", "sprint",
)


def test_translation_emits_only_treechop_keys() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=1, dz=0), _TREECHOP_KEYS
    )
    assert set(translated.keys()) == set(_TREECHOP_KEYS)
    assert translated["forward"] == 1
    assert translated["back"] == 0
    assert translated["attack"] == 0


def test_translation_emits_navigate_keys_including_place() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.WAIT), _NAVIGATE_KEYS
    )
    assert "place" in translated
    assert translated["place"] == "none"


def test_wait_action_translates_to_all_zeros() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.WAIT), _TREECHOP_KEYS
    )
    assert set(translated.keys()) == set(_TREECHOP_KEYS)
    for key in _TREECHOP_KEYS:
        if key == "camera":
            assert translated[key] == [0.0, 0.0]
        else:
            assert translated[key] == 0


def test_move_dx_positive_sets_forward_only() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=1, dz=0), _TREECHOP_KEYS
    )
    assert translated["forward"] == 1
    assert translated["back"] == 0
    assert translated["left"] == 0
    assert translated["right"] == 0


def test_move_dx_negative_sets_back_only() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=-1, dz=0), _TREECHOP_KEYS
    )
    assert translated["back"] == 1
    assert translated["forward"] == 0


def test_move_dz_positive_sets_right_only() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=0, dz=1), _TREECHOP_KEYS
    )
    assert translated["right"] == 1
    assert translated["left"] == 0


def test_move_dz_negative_sets_left_only() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=0, dz=-1), _TREECHOP_KEYS
    )
    assert translated["left"] == 1
    assert translated["right"] == 0


def test_attack_action_sets_attack_one() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.ATTACK), _TREECHOP_KEYS
    )
    assert translated["attack"] == 1
    assert translated["forward"] == 0
    assert translated["back"] == 0


def test_camera_action_sets_yaw_pitch_vector() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.CAMERA, yaw=12.5, pitch=-3.0), _TREECHOP_KEYS
    )
    assert translated["camera"] == [12.5, -3.0]


def test_place_action_with_target_emits_block_name() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.PLACE, target="cobblestone"), _NAVIGATE_KEYS
    )
    assert translated["place"] == "cobblestone"


def test_place_action_without_target_falls_back_to_dirt() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.PLACE), _NAVIGATE_KEYS
    )
    assert translated["place"] == "dirt"


def test_use_action_also_drives_place_in_navigate() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.USE, target="dirt"), _NAVIGATE_KEYS
    )
    assert translated["place"] == "dirt"


def test_translation_ignores_keys_not_in_env_action_space() -> None:
    # A Navigate-only key like ``place`` must NOT appear when the env
    # is Treechop (which has no ``place`` slot).
    partial_keys = ("forward", "back", "camera", "attack")
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=1, dz=0), partial_keys
    )
    assert "place" not in translated
    assert "jump" not in translated
    assert translated["forward"] == 1
    assert translated["attack"] == 0
