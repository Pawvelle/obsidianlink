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


def test_camera_action_sets_pitch_yaw_vector() -> None:
    """MineRL camera is ``[delta_pitch, delta_yaw]``, not ``[yaw, pitch]``."""
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.CAMERA, yaw=12.5, pitch=-3.0), _TREECHOP_KEYS
    )
    assert translated["camera"] == [-3.0, 12.5]


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


def test_use_action_does_not_drive_place_in_navigate() -> None:
    """L1 / Phase 3 fix: a USE action must NOT place a block.

    Earlier the adapter set ``place = action.target`` for both
    PLACE and USE, which leaked a ``"dirt"`` default into USE
    when the agent omitted ``target``. L1 needs USE to drive
    only the ``use`` key (e.g. right-click with
    ``flint_and_steel`` to ignite the portal) without also
    placing an unwanted block. The D1-02 water-bucket use case
    worked around the old behaviour by ignoring the
    ``place`` value; the L1 path does not have that
    workaround.
    """
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.USE, target="dirt"), _NAVIGATE_KEYS
    )
    assert translated["place"] == "none"


def test_equip_action_with_known_target_emits_equip() -> None:
    """L1 hotbar switch via the equip action."""
    keys = _NAVIGATE_KEYS + ("equip",)
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.EQUIP, target="flint_and_steel"), keys
    )
    assert translated["equip"] == "flint_and_steel"


def test_equip_action_with_unknown_target_is_noop() -> None:
    keys = _NAVIGATE_KEYS + ("equip",)
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.EQUIP, target="diamond_sword"), keys
    )
    # Unknown target is not a known L1 equippable; no equip key.
    assert "equip" not in translated or translated["equip"] == "none"


def test_use_key_drives_use_not_jump() -> None:
    keys = _TREECHOP_KEYS + ("use",)
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.USE), keys
    )
    assert translated["use"] == 1
    assert translated["jump"] == 0


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
