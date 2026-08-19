from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minerl import (
    MineRLEnvironment,
    _selected_hotbar_item,
    _summarize_inventory,
)

_TREECHOP_KEYS = (
    "attack",
    "back",
    "camera",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
)


def test_instantiation_does_not_start_jvm() -> None:
    env = MineRLEnvironment()
    assert env._env is None  # noqa: SLF001
    assert env.action_space_keys is None


def test_wait_is_noop() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.WAIT), _TREECHOP_KEYS
    )
    assert translated["forward"] == 0
    assert translated["attack"] == 0
    assert translated["camera"] == [0.0, 0.0]


def test_move_forward() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=1), _TREECHOP_KEYS
    )
    assert translated["forward"] == 1
    assert translated["back"] == 0


def test_camera_is_pitch_then_yaw() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.CAMERA, yaw=15.0, pitch=-5.0), _TREECHOP_KEYS
    )
    assert translated["camera"] == [-5.0, 15.0]


def test_use_does_not_set_place_on_treechop() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.USE), _TREECHOP_KEYS
    )
    assert "place" not in translated


def test_place_emits_none_default_when_key_exists() -> None:
    keys = _TREECHOP_KEYS + ("place", "use")
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.PLACE), keys
    )
    assert translated["place"] == "none"
    use = MineRLEnvironment._to_minerl_action(Action(type=ActionType.USE), keys)
    assert use["place"] == "none"
    assert use["use"] == 1


def test_inventory_summary() -> None:
    assert _summarize_inventory({"dirt": {"quantity": 3}}) == {"dirt": 3}
    assert _summarize_inventory({"dirt": 2, "air": 0}) == {"dirt": 2}
    assert _selected_hotbar_item({"dirt": 2}) == "dirt"
    assert _selected_hotbar_item({}) is None
