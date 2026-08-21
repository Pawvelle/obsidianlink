from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minerl import (
    MineRLEnvironment,
    _equipped_item_name,
    _hotbar_slot,
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


def test_move_can_jump_over_natural_terrain() -> None:
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.MOVE, dx=1, jump=True), _TREECHOP_KEYS
    )
    assert translated["forward"] == 1
    assert translated["jump"] == 1


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


def test_hotbar_action_sets_only_requested_slot() -> None:
    keys = _TREECHOP_KEYS + ("use", "hotbar.1", "hotbar.2", "hotbar.3")
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.HOTBAR, target="3"), keys
    )
    assert translated["hotbar.3"] == 1
    assert "hotbar.1" not in translated
    assert "equip" not in translated


def test_hotbar_action_accepts_hotbar_dot_n() -> None:
    keys = ("hotbar.4",)
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.HOTBAR, target="hotbar.4"), keys
    )
    assert translated == {"hotbar.4": 1}


def test_wait_does_not_emit_equip_or_hotbar() -> None:
    keys = _TREECHOP_KEYS + ("use", "hotbar.1", "equip")
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.WAIT), keys
    )
    assert "equip" not in translated
    assert "hotbar.1" not in translated


def test_inventory_action_is_explicit_and_wait_releases_key() -> None:
    keys = _TREECHOP_KEYS + ("inventory",)
    opened = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.INVENTORY), keys
    )
    assert opened["inventory"] == 1
    waited = MineRLEnvironment._to_minerl_action(Action(type=ActionType.WAIT), keys)
    assert waited["inventory"] == 0


def test_hotbar_slot_parser() -> None:
    assert _hotbar_slot("1") == 1
    assert _hotbar_slot("hotbar.9") == 9
    assert _hotbar_slot("0") is None
    assert _hotbar_slot("10") is None
    assert _hotbar_slot("") is None


def test_equipped_item_name_prefers_mainhand() -> None:
    raw = {"equipped_items": {"mainhand": {"type": "iron_pickaxe"}}}
    assert _equipped_item_name(raw) == "iron_pickaxe"
    assert _equipped_item_name({"equipped_items": {"mainhand": {"type": "none"}}}) is None
    assert _equipped_item_name({}) is None
