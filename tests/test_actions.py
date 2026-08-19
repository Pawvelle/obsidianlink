from obsidianlink.env.actions import Action, ActionType


def test_wait_defaults_are_inert() -> None:
    action = Action(type=ActionType.WAIT)
    assert action.dx == 0
    assert action.dz == 0
    assert action.yaw == 0.0
    assert action.pitch == 0.0
    assert action.target == ""


def test_action_type_values_are_stable() -> None:
    assert ActionType.MOVE.value == "move"
    assert ActionType.CAMERA.value == "camera"
    assert ActionType.ATTACK.value == "attack"
    assert ActionType.USE.value == "use"
    assert ActionType.PLACE.value == "place"
    assert ActionType.EQUIP.value == "equip"
    assert ActionType.HOTBAR.value == "hotbar"
    assert ActionType.WAIT.value == "wait"
