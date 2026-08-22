from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minedojo import MineDojoEnvironment


class FakeActionSpace:
    def no_op(self):
        return {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sneak": 0,
            "attack": 0,
            "use": 0,
            "camera": [0.0, 0.0],
            "equip": "none",
            "place": "none",
        }


class FakeMineDojoTask:
    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.actions: list[dict[str, object]] = []
        self.closed = False

    @staticmethod
    def _observation():
        return {
            "rgb": np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                ],
                dtype=np.uint8,
            ),
            "inventory": {
                "name": np.array(["oak log", "air", "oak log"]),
                "quantity": np.array([2, 0, 1]),
            },
            "equipment": {
                "name": np.array(["iron axe"]),
                "quantity": np.array([1]),
            },
            "location_stats": {"xpos": 99},
        }

    def reset(self):
        return self._observation()

    def step(self, action):
        self.actions.append(action)
        return self._observation(), 0.5, False, {"task_success": False}

    def close(self):
        self.closed = True


def test_minedojo_adapter_converts_agent_visible_observation_and_actions(monkeypatch) -> None:
    task = FakeMineDojoTask()
    calls: list[dict[str, object]] = []

    def make(**kwargs):
        calls.append(kwargs)
        return task

    _install_fake_minedojo(monkeypatch, make)
    env = MineDojoEnvironment("harvest_milk", image_size=(2, 2))

    observation = env.reset()
    assert calls == [
        {
            "task_id": "harvest_milk",
            "image_size": (2, 2),
            "event_level_control": True,
        }
    ]
    assert observation.inventory == {"oak_log": 3}
    assert observation.selected_item == "iron_axe"
    assert observation.frame.tolist() == [
        [[9, 5, 1], [10, 6, 2]],
        [[11, 7, 3], [12, 8, 4]],
    ]

    env.step(Action(ActionType.CAMERA, pitch=-3, yaw=8))
    assert task.actions[-1]["camera"] == [-3.0, 8.0]
    assert env.hidden_state == {"reward": 0.5, "done": False}
    assert env.last_info == {"task_success": False}


def test_minedojo_adapter_maps_event_actions_and_rejects_unavailable_ones(monkeypatch) -> None:
    task = FakeMineDojoTask()
    _install_fake_minedojo(monkeypatch, lambda **_kwargs: task)
    env = MineDojoEnvironment()
    env.reset()

    env.step(Action(ActionType.MOVE, dx=1, dz=-1, jump=True))
    assert task.actions[-1]["forward"] == 1
    assert task.actions[-1]["left"] == 1
    assert task.actions[-1]["jump"] == 1
    env.step(Action(ActionType.EQUIP, target="oak_log"))
    assert task.actions[-1]["equip"] == "oak_log"
    with pytest.raises(ValueError, match="hotbar is unavailable"):
        env.step(Action(ActionType.HOTBAR, target="3"))


def test_minedojo_adapter_requires_reset_before_observe_or_step() -> None:
    env = MineDojoEnvironment()
    with pytest.raises(RuntimeError, match="before reset"):
        env.observe()
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(Action(ActionType.WAIT))


def _install_fake_minedojo(monkeypatch, make) -> None:
    package = ModuleType("minedojo")
    package.__path__ = []
    tasks = ModuleType("minedojo.tasks")
    tasks._specific_task_make = lambda task_id, **kwargs: make(task_id=task_id, **kwargs)
    monkeypatch.setitem(sys.modules, "minedojo", package)
    monkeypatch.setitem(sys.modules, "minedojo.tasks", tasks)
