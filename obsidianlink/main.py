"""ObsidianLink entry points.

Modes:

* ``OBSIDIANLINK_OFFLINE=1`` — no Java / Minecraft
* default — MineDojo reset + no-op smoke on ``harvest_1_log``
"""

from __future__ import annotations

import os
import sys
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

NUM_LIVE_STEPS = 16


def main() -> int:
    if os.environ.get("OBSIDIANLINK_OFFLINE") == "1":
        return _run_offline()
    return _run_minedojo_smoke()


def _run_offline() -> int:
    class _StubEnvironment(Environment):
        def __init__(self) -> None:
            self._obs = Observation()
            self.closed = False

        def reset(self) -> Observation:
            self._obs = Observation(frame="offline-stub-frame", inventory={}, selected_item=None)
            return self._obs

        def observe(self) -> Observation:
            return self._obs

        def step(self, action: Action) -> Observation:
            del action
            self._obs = Observation(frame="offline-stub-frame-1", inventory={}, selected_item=None)
            return self._obs

        def close(self) -> None:
            self.closed = True

    print("ObsidianLink offline stub")
    env = _StubEnvironment()
    env.reset()
    env.observe()
    env.step(Action(type=ActionType.WAIT))
    env.close()
    print("offline: reset/observe/step/close wired")
    return 0


def _frame_shape(frame: Any) -> str:
    shape = getattr(frame, "shape", None)
    return str(shape) if shape is not None else type(frame).__name__


def _run_minedojo_smoke() -> int:
    from obsidianlink.controller.minecraft_controller import MinecraftController
    from obsidianlink.env.minedojo import MineDojoEnvironment

    task_id = os.environ.get("OBSIDIANLINK_TASK_ID", "harvest_1_log")
    print(f"ObsidianLink MineDojo smoke — {task_id}")
    sys.stdout.flush()
    environment = MineDojoEnvironment(task_id, image_size=(64, 64))
    controller = MinecraftController(environment, max_steps=NUM_LIVE_STEPS)
    try:
        observation = controller.reset()
        print(
            "reset ok: "
            f"frame={_frame_shape(observation.frame)} "
            f"inventory={observation.inventory} "
            f"selected={observation.selected_item}"
        )
        sys.stdout.flush()
        for index in range(NUM_LIVE_STEPS):
            observation = controller.step(Action(ActionType.WAIT))
            print(f"step {index + 1}/{NUM_LIVE_STEPS}: wait inventory={observation.inventory}")
            sys.stdout.flush()
        print("MineDojo smoke: reset → wait → observation OK")
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
