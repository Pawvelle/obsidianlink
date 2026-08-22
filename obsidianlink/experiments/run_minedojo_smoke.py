"""Run the active-platform MineDojo reset-and-step smoke.

Use this only after MineDojo's Minecraft backend has completed its local
first-start preparation.  It is intentionally not a task-capability run.

    /opt/anaconda3/bin/conda run -n mc-agent python \
        obsidianlink/experiments/run_minedojo_smoke.py --task-id harvest_milk
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minedojo import MineDojoEnvironment


def run_smoke(task_id: str, *, height: int, width: int, steps: int) -> dict[str, Any]:
    environment = MineDojoEnvironment(task_id, image_size=(height, width))
    controller = MinecraftController(environment, max_steps=max(1, steps))
    try:
        observation = controller.reset()
        for _ in range(steps):
            controller.step(Action(ActionType.WAIT))
        final = controller.observe()
        return {
            "task_id": task_id,
            "initial_frame_shape": list(getattr(observation.frame, "shape", ())),
            "final_frame_shape": list(getattr(final.frame, "shape", ())),
            "inventory": dict(final.inventory or {}),
            "selected_item": final.selected_item,
            "environment_steps": controller.steps,
            "hidden_state": environment.hidden_state,
        }
    finally:
        controller.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MineDojo platform smoke")
    parser.add_argument("--task-id", default="harvest_milk")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    summary = run_smoke(
        args.task_id,
        height=max(1, args.height),
        width=max(1, args.width),
        steps=max(1, args.steps),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
