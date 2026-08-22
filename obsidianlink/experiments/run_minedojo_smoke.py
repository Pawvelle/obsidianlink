"""Run the active-platform MineDojo reset-and-step smoke.

Use this only after MineDojo's Minecraft backend has completed its local
first-start preparation.  It is intentionally not a task-capability run.

    # On macOS, start this from Terminal.app so MineDojo can keep its Python
    # parent alive while it creates the Minecraft window:
    open scripts/run_minedojo_smoke_macos.command

    # Or, from an already-open normal terminal:
    /opt/anaconda3/bin/conda run --no-capture-output -n mc-agent python -m \
        obsidianlink.experiments.run_minedojo_smoke --task-id harvest_1_log

MineDojo's own macOS launcher starts Minecraft in a separate Terminal.app
window and its watchdog deliberately terminates that child when this Python
process exits.  A short-lived IDE or automation command therefore is not a
valid GUI smoke-test host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    parser.add_argument("--task-id", default="harvest_1_log")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="write a small success/error summary here; useful for Terminal.app smoke runs",
    )
    args = parser.parse_args()
    exit_code = 0
    try:
        summary = run_smoke(
            args.task_id,
            height=max(1, args.height),
            width=max(1, args.width),
            steps=max(1, args.steps),
        )
        summary["status"] = "ok"
    except Exception as exc:
        exit_code = 1
        summary = {
            "status": "error",
            "task_id": args.task_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
