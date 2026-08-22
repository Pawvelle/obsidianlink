"""Run the first live GeneralAgent smoke task on real MineDojo/Minecraft.

This runner deliberately uses no Wiki, RAG, vision model, or LLM API.  It
isolates the natural-language → planner → skill → Minecraft → observation
feedback path before adding more reasoning capabilities.

Usage:
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_general_agent.py --task "Mine 1 obsidian block"

Natural forest diagnostic (collect_wood is not yet stable there):
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_general_agent.py --natural-world
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.general_smoke import GeneralBlockSmokeEnv
from obsidianlink.env.wood_pickaxe import WoodPickaxeEnv
from obsidianlink.skills import legacy_workflow_skill_library
from obsidianlink.tasks.general_smoke import (
    COLLECT_ONE_LOG_SMOKE,
    CollectWoodSmokePlanner,
    MINE_ONE_OBSIDIAN_SMOKE,
    MineObsidianSmokePlanner,
    collect_wood_goal_verified,
    obsidian_goal_verified,
)


def _save_frame(frame: Any, path: Path) -> bool:
    if frame is None:
        return False
    try:
        from PIL import Image

        Image.fromarray(frame).save(path)
        return True
    except (TypeError, ValueError, AttributeError):
        return False


class _TraceEnvironment(Environment):
    """Save sparse agent-visible POV evidence without exposing hidden state."""

    def __init__(
        self,
        env: Environment,
        output_dir: Path | None,
        *,
        every: int = 40,
    ) -> None:
        self.env = env
        self.output_dir = output_dir
        self.every = max(1, every)
        self.steps = 0

    def _capture(self, label: str, observation: Observation) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _save_frame(observation.frame, self.output_dir / f"{label}.png")

    def reset(self) -> Observation:
        self.steps = 0
        observation = self.env.reset()
        self._capture("step_0000", observation)
        return observation

    def observe(self) -> Observation:
        return self.env.observe()

    def step(self, action: Action) -> Observation:
        observation = self.env.step(action)
        self.steps += 1
        if self.steps % self.every == 0:
            self._capture(f"step_{self.steps:04d}", observation)
        return observation

    def close(self) -> None:
        self.env.close()


def run_live_smoke(
    task: str,
    *,
    max_steps: int,
    max_skill_steps: int,
    output_dir: Path | None = None,
    natural_world: bool = False,
) -> dict[str, Any]:
    if natural_world:
        live_env: Environment = WoodPickaxeEnv()
        planner = CollectWoodSmokePlanner(max_skill_steps=max_skill_steps)
        goal_verifier = collect_wood_goal_verified
        task_id = COLLECT_ONE_LOG_SMOKE.task_id
    else:
        live_env = GeneralBlockSmokeEnv()
        planner = MineObsidianSmokePlanner(max_skill_steps=max_skill_steps)
        goal_verifier = obsidian_goal_verified
        task_id = MINE_ONE_OBSIDIAN_SMOKE.task_id
    traced_env = _TraceEnvironment(live_env, output_dir)
    controller = MinecraftController(traced_env, max_steps=max_steps)
    agent = GeneralAgent(
        planner,
        controller,
        # Historical live smoke planners intentionally exercise the earlier
        # workflow implementations. Production GeneralAgent defaults remain
        # primitive-only.
        skills=legacy_workflow_skill_library(),
        goal_verifier=goal_verifier,
        max_planning_cycles=2,
    )
    try:
        result = agent.run(task)
        final_observation = agent.memory.last_observation
        summary: dict[str, Any] = {
            "task_id": task_id,
            "world": "natural" if natural_world else "controlled_smoke",
            "result": asdict(result),
            "planner_calls": planner.planning_calls,
            "action_counts": controller.action_counts,
            "frame_shape": list(
                getattr(getattr(final_observation, "frame", None), "shape", ())
            ),
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_path = output_dir / "final_frame.png"
            summary["final_frame"] = None
            if final_observation is not None and _save_frame(
                final_observation.frame, frame_path
            ):
                summary["final_frame"] = str(frame_path)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary
    finally:
        agent.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live GeneralAgent Minecraft smoke")
    parser.add_argument("--task", default=MINE_ONE_OBSIDIAN_SMOKE.goal)
    parser.add_argument(
        "--max-steps", type=int, default=MINE_ONE_OBSIDIAN_SMOKE.max_environment_steps
    )
    parser.add_argument(
        "--max-skill-steps", type=int, default=MINE_ONE_OBSIDIAN_SMOKE.max_skill_steps
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--natural-world",
        action="store_true",
        help="Use the randomized natural forest (currently less reliable).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("general_live_%Y%m%d_%H%M%SZ")
        args.output_dir = Path("logs") / run_id
    if args.natural_world and args.task == MINE_ONE_OBSIDIAN_SMOKE.goal:
        args.task = COLLECT_ONE_LOG_SMOKE.goal
    if (
        args.natural_world
        and args.max_steps == MINE_ONE_OBSIDIAN_SMOKE.max_environment_steps
    ):
        args.max_steps = COLLECT_ONE_LOG_SMOKE.max_environment_steps
    if (
        args.natural_world
        and args.max_skill_steps == MINE_ONE_OBSIDIAN_SMOKE.max_skill_steps
    ):
        args.max_skill_steps = COLLECT_ONE_LOG_SMOKE.max_skill_steps
    summary = run_live_smoke(
        args.task,
        max_steps=max(1, args.max_steps),
        max_skill_steps=max(8, args.max_skill_steps),
        output_dir=args.output_dir,
        natural_world=args.natural_world,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["result"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
