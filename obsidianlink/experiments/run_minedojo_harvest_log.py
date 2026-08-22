"""Visible local-Qwen MineDojo trial: obtain one log with primitive actions.

Run from the repository root:

    conda run -n mc-agent python -m obsidianlink.experiments.run_minedojo_harvest_log
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obsidianlink.agents.episode_log import DEFAULT_EPISODE_ROOT, EpisodeLogger
from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import LLMSkillPlanner
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.environment import Observation
from obsidianlink.env.live_view import DisplayEpisodeLogger, LiveDesktopView, LiveProcessBoard
from obsidianlink.env.minedojo import MineDojoEnvironment
from obsidianlink.models.minimax_client import MiniMaxClient
from obsidianlink.models.qwen_client import QwenLLMClient, default_qwen_model_path
from obsidianlink.skills.mining import log_count
from obsidianlink.voyager import MineDojoVoyager, VoyagerTask, early_survival_curriculum

TASK_ID = "harvest_1_log"
TASK = "MineDojo task: obtain one log. Find a tree, aim at its trunk, and break it."


def _goal_verified(_task: str, _memory: AgentMemory, observation: Observation) -> bool:
    return log_count(dict(observation.inventory or {})) >= 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Visible LLM + MineDojo log trial")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-planning-cycles", type=int, default=8)
    parser.add_argument("--backend", choices=("qwen", "minimax"), default="qwen")
    parser.add_argument("--model-path", type=Path, default=default_qwen_model_path())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--voyager",
        action="store_true",
        help="run through the MineDojo-native Voyager adaptation instead of GeneralAgent directly",
    )
    parser.add_argument(
        "--voyager-curriculum",
        action="store_true",
        help="run the bounded early-survival curriculum in a MineDojo open-ended world",
    )
    parser.add_argument("--curriculum-max-tasks", type=int, default=5)
    args = parser.parse_args()
    if args.voyager_curriculum and not args.voyager:
        parser.error("--voyager-curriculum requires --voyager")
    output_dir = args.output_dir or Path("logs") / datetime.now(timezone.utc).strftime(
        "minedojo_qwen_harvest_log_%Y%m%d_%H%M%SZ"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment_task_id = "open-ended" if args.voyager_curriculum else TASK_ID
    environment = MineDojoEnvironment(environment_task_id, image_size=(360, 640))
    view = LiveDesktopView(environment)
    board = LiveProcessBoard(view=view)
    client = (
        QwenLLMClient(args.model_path, max_new_tokens=384)
        if args.backend == "qwen"
        else MiniMaxClient(timeout_s=120.0, max_tokens=768)
    )
    view.set_hud(task="obtain 1 log", model=client.model, status="starting MineDojo")
    board.push(f"启动 MineDojo + {args.backend}：{client.model}")
    board.push("请查看 Minecraft、ObsidianLink Agent POV、ObsidianLink Agent Process 三个窗口")
    controller = MinecraftController(view, max_steps=max(1, args.max_steps))
    voyager: MineDojoVoyager | None = None
    agent: GeneralAgent | None = None
    if args.voyager:
        voyager = MineDojoVoyager(
            view,
            planner_factory=lambda: LLMSkillPlanner(
                client, use_vision=True, allow_wiki=False
            ),
            max_steps=max(1, args.max_steps),
            max_planning_cycles=max(1, args.max_planning_cycles),
        )
    else:
        agent = GeneralAgent(
            LLMSkillPlanner(client, use_vision=True, allow_wiki=False),
            controller,
            goal_verifier=_goal_verified,
            max_planning_cycles=max(1, args.max_planning_cycles),
            episode_logger=DisplayEpisodeLogger(EpisodeLogger.create(DEFAULT_EPISODE_ROOT), board),
        )
    try:
        curriculum_episodes = ()
        if voyager is not None:
            if args.voyager_curriculum:
                curriculum_episodes = voyager.learn_curriculum(
                    early_survival_curriculum(),
                    max_tasks=max(1, args.curriculum_max_tasks),
                )
                episode = curriculum_episodes[-1]
            else:
                episode = voyager.run_task(VoyagerTask(TASK_ID, TASK, _goal_verified))
            result = episode.result
        else:
            assert agent is not None
            result = agent.run(TASK)
        summary: dict[str, Any] = {
            "task_id": TASK_ID,
            "task": TASK,
            "agent_mode": "minedojo_voyager" if voyager is not None else "general_agent",
            "environment_task_id": environment_task_id,
            "result": asdict(result),
            "model": client.model,
            "backend": args.backend,
            "model_path": getattr(client, "model_path", None),
            "model_calls": client.completions,
            "vision_calls": getattr(client, "vision_completions", None),
            "action_counts": (
                episode.action_counts if voyager is not None else controller.action_counts
            ),
        }
        if voyager is not None:
            summary["retrieved_skills"] = list(episode.retrieved_skills)
            summary["skill_memory"] = voyager.skill_memory.as_dict()
            summary["curriculum_episodes"] = [
                item.as_dict() for item in curriculum_episodes
            ]
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if result.success else 1
    finally:
        board.close()
        if voyager is not None:
            voyager.close()
        elif agent is not None:
            agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
