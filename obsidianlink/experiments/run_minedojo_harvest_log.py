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
from obsidianlink.models.qwen_client import QwenLLMClient, default_qwen_model_path
from obsidianlink.skills.mining import log_count

TASK_ID = "harvest_1_log"
TASK = "MineDojo task: obtain one log. Find a tree, aim at its trunk, and break it."


def _goal_verified(_task: str, _memory: AgentMemory, observation: Observation) -> bool:
    return log_count(dict(observation.inventory or {})) >= 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Visible Qwen + MineDojo log trial")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-planning-cycles", type=int, default=8)
    parser.add_argument("--model-path", type=Path, default=default_qwen_model_path())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("logs") / datetime.now(timezone.utc).strftime(
        "minedojo_qwen_harvest_log_%Y%m%d_%H%M%SZ"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = MineDojoEnvironment(TASK_ID, image_size=(360, 640))
    view = LiveDesktopView(environment)
    board = LiveProcessBoard(view=view)
    client = QwenLLMClient(args.model_path, max_new_tokens=384)
    view.set_hud(task="obtain 1 log", model=client.model, status="starting MineDojo")
    board.push(f"启动 MineDojo + 本地千问：{client.model}")
    board.push("请查看 Minecraft、ObsidianLink Agent POV、ObsidianLink Agent Process 三个窗口")
    controller = MinecraftController(view, max_steps=max(1, args.max_steps))
    agent = GeneralAgent(
        LLMSkillPlanner(client, use_vision=True, allow_wiki=False),
        controller,
        goal_verifier=_goal_verified,
        max_planning_cycles=max(1, args.max_planning_cycles),
        episode_logger=DisplayEpisodeLogger(EpisodeLogger.create(DEFAULT_EPISODE_ROOT), board),
    )
    try:
        result = agent.run(TASK)
        summary: dict[str, Any] = {
            "task_id": TASK_ID,
            "task": TASK,
            "result": asdict(result),
            "model": client.model,
            "model_path": client.model_path,
            "model_calls": client.completions,
            "vision_calls": client.vision_completions,
            "action_counts": controller.action_counts,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if result.success else 1
    finally:
        board.close()
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
