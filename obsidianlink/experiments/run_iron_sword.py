"""Live GeneralAgent playtest: empty inventory → craft 1 iron sword.

Uses MiniMax-M3 (China endpoint by default), primitive skills, Wiki, and a
desktop POV window so a human can watch the agent act in real time.

PYTHONPATH=. /opt/anaconda3/envs/mc-agent/bin/python \\
    obsidianlink/experiments/run_iron_sword.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obsidianlink.agents.episode_log import DEFAULT_EPISODE_ROOT, EpisodeLogger
from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import LLMSkillPlanner, PlannerDecision
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.environment import Observation
from obsidianlink.env.live_view import LiveDesktopView
from obsidianlink.env.survival import SurvivalEnv, IRON_SWORD_TASK_ID, iron_sword_count
from obsidianlink.models.minimax_client import MiniMaxClient

DEFAULT_TASK = (
    "Start with an empty inventory in survival Minecraft. "
    "From scratch, gather resources and craft 1 iron_sword. "
    "Do not finish until iron_sword is in the current inventory."
)


def iron_sword_goal_verified(
    _task: str,
    _memory: AgentMemory,
    observation: Observation,
) -> bool:
    return iron_sword_count(observation.inventory) >= 1


class LoggingPlanner:
    """Print each planner decision and push a short HUD line to the POV window."""

    def __init__(self, inner: LLMSkillPlanner, view: LiveDesktopView) -> None:
        self.inner = inner
        self.view = view
        self.calls = 0

    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        self.calls += 1
        print(
            f"[planner {self.calls}] asking MiniMax-M3 "
            f"(inventory={dict(observation.inventory or {})}, "
            f"selected={observation.selected_item})",
            flush=True,
        )
        decision = self.inner.plan(memory, observation, skill_descriptions)
        if decision.type == "skill":
            detail = f"{decision.name} {json.dumps(decision.arguments, ensure_ascii=False)}"
        elif decision.type == "wiki":
            detail = f"wiki {decision.query}"
        elif decision.type == "memory":
            detail = f"memory {decision.query}"
        else:
            detail = decision.type
        print(
            f"[planner {self.calls}] {detail} | reason={decision.reason}",
            flush=True,
        )
        self.view.set_hud(
            cycle=str(self.calls),
            decision=detail[:80],
            reason=(decision.reason or "")[:80],
            subgoal=(decision.subgoal or decision.active_subgoal_id or "")[:80],
        )
        return decision


def run_iron_sword(
    task: str,
    *,
    max_steps: int,
    max_planning_cycles: int,
    output_dir: Path,
    use_vision: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_env = SurvivalEnv(IRON_SWORD_TASK_ID)
    view = LiveDesktopView(live_env)
    view.set_hud(task="craft 1 iron_sword", status="starting Minecraft")
    controller = MinecraftController(view, max_steps=max_steps)
    client = MiniMaxClient(timeout_s=120.0, max_tokens=1024)
    planner = LoggingPlanner(
        LLMSkillPlanner(client, use_vision=use_vision, allow_wiki=True),
        view,
    )
    episode_logger = EpisodeLogger.create(DEFAULT_EPISODE_ROOT)
    agent = GeneralAgent(
        planner,
        controller,
        goal_verifier=iron_sword_goal_verified,
        max_planning_cycles=max_planning_cycles,
        max_wiki_calls=8,
        max_memory_retrievals=12,
        episode_logger=episode_logger,
    )
    print(
        "Minecraft 即将启动。请看桌面：会有游戏窗口，以及放大的 "
        f"{view.window_name} POV 窗口。",
        flush=True,
    )
    try:
        result = agent.run(task)
        summary: dict[str, Any] = {
            "task": task,
            "result": asdict(result),
            "planner_calls": planner.calls,
            "model_calls": planner.inner.model_calls,
            "minimax_completions": client.completions,
            "minimax_url": client.url,
            "minimax_model": client.model,
            "action_counts": controller.action_counts,
            "use_vision": use_vision,
            "episode_dir": str(episode_logger.directory),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        agent.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live GeneralAgent: craft an iron sword from scratch"
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-planning-cycles", type=int, default=48)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("iron_sword_%Y%m%d_%H%M%SZ")
        args.output_dir = Path("logs") / run_id
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    summary = run_iron_sword(
        args.task,
        max_steps=max(1, args.max_steps),
        max_planning_cycles=max(1, args.max_planning_cycles),
        output_dir=args.output_dir,
        use_vision=not args.no_vision,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["result"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
