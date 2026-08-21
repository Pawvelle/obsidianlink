"""Run the first autonomous loop: collect wood and craft a wooden pickaxe.

Usage:
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
    obsidianlink/experiments/run_autonomous_agent.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from obsidianlink.agents.agent import AutonomousMinecraftAgent, WOODEN_PICKAXE_GOAL
from obsidianlink.agents.planner import LLMSkillPlanner
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.wood_pickaxe import WoodPickaxeEnv
from obsidianlink.models.minimax_client import MiniMaxClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Minecraft agent: collect wood and craft a wooden pickaxe"
    )
    parser.add_argument("--goal", default=WOODEN_PICKAXE_GOAL)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--max-planning-cycles", type=int, default=16)
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()

    env = WoodPickaxeEnv()
    controller = MinecraftController(env, max_steps=max(1, args.max_steps))
    planner = LLMSkillPlanner(MiniMaxClient(), use_vision=not args.no_vision)
    agent = AutonomousMinecraftAgent(
        planner,
        controller,
        max_planning_cycles=max(1, args.max_planning_cycles),
    )
    try:
        result = agent.run(args.goal)
    finally:
        controller.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
