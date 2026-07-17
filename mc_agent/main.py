"""Small command-line entry point for the personal Minecraft Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from mc_agent.agent import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MineRL + Qwen Minecraft Agent")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=240)
    parser.add_argument("--observation-interval", type=int, default=40)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="show a live MineRL POV window while the agent is running",
    )
    args = parser.parse_args()
    run_agent(
        episodes=args.episodes,
        ticks=args.ticks,
        observation_interval=args.observation_interval,
        output_root=args.output_root,
        watch=args.watch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
