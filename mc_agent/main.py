"""Small command-line entry point for the personal Minecraft Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from mc_agent.agent import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MineRL + Qwen Minecraft Agent")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=240)
    parser.add_argument(
        "--mission-ticks",
        type=int,
        default=None,
        help=(
            "override the local FindCave mission limit in ticks; 18000 is 15 minutes "
            "at the fixed 20 Hz MineRL rate"
        ),
    )
    parser.add_argument("--observation-interval", type=int, default=40)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=None,
        help="use an explicit local model lock for an isolated planner experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="use one reproducible MineRL seed for every requested episode",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="show a live MineRL POV window while the agent is running",
    )
    parser.add_argument(
        "--cave-entry-phase",
        action="store_true",
        help=(
            "enable the optional Phase 5 bounded entry phase: after the "
            "existing double-confirmed cave gate, run a short local forward "
            "block into the validated opening, record a post-entry frame, "
            "then emit the single local ESC tick"
        ),
    )
    parser.add_argument(
        "--cave-entry-phase-max-ticks",
        type=int,
        default=None,
        help=(
            "override the Phase 5 entry forward budget in ticks (default 30; "
            "only meaningful together with --cave-entry-phase)"
        ),
    )
    args = parser.parse_args()
    run_agent(
        episodes=args.episodes,
        ticks=args.ticks,
        observation_interval=args.observation_interval,
        output_root=args.output_root,
        seed=args.seed,
        watch=args.watch,
        mission_max_ticks=args.mission_ticks,
        model_lock_path=args.model_lock,
        cave_entry_phase_enabled=args.cave_entry_phase,
        cave_entry_phase_max_ticks=(
            args.cave_entry_phase_max_ticks
            if args.cave_entry_phase_max_ticks is not None
            else 30
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
