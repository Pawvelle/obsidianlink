"""Project command-line entry point."""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from mc_agent.actions import (
    LatestActionMailbox,
    MacroExecutor,
    Watchdog,
    parse_macro_action,
)
from mc_agent.env import MineRLEnvAdapter
from mc_agent.evaluation import EpisodeLogger
from mc_agent.evaluation.phase4 import run_phase4_evaluation
from mc_agent.evaluation.phase5 import DEFAULT_SEEDS, run_phase5_frame_change_ab
from mc_agent.evaluation.phase5_forward_probe import run_phase5_forward_probe_ab
from mc_agent.evaluation.phase5_repetition import run_phase5_repetition_ab
from mc_agent.evaluation.phase5_recovery import run_phase5_recovery_ab
from mc_agent.evaluation.phase5_orientation import run_phase5_orientation_ab
from mc_agent.evaluation.phase5_hierarchical import run_phase5_hierarchical_ab
from mc_agent.evaluation.phase5_turning import run_phase5_turning_loop_ab


ROOT = Path(__file__).resolve().parents[2]


def _seed_tuple(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def scripted_macro(index: int) -> str:
    if index % 2 == 0:
        return json.dumps(
            {
                "action": "look",
                "duration_ticks": 10,
                "camera": {"pitch": 0, "yaw": 8},
                "reason": "phase3 deterministic camera sweep",
            }
        )
    return json.dumps(
        {
            "action": "wait",
            "duration_ticks": 10,
            "reason": "phase3 deterministic pause",
        }
    )


def phase3_smoke(ticks: int, output_root: Path) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    logger = EpisodeLogger(
        run_dir,
        {
            "phase": 3,
            "env_id": "MineRLBasaltFindCave-v0",
            "target_ticks": ticks,
            "policy": "alternating look/wait; no Qwen",
        },
    )
    watchdog = Watchdog(max_ticks=ticks)
    signal.signal(signal.SIGINT, lambda *_: watchdog.request_stop("SIGINT"))
    total_reward = 0.0
    completed_ticks = 0
    early_done = False
    started = time.perf_counter()

    try:
        with MineRLEnvAdapter() as adapter:
            observation = adapter.reset()
            Image.fromarray(observation["pov"]).save(run_dir / "initial.png")
            logger.event(
                "reset",
                pov_shape=list(observation["pov"].shape),
                action_fields=list(adapter.action_space.spaces),
            )
            executor = MacroExecutor(adapter.action_space, watchdog=watchdog)
            mailbox = LatestActionMailbox()
            macro_index = 0

            while completed_ticks < ticks:
                if watchdog.should_stop and watchdog.reason != "max_ticks":
                    logger.event("interrupted", reason=watchdog.reason)
                    break
                if executor.needs_action:
                    result = parse_macro_action(scripted_macro(macro_index))
                    if not result.accepted:
                        raise RuntimeError(result.error)
                    mailbox.publish(result.action)
                    latest = mailbox.take_latest()
                    if latest is None:
                        raise RuntimeError("action mailbox unexpectedly empty")
                    executor.submit(latest)
                    logger.event(
                        "macro",
                        macro_index=macro_index,
                        action=result.action.to_log_dict(),
                    )
                    macro_index += 1

                tick_action = executor.next_tick()
                step = adapter.step(tick_action)
                completed_ticks += 1
                watchdog.after_tick()
                total_reward += step.reward
                observation = step.observation
                logger.event(
                    "tick",
                    tick=completed_ticks,
                    action={
                        "ESC": tick_action["ESC"],
                        "attack": tick_action["attack"],
                        "camera": tick_action["camera"],
                        "forward": tick_action["forward"],
                        "jump": tick_action["jump"],
                        "sprint": tick_action["sprint"],
                    },
                    reward=step.reward,
                    done=step.done,
                )
                if completed_ticks % 100 == 0:
                    logger.event(
                        "progress",
                        tick=completed_ticks,
                        reward_sum=total_reward,
                    )
                if step.done:
                    early_done = completed_ticks < ticks
                    logger.event("done", tick=completed_ticks, early=early_done)
                    break

            Image.fromarray(observation["pov"]).save(run_dir / "final.png")
    except BaseException as error:
        logger.event("error", error=repr(error), tick=completed_ticks)
        logger.finish(
            {
                "accepted": False,
                "completed_ticks": completed_ticks,
                "error": repr(error),
            }
        )
        raise

    elapsed = time.perf_counter() - started
    accepted = completed_ticks == ticks and not early_done
    metrics = {
        "accepted": accepted,
        "completed_ticks": completed_ticks,
        "target_ticks": ticks,
        "early_done": early_done,
        "reward_sum": total_reward,
        "elapsed_seconds": elapsed,
        "ticks_per_second": completed_ticks / elapsed if elapsed else 0,
        "watchdog_reason": watchdog.reason,
    }
    logger.finish(metrics)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))
    if not accepted:
        raise RuntimeError("Phase-3 tick target was not completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("phase3-smoke")
    smoke.add_argument("--ticks", type=int, default=500)
    smoke.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase3" / "runs",
    )
    phase4 = subparsers.add_parser("phase4-eval")
    phase4.add_argument("--episodes", type=int, default=5)
    phase4.add_argument("--ticks", type=int, default=240)
    phase4.add_argument("--observation-interval", type=int, default=40)
    phase4.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase4" / "runs",
    )
    phase5 = subparsers.add_parser("phase5-frame-change-ab")
    phase5.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5.add_argument("--ticks", type=int, default=800)
    phase5.add_argument("--observation-interval", type=int, default=40)
    phase5.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "frame-change-ab",
    )
    phase5_turning = subparsers.add_parser("phase5-turning-loop-ab")
    phase5_turning.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_turning.add_argument("--ticks", type=int, default=800)
    phase5_turning.add_argument("--observation-interval", type=int, default=40)
    phase5_turning.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "turning-loop-ab",
    )
    phase5_repetition = subparsers.add_parser("phase5-repetition-ab")
    phase5_repetition.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_repetition.add_argument("--ticks", type=int, default=800)
    phase5_repetition.add_argument("--observation-interval", type=int, default=40)
    phase5_repetition.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "repetition-ab",
    )
    phase5_recovery = subparsers.add_parser("phase5-recovery-ab")
    phase5_recovery.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_recovery.add_argument("--ticks", type=int, default=800)
    phase5_recovery.add_argument("--observation-interval", type=int, default=40)
    phase5_recovery.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "recovery-ab",
    )
    phase5_orientation = subparsers.add_parser("phase5-orientation-ab")
    phase5_orientation.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_orientation.add_argument("--ticks", type=int, default=800)
    phase5_orientation.add_argument("--observation-interval", type=int, default=40)
    phase5_orientation.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "orientation-ab",
    )
    phase5_hierarchical = subparsers.add_parser("phase5-hierarchical-ab")
    phase5_hierarchical.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_hierarchical.add_argument("--ticks", type=int, default=800)
    phase5_hierarchical.add_argument("--observation-interval", type=int, default=40)
    phase5_hierarchical.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "hierarchical-ab",
    )
    phase5_forward_probe = subparsers.add_parser("phase5-forward-probe-ab")
    phase5_forward_probe.add_argument(
        "--seeds",
        type=_seed_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated paired MineRL seeds",
    )
    phase5_forward_probe.add_argument("--ticks", type=int, default=800)
    phase5_forward_probe.add_argument("--observation-interval", type=int, default=40)
    phase5_forward_probe.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase5" / "forward-probe-ab",
    )
    args = parser.parse_args()
    if args.command == "phase3-smoke":
        return phase3_smoke(args.ticks, args.output_root)
    if args.command == "phase4-eval":
        run_phase4_evaluation(
            episodes=args.episodes,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-frame-change-ab":
        run_phase5_frame_change_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-turning-loop-ab":
        run_phase5_turning_loop_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-repetition-ab":
        run_phase5_repetition_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-recovery-ab":
        run_phase5_recovery_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-orientation-ab":
        run_phase5_orientation_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-hierarchical-ab":
        run_phase5_hierarchical_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    if args.command == "phase5-forward-probe-ab":
        run_phase5_forward_probe_ab(
            seeds=args.seeds,
            ticks=args.ticks,
            observation_interval=args.observation_interval,
            output_root=args.output_root,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
