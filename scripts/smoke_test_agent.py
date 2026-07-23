#!/usr/bin/env python3
"""Offline protocol check for the asynchronous Qwen planner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_agent.actions import (  # noqa: E402
    has_directional_stone_bounded_dark_opening_region,
    is_cave_candidate,
    resolve_dark_opening_direction,
    resolve_cave_direction,
)
from mc_agent.qwen import QwenPlannerWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "runs" / "smoke" / "findcave-reset.png",
    )
    parser.add_argument(
        "--after-forward",
        action="store_true",
        help="Check the action-change prompt after a medium forward action.",
    )
    parser.add_argument(
        "--expect-no-cave",
        action="store_true",
        help="Fail if the validated result claims a cave candidate.",
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=None,
        help="use an explicit local model lock for an isolated planner experiment",
    )
    args = parser.parse_args()
    pov = np.asarray(Image.open(args.frame).convert("RGB"))
    worker = QwenPlannerWorker(lock_path=args.model_lock)
    worker.start()
    try:
        if not worker.ready.wait(30):
            raise TimeoutError("planner did not load within 30 seconds")
        if worker.error:
            raise RuntimeError(worker.error)
        worker.begin_episode("offline-smoke")
        previous_action = None
        visual_change = None
        if args.after_forward:
            previous_action = {
                "action": "move_forward",
                "duration_ticks": 16,
                "camera": {"pitch": 0.0, "yaw": 0.0},
                "attack": False,
                "jump": False,
                "sprint": True,
            }
            visual_change = {
                "mean_absolute_difference": 0.04,
                "changed_pixel_fraction": 0.05,
                "low_change": False,
            }
        worker.submit(
            "offline-smoke",
            0,
            pov,
            previous_action,
            visual_change,
        )
        deadline = time.monotonic() + 30
        decision = None
        while time.monotonic() < deadline and decision is None:
            if worker.error:
                raise RuntimeError(worker.error)
            decision = worker.decisions.take_latest(timeout=0.1)
        if decision is None:
            raise TimeoutError("planner did not produce a decision")
        worker.acknowledge_decision(
            decision.episode_id,
            decision.observation_tick,
        )
        cave_text_evidence_complete = is_cave_candidate(decision.action)
        cave_direction = (
            resolve_cave_direction(decision.action.reason)
            if cave_text_evidence_complete
            else None
        )
        cave_frame_plausible = bool(
            cave_direction is not None
            and has_directional_stone_bounded_dark_opening_region(pov, cave_direction)
        )
        cave_direction_source = "model_reason" if cave_direction is not None else None
        if cave_text_evidence_complete and not cave_frame_plausible:
            local_direction = resolve_dark_opening_direction(pov)
            if local_direction is not None:
                cave_direction = local_direction
                cave_direction_source = "local_dark_region"
                cave_frame_plausible = has_directional_stone_bounded_dark_opening_region(
                    pov, cave_direction
                )
        result = {
            "accepted": decision.accepted,
            "raw": decision.raw,
            "parsed": decision.action.to_log_dict(),
            "error": decision.error,
            "latency_seconds": decision.latency_seconds,
            "load_seconds": worker.load_seconds,
            "peak_mps_driver_bytes": worker.peak_mps_driver_bytes,
            "cave_text_evidence_complete": cave_text_evidence_complete,
            "cave_direction": cave_direction,
            "cave_direction_source": cave_direction_source,
            "cave_frame_plausible": cave_frame_plausible,
            "cave_candidate_validated": bool(
                cave_text_evidence_complete and cave_frame_plausible
            ),
        }
        print(json.dumps(result, indent=2))
        if not decision.accepted:
            raise RuntimeError(decision.error)
        if args.expect_no_cave and result["cave_candidate_validated"]:
            raise RuntimeError("unexpected cave candidate")
        return 0
    finally:
        worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
