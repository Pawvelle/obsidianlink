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

from mc_agent.qwen import QwenPlannerWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "runs" / "smoke" / "findcave-reset.png",
    )
    args = parser.parse_args()
    pov = np.asarray(Image.open(args.frame).convert("RGB"))
    worker = QwenPlannerWorker()
    worker.start()
    try:
        if not worker.ready.wait(30):
            raise TimeoutError("planner did not load within 30 seconds")
        if worker.error:
            raise RuntimeError(worker.error)
        worker.begin_episode("offline-smoke")
        worker.submit("offline-smoke", 0, pov, None)
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
        result = {
            "accepted": decision.accepted,
            "raw": decision.raw,
            "parsed": decision.action.to_log_dict(),
            "error": decision.error,
            "latency_seconds": decision.latency_seconds,
            "load_seconds": worker.load_seconds,
            "peak_mps_driver_bytes": worker.peak_mps_driver_bytes,
        }
        print(json.dumps(result, indent=2))
        if not decision.accepted:
            raise RuntimeError(decision.error)
        return 0
    finally:
        worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
