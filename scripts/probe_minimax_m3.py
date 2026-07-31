"""One explicitly authorised, fixed-frame MiniMax-M3 contract probe."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "runs/phase3-scripted-a0/20260731-210140/initial.png"
TASK_PATH = ROOT / "benchmark/instances/route_a_a0_phase3.json"

import sys

sys.path.insert(0, str(ROOT))

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.agents import MiniMaxM3Responder
from obsidianlink.core.types import TaskInstance


def _task() -> TaskInstance:
    return TaskInstance.from_dict(json.loads(TASK_PATH.read_text(encoding="utf-8")))


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--allow-live-request", action="store_true",
        help="required acknowledgement before the single paid API request",
    )
    args = parser.parse_args()
    if not args.allow_live_request:
        parser.error("--allow-live-request is required; this sends one paid API request")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")

    run_dir = ROOT / "runs/phase3-minimax-m3-probe" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        task = _task()
        frame = np.asarray(Image.open(args.image).convert("RGB"))
        prompt = {
            "instruction": task.instruction,
            "workflow": task.workflow,
            "current_stage": "initial",
            "observation": {
                "episode_id": task.task_id,
                "agent_id": "agent_1",
                "step_id": 0,
                "visible_inventory": dict(task.initial_inventories["agent_1"]),
                "messages": [],
                "workflow_stage": "initial",
                "frame": frame,
            },
        }
        responder = MiniMaxM3Responder(timeout_seconds=args.timeout_seconds)
        raw = responder(prompt)
        parsed = parse_macro_action(raw)
        summary = {
            "status": "passed",
            "request_completed": True,
            "action_accepted": parsed.accepted,
            "parse_error": parsed.error,
            "action": {
                "action_type": parsed.action.action_type,
                "target": parsed.action.target,
                "duration_ticks": parsed.action.duration_ticks,
                "parameters": dict(parsed.action.parameters),
            },
            "remote_request": _json_ready(responder.last_request.__dict__),
            "image": str(args.image),
            "run_dir": str(run_dir),
        }
        exit_code = 0
    except Exception as error:
        summary = {
            "status": "failed",
            "request_completed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir),
        }
        exit_code = 1
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
