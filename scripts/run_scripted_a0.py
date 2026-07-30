from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.core.types import TaskInstance
from obsidianlink.drivers.scripted_a0 import run_scripted_a0
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend


TASK_PATH = ROOT / "benchmark/instances/route_a_a0_development.json"


def _load_task() -> TaskInstance:
    return TaskInstance.from_dict(
        json.loads(TASK_PATH.read_text(encoding="utf-8"))
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/phase1-scripted-a0",
    )
    parser.add_argument("--max-portal-wait-steps", type=int, default=120)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    backend = MineRLEnvironmentBackend()
    backend.open()
    try:
        with (run_dir / "events.jsonl").open(
            "w",
            encoding="utf-8",
        ) as event_handle:
            def write_event(event: dict[str, Any]) -> None:
                event_handle.write(
                    json.dumps(
                        _json_ready(dict(event)),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                event_handle.flush()

            result = run_scripted_a0(
                backend,
                _load_task(),
                max_portal_wait_steps=args.max_portal_wait_steps,
                event_sink=write_event,
            )
        Image.fromarray(result.final_observation.frame).save(
            run_dir / "final.png"
        )
        summary = _json_ready(
            {
                "status": result.status,
                "steps_completed": result.steps_completed,
                "planned_steps": result.planned_steps,
                "wait_steps": result.wait_steps,
                "final_dimension": result.final_dimension,
                "portal_activated": result.portal_activated,
                "entered_nether": result.entered_nether,
                "terminated": result.terminated,
                "evaluation_evidence": dict(result.evaluation_evidence),
                "blocked_reason": result.blocked_reason,
            }
        )
        summary["run_dir"] = str(run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if result.status == "passed" else 2
    except Exception as error:
        summary = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
