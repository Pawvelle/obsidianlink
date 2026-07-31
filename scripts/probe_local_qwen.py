"""Isolated local-Qwen load probe that survives a child-process failure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT = ROOT / "runs/phase3-vlm-a0-preflight"


def _child() -> int:
    sys.path.insert(0, str(ROOT))
    from obsidianlink.agents import LocalQwenResponder

    responder = LocalQwenResponder(MODEL_PATH)
    responder._ensure_loaded()
    print(json.dumps({"status": "loaded", "device": responder.device}))
    return 0


def _parent(timeout_seconds: float) -> int:
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        summary = {
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout_seconds": timeout_seconds,
            "model_path": str(MODEL_PATH.relative_to(ROOT)),
            "run_dir": str(run_dir),
        }
    except subprocess.TimeoutExpired as error:
        summary = {
            "status": "failed", "failure_type": "TimeoutExpired",
            "timeout_seconds": timeout_seconds,
            "stdout": error.stdout, "stderr": error.stderr,
            "model_path": str(MODEL_PATH.relative_to(ROOT)), "run_dir": str(run_dir),
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return _child() if args.child else _parent(args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
