"""Flush-on-write JSONL episode logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class EpisodeLogger:
    def __init__(self, run_dir: Path, config: dict[str, Any]):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        self._events = (self.run_dir / "events.jsonl").open("x", encoding="utf-8")
        self.started = time.perf_counter()

    def event(self, kind: str, **fields: Any) -> None:
        payload = {
            "elapsed_seconds": time.perf_counter() - self.started,
            "kind": kind,
            **fields,
        }
        self._events.write(json.dumps(payload, default=_json_default) + "\n")
        self._events.flush()

    def finish(self, metrics: dict[str, Any]) -> None:
        self._events.close()
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
