#!/usr/bin/env python3
"""Capture reproducible MineRL FindCave starting views for manual review.

This tool deliberately does not load Qwen.  Its optional panorama mode only
uses fixed camera turns with every interaction key left at the action-space
no-op, so a starting area can be screened before Phase 4 validates a candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_agent.env import MineRLEnvAdapter


def _write_index(session_dir: Path, manifest: dict[str, object]) -> None:
    """Atomically checkpoint capture progress for an interruptible MineRL run."""
    index_path = session_dir / "index.json"
    temporary_path = session_dir / ".index.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(index_path)


def capture_starts(
    seed_start: int, count: int, output_root: Path, *, panorama: bool = False
) -> Path:
    """Save reset POVs, checkpointing every view for interrupted-run recovery."""
    if not 1 <= count <= 64:
        raise ValueError("count must be between 1 and 64")

    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "env_id": "MineRLBasaltFindCave-v0",
        "seed_start": seed_start,
        "count": count,
        "panorama": panorama,
        "status": "in_progress",
        "captures": [],
        "note": (
            "Frames require manual review; no model was loaded. Panorama uses only "
            "fixed camera turns with interaction keys disabled. The index is updated "
            "after every saved frame, so incomplete sessions remain reviewable."
        ),
    }
    _write_index(session_dir, manifest)

    try:
        with MineRLEnvAdapter() as adapter:
            for offset in range(count):
                seed = seed_start + offset
                adapter.seed(seed)
                observation = adapter.reset()
                capture: dict[str, object] = {
                    "seed": seed,
                    "pov_shape": list(observation["pov"].shape),
                    "status": "in_progress",
                    "views": [],
                }
                captures = manifest["captures"]
                assert isinstance(captures, list)
                captures.append(capture)
                _write_index(session_dir, manifest)

                def save_view(heading_degrees: int) -> None:
                    filename = f"seed-{seed}-yaw-{heading_degrees:03d}.png"
                    Image.fromarray(observation["pov"]).save(session_dir / filename)
                    views = capture["views"]
                    assert isinstance(views, list)
                    views.append({"heading_degrees": heading_degrees, "frame": filename})
                    _write_index(session_dir, manifest)

                save_view(0)
                if panorama:
                    for heading_degrees in (90, 180, 270):
                        for _ in range(3):
                            action = adapter.action_space.no_op()
                            action["camera"] = np.asarray([0.0, 30.0], dtype=np.float32)
                            action["attack"] = 0
                            action["jump"] = 0
                            action["sprint"] = 0
                            action["ESC"] = 0
                            observation = adapter.step(action).observation
                        save_view(heading_degrees)
                capture["status"] = "captured"
                _write_index(session_dir, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_index(session_dir, manifest)
        raise

    manifest["status"] = "completed"
    _write_index(session_dir, manifest)
    return session_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture reproducible MineRLBasaltFindCave starting views"
    )
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("runs/cave-starts"))
    parser.add_argument(
        "--panorama",
        action="store_true",
        help="capture yaw 0, 90, 180, and 270 views using safe camera-only turns",
    )
    args = parser.parse_args()

    session_dir = capture_starts(
        args.seed_start, args.count, args.output_root, panorama=args.panorama
    )
    print(json.dumps({"session_dir": str(session_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
