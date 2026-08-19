"""D2-01 scene-validity debug — capture left / center / right frames.

NOT a model run. Starts each D2-01 spawn-yaw scene, takes the
observation the Agent would see on step 1, and writes native
640×360 PNGs. Confirms lava is visually distinct on the left,
center, or right of the frame.

No camera or movement is applied: D2 Grounding is visual only.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/debug_d2_01_scenes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.env.d2_01_scene import (
    D2_01_CONDITIONS,
    D2_01_ENV_IDS,
    D2_01_SPAWN_YAWS,
)
from obsidianlink.tasks.diagnostic import D2_01_WARMUP_STEPS


_DEFAULT_OUT = os.path.join(
    "obsidianlink",
    "experiments",
    "runs",
    "d2_01_scene_validity",
)


def _as_uint8_hwc(frame: Any):
    import numpy as np

    if frame is None:
        raise RuntimeError("observation.frame is None; nothing to save")
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise RuntimeError(f"unexpected frame shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype("uint8")
    return arr


def _orange_mask(arr: Any):
    r = arr[..., 0].astype("int16")
    g = arr[..., 1].astype("int16")
    b = arr[..., 2].astype("int16")
    return (r > 140) & (r > g + 40) & (b < 80)


def _orange_stats(arr: Any) -> dict[str, Any]:
    import numpy as np

    mask = _orange_mask(arr)
    frac = float(mask.mean())
    ys, xs = np.where(mask)
    centroid_x = float(xs.mean()) if xs.size else None
    nx = (centroid_x / arr.shape[1]) if centroid_x is not None else None
    return {
        "orange_fraction": frac,
        "orange_centroid_x": centroid_x,
        "orange_centroid_nx": nx,
    }


def _save_frame(label: str, frame: Any, out_dir: str) -> dict[str, Any]:
    from PIL import Image

    arr = _as_uint8_hwc(frame)
    h, w = arr.shape[:2]
    native_path = os.path.join(out_dir, f"{label}_native_{w}x{h}.png")
    Image.fromarray(arr).save(native_path)
    stats = {
        "label": label,
        "shape": list(arr.shape),
        "mean_rgb": [float(x) for x in arr.reshape(-1, 3).mean(axis=0)],
        "native_png": native_path,
        **_orange_stats(arr),
    }
    return stats


def _capture_condition(condition: str, out_dir: str) -> dict[str, Any]:
    env_id = D2_01_ENV_IDS[condition]
    print(f"\n=== capturing {condition} ({env_id}) ===")
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D2_01_WARMUP_STEPS)
    t0 = time.perf_counter()
    try:
        observation = env.reset()
        elapsed = time.perf_counter() - t0
        stats = _save_frame(condition, observation.frame, out_dir)
        stats["env_id"] = env_id
        stats["spawn_yaw"] = D2_01_SPAWN_YAWS[condition]
        stats["ground_truth_direction"] = condition
        stats["hidden_state"] = dict(env.hidden_state)
        stats["reset_seconds"] = elapsed
        stats["warmup_steps"] = D2_01_WARMUP_STEPS
        with open(os.path.join(out_dir, f"{condition}_stats.json"), "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        print(f"  gt_direction     : {condition}")
        print(f"  spawn_yaw        : {stats['spawn_yaw']}")
        print(f"  orange_fraction  : {stats['orange_fraction']:.3f}")
        print(f"  orange_centroid  : {stats['orange_centroid_nx']!r}")
        print(f"  native           : {stats['native_png']}")
        sys.stdout.flush()
        return stats
    finally:
        env.close()
        print("  env.close() ok")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print("ObsidianLink — D2-01 Direction Grounding scene-validity debug")
    print("  NO VLM is loaded. This only saves the RGB the Agent would see.")
    print("  No camera / movement is applied.")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  warmup_steps  : {D2_01_WARMUP_STEPS}")
    sys.stdout.flush()

    records = [_capture_condition(c, args.output_dir) for c in D2_01_CONDITIONS]
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "scenes": records,
                "vlm_ran": False,
                "motor_execution": False,
                "design": "direction_grounding",
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"\nsummary: {summary_path}")
    print("STOP. Inspect the PNGs; then run run_d2_01.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
