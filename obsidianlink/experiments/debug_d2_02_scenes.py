"""D2-02 scene-validity debug — capture 3×3 region frames.

NOT a model run. Starts each D2-02 spawn-pose scene, takes the
observation the Agent would see on step 1, and writes native
640×360 PNGs. Confirms lava is visually distinct in the intended
3×3 cell (orange centroid bin vs scene GT).

No camera or movement is applied: D2 Grounding is visual only.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/debug_d2_02_scenes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.env.d2_02_scene import (
    D2_02_ENV_IDS,
    D2_02_REGIONS,
    D2_02_SPAWN_POSES,
    d2_02_region_from_norm,
)
from obsidianlink.tasks.diagnostic import D2_02_WARMUP_STEPS


_DEFAULT_OUT = os.path.join(
    "obsidianlink",
    "experiments",
    "runs",
    "d2_02_region_validity",
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
    centroid_y = float(ys.mean()) if ys.size else None
    h, w = arr.shape[:2]
    nx = (centroid_x / w) if centroid_x is not None else None
    ny = (centroid_y / h) if centroid_y is not None else None
    return {
        "orange_fraction": frac,
        "orange_centroid_x": centroid_x,
        "orange_centroid_y": centroid_y,
        "orange_centroid_nx": nx,
        "orange_centroid_ny": ny,
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


def _capture_condition(region: str, out_dir: str) -> dict[str, Any]:
    env_id = D2_02_ENV_IDS[region]
    yaw, pitch = D2_02_SPAWN_POSES[region]
    print(f"\n=== capturing {region} ({env_id}) ===")
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D2_02_WARMUP_STEPS)
    t0 = time.perf_counter()
    try:
        observation = env.reset()
        elapsed = time.perf_counter() - t0
        stats = _save_frame(region, observation.frame, out_dir)
        stats["env_id"] = env_id
        stats["spawn_yaw"] = yaw
        stats["spawn_pitch"] = pitch
        stats["ground_truth_region"] = region
        nx = stats.get("orange_centroid_nx")
        ny = stats.get("orange_centroid_ny")
        observed = (
            d2_02_region_from_norm(nx, ny)
            if nx is not None and ny is not None
            else None
        )
        stats["observed_region_bin"] = observed
        stats["centroid_matches_gt"] = observed == region
        stats["hidden_state"] = dict(env.hidden_state)
        stats["reset_seconds"] = elapsed
        stats["warmup_steps"] = D2_02_WARMUP_STEPS
        with open(os.path.join(out_dir, f"{region}_stats.json"), "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        match = "MATCH" if stats["centroid_matches_gt"] else "MISMATCH"
        print(f"  gt_region        : {region}")
        print(f"  spawn yaw/pitch  : {yaw} / {pitch}")
        print(f"  orange_fraction  : {stats['orange_fraction']:.3f}")
        print(
            f"  centroid nx, ny  : {nx!r}, {ny!r}  "
            f"bin={observed!r}  {match}"
        )
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
    parser.add_argument(
        "--condition",
        choices=("all",) + D2_02_REGIONS,
        default="all",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    conditions = D2_02_REGIONS if args.condition == "all" else (args.condition,)
    print("ObsidianLink — D2-02 Spatial Region Grounding scene-validity debug")
    print("  NO VLM is loaded. This only saves the RGB the Agent would see.")
    print("  No camera / movement is applied.")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  warmup_steps  : {D2_02_WARMUP_STEPS}")
    print(f"  conditions    : {', '.join(conditions)}")
    sys.stdout.flush()

    records = [_capture_condition(c, args.output_dir) for c in conditions]
    n_match = sum(1 for r in records if r.get("centroid_matches_gt"))
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "scenes": records,
                "n_match": n_match,
                "n_scenes": len(records),
                "vlm_ran": False,
                "motor_execution": False,
                "design": "spatial_region_grounding",
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"\nsummary: {summary_path}")
    print(f"centroid bin match: {n_match}/{len(records)}")
    print("STOP. Inspect the PNGs; then run run_d2_02.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
