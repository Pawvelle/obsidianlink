"""D3-02 scene-validity debug — capture start frame and scripted walk.

NOT a model run. Starts the D3-02 approach scene, saves the RGB
the Agent would see on step 1, then walks forward for a fixed
number of steps and records evaluator-only distance to the lava
AABB. Confirms lava is centered, starts far, and that movement
updates hidden xyz.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/debug_d3_02_scenes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.env.d3_02_scene import (
    D3_02_ENV_ID,
    D3_02_GOAL_DISTANCE,
    D3_02_MIN_DISTANCE,
    D3_02_PLAYER_Z,
    distance_to_lava,
)
from obsidianlink.tasks.diagnostic import D3_02_WARMUP_STEPS


_DEFAULT_OUT = os.path.join(
    "obsidianlink",
    "experiments",
    "runs",
    "d3_02_scene_validity",
)
_WALK_STEPS = 20


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
    return {
        "label": label,
        "shape": list(arr.shape),
        "mean_rgb": [float(x) for x in arr.reshape(-1, 3).mean(axis=0)],
        "native_png": native_path,
        **_orange_stats(arr),
    }


def _pose_record(env: ControlledSceneEnv) -> dict[str, Any]:
    hidden = dict(env.hidden_state)
    xpos = hidden.get("xpos")
    zpos = hidden.get("zpos")
    dist = (
        distance_to_lava(float(xpos), float(zpos))
        if xpos is not None and zpos is not None
        else None
    )
    return {"hidden_state": hidden, "distance_to_lava": dist}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=_DEFAULT_OUT)
    parser.add_argument("--walk-steps", type=int, default=_WALK_STEPS)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print("ObsidianLink — D3-02 Target Approach scene-validity debug")
    print("  NO VLM is loaded. This only saves RGB and walks forward.")
    print(f"  env_id        : {D3_02_ENV_ID}")
    print(f"  spawn_z       : {D3_02_PLAYER_Z}")
    print(f"  warmup_steps  : {D3_02_WARMUP_STEPS}")
    print(f"  walk_steps    : {args.walk_steps}")
    print(f"  success band  : {D3_02_MIN_DISTANCE}–{D3_02_GOAL_DISTANCE}")
    sys.stdout.flush()

    env = ControlledSceneEnv(env_id=D3_02_ENV_ID, warmup_steps=D3_02_WARMUP_STEPS)
    t0 = time.perf_counter()
    try:
        observation = env.reset()
        start = _save_frame("start", observation.frame, env_dir := args.output_dir)
        start.update(_pose_record(env))
        start["reset_seconds"] = time.perf_counter() - t0
        print(f"\n  start distance : {start['distance_to_lava']!r}")
        print(f"  start hidden   : {start['hidden_state']}")
        print(f"  start orange nx: {start['orange_centroid_nx']!r}")
        sys.stdout.flush()

        walk_log: list[dict[str, Any]] = []
        forward = Action(type=ActionType.MOVE, dx=1, dz=0)
        for i in range(args.walk_steps):
            observation = env.step(forward)
            rec = _pose_record(env)
            rec["step"] = i + 1
            walk_log.append(rec)
            print(
                f"  step {i + 1:02d}  z={rec['hidden_state'].get('zpos')!r}  "
                f"d={rec['distance_to_lava']!r}"
            )
            sys.stdout.flush()

        after = _save_frame("after_walk", observation.frame, env_dir)
        after.update(_pose_record(env))
        in_band = [
            r["step"]
            for r in walk_log
            if r["distance_to_lava"] is not None
            and D3_02_MIN_DISTANCE <= r["distance_to_lava"] <= D3_02_GOAL_DISTANCE
        ]
        summary = {
            "env_id": D3_02_ENV_ID,
            "spawn_z": D3_02_PLAYER_Z,
            "warmup_steps": D3_02_WARMUP_STEPS,
            "goal_distance": D3_02_GOAL_DISTANCE,
            "min_distance": D3_02_MIN_DISTANCE,
            "start": start,
            "after_walk": after,
            "walk_log": walk_log,
            "steps_in_success_band": in_band,
            "hidden_xyz_updated": (
                start.get("distance_to_lava") is not None
                and after.get("distance_to_lava") is not None
                and after["distance_to_lava"] < start["distance_to_lava"] - 0.5
            ),
            "vlm_ran": False,
            "motor_execution": True,
            "design": "target_approach",
        }
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\n  after distance : {after['distance_to_lava']!r}")
        print(f"  after orange nx: {after['orange_centroid_nx']!r}")
        print(f"  band steps     : {in_band}")
        print(f"  xyz updated    : {summary['hidden_xyz_updated']}")
        print(f"\nsummary: {summary_path}")
        print("STOP. Inspect the PNGs; then run run_d3_02.py.")
        return 0
    finally:
        env.close()
        print("  env.close() ok")
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
