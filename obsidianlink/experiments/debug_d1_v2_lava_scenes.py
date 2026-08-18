"""D1-01 scene-validity debug — capture the exact VLM input frames.

NOT a model run. This script starts the D1-01 Lava Presence
positive and negative Minecraft scenes, takes the same observation
the Agent would receive on its single step, and writes:

* ``<label>_native_<W>x<H>.png`` — the exact RGB bytes that would
  be sent to the VLM (D1-01 POV is 640×360).
* ``<label>_upscaled.png`` — nearest-neighbour upscale, for human
  inspection only. The VLM never sees this.
* ``<label>_stats.json`` — shape / mean RGB / a coarse orange-pixel
  fraction (debug aid, not ground truth).

Old 64×64 captures under ``d1_v2_scene_validity/`` are kept as
pilot artefacts and are not overwritten.

Usage
-----

::

    PYTHONPATH=/Users/joey/Documents/Projects/ObsidianLink \\
        /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/debug_d1_v2_lava_scenes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_ENV_IDS,
    D1_01_WARMUP_STEPS,
)


_DEFAULT_OUT = os.path.join(
    "obsidianlink",
    "experiments",
    "runs",
    "d1_01_scene_validity",
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


def _orange_fraction(arr: Any) -> float:
    """Loose lava-like mask: high red, red dominates green, low blue.

    Debug-only. Not used by the Evaluator.
    """
    r = arr[..., 0].astype("int16")
    g = arr[..., 1].astype("int16")
    b = arr[..., 2].astype("int16")
    mask = (r > 140) & (r > g + 40) & (b < 80)
    return float(mask.mean())


def _save_scene(label: str, frame: Any, out_dir: str) -> dict[str, Any]:
    from PIL import Image

    arr = _as_uint8_hwc(frame)
    h, w = arr.shape[:2]
    native_path = os.path.join(out_dir, f"{label}_native_{w}x{h}.png")
    upscaled_path = os.path.join(out_dir, f"{label}_upscaled.png")
    stats_path = os.path.join(out_dir, f"{label}_stats.json")

    Image.fromarray(arr).save(native_path)
    # 64×64 used ×8; 640×360 is already inspectable at ×2.
    scale = 8 if max(h, w) <= 64 else 2
    Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST).save(
        upscaled_path
    )

    stats = {
        "label": label,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean_rgb": [float(x) for x in arr.reshape(-1, 3).mean(axis=0)],
        "min_rgb": [int(x) for x in arr.reshape(-1, 3).min(axis=0)],
        "max_rgb": [int(x) for x in arr.reshape(-1, 3).max(axis=0)],
        "orange_fraction": _orange_fraction(arr),
        "native_png": native_path,
        "upscaled_png": upscaled_path,
        "note": (
            "native_png is the exact VLM input. "
            "upscaled_png is human-only (nearest-neighbour)."
        ),
    }
    # numpy types can sneak into mean_rgb; json.dump default=str is enough
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats


def _capture(label: str, env_id: str, out_dir: str) -> dict[str, Any]:
    print(f"\n=== capturing {label} ({env_id}) ===")
    sys.stdout.flush()
    env = ControlledSceneEnv(env_id=env_id, warmup_steps=D1_01_WARMUP_STEPS)
    t0 = time.perf_counter()
    try:
        observation = env.reset()
        elapsed = time.perf_counter() - t0
        print(f"  reset+warmup ({D1_01_WARMUP_STEPS} ticks) in {elapsed:.1f}s")
        stats = _save_scene(label, observation.frame, out_dir)
        stats["env_id"] = env_id
        stats["warmup_steps"] = D1_01_WARMUP_STEPS
        stats["reset_seconds"] = elapsed
        stats["inventory"] = dict(observation.inventory or {})
        stats["selected_item"] = observation.selected_item
        # Re-write stats with the extra env fields.
        with open(os.path.join(out_dir, f"{label}_stats.json"), "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        print(f"  shape            : {stats['shape']}")
        print(f"  mean_rgb         : {stats['mean_rgb']}")
        print(f"  orange_fraction  : {stats['orange_fraction']:.3f}")
        print(f"  native           : {stats['native_png']}")
        print(f"  upscaled         : {stats['upscaled_png']}")
        sys.stdout.flush()
        return stats
    finally:
        env.close()
        print("  env.close() ok")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUT,
        help="Directory for the captured frames (created if missing).",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print("ObsidianLink — D1 v2 Lava Presence scene-validity debug")
    print("  NO VLM is loaded. NO prompt is sent. NO evaluation.")
    print("  This only saves the RGB the Agent would see at max_steps=1.")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  warmup_steps  : {D1_01_WARMUP_STEPS}")
    print(f"  positive env  : {D1_01_LAVA_ENV_IDS['positive']}")
    print(f"  negative env  : {D1_01_LAVA_ENV_IDS['negative']}")
    sys.stdout.flush()

    records = []
    for label in ("positive", "negative"):
        records.append(_capture(label, D1_01_LAVA_ENV_IDS[label], args.output_dir))

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"scenes": records, "vlm_ran": False}, fh, indent=2, default=str)
    print(f"\nsummary: {summary_path}")
    print("STOP. Inspect the PNGs; then run run_d1_01.py for 2B / 4B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
