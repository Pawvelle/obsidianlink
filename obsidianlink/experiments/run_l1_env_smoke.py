"""L1 controlled environment v0.1 live smoke test.

Validates reset, spawn, lava pool visibility, inventory, hotbar select,
RGB/inventory/selected_item, close, and a fresh reset.

This is not a portal solver, Oracle, or ReactiveAgent run.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_env_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import (
    L1_ENV_ID,
    L1_INVENTORY,
    PLAYER_Y,
    L1ControlledEnv,
)

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

REQUIRED_INVENTORY = {
    item["type"]: int(item["quantity"]) for item in L1_INVENTORY.values()
}
HOTBAR_EXPECT = {
    1: "water_bucket",
    2: "bucket",
    3: "cobblestone",
    4: "iron_pickaxe",
    5: "flint_and_steel",
}


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return _jsonish(value.tolist())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _frame_stats(frame: Any) -> dict[str, Any]:
    if frame is None:
        return {"present": False}
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {"present": bool(arr.size), "shape": list(arr.shape)}
    h, w = arr.shape[:2]
    region = arr[int(h * 0.40) : int(h * 0.88), int(w * 0.30) : int(w * 0.70)]
    r = region[:, :, 0].astype(np.float32)
    g = region[:, :, 1].astype(np.float32)
    b = region[:, :, 2].astype(np.float32)
    lava = (r > 140) & (g > 50) & (b < 90) & (r > g) & (g >= b * 0.8)
    portal = (r > 70) & (b > 90) & (g < 55) & (b > g) & (r > g)
    # Lower FOV: grass is green; the old obsidian courtyard was dark purple.
    ground = arr[int(h * 0.62) : int(h * 0.92), int(w * 0.15) : int(w * 0.85)]
    gr = ground[:, :, 0].astype(np.float32)
    gg = ground[:, :, 1].astype(np.float32)
    gb = ground[:, :, 2].astype(np.float32)
    grass = (gg > 70) & (gg > gr) & (gg > gb)
    obsidian = (gr < 45) & (gg < 35) & (gb < 50) & (gb >= gr * 0.7)
    return {
        "present": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "frame_mean": float(arr.mean()),
        "lava_frac": float(lava.mean()),
        "portal_frac": float(portal.mean()),
        "grass_frac": float(grass.mean()),
        "obsidian_frac": float(obsidian.mean()),
        "region_mean_r": float(r.mean()),
        "region_mean_g": float(g.mean()),
        "region_mean_b": float(b.mean()),
    }


def _save_frame(path: str, frame: Any) -> bool:
    if frame is None:
        return False
    try:
        from PIL import Image

        arr = np.asarray(frame)
        if arr.ndim != 3:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(arr.astype(np.uint8)).save(path)
        return True
    except Exception:
        return False


def _inventory_ok(inventory: dict[str, int] | None) -> tuple[bool, dict[str, Any]]:
    got = dict(inventory or {})
    missing = []
    wrong = {}
    for name, qty in REQUIRED_INVENTORY.items():
        actual = int(got.get(name, 0))
        if actual < qty:
            missing.append(name)
            wrong[name] = {"expected": qty, "got": actual}
    extra_lava = int(got.get("lava_bucket", 0))
    ok = not missing and extra_lava == 0
    return ok, {
        "ok": ok,
        "got": got,
        "missing": missing,
        "mismatches": wrong,
        "lava_bucket": extra_lava,
    }


def _pose(hidden: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("xpos", "ypos", "zpos", "yaw", "pitch"):
        val = hidden.get(key)
        if val is None:
            continue
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _near_spawn(pose: dict[str, float]) -> bool:
    if "xpos" not in pose or "ypos" not in pose or "zpos" not in pose:
        return False
    dx = abs(pose["xpos"] - 0.5)
    dy = abs(pose["ypos"] - float(PLAYER_Y))
    dz = abs(pose["zpos"] - 0.5)
    return dx < 2.5 and dy < 2.5 and dz < 2.5


def _snapshot(env: L1ControlledEnv) -> dict[str, Any]:
    obs = env.observe()
    return {
        "inventory": dict(obs.inventory or {}),
        "selected_item": obs.selected_item,
        "visual": _frame_stats(obs.frame),
        "pose": _pose(env.hidden_state),
        "action_space_keys": list(env.action_space_keys or ()),
    }


def _check_checks(checks: dict[str, Any]) -> bool:
    return all(bool(item.get("ok")) for item in checks.values())


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    frames_dir = os.path.join(_RUNS_DIR, f"l1_env_smoke_{stamp}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)

    report: dict[str, Any] = {
        "kind": "l1_controlled_env_v0.1_smoke",
        "env_id": L1_ENV_ID,
        "started_utc": stamp,
        "prebuilt_portal_frame": False,
        "used_equip_action": False,
        "used_observation_from_grid": False,
        "oracle_or_agent_run": False,
        "checks": {},
        "limitations": [],
        "error": None,
    }

    env: L1ControlledEnv | None = None
    t0 = time.perf_counter()
    try:
        env = L1ControlledEnv()
        print("[l1-smoke] reset 1")
        sys.stdout.flush()
        obs = env.reset()
        snap = _snapshot(env)
        _save_frame(os.path.join(frames_dir, "00_reset.png"), obs.frame)
        report["reset1"] = snap
        keys = set(snap["action_space_keys"])
        report["action_space_keys"] = sorted(keys)

        inv_ok, inv_detail = _inventory_ok(obs.inventory)
        rgb_ok = bool(snap["visual"].get("present")) and snap["visual"].get("shape")
        pose = snap["pose"]
        spawn_ok = _near_spawn(pose)
        lava_frac = float(snap["visual"].get("lava_frac") or 0.0)
        portal_frac = float(snap["visual"].get("portal_frac") or 0.0)

        report["checks"]["A_reset"] = {
            "ok": obs.frame is not None,
            "selected_item": obs.selected_item,
        }
        report["checks"]["B_spawn"] = {"ok": spawn_ok, "pose": pose}
        report["checks"]["D_inventory"] = inv_detail
        report["checks"]["F_rgb"] = {
            "ok": bool(rgb_ok),
            "visual": snap["visual"],
        }
        report["checks"]["no_equip_in_action_space"] = {
            "ok": "equip" not in keys,
            "has_hotbar": all(f"hotbar.{i}" in keys for i in range(1, 10)),
        }
        report["checks"]["no_prebuilt_portal_visual"] = {
            "ok": portal_frac < 0.01,
            "portal_frac": portal_frac,
        }

        if lava_frac < 0.01:
            print("[l1-smoke] lava_frac low; pitching down")
            sys.stdout.flush()
            env.step(Action(type=ActionType.CAMERA, pitch=20.0))
            env.step(Action(type=ActionType.WAIT))
            obs = env.observe()
            snap = _snapshot(env)
            lava_frac = float(snap["visual"].get("lava_frac") or 0.0)
            _save_frame(os.path.join(frames_dir, "01_look_down.png"), obs.frame)
            report["after_look_down"] = snap

        report["checks"]["C_lava_visible"] = {
            "ok": lava_frac >= 0.01,
            "lava_frac": lava_frac,
        }
        grass_frac = float(snap["visual"].get("grass_frac") or 0.0)
        obsidian_frac = float(snap["visual"].get("obsidian_frac") or 0.0)
        report["checks"]["C2_grass_floor"] = {
            "ok": grass_frac > obsidian_frac and obsidian_frac < 0.35,
            "grass_frac": grass_frac,
            "obsidian_frac": obsidian_frac,
            "floor_surface": "grass",
        }

        print("[l1-smoke] hotbar select")
        sys.stdout.flush()
        hotbar_results: dict[str, Any] = {}
        hotbar_ok = True
        for slot, expected in HOTBAR_EXPECT.items():
            env.step(Action(type=ActionType.HOTBAR, target=str(slot)))
            env.step(Action(type=ActionType.WAIT))
            env.step(Action(type=ActionType.WAIT))
            cur = env.observe()
            selected = cur.selected_item
            matched = selected == expected
            hotbar_results[f"hotbar.{slot}"] = {
                "expected": expected,
                "selected_item": selected,
                "ok": matched,
            }
            if not matched:
                hotbar_ok = False
            _save_frame(
                os.path.join(frames_dir, f"hotbar_{slot}_{expected}.png"),
                cur.frame,
            )
        report["checks"]["E_hotbar"] = {"ok": hotbar_ok, "slots": hotbar_results}
        report["checks"]["F_selected_item"] = {
            "ok": hotbar_ok,
            "note": "selected_item tracked EquippedItemObservation after hotbar keys",
        }

        print("[l1-smoke] small move")
        sys.stdout.flush()
        env.step(Action(type=ActionType.MOVE, dz=-1))
        env.step(Action(type=ActionType.WAIT))
        moved = _snapshot(env)
        _save_frame(os.path.join(frames_dir, "02_after_strafe.png"), env.observe().frame)
        report["after_strafe"] = moved

        print("[l1-smoke] close then fresh reset")
        sys.stdout.flush()
        env.close()
        report["checks"]["G_close"] = {"ok": True}

        env = L1ControlledEnv()
        obs2 = env.reset()
        snap2 = _snapshot(env)
        _save_frame(os.path.join(frames_dir, "03_fresh_reset.png"), obs2.frame)
        inv2_ok, inv2_detail = _inventory_ok(obs2.inventory)
        lava2 = float(snap2["visual"].get("lava_frac") or 0.0)
        if lava2 < 0.01:
            env.step(Action(type=ActionType.CAMERA, pitch=20.0))
            env.step(Action(type=ActionType.WAIT))
            snap2 = _snapshot(env)
            lava2 = float(snap2["visual"].get("lava_frac") or 0.0)
            _save_frame(os.path.join(frames_dir, "03b_fresh_look_down.png"), env.observe().frame)
        report["reset2"] = snap2
        report["checks"]["H_fresh_reset"] = {
            "ok": inv2_ok and _near_spawn(snap2["pose"]) and lava2 >= 0.01,
            "inventory": inv2_detail,
            "pose": snap2["pose"],
            "lava_frac": lava2,
            "grass_frac": snap2["visual"].get("grass_frac"),
            "obsidian_frac": snap2["visual"].get("obsidian_frac"),
        }
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["limitations"].append(report["error"])
        print(report["traceback"])
        sys.stdout.flush()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["limitations"].append(f"close: {type(exc).__name__}: {exc}")

    report["wall_time"] = time.perf_counter() - t0
    report["frames_dir"] = frames_dir
    report["success"] = report.get("error") is None and _check_checks(report["checks"])
    out_path = os.path.join(_RUNS_DIR, f"l1_env_smoke_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    print("[l1-smoke] wrote", out_path)
    print(
        json.dumps(
            _jsonish(
                {
                    "success": report["success"],
                    "checks": {
                        name: {"ok": item.get("ok")}
                        for name, item in report.get("checks", {}).items()
                    },
                    "error": report.get("error"),
                    "limitations": report.get("limitations"),
                    "wall_time": report.get("wall_time"),
                    "action_space_keys": report.get("action_space_keys"),
                }
            ),
            indent=2,
        )
    )
    sys.stdout.flush()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
