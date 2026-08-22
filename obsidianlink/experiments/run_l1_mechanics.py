"""L1 mechanical interaction live test.

Scripted validation that Formal L1 (`minedojo_l1_portal`) can scoop
lava with an empty bucket, place lava/water via native USE, form a new
obsidian block, place cobblestone, and mine it with an iron pickaxe.

Not an Oracle, Evaluator, or ReactiveAgent run.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_mechanics.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import L1_ENV_ID, L1ControlledEnv
from obsidianlink.experiments.l1_mechanics import (
    HOTBAR_BUCKET,
    HOTBAR_COBBLE,
    HOTBAR_PICKAXE,
    HOTBAR_WATER,
    cobble_broken,
    cobble_placed,
    frame_stats,
    new_obsidian_from_evidence,
    poured_lava,
    qty,
    scooped_lava,
    used_water,
)

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
SCOOP_Z = 4.15
MAX_FORWARD = 36
LOOK_LAVA_PITCH = 45.0
LOOK_DOWN_PITCH = 58.0
LOOK_COBBLE_PITCH = 42.0
# North of spawn, away from the lava pool (z=5..8) and the flooded pour site.
DRY_PAD_X = 0.5
DRY_PAD_Z = -1.0


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


def _yaw_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


class MechanicsSession:
    def __init__(self, env: L1ControlledEnv, frames_dir: str) -> None:
        self.env = env
        self.frames_dir = frames_dir
        self.steps = 0
        self.action_log: list[dict[str, Any]] = []
        self.failed_at: str | None = None

    def snap(self) -> dict[str, Any]:
        obs = self.env.observe()
        return {
            "inventory": dict(obs.inventory or {}),
            "selected_item": obs.selected_item,
            "visual": frame_stats(obs.frame),
            "pose": _pose(self.env.hidden_state),
        }

    def save(self, name: str) -> None:
        _save_frame(os.path.join(self.frames_dir, name), self.env.observe().frame)

    def step(self, action: Action) -> dict[str, Any]:
        if action.type is ActionType.EQUIP:
            raise RuntimeError("EquipAction is forbidden on L1 mechanics test")
        self.env.step(action)
        self.steps += 1
        recorded = {
            "step": self.steps,
            "type": action.type.value,
            "dx": action.dx,
            "dz": action.dz,
            "yaw": action.yaw,
            "pitch": action.pitch,
            "target": action.target,
            "sneak": action.sneak,
        }
        after = self.snap()
        recorded["after"] = {
            "inventory": after["inventory"],
            "selected_item": after["selected_item"],
            "pose": after["pose"],
        }
        self.action_log.append(recorded)
        return after

    def wait(self, n: int) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(n):
            snap = self.step(Action(type=ActionType.WAIT))
        return snap

    def hotbar(self, slot: str) -> dict[str, Any]:
        from obsidianlink.env.l1_scene import l1_equip_target

        inventory = dict(self.snap().get("inventory") or {})
        snap = self.step(
            Action(type=ActionType.EQUIP, target=l1_equip_target(slot, inventory))
        )
        return self.wait(2)

    def use(self, ticks: int = 4, *, sneak: bool = False) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(ticks):
            snap = self.step(Action(type=ActionType.USE, sneak=sneak))
        return snap

    def attack(self, ticks: int, *, sneak: bool = False) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(ticks):
            snap = self.step(Action(type=ActionType.ATTACK, sneak=sneak))
        return snap

    def walk_toward(
        self, x_target: float, z_target: float, max_steps: int = 80
    ) -> dict[str, Any]:
        """Scripted navigation from evaluator-only location_stats. Not an Agent."""
        self.look_pitch(12.0)
        snap = self.snap()
        for _ in range(max_steps):
            pose = snap["pose"]
            x = pose.get("xpos")
            z = pose.get("zpos")
            if x is None or z is None:
                snap = self.step(Action(type=ActionType.MOVE, dx=1))
                continue
            dx = float(x_target) - float(x)
            dz = float(z_target) - float(z)
            if dx * dx + dz * dz < 0.45:
                return snap
            desired_yaw = math.degrees(math.atan2(-dx, dz))
            current_yaw = pose.get("yaw")
            if current_yaw is None or abs(_yaw_delta(desired_yaw, float(current_yaw))) > 8.0:
                snap = self.turn_yaw(desired_yaw, max_tries=6)
            snap = self.step(Action(type=ActionType.MOVE, dx=1, sneak=True))
        return snap

    def look_pitch(self, target: float, max_tries: int = 8) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(max_tries):
            current = snap["pose"].get("pitch")
            if current is None:
                snap = self.step(
                    Action(type=ActionType.CAMERA, pitch=target - 25.0)
                )
                break
            delta = target - float(current)
            if abs(delta) < 1.5:
                break
            snap = self.step(
                Action(
                    type=ActionType.CAMERA,
                    pitch=max(-30.0, min(30.0, delta)),
                )
            )
        return snap

    def turn_yaw(self, target: float, max_tries: int = 10) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(max_tries):
            current = snap["pose"].get("yaw")
            if current is None:
                snap = self.step(Action(type=ActionType.CAMERA, yaw=target))
                break
            delta = _yaw_delta(target, float(current))
            if abs(delta) < 3.0:
                break
            snap = self.step(
                Action(
                    type=ActionType.CAMERA,
                    yaw=max(-30.0, min(30.0, delta)),
                )
            )
        return snap

    def walk_forward_until_z(self, z_min: float, max_steps: int = MAX_FORWARD) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(max_steps):
            zpos = snap["pose"].get("zpos")
            if zpos is not None and float(zpos) >= z_min:
                return snap
            if zpos is not None and float(zpos) >= 4.55:
                return snap
            snap = self.step(Action(type=ActionType.MOVE, dx=1))
        return snap


def _fail(report: dict[str, Any], gate: str, detail: dict[str, Any]) -> None:
    report["failed_at"] = gate
    report["checks"][gate] = {"ok": False, **detail}
    print(f"[l1-mech] FAIL {gate} {detail}")
    sys.stdout.flush()


def _ok(report: dict[str, Any], gate: str, detail: dict[str, Any]) -> None:
    report["checks"][gate] = {"ok": True, **detail}
    print(f"[l1-mech] OK {gate}")
    sys.stdout.flush()


def run_mechanics(session: MechanicsSession, report: dict[str, Any]) -> None:
    keys = set(session.env.action_space_keys or ())
    report["action_space_keys"] = sorted(keys)
    if "equip" in keys:
        _fail(report, "no_equip", {"keys": sorted(keys)})
        return
    _ok(report, "event_actions", {"has_equip": "equip" in keys or not keys})
    report["used_observation_from_grid"] = False

    print("[l1-mech] reset")
    sys.stdout.flush()
    session.save("00_reset.png")
    reset = session.snap()
    report["reset"] = reset
    if qty(reset["inventory"], "lava_bucket") != 0:
        _fail(report, "start_inventory", {"got": reset["inventory"]})
        return
    start_ok = (
        qty(reset["inventory"], "bucket") >= 1
        and qty(reset["inventory"], "water_bucket") >= 1
        and qty(reset["inventory"], "cobblestone") >= 1
        and qty(reset["inventory"], "iron_pickaxe") >= 1
    )
    if not start_ok:
        _fail(report, "start_inventory", {"got": reset["inventory"]})
        return
    _ok(report, "start_inventory", {"got": reset["inventory"]})

    session.hotbar(HOTBAR_BUCKET)
    session.walk_forward_until_z(SCOOP_Z)
    session.look_pitch(LOOK_LAVA_PITCH)
    session.wait(2)
    session.save("01_aimed_lava.png")
    before_scoop = session.snap()
    report["before_scoop"] = before_scoop
    scooped = False
    selected_lava = False
    after_scoop = before_scoop
    for pitch in (LOOK_LAVA_PITCH, 40.0, 48.0, 52.0, 36.0):
        session.look_pitch(pitch)
        session.use(3)
        session.wait(5)
        after_scoop = session.snap()
        scooped = scooped_lava(before_scoop["inventory"], after_scoop["inventory"])
        selected_lava = after_scoop["selected_item"] == "lava_bucket"
        if scooped:
            break
        session.use(3, sneak=True)
        session.wait(5)
        after_scoop = session.snap()
        scooped = scooped_lava(before_scoop["inventory"], after_scoop["inventory"])
        selected_lava = after_scoop["selected_item"] == "lava_bucket"
        if scooped:
            break
    report["after_scoop"] = after_scoop
    session.save("02_after_scoop.png")
    if not scooped:
        _fail(
            report,
            "scoop_lava",
            {
                "before": before_scoop["inventory"],
                "after": after_scoop["inventory"],
                "selected_item": after_scoop["selected_item"],
                "pose": after_scoop["pose"],
                "note": "crosshair must hit lava source, not the grass rim",
            },
        )
        return
    _ok(
        report,
        "scoop_lava",
        {
            "before": before_scoop["inventory"],
            "after": after_scoop["inventory"],
            "selected_item": after_scoop["selected_item"],
            "selected_is_lava_bucket": selected_lava,
        },
    )

    # Leave the pool, face construction grass (west, -X), look down.
    for _ in range(8):
        session.step(Action(type=ActionType.MOVE, dx=-1))
    session.turn_yaw(90.0)
    for _ in range(6):
        session.step(Action(type=ActionType.MOVE, dx=1))
    session.look_pitch(LOOK_DOWN_PITCH)
    session.wait(2)
    session.save("03_before_pour.png")
    before_pour = session.snap()
    report["before_pour"] = before_pour
    session.use(4, sneak=True)
    session.wait(8)
    after_pour = session.snap()
    report["after_pour"] = after_pour
    session.save("04_after_lava_place.png")
    poured = poured_lava(before_pour["inventory"], after_pour["inventory"])
    if not poured:
        session.look_pitch(LOOK_DOWN_PITCH + 10.0)
        session.use(5, sneak=True)
        session.wait(8)
        after_pour = session.snap()
        report["after_pour_retry"] = after_pour
        session.save("04b_pour_retry.png")
        poured = poured_lava(before_pour["inventory"], after_pour["inventory"])
    if not poured:
        _fail(
            report,
            "place_lava",
            {"before": before_pour["inventory"], "after": after_pour["inventory"]},
        )
        return
    _ok(
        report,
        "place_lava",
        {
            "before": before_pour["inventory"],
            "after": after_pour["inventory"],
            "visual": after_pour["visual"],
        },
    )

    session.hotbar(HOTBAR_WATER)
    session.look_pitch(LOOK_DOWN_PITCH)
    before_water = session.snap()
    report["before_water"] = before_water
    session.use(4, sneak=True)
    session.wait(12)
    after_water = session.snap()
    report["after_water"] = after_water
    session.save("05_after_water.png")
    watered = used_water(before_water["inventory"], after_water["inventory"])
    if not watered:
        session.use(4, sneak=True)
        session.wait(12)
        after_water = session.snap()
        report["after_water_retry"] = after_water
        session.save("05b_water_retry.png")
        watered = used_water(before_water["inventory"], after_water["inventory"])
    if not watered:
        _fail(
            report,
            "place_water",
            {"before": before_water["inventory"], "after": after_water["inventory"]},
        )
        return
    _ok(
        report,
        "place_water",
        {"before": before_water["inventory"], "after": after_water["inventory"]},
    )

    session.wait(8)
    after_cast = session.snap()
    report["after_cast"] = after_cast
    session.save("06_after_obsidian.png")
    obsidian = new_obsidian_from_evidence(
        scooped=scooped,
        poured=poured,
        watered=watered,
        visual_before_water=after_pour["visual"],
        visual_after_water=after_cast["visual"],
    )
    report["new_obsidian"] = obsidian
    if not obsidian["ok"]:
        _fail(report, "new_obsidian", obsidian)
        return
    _ok(report, "new_obsidian", obsidian)

    # Leave the flooded pour site. Flowing water cannot be bucketed and
    # pushes the player off the cobble crosshair.
    session.hotbar(HOTBAR_WATER)
    session.look_pitch(LOOK_DOWN_PITCH)
    session.use(4, sneak=True)
    session.wait(4)
    session.save("06b_after_water_pickup.png")
    session.walk_toward(DRY_PAD_X, DRY_PAD_Z)
    session.look_pitch(LOOK_DOWN_PITCH)
    session.wait(2)
    session.save("06c_dry_pad.png")
    report["dry_pad"] = session.snap()

    session.hotbar(HOTBAR_COBBLE)
    session.look_pitch(LOOK_DOWN_PITCH)
    before_cobble = session.snap()
    report["before_cobble"] = before_cobble
    session.use(4, sneak=True)
    session.wait(6)
    after_cobble = session.snap()
    report["after_cobble"] = after_cobble
    session.save("07_after_cobble_place.png")
    placed = cobble_placed(before_cobble["inventory"], after_cobble["inventory"])
    if not placed:
        session.look_pitch(LOOK_DOWN_PITCH + 8.0)
        session.use(5, sneak=True)
        session.wait(6)
        after_cobble = session.snap()
        report["after_cobble_retry"] = after_cobble
        session.save("07b_cobble_retry.png")
        placed = cobble_placed(before_cobble["inventory"], after_cobble["inventory"])
    if not placed:
        _fail(
            report,
            "place_cobble",
            {
                "before": before_cobble["inventory"],
                "after": after_cobble["inventory"],
                "note": "native USE only; PlaceBlock not enabled",
            },
        )
        return
    _ok(
        report,
        "place_cobble",
        {"before": before_cobble["inventory"], "after": after_cobble["inventory"]},
    )

    # Step back so the new block is in front, then aim at its face.
    for _ in range(2):
        session.step(Action(type=ActionType.MOVE, dx=-1, sneak=True))
    session.hotbar(HOTBAR_PICKAXE)
    session.look_pitch(LOOK_COBBLE_PITCH)
    session.wait(2)
    session.save("07c_aimed_cobble.png")
    before_mine = session.snap()
    report["before_mine"] = before_mine
    after_mine = before_mine
    for pitch, yaw_nudge in (
        (LOOK_COBBLE_PITCH, 0.0),
        (LOOK_DOWN_PITCH, 0.0),
        (36.0, 0.0),
        (LOOK_COBBLE_PITCH, 12.0),
        (LOOK_COBBLE_PITCH, -12.0),
        (50.0, 0.0),
    ):
        if yaw_nudge:
            session.step(Action(type=ActionType.CAMERA, yaw=yaw_nudge))
        session.look_pitch(pitch)
        session.attack(24, sneak=True)
        session.wait(6)
        for _ in range(2):
            session.step(Action(type=ActionType.MOVE, dx=1, sneak=True))
        session.wait(6)
        after_mine = session.snap()
        if cobble_broken(before_mine["inventory"], after_mine["inventory"]):
            break
    report["after_mine"] = after_mine
    session.save("08_after_cobble_break.png")
    broken = cobble_broken(before_mine["inventory"], after_mine["inventory"])
    if not broken:
        _fail(
            report,
            "break_cobble",
            {
                "before": before_mine["inventory"],
                "after": after_mine["inventory"],
                "pose": after_mine["pose"],
                "note": "require cobblestone count to rise after mining (drop pickup)",
            },
        )
        return
    _ok(
        report,
        "break_cobble",
        {
            "inventory_picked_up": True,
            "before": before_mine["inventory"],
            "after": after_mine["inventory"],
        },
    )


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    frames_dir = os.path.join(_RUNS_DIR, f"l1_mechanics_{stamp}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "l1_mechanical_interaction",
        "env_id": L1_ENV_ID,
        "started_utc": stamp,
        "prebuilt_portal_frame": False,
        "used_equip_action": False,
        "used_observation_from_grid": False,
        "used_placeblock": False,
        "drawblock_obsidian": False,
        "preloaded_lava_bucket": False,
        "oracle_or_agent_run": False,
        "checks": {},
        "limitations": [],
        "error": None,
        "failed_at": None,
        "new_obsidian": {"ok": False},
    }
    env: L1ControlledEnv | None = None
    session: MechanicsSession | None = None
    t0 = time.perf_counter()
    try:
        env = L1ControlledEnv()
        env.reset()
        session = MechanicsSession(env, frames_dir)
        run_mechanics(session, report)
        report["action_sequence"] = session.action_log
        report["steps"] = session.steps
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
    checks = report.get("checks") or {}
    report["scooped_lava"] = bool(checks.get("scoop_lava", {}).get("ok"))
    report["placed_lava"] = bool(checks.get("place_lava", {}).get("ok"))
    report["placed_water"] = bool(checks.get("place_water", {}).get("ok"))
    report["observed_new_obsidian"] = bool(
        (report.get("new_obsidian") or {}).get("ok")
        or checks.get("new_obsidian", {}).get("ok")
    )
    report["placed_cobblestone"] = bool(checks.get("place_cobble", {}).get("ok"))
    report["broke_cobblestone"] = bool(checks.get("break_cobble", {}).get("ok"))
    report["success"] = (
        report.get("error") is None
        and report["failed_at"] is None
        and report["observed_new_obsidian"]
        and report["placed_cobblestone"]
        and report["broke_cobblestone"]
    )
    out_path = os.path.join(_RUNS_DIR, f"l1_mechanics_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    print("[l1-mech] wrote", out_path)
    print(
        json.dumps(
            _jsonish(
                {
                    "success": report["success"],
                    "failed_at": report["failed_at"],
                    "scooped_lava": report["scooped_lava"],
                    "placed_lava": report["placed_lava"],
                    "placed_water": report["placed_water"],
                    "observed_new_obsidian": report["observed_new_obsidian"],
                    "placed_cobblestone": report["placed_cobblestone"],
                    "broke_cobblestone": report["broke_cobblestone"],
                    "error": report.get("error"),
                    "limitations": report.get("limitations"),
                    "wall_time": report.get("wall_time"),
                }
            ),
            indent=2,
        )
    )
    sys.stdout.flush()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
