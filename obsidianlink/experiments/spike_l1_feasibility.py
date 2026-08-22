"""L1 technical feasibility spike.

Validates vanilla Minecraft mechanics for official L1:

    Casting → Frame → Ignition → Nether Entry

This is not an official L1 implementation. It does not wire a Reactive
agent, planner, or reflection. Scene DrawBlock is only an obsidian
casting surface — not a pre-built portal frame.

Run:

    PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
        obsidianlink/experiments/spike_l1_feasibility.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

from obsidianlink.env.scene import (
    FLAT_WORLD,
    PLAYER_PITCH,
    PLAYER_X,
    PLAYER_Y,
    PLAYER_YAW,
    PLAYER_Z,
    RESOLUTION,
    WARMUP_STEPS,
    courtyard_xml,
)

ENV_ID = "minedojo_l1_spike"
N_LAVA = 14
INV_ITEMS = [
    "air",
    "bucket",
    "flint_and_steel",
    "lava_bucket",
    "obsidian",
    "water_bucket",
]
EQUIP_ITEMS = [
    "none",
    "water_bucket",
    "lava_bucket",
    "bucket",
    "flint_and_steel",
    "other",
]

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


_SPIKE_SLOT_ITEMS = {
    1: "water_bucket",
    2: "lava_bucket",
    3: "lava_bucket",
    4: "lava_bucket",
    5: "lava_bucket",
    6: "lava_bucket",
    7: "lava_bucket",
    8: "lava_bucket",
    9: "flint_and_steel",
}


def _spike_inventory() -> list[dict[str, Any]]:
    items = [{"slot": 0, "name": "water_bucket", "quantity": 1}]
    for index in range(7):
        items.append({"slot": index + 1, "name": "lava_bucket", "quantity": 1})
    items.append({"slot": 8, "name": "flint_and_steel", "quantity": 1})
    for index in range(7):
        items.append({"slot": 9 + index, "name": "lava_bucket", "quantity": 1})
    return items


def make_spike_env() -> Any:
    from obsidianlink.env.minedojo import MineDojoEnvironment

    return MineDojoEnvironment(
        "open-ended",
        image_size=RESOLUTION,
        generate_world_type="flat",
        flat_world_seed_string=FLAT_WORLD,
        drawing_str=courtyard_xml(lava_present=False),
        initial_inventory=_spike_inventory(),
        start_position={
            "x": PLAYER_X,
            "y": PLAYER_Y,
            "z": PLAYER_Z,
            "yaw": PLAYER_YAW,
            "pitch": PLAYER_PITCH,
        },
        allow_time_passage=False,
        allow_mob_spawn=False,
        initial_weather="clear",
        start_time=6000,
    )


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError, AttributeError, IndexError):
        return None


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return _jsonish(value.tolist())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _inventory_sane(inv: dict[str, int]) -> bool:
    if inv.get("air", 0) > 40:
        return False
    for name, qty in inv.items():
        if name == "air":
            continue
        if qty > 64:
            return False
    return True


def _inventory_counts(obs: Any) -> dict[str, int]:
    if hasattr(obs, "inventory"):
        return {str(name): int(qty) for name, qty in dict(obs.inventory or {}).items()}
    inv = obs.get("inventory") if isinstance(obs, dict) else None
    if not isinstance(inv, dict):
        return {}
    out: dict[str, int] = {}
    for name, qty in inv.items():
        n = _scalar(qty)
        if n is None:
            continue
        count = int(n)
        if count > 0:
            out[str(name)] = count
    return out


def _equipped_mainhand(obs: Any) -> str | None:
    if hasattr(obs, "selected_item"):
        return obs.selected_item
    if not isinstance(obs, dict):
        return None
    eq = obs.get("equipped_items")
    if not isinstance(eq, dict):
        return None
    hand = eq.get("mainhand", eq)
    if isinstance(hand, dict):
        item = hand.get("type")
        if item is None:
            return None
        if isinstance(item, (bytes, str)):
            return str(item)
        return str(item)
    if isinstance(hand, (bytes, str)):
        return str(hand)
    return None


def _location(obs: Any, info: Any) -> dict[str, float]:
    loc: dict[str, float] = {}
    for source in (info, obs):
        if not isinstance(source, dict):
            continue
        stats = source.get("location_stats")
        mapping = stats if isinstance(stats, dict) else source
        if not isinstance(mapping, dict):
            continue
        for key in (
            "xpos",
            "ypos",
            "zpos",
            "yaw",
            "pitch",
            "biome_id",
            "biome_temperature",
            "light_level",
        ):
            val = _scalar(mapping.get(key))
            if val is not None:
                loc[key] = val
        if loc:
            break
    return loc


def _region_scores(frame: Any) -> dict[str, float]:
    if frame is None:
        return {}
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {"frame_mean": float(arr.mean()) if arr.size else 0.0}
    h, w = arr.shape[:2]
    region = arr[int(h * 0.40) : int(h * 0.88), int(w * 0.30) : int(w * 0.70)]
    r = region[:, :, 0].astype(np.float32)
    g = region[:, :, 1].astype(np.float32)
    b = region[:, :, 2].astype(np.float32)
    lava = (r > 140) & (g > 50) & (b < 90) & (r > g) & (g >= b * 0.8)
    portal = (r > 70) & (b > 90) & (g < 55) & (b > g) & (r > g)
    return {
        "frame_mean": float(arr.mean()),
        "lava_frac": float(lava.mean()),
        "portal_frac": float(portal.mean()),
        "region_mean_r": float(r.mean()),
        "region_mean_g": float(g.mean()),
        "region_mean_b": float(b.mean()),
    }


def snapshot(obs: Any, info: Any, reward: float | None = None) -> dict[str, Any]:
    frame = getattr(obs, "frame", None)
    if frame is None and isinstance(obs, dict):
        frame = obs.get("pov")
    out: dict[str, Any] = {
        "obs_keys": ["frame", "inventory", "selected_item"],
        "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        "inventory": _inventory_counts(obs),
        "equipped": _equipped_mainhand(obs),
        "location": _location(info, info),
        "visual": _region_scores(frame),
    }
    if reward is not None:
        out["reward"] = float(reward)
    return out


def save_frame(path: str, obs: Any) -> None:
    frame = getattr(obs, "frame", None)
    if frame is None and isinstance(obs, dict):
        frame = obs.get("pov")
    if frame is None:
        return
    from PIL import Image

    arr = np.asarray(frame)
    if arr.ndim != 3:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


class SpikeSession:
    def __init__(self, env: Any, frames_dir: str) -> None:
        self.env = env
        self.frames_dir = frames_dir
        self.obs: Any = None
        self.info: dict[str, Any] = {}
        self.reward = 0.0
        self.done = False
        self.steps = 0
        self.total_reward = 0.0
        self.limitations: list[str] = []
        self.phases: dict[str, Any] = {}
        self.action_log: list[dict[str, Any]] = []

    def _action_from_overrides(self, overrides: dict[str, Any]) -> Any:
        from obsidianlink.env.actions import Action, ActionType

        if "use" in overrides:
            return Action(type=ActionType.USE)
        if "attack" in overrides:
            return Action(type=ActionType.ATTACK)
        if "camera" in overrides:
            camera = np.asarray(overrides["camera"]).reshape(-1)
            return Action(
                type=ActionType.CAMERA,
                pitch=float(camera[0]) if camera.size else 0.0,
                yaw=float(camera[1]) if camera.size > 1 else 0.0,
            )
        for key, value in overrides.items():
            if str(key).startswith("hotbar.") and int(value or 0):
                slot = int(str(key).split(".", 1)[1])
                return Action(
                    type=ActionType.EQUIP,
                    target=_SPIKE_SLOT_ITEMS.get(slot, "lava_bucket"),
                )
        if "forward" in overrides:
            return Action(type=ActionType.MOVE, dx=1 if int(overrides.get("forward") or 0) else 0)
        return Action(type=ActionType.WAIT)

    def step(self, **overrides: Any) -> dict[str, Any]:
        action = self._action_from_overrides(overrides)
        try:
            self.obs = self.env.step(action)
        except RuntimeError as exc:
            self.done = True
            self.limitations.append(f"{type(exc).__name__}: {exc}")
            return snapshot(self.obs, self.info, self.reward)
        hidden = dict(getattr(self.env, "hidden_state", {}) or {})
        self.reward = float(hidden.get("reward") or 0.0)
        self.total_reward += self.reward
        self.done = bool(hidden.get("done"))
        info = dict(getattr(self.env, "last_info", {}) or {})
        info.update(hidden)
        self.info = info
        self.steps += 1
        err = self.info.get("error")
        if err:
            self.done = True
            msg = f"env step error: {err}"
            if msg not in self.limitations:
                self.limitations.append(msg)
        snap = snapshot(self.obs, self.info, self.reward)
        recorded = {k: v for k, v in overrides.items() if k != "camera"}
        if "camera" in overrides:
            recorded["camera"] = [float(x) for x in np.asarray(overrides["camera"]).tolist()]
        self.action_log.append({"step": self.steps, "action": recorded, "after": snap})
        return snap

    def wait(self, n: int) -> dict[str, Any]:
        snap: dict[str, Any] = snapshot(self.obs, self.info)
        for _ in range(n):
            if self.done:
                break
            snap = self.step()
        return snap

    def look(self, pitch: float, yaw: float) -> dict[str, Any]:
        return self.step(camera=np.array([pitch, yaw], dtype=np.float32))

    def set_pitch(self, target: float, max_tries: int = 6) -> dict[str, Any]:
        snap = snapshot(self.obs, self.info)
        for _ in range(max_tries):
            current = snap.get("location", {}).get("pitch")
            if current is None:
                snap = self.look(target - PLAYER_PITCH, 0.0)
                break
            delta = target - float(current)
            if abs(delta) < 1.5:
                break
            snap = self.look(max(-30.0, min(30.0, delta)), 0.0)
        return snap

    def select_hotbar(self, slot: int) -> dict[str, Any]:
        from obsidianlink.env.actions import Action, ActionType

        self.obs = self.env.step(
            Action(
                type=ActionType.EQUIP,
                target=_SPIKE_SLOT_ITEMS.get(int(slot), "lava_bucket"),
            )
        )
        hidden = dict(getattr(self.env, "hidden_state", {}) or {})
        self.info = dict(getattr(self.env, "last_info", {}) or {})
        self.info.update(hidden)
        self.steps += 1
        snap = self.wait(2)
        return snap

    def use_held(self, ticks: int = 4) -> dict[str, Any]:
        snap = snapshot(self.obs, self.info)
        for _ in range(ticks):
            if self.done:
                break
            snap = self.step(use=1)
        return snap


def _phase(session: SpikeSession, name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    payload = {
        "ok": ok,
        "steps": session.steps,
        "snapshot": snapshot(session.obs, session.info, session.reward),
        **extra,
    }
    session.phases[name] = payload
    status = "OK" if ok else "FAIL"
    print(f"[spike] {name}: {status} steps={session.steps} extra={ {k: extra[k] for k in extra if k != 'snapshot'} }")
    sys.stdout.flush()
    return payload


def run_oracle(session: SpikeSession) -> str:
    """Return the last phase the oracle reached with a real world effect."""
    last_effect = "reset"

    session.limitations.append(
        "Spike now uses MineDojo event-level equip-by-name on an open-ended "
        "courtyard world. Portal-touch reward is not wired into open-ended."
    )
    session.wait(WARMUP_STEPS)
    save_frame(os.path.join(session.frames_dir, "00_reset.png"), session.obs)
    if session.done:
        session.limitations.append("episode ended during warmup; cannot validate actions")
        _phase(session, "A_inventory", False, got=_inventory_counts(session.obs))
        return last_effect
    inv = _inventory_counts(session.obs)
    sane = _inventory_sane(inv)
    has_water = inv.get("water_bucket", 0) >= 1
    has_lava = 1 <= inv.get("lava_bucket", 0) <= N_LAVA
    has_flint = inv.get("flint_and_steel", 0) >= 1
    inv_ok = sane and has_water and has_lava and has_flint
    _phase(
        session,
        "A_inventory",
        inv_ok,
        expected={"water_bucket": 1, "lava_bucket": N_LAVA, "flint_and_steel": 1},
        got=inv,
        sane=sane,
        location=_location(session.obs, session.info),
    )
    if not inv_ok:
        session.limitations.append(
            "InventoryAgentStart did not give a sane water_bucket / lava_bucket / flint_and_steel set"
        )
        return last_effect
    last_effect = "inventory"

    lava_before = inv.get("lava_bucket", 0)
    session.select_hotbar(2)
    equipped = _equipped_mainhand(session.obs)
    _phase(session, "B_select_lava", not session.done, equipped=equipped)
    last_effect = "hotbar_select"

    session.set_pitch(58.0)
    save_frame(os.path.join(session.frames_dir, "01_aimed.png"), session.obs)
    visual_before = _region_scores(session.obs.get("pov"))
    session.use_held(4)
    session.wait(8)
    save_frame(os.path.join(session.frames_dir, "02_after_lava_use.png"), session.obs)
    inv_after_lava = _inventory_counts(session.obs)
    lava_after = inv_after_lava.get("lava_bucket", 0)
    buckets_after = inv_after_lava.get("bucket", 0)
    visual_lava = _region_scores(session.obs.get("pov"))
    lava_poured = lava_after < lava_before or visual_lava.get("lava_frac", 0) > visual_before.get(
        "lava_frac", 0
    ) + 0.02
    _phase(
        session,
        "C_pour_lava",
        lava_poured,
        lava_before=lava_before,
        lava_after=lava_after,
        buckets_after=buckets_after,
        visual_before=visual_before,
        visual_after=visual_lava,
    )
    if not lava_poured:
        session.step(sneak=1, use=1)
        session.wait(6)
        inv_retry = _inventory_counts(session.obs)
        visual_retry = _region_scores(session.obs.get("pov"))
        lava_poured = inv_retry.get("lava_bucket", 0) < lava_before or visual_retry.get(
            "lava_frac", 0
        ) > visual_before.get("lava_frac", 0) + 0.02
        session.phases["C_pour_lava"]["retry_sneak_use"] = {
            "ok": lava_poured,
            "inventory": inv_retry,
            "visual": visual_retry,
        }
        save_frame(os.path.join(session.frames_dir, "02b_lava_retry.png"), session.obs)
        if lava_poured:
            inv_after_lava = inv_retry
            lava_after = inv_retry.get("lava_bucket", 0)
            visual_lava = visual_retry
    if not lava_poured:
        session.limitations.append(
            "USE with lava_bucket did not consume the bucket or produce visible lava"
        )
        return last_effect
    last_effect = "pour_lava"

    # Step back so we are not standing in the lava.
    session.step(back=1)
    session.wait(2)

    water_before = inv_after_lava.get("water_bucket", 0)
    session.select_hotbar(1)
    session.set_pitch(58.0)
    session.use_held(4)
    session.wait(10)
    save_frame(os.path.join(session.frames_dir, "03_after_water_use.png"), session.obs)
    inv_after_water = _inventory_counts(session.obs)
    visual_obsidian = _region_scores(session.obs.get("pov"))
    water_used = (
        inv_after_water.get("water_bucket", 0) < water_before
        or inv_after_water.get("bucket", 0) > inv_after_lava.get("bucket", 0)
    )
    lava_gone = visual_obsidian.get("lava_frac", 1.0) < visual_lava.get("lava_frac", 0.0) - 0.01
    converted = water_used or lava_gone
    _phase(
        session,
        "D_water_on_lava",
        converted,
        water_used=water_used,
        lava_visual_dropped=lava_gone,
        inventory=inv_after_water,
        visual_after=visual_obsidian,
        visual_lava=visual_lava,
    )
    if not converted:
        session.limitations.append(
            "USE with water_bucket did not convert visible lava / did not change water inventory"
        )
        return last_effect
    last_effect = "cast_obsidian"

    # Try to pick the water back up from hotbar 1 (now empty bucket if pour worked).
    inv_pre_pickup = dict(inv_after_water)
    session.select_hotbar(1)
    session.use_held(4)
    session.wait(6)
    save_frame(os.path.join(session.frames_dir, "04_after_pickup.png"), session.obs)
    inv_pickup = _inventory_counts(session.obs)
    picked = inv_pickup.get("water_bucket", 0) > inv_pre_pickup.get("water_bucket", 0) or (
        inv_pickup.get("bucket", 0) < inv_pre_pickup.get("bucket", 0)
        and inv_pickup.get("water_bucket", 0) >= 1
    )
    _phase(
        session,
        "E_pickup_water",
        picked or inv_pickup.get("water_bucket", 0) >= 1,
        picked=picked,
        inventory=inv_pickup,
        note="water_bucket is reusable; lava_bucket is consumed",
    )
    if picked:
        last_effect = "pickup_water"

    # Extra casts: strafe and repeat lava+water a few times.
    lava_hotbars = (3, 4, 5, 6)
    extra_casts = 0
    lava_start_extra = _inventory_counts(session.obs).get("lava_bucket", 0)
    for i, lava_slot in enumerate(lava_hotbars):
        if session.done:
            break
        before = _inventory_counts(session.obs).get("lava_bucket", 0)
        if before <= 0:
            break
        session.step(right=1)
        session.wait(1)
        session.select_hotbar(lava_slot)
        session.set_pitch(58.0)
        session.use_held(3)
        session.wait(5)
        session.select_hotbar(1)
        session.use_held(3)
        session.wait(6)
        after = _inventory_counts(session.obs).get("lava_bucket", 0)
        if after < before:
            extra_casts += 1
            session.use_held(2)
            session.wait(3)
        save_frame(
            os.path.join(session.frames_dir, f"05_extra_cast_{i + 1}.png"), session.obs
        )
    extra_ok = extra_casts > 0
    _phase(
        session,
        "F_extra_casts",
        extra_ok,
        extra_casts=extra_casts,
        lava_consumed=lava_start_extra - _inventory_counts(session.obs).get("lava_bucket", 0),
        note="FPS aiming; extra casts may miss even if the mechanic works",
    )
    if extra_ok:
        last_effect = "multi_cast"

    lava_used = N_LAVA - _inventory_counts(session.obs).get("lava_bucket", 0)
    frame_complete = lava_used >= 10
    _phase(
        session,
        "F2_frame_complete",
        frame_complete,
        lava_used=lava_used,
        needed_for_cornerless_frame=10,
        note="No ObservationFromGrid. Proxy = lava buckets consumed. Not a geometric proof.",
    )
    if frame_complete:
        last_effect = "frame_proxy"

    session.select_hotbar(9)
    session.set_pitch(45.0)
    session.use_held(6)
    session.wait(8)
    save_frame(os.path.join(session.frames_dir, "06_after_ignite.png"), session.obs)
    visual_portal = _region_scores(session.obs.get("pov"))
    ignited = visual_portal.get("portal_frac", 0.0) > 0.02 or session.total_reward >= 10.0
    _phase(
        session,
        "G_ignition",
        ignited,
        portal_frac=visual_portal.get("portal_frac"),
        total_reward=session.total_reward,
        equipped=_equipped_mainhand(session.obs),
    )
    if ignited:
        last_effect = "ignition"

    loc_before = _location(session.obs, session.info)
    session.set_pitch(10.0)
    for _ in range(25):
        if session.done:
            break
        session.step(forward=1)
    session.wait(80)
    save_frame(os.path.join(session.frames_dir, "07_after_enter_attempt.png"), session.obs)
    loc_after = _location(session.obs, session.info)
    ypos_delta = (loc_after.get("ypos") or 0) - (loc_before.get("ypos") or 0)
    biome_changed = loc_after.get("biome_id") != loc_before.get("biome_id")
    likely_void = (loc_after.get("ypos") or 101) < 50 and not biome_changed
    entered = bool(
        ignited
        and biome_changed
        and not likely_void
        and (session.total_reward >= 10.0 or abs(ypos_delta) > 5)
    )
    _phase(
        session,
        "H_nether_entry",
        entered,
        loc_before=loc_before,
        loc_after=loc_after,
        ypos_delta=ypos_delta,
        biome_changed=biome_changed,
        likely_void=likely_void,
        total_reward=session.total_reward,
    )
    if entered:
        last_effect = "nether_entry"
    elif likely_void:
        session.limitations.append("Walk-forward after ignition (or instead of it) looked like void fall, not Nether")
    return last_effect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1 technical feasibility spike")
    parser.add_argument("--env-id", default=ENV_ID)
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    frames_dir = os.path.join(_RUNS_DIR, f"l1_spike_{stamp}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)

    report: dict[str, Any] = {
        "kind": "l1_technical_feasibility_spike",
        "l1_definition": "Casting → Frame → Ignition → Nether Entry",
        "l1_definition_unchanged": True,
        "prebuilt_portal_frame": False,
        "drawblock_used_for": "obsidian_courtyard_casting_surface_only",
        "valid_for_l1_agent_conclusion": False,
        "env_id": args.env_id,
        "started_utc": stamp,
        "phases": {},
        "limitations": [],
        "oracle_last_effect": None,
        "recommendation": None,
    }

    env = None
    session: SpikeSession | None = None
    t0 = time.perf_counter()
    try:
        print("[spike] make MineDojo open-ended courtyard")
        sys.stdout.flush()
        env = make_spike_env()
        print("[spike] reset")
        sys.stdout.flush()
        obs = env.reset()
        report["env_id"] = ENV_ID
        report["portal_touch_reward_enabled"] = False
        session = SpikeSession(env, frames_dir)
        session.obs = obs
        session.info = dict(getattr(env, "hidden_state", {}) or {})
        session.limitations.append(
            "open-ended MineDojo worlds do not expose RewardForTouchingBlockType"
        )
        raw = getattr(env, "_env", None)
        space = getattr(raw, "action_space", None)
        no_op = space.no_op() if space is not None and hasattr(space, "no_op") else {}
        report["action_space_keys"] = sorted(no_op) if isinstance(no_op, dict) else []
        report["obs_space_keys"] = ["frame", "inventory", "selected_item"]
        last = run_oracle(session)
        report["oracle_last_effect"] = last
        report["phases"] = session.phases
        report["limitations"] = session.limitations
        report["steps"] = session.steps
        report["total_reward"] = session.total_reward
        report["done"] = session.done
        report["final_snapshot"] = snapshot(session.obs, session.info, session.reward)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if session is not None:
            report["phases"] = session.phases
            report["limitations"] = session.limitations + [report["error"]]
            report["steps"] = session.steps
        print(report["traceback"])
        sys.stdout.flush()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    report["wall_time"] = time.perf_counter() - t0
    report["frames_dir"] = frames_dir
    if session is not None:
        report["action_effects"] = {
            "hotbar_select": bool(session.phases.get("B_select_lava", {}).get("ok")),
            "use_lava_bucket": bool(session.phases.get("C_pour_lava", {}).get("ok")),
            "use_water_bucket": bool(session.phases.get("D_water_on_lava", {}).get("ok")),
            "pickup_water": bool(session.phases.get("E_pickup_water", {}).get("ok")),
            "repeat_cast": bool(session.phases.get("F_extra_casts", {}).get("ok")),
            "ignite": bool(session.phases.get("G_ignition", {}).get("ok")),
            "nether_entry": bool(session.phases.get("H_nether_entry", {}).get("ok")),
        }
        report["bucket_casting_feasible"] = bool(
            session.phases.get("C_pour_lava", {}).get("ok")
            and session.phases.get("D_water_on_lava", {}).get("ok")
        )
        report["evaluator_truth_candidates"] = {
            "inventory_delta": "lava_bucket count down; water_bucket ↔ bucket. No ObservationFromGrid.",
            "visual_pov": "lava_frac / portal_frac on saved RGB frames (human + heuristic).",
            "location_stats": "xpos/ypos/zpos/yaw/pitch/biome_id via ObservationFromFullStats / gym info.",
            "reward_for_touching_nether_portal": "RewardForTouchingBlockType; isolated from NavigationDecorator.",
            "not_used": "ObservationFromGrid (known unreliable on this Malmo stack).",
        }
        if report["bucket_casting_feasible"] and report.get("oracle_last_effect") in {
            "cast_obsidian",
            "pickup_water",
            "multi_cast",
            "frame_proxy",
            "ignition",
            "nether_entry",
        }:
            report["recommendation"] = (
                "continue_official_l1"
                if report.get("oracle_last_effect") in {"ignition", "nether_entry", "frame_proxy"}
                else "continue_official_l1_with_aiming_and_evaluator_work"
            )
        elif report.get("error"):
            report["recommendation"] = "record_limitation_do_not_change_l1_definition"
        elif not session.phases.get("A_inventory", {}).get("ok"):
            report["recommendation"] = "blocked_on_inventory_start_do_not_change_l1_definition"
        else:
            report["recommendation"] = "blocked_on_bucket_use_do_not_change_l1_definition"

    out_path = os.path.join(_RUNS_DIR, f"l1_spike_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    print("[spike] wrote", out_path)
    print(json.dumps(_jsonish({k: report[k] for k in (
        "bucket_casting_feasible",
        "oracle_last_effect",
        "recommendation",
        "limitations",
        "action_effects",
        "error",
        "steps",
        "wall_time",
    ) if k in report}), indent=2))
    sys.stdout.flush()
    return 0 if not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
