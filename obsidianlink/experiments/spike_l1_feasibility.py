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

ENV_ID = "MineRLL1Spike-v0"
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


def register_spike_spec(*, name: str = ENV_ID, portal_reward: bool = True) -> str:
    import gym  # type: ignore[import-untyped]
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import handlers
    from minerl.herobraine.hero.handler import Handler
    from minerl.herobraine.hero.mc import INVERSE_KEYMAP

    class _SafeDrawingDecorator(handlers.DrawingDecorator):
        def xml_template(self) -> str:
            return """<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"""

    class L1SpikeSpec(Treechop):
        _portal_reward = portal_reward

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_server_world_generators(self) -> list:
            return [
                handlers.FlatWorldGenerator(
                    force_reset=True, generatorString=FLAT_WORLD
                )
            ]

        def create_server_decorators(self) -> list:
            # Obsidian courtyard only. lava_present=False → no lava patch,
            # and courtyard_xml never draws a portal frame.
            return [_SafeDrawingDecorator(courtyard_xml(lava_present=False))]

        def create_server_initial_conditions(self) -> list:
            return [
                handlers.TimeInitialCondition(
                    allow_passage_of_time=False, start_time=6000
                ),
                handlers.SpawningInitialCondition(allow_spawning=False),
                handlers.WeatherInitialCondition(weather="clear"),
            ]

        def create_agent_start(self) -> list:
            # Hotbar 1=water, 2-8=lava, 9=flint. Remaining lava is off-hotbar.
            # EquipAction cannot be used on this stack (see create_actionables).
            inventory: dict[int, dict[str, Any]] = {
                0: {"type": "water_bucket", "quantity": 1},
            }
            for i in range(7):
                inventory[i + 1] = {"type": "lava_bucket", "quantity": 1}
            inventory[8] = {"type": "flint_and_steel", "quantity": 1}
            for i in range(7):
                inventory[9 + i] = {"type": "lava_bucket", "quantity": 1}
            return [
                handlers.GuiScale(1.0),
                handlers.GammaSetting(2.0),
                handlers.FOVSetting(70.0),
                handlers.FakeCursorSize(0),
                handlers.AgentStartPlacement(
                    x=PLAYER_X,
                    y=PLAYER_Y,
                    z=PLAYER_Z,
                    yaw=PLAYER_YAW,
                    pitch=PLAYER_PITCH,
                ),
                handlers.InventoryAgentStart(inventory),
            ]

        def create_observables(self) -> list:
            return [
                handlers.POVObservation(self.resolution),
                handlers.FlatInventoryObservation(list(INV_ITEMS)),
                handlers.EquippedItemObservation(list(EQUIP_ITEMS), mainhand=True),
            ]

        def create_actionables(self) -> list:
            # EquipAction cannot be used on this MineRL 1.0.2 / MCP-Reborn stack.
            # no_op sends ``equip none``; constructKeyboardState does
            # Integer.parseInt("none") and kills the episode.
            acts = super().create_actionables()
            names = {a.to_string() for a in acts}
            if "use" not in names:
                acts.append(
                    handlers.KeybasedCommandAction("use", INVERSE_KEYMAP["use"])
                )
            for i in range(1, 10):
                key = f"hotbar.{i}"
                if key not in names:
                    acts.append(handlers.KeybasedCommandAction(key, str(i)))
            return acts

        def create_monitors(self) -> list:
            return [handlers.ObservationFromCurrentLocation()]

        def create_rewardables(self) -> list:
            if not self._portal_reward:
                return []
            # Evaluator-side candidate. Isolated from NavigationDecorator.
            return [
                handlers.RewardForTouchingBlockType(
                    [
                        {
                            "type": "nether_portal",
                            "behaviour": "onceOnly",
                            "reward": 10.0,
                        }
                    ]
                )
            ]

        def create_agent_handlers(self) -> list:
            return []

        def create_server_quit_producers(self) -> list[Handler]:
            return [
                handlers.ServerQuitFromTimeUp(400000),
                handlers.ServerQuitWhenAnyAgentFinishes(),
            ]

    spec = L1SpikeSpec()
    if spec.name not in gym.envs.registry.env_specs:
        spec.register()
    return spec.name


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
    frame = obs.get("pov") if isinstance(obs, dict) else None
    out: dict[str, Any] = {
        "obs_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        "inventory": _inventory_counts(obs),
        "equipped": _equipped_mainhand(obs),
        "location": _location(obs, info),
        "visual": _region_scores(frame),
    }
    if reward is not None:
        out["reward"] = float(reward)
    return out


def save_frame(path: str, obs: Any) -> None:
    frame = obs.get("pov") if isinstance(obs, dict) else None
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

    def _noop(self) -> dict[str, Any]:
        return self.env.action_space.no_op()

    def step(self, **overrides: Any) -> dict[str, Any]:
        action = self._noop()
        for key, value in overrides.items():
            action[key] = value
        try:
            self.obs, reward, done, info = self.env.step(action)
        except RuntimeError as exc:
            self.done = True
            self.limitations.append(f"{type(exc).__name__}: {exc}")
            return snapshot(self.obs, self.info, self.reward)
        self.reward = float(reward or 0.0)
        self.total_reward += self.reward
        self.done = bool(done)
        self.info = info if isinstance(info, dict) else {}
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
        snap = self.step(**{f"hotbar.{slot}": 1})
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
        "EquipAction is unusable on MineRL 1.0.2 MCP-Reborn: no_op emits "
        "'equip none' and constructKeyboardState Integer.parseInt crashes the episode. "
        "Spike uses hotbar.1-9 instead."
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
        import gym  # type: ignore[import-untyped]
        import minerl  # type: ignore[import-untyped]  # noqa: F401

        last_make_error: Exception | None = None
        obs = None
        for portal_reward, env_id in (
            (True, args.env_id),
            (False, args.env_id.replace("-v0", "NoReward-v0")),
        ):
            print("[spike] registering", env_id, "portal_reward=", portal_reward)
            sys.stdout.flush()
            register_spike_spec(name=env_id, portal_reward=portal_reward)
            try:
                print("[spike] gym.make")
                sys.stdout.flush()
                env = gym.make(env_id)
                print("[spike] reset")
                sys.stdout.flush()
                obs = env.reset()
                report["env_id"] = env_id
                report["portal_touch_reward_enabled"] = portal_reward
                last_make_error = None
                break
            except Exception as exc:
                last_make_error = exc
                report.setdefault("reset_attempts", []).append(
                    {
                        "env_id": env_id,
                        "portal_reward": portal_reward,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = None
                if portal_reward:
                    print("[spike] reset failed with portal reward; retrying without")
                    sys.stdout.flush()
                    continue
                raise
        if env is None or obs is None:
            raise last_make_error or RuntimeError("gym.make/reset failed")
        session = SpikeSession(env, frames_dir)
        session.obs = obs
        if not report.get("portal_touch_reward_enabled", True):
            session.limitations.append(
                "RewardForTouchingBlockType(nether_portal) failed at reset; retried without it"
            )
        report["action_space_keys"] = sorted(env.action_space.spaces.keys())
        report["obs_space_keys"] = (
            sorted(env.observation_space.spaces.keys())
            if hasattr(env.observation_space, "spaces")
            else []
        )
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
