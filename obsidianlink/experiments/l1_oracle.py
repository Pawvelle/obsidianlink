"""Full Scripted Oracle session helpers for the Formal L1 Portal task.

Not an Agent. Not a Planner. The Oracle may use fixed world coordinates,
evaluator-only ``location_stats`` for scripted navigation/aiming, and a
hard-coded reference portal geometry (``obsidianlink.tasks.portal``).
It must still change the world only through the same legal Minecraft
action interface a future Agent can use: MOVE / CAMERA / USE / ATTACK /
HOTBAR / WAIT (+ sneak). No DrawBlock portal, world editing, command,
teleport, or inventory injection.

Builds on the mechanics already proven live in
``obsidianlink/experiments/run_l1_mechanics.py`` (empty-bucket lava
scoop, native ``use`` fluid placement, cobblestone place/break).

2026-08-20 Gate 1 stability pass: two independent live Oracle runs hit
``TimeoutError`` -> ``RuntimeError: Attempted to step an environment
server with done=True`` at ~270-280s wall time (see ROADMAP Blocked). A
pure-``WAIT`` control loop ran 93,200 steps / 340s with no issue, so the
fix here is not a bigger socket timeout — it is fewer, more deliberate
actions (short deterministic camera correction, retry-on-failure only,
tighter travel) plus fast, correct ``done`` handling so a broken
episode is never stepped twice.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import L1ControlledEnv
from obsidianlink.experiments.l1_mechanics import (
    cobble_broken,
    cobble_placed,
    frame_stats,
    poured_lava,
    qty,
    scooped_lava,
    used_water,
)

EYE_HEIGHT = 1.62

HOTBAR_WATER = "1"
HOTBAR_BUCKET = "2"
HOTBAR_COBBLE = "3"
HOTBAR_PICKAXE = "4"
HOTBAR_FLINT = "5"


class EpisodeAborted(RuntimeError):
    """Raised once the backend reports ``done`` or a step raises.

    Session.step() short-circuits on every subsequent call instead of
    re-entering ``env.step()`` on an already-finished episode (that
    second call is what previously raised a *different*, misleading
    ``RuntimeError`` and buried the real root cause).
    """


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
    for key in ("xpos", "ypos", "zpos", "yaw", "pitch", "biome_id", "reward"):
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


class OracleSession:
    """Scripted, retryable primitives over :class:`L1ControlledEnv`.

    Tracks per-action-type counts, per-step wall-clock duration, and a
    ``stage(name)`` context manager so a Gate can be profiled without a
    separate instrumentation layer.
    """

    def __init__(self, env: L1ControlledEnv, frames_dir: str) -> None:
        self.env = env
        self.frames_dir = frames_dir
        self.steps = 0
        self.lava_sources_used = 0
        self.action_log: list[dict[str, Any]] = []
        self.action_counts: Counter[str] = Counter()
        self.step_durations: list[float] = []
        self.stage_log: list[dict[str, Any]] = []
        self.aborted = False
        self.abort_reason: str | None = None
        self._stage_start_steps = 0
        self._stage_start_counts: Counter[str] = Counter()
        self._trace_inventory = False
        self.inv_trace: list[dict[str, Any]] = []
        self.inv_trace_phase = ""

    def begin_inventory_trace(self, phase: str = "trace") -> dict[str, Any]:
        """Record every subsequent ``step`` inventory (and print one line).

        Used by the water-rollback isolation probe: the live Gate 1 run
        saw ``water_bucket`` appear after recover, then disappear during
        a later ``look_at``+``wait`` with no explicit ``USE``. Tick-level
        inventory is the only way to tell observation lag from a real
        second use.
        """
        self._trace_inventory = True
        self.inv_trace = []
        self.inv_trace_phase = phase
        snap = self.snap()
        self._record_inv_tick(None, snap)
        return snap

    def end_inventory_trace(self) -> list[dict[str, Any]]:
        self._trace_inventory = False
        return self.inv_trace

    def _record_inv_tick(self, action: Action | None, snap: dict[str, Any]) -> None:
        prev = self.inv_trace[-1] if self.inv_trace else None
        inv = dict(snap["inventory"])
        selected = snap["selected_item"]
        changed = (
            prev is None
            or prev.get("inventory") != inv
            or prev.get("selected_item") != selected
        )
        entry = {
            "step": self.steps,
            "phase": self.inv_trace_phase,
            "type": action.type.value if action is not None else "baseline",
            "sneak": bool(action.sneak) if action is not None else False,
            "inventory": inv,
            "selected_item": selected,
            "changed": changed,
            "water_bucket": qty(inv, "water_bucket"),
            "bucket": qty(inv, "bucket"),
            "lava_bucket": qty(inv, "lava_bucket"),
        }
        self.inv_trace.append(entry)
        mark = " CHANGED" if changed else ""
        print(
            f"[inv-tick] step={entry['step']:4d} phase={entry['phase']:<22} "
            f"type={entry['type']:<8} sneak={int(entry['sneak'])} "
            f"selected={str(selected):<14} wb={entry['water_bucket']} "
            f"b={entry['bucket']} lb={entry['lava_bucket']} inv={inv}{mark}"
        )
        sys.stdout.flush()

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
        if action.type in (ActionType.EQUIP, ActionType.PLACE):
            raise RuntimeError(f"{action.type} is forbidden for the Oracle")
        if self.aborted:
            # Never re-enter env.step() on a finished/broken episode: that
            # previously raised a second, misleading RuntimeError.
            raise EpisodeAborted(self.abort_reason or "episode already aborted")
        t0 = time.perf_counter()
        try:
            self.env.step(action)
        except Exception as exc:  # noqa: BLE001
            self.aborted = True
            self.abort_reason = f"{type(exc).__name__}: {exc}"
            self.step_durations.append(time.perf_counter() - t0)
            raise
        self.step_durations.append(time.perf_counter() - t0)
        self.steps += 1
        self.action_counts[action.type.value] += 1
        hidden = self.env.hidden_state
        if bool(hidden.get("done")):
            self.aborted = True
            self.abort_reason = "env hidden_state done=True"
        snap = self.snap()
        self.action_log.append(
            {
                "step": self.steps,
                "type": action.type.value,
                "dx": action.dx,
                "dz": action.dz,
                "yaw": action.yaw,
                "pitch": action.pitch,
                "target": action.target,
                "sneak": action.sneak,
                "pose": snap["pose"],
                "step_seconds": self.step_durations[-1],
            }
        )
        if self._trace_inventory:
            self._record_inv_tick(action, snap)
        return snap

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Profile one named phase: elapsed seconds + per-verb action delta."""
        start_steps = self.steps
        start_counts = Counter(self.action_counts)
        t0 = time.perf_counter()
        error: str | None = None
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            delta = {
                verb: self.action_counts[verb] - start_counts.get(verb, 0)
                for verb in self.action_counts
                if self.action_counts[verb] - start_counts.get(verb, 0) > 0
            }
            self.stage_log.append(
                {
                    "name": name,
                    "elapsed_seconds": elapsed,
                    "steps": self.steps - start_steps,
                    "action_counts": delta,
                    "error": error,
                }
            )

    def wait(self, n: int) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(n):
            snap = self.step(Action(type=ActionType.WAIT))
        return snap

    def hotbar(self, slot: str) -> dict[str, Any]:
        self.step(Action(type=ActionType.HOTBAR, target=slot))
        return self.wait(1)

    def use(self, ticks: int = 3, *, sneak: bool = False) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(ticks):
            snap = self.step(Action(type=ActionType.USE, sneak=sneak))
        return snap

    def attack(self, ticks: int, *, sneak: bool = False) -> dict[str, Any]:
        snap = self.snap()
        for _ in range(ticks):
            snap = self.step(Action(type=ActionType.ATTACK, sneak=sneak))
        return snap

    def _aim_error(self, target: tuple[float, float, float], pose: dict[str, float]) -> tuple[float, float] | None:
        tx, ty, tz = target
        x, y, z = pose.get("xpos"), pose.get("ypos"), pose.get("zpos")
        if x is None or y is None or z is None:
            return None
        dx = tx - float(x)
        dz = tz - float(z)
        dy = ty - (float(y) + EYE_HEIGHT)
        dist = math.sqrt(dx * dx + dz * dz)
        desired_yaw = math.degrees(math.atan2(-dx, dz))
        desired_pitch = max(-89.0, min(89.0, math.degrees(math.atan2(-dy, max(dist, 1e-6)))))
        cur_yaw = pose.get("yaw")
        cur_pitch = pose.get("pitch")
        yaw_delta = _yaw_delta(desired_yaw, float(cur_yaw)) if cur_yaw is not None else desired_yaw
        pitch_delta = desired_pitch - float(cur_pitch) if cur_pitch is not None else desired_pitch
        return yaw_delta, pitch_delta

    def look_at(
        self, target: tuple[float, float, float], max_tries: int = 4
    ) -> dict[str, Any]:
        """Turn camera to face a fixed world-space point, evaluator-only nav.

        Deterministic: the desired yaw/pitch is computed analytically
        from known geometry + current pose every iteration. ``max_tries``
        must be large enough to cover the worst-case turn this call can
        be asked to make (e.g. up to a ~180 degree swing right after
        scooping lava, camera still facing the pool) — each CAMERA step
        is clamped to +/-45 degrees, so a full about-face needs ~4 steps
        before fine convergence even starts.
        """
        snap = self.snap()
        for _ in range(max_tries):
            error = self._aim_error(target, snap["pose"])
            if error is None:
                break
            yaw_delta, pitch_delta = error
            if abs(yaw_delta) < 2.5 and abs(pitch_delta) < 2.5:
                break
            snap = self.step(
                Action(
                    type=ActionType.CAMERA,
                    yaw=max(-45.0, min(45.0, yaw_delta)),
                    pitch=max(-45.0, min(45.0, pitch_delta)),
                )
            )
        return snap

    def is_aimed_at(self, target: tuple[float, float, float], *, tolerance: float = 4.0) -> bool:
        """True if the camera is already converged on ``target``.

        Used to gate ``use``/``attack`` bursts: never fire blind — if a
        cast primitive's aim did not converge within its step budget, it
        must report failure (so the caller retries deliberately) instead
        of spending USE ticks pointed at whatever the camera last landed
        on (e.g. a mold wall or open air).
        """
        error = self._aim_error(target, self.snap()["pose"])
        if error is None:
            return False
        yaw_delta, pitch_delta = error
        return abs(yaw_delta) < tolerance and abs(pitch_delta) < tolerance

    def turn_yaw(self, target: float, max_tries: int = 4) -> dict[str, Any]:
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
                Action(type=ActionType.CAMERA, yaw=max(-45.0, min(45.0, delta)))
            )
        return snap

    def walk_toward(
        self, x_target: float, z_target: float, max_steps: int = 20, *, sneak: bool = False
    ) -> dict[str, Any]:
        debug = bool(os.environ.get("ORACLE_DEBUG_NAV"))
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
            if dx * dx + dz * dz < 0.36:
                return snap
            desired_yaw = math.degrees(math.atan2(-dx, dz))
            current_yaw = pose.get("yaw")
            if current_yaw is None or abs(_yaw_delta(desired_yaw, float(current_yaw))) > 8.0:
                snap = self.turn_yaw(desired_yaw, max_tries=3)
            before_xz = (x, z)
            snap = self.step(Action(type=ActionType.MOVE, dx=1, sneak=sneak))
            if debug:
                after_xz = (snap["pose"].get("xpos"), snap["pose"].get("zpos"))
                print(f"[debug] walk_toward target=({x_target},{z_target}) {before_xz} -> {after_xz} yaw={snap['pose'].get('yaw')}")
        return snap

    def cast_fluid_at(
        self,
        target: tuple[int, int, int],
        *,
        backing_face_from: tuple[float, float, float],
        ticks: int = 3,
        settle_wait: int = 2,
        aim_tries: int = 8,
        aim_tolerance: float = 4.0,
    ) -> dict[str, Any]:
        """One attempt: aim through ``target`` at a known solid backing
        face and USE. No internal retry loop — the caller checks the
        inventory/visual result and only re-invokes (with a nudged aim
        or more ticks) if this single attempt failed. This is the
        "retry only on failure" rule: the common-case path must stay
        short. USE only fires if the camera actually converged on the
        aim point (never fires blind at wherever the camera last was).
        """
        self.look_at(backing_face_from, max_tries=aim_tries)
        self.wait(1)  # let location_stats catch up to the final CAMERA step
        if not self.is_aimed_at(backing_face_from, tolerance=aim_tolerance):
            return self.snap()
        snap = self.use(ticks, sneak=True)
        if settle_wait:
            snap = self.wait(settle_wait)
        return snap

    def place_solid_at(
        self,
        target: tuple[int, int, int],
        *,
        backing_face_from: tuple[float, float, float],
        hotbar_slot: str,
        ticks: int = 3,
        settle_wait: int = 1,
        aim_tries: int = 8,
    ) -> dict[str, Any]:
        """Same aim-at-solid-face mechanic as :meth:`cast_fluid_at`, for a
        solid block (e.g. cobblestone mold walls) instead of a fluid.
        One attempt; caller retries on failure only.
        """
        self.hotbar(hotbar_slot)
        self.look_at(backing_face_from, max_tries=aim_tries)
        self.wait(1)  # let location_stats catch up to the final CAMERA step
        if not self.is_aimed_at(backing_face_from):
            snap = self.snap()
            if os.environ.get("ORACLE_DEBUG"):
                err = self._aim_error(backing_face_from, snap["pose"])
                print(f"[debug] place_solid_at NOT AIMED target={backing_face_from} pose={snap['pose']} error={err} selected={snap['selected_item']}")
            return snap
        snap = self.use(ticks, sneak=True)
        if settle_wait:
            snap = self.wait(settle_wait)
        return snap

    def build_mold(
        self,
        walls: list[tuple[tuple[int, int, int], tuple[float, float, float]]],
        *,
        hotbar_slot: str,
    ) -> list[bool]:
        """Place cobblestone at each ``(cell, backing_face_from)`` pair.

        A mold physically contains poured lava to the intended cell(s):
        on flat open ground, unconstrained lava spreads across several
        cells before it can be watered (live-observed 2026-08-20 Oracle
        Gate 1 attempt), so casting a specific frame cell requires walling
        off its open sides first. One placement attempt per wall; a
        single close-range retry only if the first attempt did not
        consume cobblestone.
        """
        results: list[bool] = []
        debug = bool(os.environ.get("ORACLE_DEBUG"))
        for cell, backing in walls:
            bx, _by, bz = backing
            self.walk_toward(bx, bz - 1.2, max_steps=15)
            before_snap = self.snap()
            before = qty(before_snap["inventory"], "cobblestone")
            self.place_solid_at(cell, backing_face_from=backing, hotbar_slot=hotbar_slot)
            after_snap = self.snap()
            placed = qty(after_snap["inventory"], "cobblestone") < before
            if debug and not placed:
                print(
                    f"[debug] mold wall {cell} attempt1 failed: "
                    f"before_pose={before_snap['pose']} after_pose={after_snap['pose']} "
                    f"selected={after_snap['selected_item']} inv={after_snap['inventory']}"
                )
            if not placed:
                self.place_solid_at(
                    cell, backing_face_from=backing, hotbar_slot=hotbar_slot, ticks=4
                )
                after_snap = self.snap()
                placed = qty(after_snap["inventory"], "cobblestone") < before
                if debug and not placed:
                    print(
                        f"[debug] mold wall {cell} attempt2 failed: "
                        f"after_pose={after_snap['pose']} selected={after_snap['selected_item']} "
                        f"inv={after_snap['inventory']}"
                    )
            results.append(placed)
        return results

    def scoop_lava_at(
        self,
        source: tuple[int, int, int],
        *,
        aim_from: tuple[float, float, float],
        retries: int = 2,
    ) -> tuple[bool, dict[str, Any]]:
        """One aim+use attempt; retry only if the first did not scoop."""
        self.hotbar(HOTBAR_BUCKET)
        before = self.snap()
        scooped = False
        after = before
        debug = bool(os.environ.get("ORACLE_DEBUG"))
        for _attempt in range(retries):
            self.look_at(aim_from, max_tries=12)
            self.wait(1)
            # The pool is a 4x4 source area, not a single precise cell, so
            # a looser convergence tolerance than the default 4.0 is still
            # a safe aim: live-observed steep-pitch aims (~56-60 degrees)
            # can plateau ~4-5 degrees short of the analytic target for a
            # few extra ticks without ever missing the pool itself.
            if not self.is_aimed_at(aim_from, tolerance=6.0):
                after = self.snap()
                if debug:
                    err = self._aim_error(aim_from, after["pose"])
                    print(f"[debug] scoop_lava_at NOT AIMED source={source} aim_from={aim_from} pose={after['pose']} error={err}")
                continue
            self.use(3)
            after = self.wait(1)
            scooped = scooped_lava(before["inventory"], after["inventory"])
            if debug and not scooped:
                print(
                    f"[debug] scoop_lava_at aimed but not scooped source={source} "
                    f"pose={after['pose']} selected={after['selected_item']} inv={after['inventory']}"
                )
            if scooped:
                self.lava_sources_used += 1
                break
        return scooped, after


__all__ = [
    "EYE_HEIGHT",
    "EpisodeAborted",
    "HOTBAR_BUCKET",
    "HOTBAR_COBBLE",
    "HOTBAR_FLINT",
    "HOTBAR_PICKAXE",
    "HOTBAR_WATER",
    "OracleSession",
    "cobble_broken",
    "cobble_placed",
    "poured_lava",
    "qty",
    "used_water",
]
