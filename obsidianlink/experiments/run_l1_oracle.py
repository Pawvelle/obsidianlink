"""Full Scripted Oracle for the Formal L1 Portal task — Gate 1 only.

Gate 1 is the shortest live casting sequence that produces **one**
new obsidian block:

    scoop lava → place lava → place water on lava → wait for obsidian

Not a portal frame. Not ignition. Not Nether entry. Starting inventory
already has ``water_bucket``; do not place/recover a temporary water
source before the lava pour. Bucket interactions are a **single**
``USE`` (water-recovery isolation 2026-08-20: a 3-tick burst queues a
second click). Success is ``observed_new_obsidian`` from inventory
delta + POV, not ``L1Evaluator`` Nether entry.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_oracle.py [--runs 1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

from obsidianlink.benchmark.l1_evaluator import L1Evaluator
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import L1_ENV_ID, L1ControlledEnv
from obsidianlink.experiments.l1_mechanics import (
    new_obsidian_from_evidence,
    poured_lava,
    qty,
    scooped_lava,
    used_water,
)
from obsidianlink.experiments.l1_oracle import (
    EpisodeAborted,
    HOTBAR_BUCKET,
    HOTBAR_WATER,
    OracleSession,
)
from obsidianlink.tasks.portal import L1_PORTAL_TASK, PortalGeometry

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# Reference portal geometry is kept for later gates. Gate 1 does not
# build this frame — it only needs one new obsidian block.
GEOMETRY = PortalGeometry(base_x=-1, base_y=4, z=3, backing_z=4)

# Proven live path from ``run_l1_mechanics.py`` (NEW OBSIDIAN = TRUE).
SCOOP_Z = 4.15
LOOK_LAVA_PITCH = 45.0
LOOK_DOWN_PITCH = 58.0
MAX_SCOOP_PITCHES = (LOOK_LAVA_PITCH, 40.0, 48.0, 52.0)
OBS_WAIT_TICKS = 16


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


def _fail(report: dict[str, Any], gate: str, detail: dict[str, Any]) -> None:
    report["failed_at"] = gate
    report["gates"][gate] = {"ok": False, **detail}
    print(f"[oracle] FAIL {gate}: {detail}")
    sys.stdout.flush()


def _ok(report: dict[str, Any], gate: str, detail: dict[str, Any]) -> None:
    report["gates"][gate] = {"ok": True, **detail}
    print(f"[oracle] OK {gate}")
    sys.stdout.flush()


def _look_pitch(session: OracleSession, target: float, max_tries: int = 6) -> dict[str, Any]:
    snap = session.snap()
    for _ in range(max_tries):
        current = snap["pose"].get("pitch")
        if current is None:
            snap = session.step(Action(type=ActionType.CAMERA, pitch=target - 25.0))
            break
        delta = target - float(current)
        if abs(delta) < 1.5:
            break
        snap = session.step(
            Action(type=ActionType.CAMERA, pitch=max(-30.0, min(30.0, delta)))
        )
    return snap


def _walk_forward_until_z(
    session: OracleSession, z_min: float, max_steps: int = 24
) -> dict[str, Any]:
    snap = session.snap()
    for _ in range(max_steps):
        zpos = snap["pose"].get("zpos")
        if zpos is not None and float(zpos) >= min(z_min, 4.55):
            return snap
        snap = session.step(Action(type=ActionType.MOVE, dx=1))
    return snap


def _use_once(session: OracleSession, *, sneak: bool = False) -> dict[str, Any]:
    """One bucket click. Do not burst — extra USE dumps a just-filled bucket."""
    return session.use(1, sneak=sneak)


def gate1_cast_obsidian(
    session: OracleSession, evaluator: L1Evaluator, report: dict[str, Any]
) -> bool:
    """Cast at least one new obsidian block. Not a portal frame.

    Evidence: inventory deltas (scoop / pour lava / use water) plus POV
    lava/obsidian fractions via ``new_obsidian_from_evidence``.
    ``exact_block_truth`` is unavailable on this stack without
    ObservationFromGrid (known unreliable, not used).
    """
    reset = session.snap()
    report["reset_inventory"] = dict(reset["inventory"])
    if qty(reset["inventory"], "lava_bucket") != 0:
        _fail(report, "start_inventory", {"got": reset["inventory"]})
        return False
    if qty(reset["inventory"], "bucket") < 1 or qty(reset["inventory"], "water_bucket") < 1:
        _fail(report, "start_inventory", {"got": reset["inventory"]})
        return False
    _ok(report, "start_inventory", {"got": reset["inventory"]})
    evaluator.observe_step(session.env.hidden_state)

    with session.stage("scoop_lava"):
        session.inv_trace_phase = "scoop_lava"
        session.hotbar(HOTBAR_BUCKET)
        _walk_forward_until_z(session, SCOOP_Z)
        _look_pitch(session, LOOK_LAVA_PITCH)
        before_scoop = session.snap()
        report["before_scoop"] = {
            "inventory": dict(before_scoop["inventory"]),
            "pose": before_scoop["pose"],
        }
        scooped = False
        after_scoop = before_scoop
        for pitch in MAX_SCOOP_PITCHES:
            _look_pitch(session, pitch)
            _use_once(session)
            after_scoop = session.snap()
            scooped = scooped_lava(before_scoop["inventory"], after_scoop["inventory"])
            if not scooped:
                after_scoop = session.wait(2)
                scooped = scooped_lava(before_scoop["inventory"], after_scoop["inventory"])
            if scooped:
                break
        evaluator.observe_step(session.env.hidden_state)
        session.save("01_after_scoop.png")
        report["after_scoop"] = {
            "inventory": dict(after_scoop["inventory"]),
            "selected_item": after_scoop["selected_item"],
            "pose": after_scoop["pose"],
        }
    if not scooped:
        _fail(
            report,
            "scoop_lava",
            {
                "before": before_scoop["inventory"],
                "after": after_scoop["inventory"],
                "pose": after_scoop["pose"],
            },
        )
        return False
    session.lava_sources_used += 1
    _ok(report, "scoop_lava", {"after": after_scoop["inventory"]})

    with session.stage("place_lava"):
        session.inv_trace_phase = "place_lava"
        for _ in range(8):
            session.step(Action(type=ActionType.MOVE, dx=-1))
        session.turn_yaw(90.0)
        for _ in range(6):
            session.step(Action(type=ActionType.MOVE, dx=1))
        _look_pitch(session, LOOK_DOWN_PITCH)
        before_pour = session.snap()
        _use_once(session, sneak=True)
        after_pour = session.snap()
        poured = poured_lava(before_pour["inventory"], after_pour["inventory"])
        if not poured:
            after_pour = session.wait(2)
            poured = poured_lava(before_pour["inventory"], after_pour["inventory"])
        if not poured:
            _look_pitch(session, LOOK_DOWN_PITCH + 8.0)
            _use_once(session, sneak=True)
            after_pour = session.wait(2)
            poured = poured_lava(before_pour["inventory"], after_pour["inventory"])
        evaluator.observe_step(session.env.hidden_state)
        session.save("02_after_lava.png")
        report["after_lava"] = {
            "inventory": dict(after_pour["inventory"]),
            "visual": after_pour["visual"],
            "pose": after_pour["pose"],
        }
    if not poured:
        _fail(
            report,
            "place_lava",
            {"before": before_pour["inventory"], "after": after_pour["inventory"]},
        )
        return False
    _ok(report, "place_lava", {"after": after_pour["inventory"]})
    # Let the lava source start spreading. Aiming water at the same cell
    # immediately replaces the lava with water (no obsidian) — live Run 1
    # 20260820_113730Z, lava_frac 0.38→0, obsidian_frac unchanged.
    session.inv_trace_phase = "lava_settle"
    session.wait(8)

    with session.stage("place_water"):
        session.inv_trace_phase = "place_water"
        session.hotbar(HOTBAR_WATER)
        _look_pitch(session, LOOK_DOWN_PITCH)
        # Neighbor cell, not the lava source itself.
        session.step(Action(type=ActionType.CAMERA, yaw=12.0))
        before_water = session.snap()
        _use_once(session, sneak=True)
        after_water = session.snap()
        watered = used_water(before_water["inventory"], after_water["inventory"])
        if not watered:
            after_water = session.wait(2)
            watered = used_water(before_water["inventory"], after_water["inventory"])
        if not watered:
            _use_once(session, sneak=True)
            after_water = session.wait(2)
            watered = used_water(before_water["inventory"], after_water["inventory"])
        # Look away so a sticky empty-bucket click cannot scoop the water
        # back on the next WAIT (Run 1: water_bucket reappeared on wait).
        _look_pitch(session, 18.0)
        session.wait(2)
        _look_pitch(session, LOOK_DOWN_PITCH)
        evaluator.observe_step(session.env.hidden_state)
        session.save("03_after_water.png")
        report["after_water"] = {
            "inventory": dict(session.snap()["inventory"]),
            "visual": session.snap()["visual"],
        }
        after_water = session.snap()
    if not watered:
        _fail(
            report,
            "place_water",
            {"before": before_water["inventory"], "after": after_water["inventory"]},
        )
        return False
    _ok(report, "place_water", {"after": after_water["inventory"]})

    with session.stage("wait_obsidian"):
        session.inv_trace_phase = "wait_obsidian"
        obsidian = {"ok": False}
        after_cast = after_water
        waited = 0
        while waited < OBS_WAIT_TICKS:
            after_cast = session.wait(2)
            waited += 2
            evaluator.observe_step(session.env.hidden_state)
            obsidian = new_obsidian_from_evidence(
                scooped=scooped,
                poured=poured,
                watered=watered,
                visual_before_water=after_pour["visual"],
                visual_after_water=after_cast["visual"],
            )
            if obsidian["ok"] and obsidian["obsidian_visual_rose"]:
                break
        session.save("04_after_obsidian.png")
        report["new_obsidian"] = obsidian
        report["after_cast"] = {
            "inventory": dict(after_cast["inventory"]),
            "visual": after_cast["visual"],
            "waited_ticks": waited,
        }
    if not (obsidian["ok"] and obsidian["obsidian_visual_rose"]):
        _fail(
            report,
            "new_obsidian",
            {
                **obsidian,
                "note": (
                    "lava_frac drop alone is not enough: replacing lava "
                    "with water also drops lava_frac"
                ),
            },
        )
        return False
    _ok(report, "new_obsidian", obsidian)
    report["observed_new_obsidian"] = True
    return True


def gate1_bottom_row(
    session: OracleSession, evaluator: L1Evaluator, report: dict[str, Any]
) -> bool:
    """Compatibility alias. Gate 1 no longer casts the 2-cell bottom row.

    ``exact_block_truth`` is unavailable without ObservationFromGrid.
    """
    return gate1_cast_obsidian(session, evaluator, report)


def run_once(stamp: str, run_index: int) -> dict[str, Any]:
    frames_dir = os.path.join(_RUNS_DIR, f"l1_oracle_{stamp}_run{run_index}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)

    report: dict[str, Any] = {
        "kind": "l1_scripted_oracle_gate1_one_obsidian",
        "run_index": run_index,
        "env_id": L1_ENV_ID,
        "started_utc": stamp,
        "portal_frame": False,
        "ignition": False,
        "nether_entry": False,
        "geometry": {
            "base_x": GEOMETRY.base_x,
            "base_y": GEOMETRY.base_y,
            "z": GEOMETRY.z,
            "frame_cells": [list(c) for c in GEOMETRY.frame],
            "used_by_gate1": False,
        },
        "constraints": {
            "no_equip_action": True,
            "no_placeblock": True,
            "no_observation_from_grid": True,
            "no_prebuilt_portal": True,
            "no_drawblock_obsidian_frame": True,
            "no_teleport": True,
            "no_command": True,
            "single_use_bucket_clicks": True,
        },
        "gates": {},
        "failed_at": None,
        "error": None,
        "success": False,
        "observed_new_obsidian": False,
    }

    env: L1ControlledEnv | None = None
    session: OracleSession | None = None
    evaluator = L1Evaluator()
    t0 = time.perf_counter()
    try:
        with_stage_reset = time.perf_counter()
        env = L1ControlledEnv()
        env.reset()
        report["reset_seconds"] = time.perf_counter() - with_stage_reset
        session = OracleSession(env, frames_dir)
        evaluator.observe_step(env.hidden_state)
        session.begin_inventory_trace("reset")
        session.save("00_reset.png")

        gate1_ok = gate1_cast_obsidian(session, evaluator, report)
        report["success"] = gate1_ok
        report["observed_new_obsidian"] = bool(
            gate1_ok or (report.get("new_obsidian") or {}).get("ok")
        )
        ev = evaluator.evaluate(
            L1_PORTAL_TASK,
            steps=session.steps,
            model_calls=0,
            invalid_actions=0,
            elapsed_time=time.perf_counter() - t0,
            observation=env.observe(),
            hidden_state=env.hidden_state,
        )
        report["evaluator_result"] = {
            "task_id": ev.task_id,
            "success": ev.success,
            "steps": ev.steps,
            "evidence": dict(ev.evidence),
            "note": (
                "L1Evaluator success is Nether entry. Gate 1 success is "
                "observed_new_obsidian, not this field."
            ),
        }
    except EpisodeAborted as exc:
        report["error"] = f"EpisodeAborted: {exc}"
        report["failed_at"] = report["failed_at"] or "episode_aborted"
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
        sys.stdout.flush()
    finally:
        if session is not None:
            report["steps"] = session.steps
            report["action_counts"] = dict(session.action_counts)
            report["action_sequence"] = session.action_log
            report["inventory_trace"] = session.end_inventory_trace() or session.inv_trace
            report["stages"] = session.stage_log
            if session.step_durations:
                report["step_seconds_min"] = min(session.step_durations)
                report["step_seconds_max"] = max(session.step_durations)
                report["step_seconds_last_10_avg"] = sum(
                    session.step_durations[-10:]
                ) / len(session.step_durations[-10:])
            report["lava_sources_used"] = session.lava_sources_used
            report["aborted"] = session.aborted
            report["abort_reason"] = session.abort_reason
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    report["wall_time"] = time.perf_counter() - t0
    report["frames_dir"] = frames_dir
    out_path = os.path.join(_RUNS_DIR, f"l1_oracle_{stamp}_run{run_index}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    report["_out_path"] = out_path
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted Oracle: Gate 1 one obsidian")
    parser.add_argument("--runs", type=int, default=1, help="fresh episodes (max 3 live)")
    args = parser.parse_args()
    n_runs = max(1, min(3, int(args.runs)))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    results: list[dict[str, Any]] = []
    for i in range(1, n_runs + 1):
        print(f"[oracle] === run {i}/{n_runs} ===")
        sys.stdout.flush()
        result = run_once(stamp, i)
        results.append(result)
        slowest = max(result.get("stages", []), key=lambda s: s["elapsed_seconds"], default=None)
        print(
            json.dumps(
                _jsonish(
                    {
                        "run_index": i,
                        "success": result["success"],
                        "observed_new_obsidian": result.get("observed_new_obsidian"),
                        "failed_at": result.get("failed_at"),
                        "steps": result.get("steps"),
                        "wall_time": result.get("wall_time"),
                        "action_counts": result.get("action_counts"),
                        "lava_sources_used": result.get("lava_sources_used"),
                        "aborted": result.get("aborted"),
                        "abort_reason": result.get("abort_reason"),
                        "slowest_stage": slowest,
                        "error": result.get("error"),
                    }
                ),
                indent=2,
            )
        )
        sys.stdout.flush()

    n_success = sum(1 for r in results if r["success"])
    summary = {
        "kind": "l1_oracle_gate1_one_obsidian_summary",
        "runs": n_runs,
        "successes": n_success,
        "avg_wall_time": sum(r.get("wall_time", 0.0) for r in results) / len(results),
        "avg_steps": sum(r.get("steps", 0) for r in results) / len(results),
        "results": [
            {
                "run_index": r["run_index"],
                "success": r["success"],
                "observed_new_obsidian": r.get("observed_new_obsidian"),
                "failed_at": r.get("failed_at"),
                "wall_time": r.get("wall_time"),
                "steps": r.get("steps"),
                "action_counts": r.get("action_counts"),
                "lava_sources_used": r.get("lava_sources_used"),
                "aborted": r.get("aborted"),
                "abort_reason": r.get("abort_reason"),
                "error": r.get("error"),
                "out_path": r.get("_out_path"),
            }
            for r in results
        ],
    }
    summary_path = os.path.join(_RUNS_DIR, f"l1_oracle_{stamp}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(summary), fh, indent=2)
    print("[oracle] wrote", summary_path)
    print(json.dumps(_jsonish(summary), indent=2))
    sys.stdout.flush()
    return 0 if n_success == n_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
