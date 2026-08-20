"""Isolation probe: locate the water_bucket inventory rollback tick.

Live Gate 1 (2026-08-20) recovered a water_bucket after cell 0, then
lost it during a later look_at+wait that sent no USE. This script runs
one cell only (scoop lava → pour lava → pour water → recover water)
and then holds still: WAIT-only ticks, then CAMERA look_at, then more
WAIT. OracleSession prints inventory every traced tick.

Not a Gate 1 success claim. Not an Agent run.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_water_rollback_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from obsidianlink.benchmark.l1_evaluator import L1Evaluator
from obsidianlink.env.l1_scene import L1_ENV_ID, L1ControlledEnv, LAVA_Y, LAVA_Z1
from obsidianlink.experiments.l1_mechanics import scooped_water, used_water
from obsidianlink.experiments.l1_oracle import (
    EpisodeAborted,
    HOTBAR_BUCKET,
    HOTBAR_COBBLE,
    HOTBAR_WATER,
    OracleSession,
)
from obsidianlink.experiments.run_l1_oracle import GEOMETRY, _jsonish

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

HOLD_WAIT_TICKS = 12
POST_LOOK_WAIT_TICKS = 8


def _first_wb_loss(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First tick where water_bucket drops after having been >= 1."""
    seen = False
    prev_wb = 0
    for entry in trace:
        wb = int(entry.get("water_bucket") or 0)
        if wb >= 1:
            seen = True
        if seen and wb < prev_wb:
            return dict(entry)
        prev_wb = wb
    return None


def run_probe() -> dict[str, Any]:
    raise RuntimeError(
        "Water rollback probe is retired. Isolation concluded there is no "
        "WAIT-only inventory rollback. Use run_water_recovery_isolation.py."
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    frames_dir = os.path.join(_RUNS_DIR, f"l1_water_rollback_{stamp}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)

    report: dict[str, Any] = {
        "kind": "l1_water_rollback_probe",
        "env_id": L1_ENV_ID,
        "started_utc": stamp,
        "success": False,
        "note": (
            "Isolation only: one cell scoop/pour/water/recover, then WAIT "
            "vs CAMERA to locate water_bucket rollback. Not Gate 1."
        ),
    }

    targets = [c for c in GEOMETRY.frame if c[1] == GEOMETRY.base_y]
    # Same cell 0 as Gate 1: first bottom-row frame cell, scooping from
    # the east pool column. Keep the probe path identical to the run that
    # first showed the rollback.
    target = targets[0]
    tx, ty, tz = target
    tx0 = min(c[0] for c in targets)
    tx1 = max(c[0] for c in targets)
    source = (tx1, LAVA_Y, LAVA_Z1)
    water_backing = (tx + 0.5, ty - 0.05, tz - 0.5)
    verify_aim = (tx + 0.5, ty + 0.5, tz + 0.5)
    report["target"] = list(target)
    report["source"] = list(source)

    env: L1ControlledEnv | None = None
    session: OracleSession | None = None
    evaluator = L1Evaluator()
    t0 = time.perf_counter()
    try:
        env = L1ControlledEnv()
        env.reset()
        session = OracleSession(env, frames_dir)
        evaluator.observe_step(env.hidden_state)
        session.save("00_reset.png")

        with session.stage("build_mold"):
            mold_walls = [
                ((tx0 - 1, ty, tz), (tx0 - 0.5, ty - 0.05, tz + 0.5)),
                ((tx1 + 1, ty, tz), (tx1 + 1.5, ty - 0.05, tz + 0.5)),
            ]
            session.walk_toward(tx0 + 0.5, tz - 1.0, max_steps=10)
            mold_ok = session.build_mold(mold_walls, hotbar_slot=HOTBAR_COBBLE)
            report["mold_ok"] = mold_ok
            if not all(mold_ok):
                report["failed_at"] = "mold_build_failed"
                return report

        result = _scoop_and_pour_lava(
            session, evaluator, report, index=0, target=target, source=source
        )
        if result is None:
            report["failed_at"] = "scoop_lava_failed"
            return report
        poured, _verify_aim, _before_visual = result
        report["lava_poured"] = poured
        if not poured:
            report["failed_at"] = "pour_lava_failed"
            return report

        with session.stage("pour_water_0"):
            session.hotbar(HOTBAR_WATER)
            before_water = session.snap()
            session.cast_fluid_at(
                target, backing_face_from=water_backing, aim_tolerance=6.0, settle_wait=0
            )
            watered = used_water(before_water["inventory"], session.snap()["inventory"])
            if not watered:
                session.cast_fluid_at(
                    target,
                    backing_face_from=water_backing,
                    ticks=4,
                    settle_wait=0,
                    aim_tolerance=6.0,
                )
                watered = used_water(before_water["inventory"], session.snap()["inventory"])
            report["watered"] = watered
            report["inv_after_pour_water"] = dict(session.snap()["inventory"])
            session.save("after_pour_water.png")
        if not watered:
            report["failed_at"] = "pour_water_failed"
            return report

        session.begin_inventory_trace("recover_setup")
        session.hotbar(HOTBAR_BUCKET)
        before_recover = session.snap()
        recovered = False
        after_recover = before_recover

        session.inv_trace_phase = "recover_aim"
        session.look_at(water_backing, max_tries=8)
        aimed = session.is_aimed_at(water_backing, tolerance=6.0)
        report["recover_aimed"] = aimed
        if aimed:
            session.inv_trace_phase = "recover_use"
            session.use(3)
            session.inv_trace_phase = "recover_wait1"
            after_recover = session.wait(1)
            recovered = scooped_water(before_recover["inventory"], after_recover["inventory"])
        report["recovered"] = recovered
        report["inv_after_recover"] = dict(after_recover["inventory"])
        session.save("after_recover.png")

        session.inv_trace_phase = "hold_wait"
        session.wait(HOLD_WAIT_TICKS)
        report["inv_after_hold_wait"] = dict(session.snap()["inventory"])
        session.save("after_hold_wait.png")

        session.inv_trace_phase = "look_at_verify"
        session.look_at(verify_aim, max_tries=8)
        report["inv_after_look_at"] = dict(session.snap()["inventory"])
        session.save("after_look_at.png")

        session.inv_trace_phase = "hold_wait_after_look"
        session.wait(POST_LOOK_WAIT_TICKS)
        report["inv_after_post_look_wait"] = dict(session.snap()["inventory"])
        session.save("after_post_look_wait.png")

        trace = session.end_inventory_trace()
        report["inv_trace"] = trace
        loss = _first_wb_loss(trace)
        report["first_water_bucket_loss"] = loss
        report["water_bucket_ever_present"] = any(
            int(e.get("water_bucket") or 0) >= 1 for e in trace
        )
        report["water_bucket_present_at_end"] = (
            int(session.snap()["inventory"].get("water_bucket") or 0) >= 1
        )
        if loss is None:
            report["rollback_located"] = False
            report["rollback_summary"] = (
                "water_bucket never dropped after appearing"
                if report["water_bucket_ever_present"]
                else "water_bucket never appeared in the traced window"
            )
        else:
            report["rollback_located"] = True
            report["rollback_summary"] = (
                f"water_bucket dropped at step={loss['step']} "
                f"phase={loss['phase']} type={loss['type']} sneak={loss['sneak']}"
            )
        print("[probe]", report["rollback_summary"])
        sys.stdout.flush()
        report["success"] = True
    except EpisodeAborted as exc:
        report["error"] = f"EpisodeAborted: {exc}"
        report["failed_at"] = "episode_aborted"
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
        sys.stdout.flush()
    finally:
        if session is not None:
            report["steps"] = session.steps
            report["action_counts"] = dict(session.action_counts)
            report["stages"] = session.stage_log
            report["aborted"] = session.aborted
            report["abort_reason"] = session.abort_reason
            if session.inv_trace and "inv_trace" not in report:
                report["inv_trace"] = session.inv_trace
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    report["wall_time"] = time.perf_counter() - t0
    report["frames_dir"] = frames_dir
    out_path = os.path.join(_RUNS_DIR, f"l1_water_rollback_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    report["_out_path"] = out_path
    print("[probe] wrote", out_path)
    sys.stdout.flush()
    return report


def main() -> int:
    result = run_probe()
    print(
        json.dumps(
            _jsonish(
                {
                    "success": result.get("success"),
                    "failed_at": result.get("failed_at"),
                    "watered": result.get("watered"),
                    "recovered": result.get("recovered"),
                    "rollback_located": result.get("rollback_located"),
                    "rollback_summary": result.get("rollback_summary"),
                    "first_water_bucket_loss": result.get("first_water_bucket_loss"),
                    "inv_after_recover": result.get("inv_after_recover"),
                    "inv_after_hold_wait": result.get("inv_after_hold_wait"),
                    "inv_after_look_at": result.get("inv_after_look_at"),
                    "inv_after_post_look_wait": result.get("inv_after_post_look_wait"),
                    "steps": result.get("steps"),
                    "wall_time": result.get("wall_time"),
                    "error": result.get("error"),
                    "out_path": result.get("_out_path"),
                }
            ),
            indent=2,
        )
    )
    sys.stdout.flush()
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
