"""Water Recovery Isolation Test.

Minimal live experiment: place one water source on grass, recover it
with a single USE, then hold still with WAIT-only ticks.

Not Gate 1. Not portal construction. Not an Oracle or Agent run.

The 20-tick observation window after recovery forbids USE / ATTACK /
MOVE / HOTBAR / CAMERA. Every wait tick records the mapped MineRL
``use`` bit so a disappearing ``water_bucket`` cannot be blamed on an
unlogged click.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_water_recovery_isolation.py [--runs 1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import L1_ENV_ID, L1ControlledEnv
from obsidianlink.env.minerl import MineRLEnvironment
from obsidianlink.experiments.l1_mechanics import qty, scooped_water, used_water

_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

LOOK_DOWN = 58.0
FLUID_WAIT = 8
WAIT_WINDOW = 20
MAX_POUR_USES = 3
HIDDEN_KEEP = (
    "xpos",
    "ypos",
    "zpos",
    "yaw",
    "pitch",
    "biome_id",
    "can_see_sky",
    "light_level",
    "reward",
    "done",
)
FORBIDDEN_WINDOW_TYPES = frozenset(
    {
        ActionType.USE,
        ActionType.ATTACK,
        ActionType.MOVE,
        ActionType.HOTBAR,
        ActionType.CAMERA,
    }
)


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return value.item()
    except Exception:
        return str(value)


def _pose(hidden: dict[str, Any]) -> dict[str, Any]:
    return {k: hidden.get(k) for k in ("xpos", "ypos", "zpos", "yaw", "pitch")}


def _map_action(action: Action, keys: tuple[str, ...] | None) -> dict[str, Any]:
    if not keys:
        return {}
    mapped = MineRLEnvironment._to_minerl_action(action, keys)
    return {
        "use": int(mapped.get("use", 0) or 0),
        "attack": int(mapped.get("attack", 0) or 0),
        "forward": int(mapped.get("forward", 0) or 0),
        "back": int(mapped.get("back", 0) or 0),
        "left": int(mapped.get("left", 0) or 0),
        "right": int(mapped.get("right", 0) or 0),
        "camera": list(mapped.get("camera", [0.0, 0.0])),
        "sneak": int(mapped.get("sneak", 0) or 0),
        "hotbar_pressed": sorted(
            k for k, v in mapped.items() if str(k).startswith("hotbar.") and int(v or 0)
        ),
    }


def consecutive_water_bucket_run(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """First consecutive ``water_bucket >= 1`` run in ``trace``."""
    first: int | None = None
    last: int | None = None
    disappear: int | None = None
    for rec in trace:
        tick = int(rec["tick"])
        wb = int(rec.get("water_bucket") or 0)
        if wb >= 1:
            if first is None:
                first = tick
            last = tick
        elif first is not None and disappear is None:
            disappear = tick
            break
    duration = 0 if first is None or last is None else (last - first + 1)
    return {
        "water_bucket_first_tick": first,
        "water_bucket_last_present_tick": last,
        "water_bucket_duration_ticks": duration,
        "water_bucket_disappear_tick": disappear,
        "rollback": disappear is not None,
        "stable_at_end": bool(trace) and int(trace[-1].get("water_bucket") or 0) >= 1,
    }


def inventory_stable_tick(trace: list[dict[str, Any]]) -> int | None:
    """First tick after which inventory no longer changes in ``trace``."""
    if not trace:
        return None
    last_inv = None
    last_change = int(trace[0]["tick"])
    for rec in trace:
        inv = rec.get("inventory")
        if inv != last_inv:
            last_change = int(rec["tick"])
            last_inv = inv
    return last_change


def analyze_window(trace: list[dict[str, Any]]) -> dict[str, Any]:
    run = consecutive_water_bucket_run(trace)
    actions = [str(r.get("action")) for r in trace]
    minerl_uses = [int((r.get("minerl") or {}).get("use") or 0) for r in trace]
    disappear = run["water_bucket_disappear_tick"]
    at_drop = next((r for r in trace if r["tick"] == disappear), None)
    return {
        **run,
        "n_ticks": len(trace),
        "actions": actions,
        "any_use": any(a == "use" for a in actions),
        "any_forbidden": any(a in {"use", "attack", "move", "hotbar", "camera"} for a in actions),
        "minerl_use_max": max(minerl_uses, default=0),
        "done_changed": len({bool(r.get("done")) for r in trace}) > 1,
        "reward_changed": len({str(r.get("reward")) for r in trace}) > 1,
        "selected_item_changed": len({str(r.get("selected_item")) for r in trace}) > 1,
        "pose_changed": len({json.dumps(r.get("pose"), sort_keys=True) for r in trace}) > 1,
        "inventory_stable_tick": inventory_stable_tick(trace),
        "max_step_latency": max((float(r.get("step_latency") or 0.0) for r in trace), default=0.0),
        "at_disappear": None
        if at_drop is None
        else {
            "tick": at_drop["tick"],
            "action": at_drop.get("action"),
            "minerl_use": (at_drop.get("minerl") or {}).get("use"),
            "inventory": at_drop.get("inventory"),
            "selected_item": at_drop.get("selected_item"),
            "reward": at_drop.get("reward"),
            "done": at_drop.get("done"),
            "pose": at_drop.get("pose"),
        },
    }


class IsolationSession:
    def __init__(self, env: L1ControlledEnv) -> None:
        self.env = env
        self.tick = -1
        self.trace: list[dict[str, Any]] = []
        self.keys = tuple(env.action_space_keys or ())

    def snapshot(self, *, phase: str, action: str, mapped: dict[str, Any], latency: float) -> dict[str, Any]:
        obs = self.env.observe()
        hidden = self.env.hidden_state
        inv = dict(obs.inventory or {})
        rec = {
            "tick": self.tick,
            "phase": phase,
            "action": action,
            "inventory": inv,
            "bucket": qty(inv, "bucket"),
            "water_bucket": qty(inv, "water_bucket"),
            "selected_item": obs.selected_item,
            "pose": _pose(hidden),
            "reward": hidden.get("reward"),
            "done": bool(hidden.get("done")),
            "step_latency": latency,
            "wall_clock": time.time(),
            "wall_clock_iso": datetime.now(timezone.utc).isoformat(),
            "minerl": mapped,
            "hidden": {k: hidden.get(k) for k in HIDDEN_KEEP},
        }
        self.trace.append(rec)
        print(
            f"[{phase}] tick={self.tick:3d} action={action:<8} "
            f"bucket={rec['bucket']} water_bucket={rec['water_bucket']} "
            f"selected={rec['selected_item']} done={rec['done']} "
            f"reward={rec['reward']} minerl_use={mapped.get('use')} "
            f"latency={latency:.4f} pose={rec['pose']}"
        )
        sys.stdout.flush()
        return rec

    def step(self, action: Action, *, phase: str) -> dict[str, Any]:
        if phase == "wait_window" and action.type in FORBIDDEN_WINDOW_TYPES:
            raise RuntimeError(f"wait_window forbids {action.type.value}")
        mapped = _map_action(action, self.keys or self.env.action_space_keys)
        t0 = time.perf_counter()
        self.env.step(action)
        latency = time.perf_counter() - t0
        if not self.keys:
            self.keys = tuple(self.env.action_space_keys or ())
            mapped = _map_action(action, self.keys)
        self.tick += 1
        return self.snapshot(
            phase=phase, action=action.type.value, mapped=mapped, latency=latency
        )

    def look_down(self, *, phase: str) -> None:
        for _ in range(3):
            hidden = self.env.hidden_state
            current = hidden.get("pitch")
            if current is None:
                self.step(Action(type=ActionType.CAMERA, pitch=LOOK_DOWN - 25.0), phase=phase)
                return
            delta = LOOK_DOWN - float(current)
            if abs(delta) < 1.5:
                return
            self.step(
                Action(
                    type=ActionType.CAMERA,
                    pitch=max(-30.0, min(30.0, delta)),
                ),
                phase=phase,
            )


def _save_frame(path: str, frame: Any) -> None:
    if frame is None:
        return
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(frame)
        if arr.ndim != 3:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(arr.astype(np.uint8)).save(path)
    except Exception:
        return


def _phase(trace: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [r for r in trace if r.get("phase") == name]


def _answers(report: dict[str, Any]) -> dict[str, Any]:
    reset = report.get("reset_inventory") or {}
    pour = analyze_window(_phase(report.get("trace") or [], "pour"))
    fluid = analyze_window(_phase(report.get("trace") or [], "fluid_wait"))
    recover = analyze_window(_phase(report.get("trace") or [], "recover"))
    window = report.get("wait_window_analysis") or {}
    post = _phase(report.get("trace") or [], "recover") + _phase(
        report.get("trace") or [], "wait_window"
    )
    post_run = consecutive_water_bucket_run(post)
    return {
        "1_before_pour_inventory": {
            "bucket": qty(reset, "bucket"),
            "water_bucket": qty(reset, "water_bucket"),
            "inventory": reset,
        },
        "2_after_pour_inventory_stable_tick": report.get("pour_stable_tick"),
        "inv_after_pour": report.get("inv_after_pour"),
        "pour_analysis": pour,
        "fluid_wait_analysis": fluid,
        "3_water_bucket_first_tick_after_recover": post_run["water_bucket_first_tick"],
        "4_water_bucket_consecutive_ticks": post_run["water_bucket_duration_ticks"],
        "5_disappear_tick": post_run["water_bucket_disappear_tick"],
        "6_any_use_during_wait_window": window.get("any_use"),
        "6_minerl_use_max_in_wait_window": window.get("minerl_use_max"),
        "7_at_disappear": window.get("at_disappear") or recover.get("at_disappear"),
        "7_reward_done_pose_selected_changed_in_wait": {
            "reward_changed": window.get("reward_changed"),
            "done_changed": window.get("done_changed"),
            "pose_changed": window.get("pose_changed"),
            "selected_item_changed": window.get("selected_item_changed"),
        },
        "9_looks_transient": bool(
            post_run["rollback"] and not window.get("any_use") and window.get("minerl_use_max") == 0
        ),
        "wait_window_stable_at_end": window.get("stable_at_end"),
    }


def run_once(stamp: str, run_index: int) -> dict[str, Any]:
    frames_dir = os.path.join(
        _RUNS_DIR, f"water_recovery_iso_{stamp}_run{run_index}_frames"
    )
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(_RUNS_DIR, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "water_recovery_isolation",
        "run_index": run_index,
        "env_id": L1_ENV_ID,
        "success": False,
        "oracle_or_agent_run": False,
        "gate1": False,
        "wait_window_ticks": WAIT_WINDOW,
        "recover_use_ticks": 1,
    }
    env: L1ControlledEnv | None = None
    session: IsolationSession | None = None
    t0 = time.perf_counter()
    try:
        env = L1ControlledEnv()
        env.reset()
        session = IsolationSession(env)
        reset_obs = env.observe()
        reset_inv = dict(reset_obs.inventory or {})
        report["reset_inventory"] = reset_inv
        report["reset_selected_item"] = reset_obs.selected_item
        print("[iso] reset inventory", reset_inv)
        sys.stdout.flush()
        if qty(reset_inv, "water_bucket") < 1 or qty(reset_inv, "bucket") < 1:
            report["failed_at"] = "start_inventory"
            return report
        _save_frame(os.path.join(frames_dir, "00_reset.png"), reset_obs.frame)

        session.step(Action(type=ActionType.HOTBAR, target="1"), phase="setup")
        session.look_down(phase="setup")
        before_pour = dict(env.observe().inventory or {})
        report["inv_before_pour"] = before_pour

        watered = False
        for _ in range(MAX_POUR_USES):
            after = session.step(Action(type=ActionType.USE, sneak=True), phase="pour")
            watered = used_water(before_pour, after["inventory"])
            if watered:
                break
        report["watered"] = watered
        report["inv_after_pour_use"] = dict(env.observe().inventory or {})
        if not watered:
            report["failed_at"] = "pour_water_failed"
            _save_frame(os.path.join(frames_dir, "01_pour_failed.png"), env.observe().frame)
            return report

        for _ in range(FLUID_WAIT):
            session.step(Action(type=ActionType.WAIT), phase="fluid_wait")
        report["inv_after_pour"] = dict(env.observe().inventory or {})
        report["pour_stable_tick"] = inventory_stable_tick(
            _phase(session.trace, "pour") + _phase(session.trace, "fluid_wait")
        )
        _save_frame(os.path.join(frames_dir, "02_after_pour.png"), env.observe().frame)

        session.step(Action(type=ActionType.HOTBAR, target="2"), phase="recover_setup")
        session.look_down(phase="recover_setup")
        before_recover = dict(env.observe().inventory or {})
        report["inv_before_recover"] = before_recover

        recover_rec = session.step(Action(type=ActionType.USE, sneak=True), phase="recover")
        report["recovered_immediate"] = scooped_water(before_recover, recover_rec["inventory"])
        report["inv_after_recover_use"] = dict(recover_rec["inventory"])
        _save_frame(os.path.join(frames_dir, "03_after_recover_use.png"), env.observe().frame)

        for _ in range(WAIT_WINDOW):
            rec = session.step(Action(type=ActionType.WAIT), phase="wait_window")
            if rec.get("minerl", {}).get("use"):
                report["failed_at"] = "wait_window_emitted_use"
                break
        _save_frame(os.path.join(frames_dir, "04_after_wait_window.png"), env.observe().frame)

        report["trace"] = session.trace
        report["wait_window_analysis"] = analyze_window(_phase(session.trace, "wait_window"))
        report["recover_analysis"] = analyze_window(_phase(session.trace, "recover"))
        report["post_recover_analysis"] = analyze_window(
            _phase(session.trace, "recover") + _phase(session.trace, "wait_window")
        )
        report["answers"] = _answers(report)
        report["success"] = report.get("failed_at") is None
        print("[iso] wait_window", report["wait_window_analysis"])
        print("[iso] post_recover", report["post_recover_analysis"])
        print("[iso] answers", json.dumps(_jsonish(report["answers"]), indent=2))
        sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        print("[iso] error", report["error"])
        sys.stdout.flush()
    finally:
        if session is not None and "trace" not in report:
            report["trace"] = session.trace
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    report["wall_time"] = time.perf_counter() - t0
    report["frames_dir"] = frames_dir
    out_path = os.path.join(_RUNS_DIR, f"water_recovery_iso_{stamp}_run{run_index}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(report), fh, indent=2)
    report["_out_path"] = out_path
    print("[iso] wrote", out_path)
    sys.stdout.flush()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Water recovery isolation (WAIT-only window)")
    parser.add_argument("--runs", type=int, default=1, help="fresh episodes (max 2)")
    args = parser.parse_args()
    n_runs = max(1, min(2, int(args.runs)))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    results: list[dict[str, Any]] = []
    for i in range(1, n_runs + 1):
        print(f"[iso] === run {i}/{n_runs} ===")
        sys.stdout.flush()
        results.append(run_once(stamp, i))
    summary = {
        "kind": "water_recovery_isolation_summary",
        "runs": [
            {
                "run_index": r["run_index"],
                "success": r.get("success"),
                "watered": r.get("watered"),
                "recovered_immediate": r.get("recovered_immediate"),
                "inv_before_pour": r.get("inv_before_pour"),
                "inv_after_pour": r.get("inv_after_pour"),
                "inv_after_recover_use": r.get("inv_after_recover_use"),
                "wait_window_analysis": r.get("wait_window_analysis"),
                "post_recover_analysis": r.get("post_recover_analysis"),
                "answers": r.get("answers"),
                "failed_at": r.get("failed_at"),
                "error": r.get("error"),
                "wall_time": r.get("wall_time"),
                "out_path": r.get("_out_path"),
            }
            for r in results
        ],
    }
    path = os.path.join(_RUNS_DIR, f"water_recovery_iso_{stamp}_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonish(summary), fh, indent=2)
    print("[iso] summary", path)
    print(json.dumps(_jsonish(summary), indent=2))
    sys.stdout.flush()
    return 0 if all(r.get("success") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
