from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VENDORED_MINERL = ROOT / "vendor/minerl"
PINNED_JAVA_HOME = Path("/opt/anaconda3/envs/mc-agent")


def _configure_pinned_java_home() -> None:
    """Keep MineRL on the repository's locked Java 8 runtime."""
    java = PINNED_JAVA_HOME / "bin/java"
    if not java.is_file():
        raise RuntimeError(f"missing pinned Java runtime: {java}")
    os.environ["JAVA_HOME"] = str(PINNED_JAVA_HOME)
    os.environ["PATH"] = (
        f"{PINNED_JAVA_HOME / 'bin'}:{os.environ.get('PATH', '')}"
    )


_configure_pinned_java_home()
for import_root in (ROOT, VENDORED_MINERL):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from obsidianlink.core.types import TaskInstance
from obsidianlink.drivers.scripted_a0 import (
    FAILURE_INJECTIONS,
    run_scripted_a0,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation import PortalEvaluator


TASK_PATH = ROOT / "benchmark/instances/route_a_a0_phase3.json"
EXPERIMENT_PATH = ROOT / "configs/experiments/phase3_scripted_a0.json"
LIVE_WINDOW_TITLE = "ObsidianLink Live View - AI First Person"


class _LiveViewer:
    """Best-effort Cocoa window for the exact RGB frames seen by the agent."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self._enabled = True
        try:
            cv2.namedWindow(LIVE_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(LIVE_WINDOW_TITLE, 960, 540)
            loading = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(
                loading,
                "Starting Minecraft / MineRL...",
                (145, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(LIVE_WINDOW_TITLE, loading)
            cv2.waitKey(1)
        except cv2.error as error:
            self._enabled = False
            print(f"Live viewer unavailable: {error}", file=sys.stderr)

    def show(self, observation: Any, context: dict[str, Any]) -> None:
        if not self._enabled:
            return
        cv2 = self._cv2
        try:
            frame = cv2.cvtColor(observation.frame, cv2.COLOR_RGB2BGR)
            frame = frame.copy()
            label = str(context.get("label", ""))
            phase = str(context.get("phase", ""))
            status = f"step {observation.step_id} | {phase} | {label}"
            cv2.rectangle(frame, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(
                frame,
                status[:95],
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(LIVE_WINDOW_TITLE, frame)
            cv2.waitKey(1)
        except cv2.error as error:
            self._enabled = False
            print(f"Live viewer stopped: {error}", file=sys.stderr)

    def close(self) -> None:
        if self._enabled:
            try:
                self._cv2.destroyWindow(LIVE_WINDOW_TITLE)
                self._cv2.waitKey(1)
            except self._cv2.error:
                pass


def _load_task() -> TaskInstance:
    return TaskInstance.from_dict(
        json.loads(TASK_PATH.read_text(encoding="utf-8"))
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "frame"


class _RunFrameRecorder:
    """Persist only agent-visible frames; evaluator truth is never accepted."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._initial_saved = False

    def record(self, observation: Any, context: dict[str, Any]) -> None:
        label = str(context.get("label", "frame"))
        action_type = str(context.get("action_type", "wait"))
        if not self._initial_saved:
            destination = self._run_dir / "initial.png"
            self._initial_saved = True
        elif action_type == "wait":
            return
        else:
            destination = (
                self._run_dir
                / "decision_frames"
                / f"{observation.step_id:04d}-{_safe_artifact_name(label)}.png"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation.frame).save(destination)


def _write_reproducibility_snapshot(run_dir: Path, args: argparse.Namespace) -> None:
    shutil.copyfile(TASK_PATH, run_dir / "task_instance.json")
    shutil.copyfile(EXPERIMENT_PATH, run_dir / "experiment_config.json")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    version = {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "runner_arguments": _json_ready(vars(args)),
        "task_instance": str(TASK_PATH.relative_to(ROOT)),
        "experiment_config": str(EXPERIMENT_PATH.relative_to(ROOT)),
    }
    (run_dir / "code_version.json").write_text(
        json.dumps(version, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _foreground_minecraft_window(timeout_seconds: float = 60.0) -> None:
    """Bring the MineRL Java window forward on macOS.

    MineRL launches Minecraft in a separate process group, which can leave its
    window behind Terminal or Codex even though rendering is enabled. Failure
    to activate the window must not affect the controlled environment run.
    """
    if sys.platform != "darwin":
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process = subprocess.run(
            ["pgrep", "-n", "-f", "mcprec-6.13.jar"],
            check=False,
            capture_output=True,
            text=True,
        )
        pid = process.stdout.strip()
        if pid.isdigit():
            script = (
                'tell application "System Events" to set frontmost of '
                f"first process whose unix id is {pid} to true"
            )
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/phase3-scripted-a0",
    )
    parser.add_argument("--max-portal-wait-steps", type=int, default=120)
    parser.add_argument("--max-placement-retries", type=int, default=0)
    parser.add_argument("--step-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--failure-injection",
        choices=sorted(FAILURE_INJECTIONS),
        help="run one bounded deterministic negative-path injection",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="show a live window with the exact first-person frames seen by AI",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_reproducibility_snapshot(run_dir, args)
    if args.watch:
        threading.Thread(
            target=_foreground_minecraft_window,
            name="minecraft-window-foreground",
            daemon=True,
        ).start()
    live_viewer = _LiveViewer() if args.watch else None
    frame_recorder = _RunFrameRecorder(run_dir)

    def record_observation(observation: Any, context: dict[str, Any]) -> None:
        frame_recorder.record(observation, context)
        if live_viewer is not None:
            live_viewer.show(observation, context)

    backend = MineRLEnvironmentBackend()
    backend.open()
    try:
        with (run_dir / "events.jsonl").open(
            "w",
            encoding="utf-8",
        ) as event_handle:
            def write_event(event: dict[str, Any]) -> None:
                event_handle.write(
                    json.dumps(
                        _json_ready(dict(event)),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                event_handle.flush()

            result = run_scripted_a0(
                backend,
                _load_task(),
                max_portal_wait_steps=args.max_portal_wait_steps,
                max_placement_retries=args.max_placement_retries,
                step_timeout_seconds=args.step_timeout_seconds,
                failure_injection=args.failure_injection,
                event_sink=write_event,
                observation_sink=record_observation,
            )
        backend.mark_terminated(
            step_id=result.steps_completed,
            reason="scripted_a0_driver_complete",
        )
        evaluation_state = backend.get_evaluation_state()
        evaluation_result = PortalEvaluator().evaluate(evaluation_state)
        evaluator_events = tuple(evaluation_state.milestone_events())
        with (run_dir / "evaluator_events.jsonl").open(
            "w",
            encoding="utf-8",
        ) as evaluator_handle:
            for event in evaluator_events:
                evaluator_handle.write(
                    json.dumps(
                        _json_ready(event.to_dict()),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        Image.fromarray(result.final_observation.frame).save(
            run_dir / "final.png"
        )
        if result.status == "failed":
            status = "failed"
        elif evaluation_result.success:
            status = "passed"
        else:
            status = "blocked"
        blocked_reason = result.blocked_reason
        if blocked_reason is None and not evaluation_result.success:
            blocked_reason = (
                "formal PortalEvaluator blocked success: "
                + ", ".join(evaluation_result.blocking_conditions)
            )
        summary = _json_ready(
            {
                "status": status,
                "driver_status": result.status,
                "steps_completed": result.steps_completed,
                "planned_steps": result.planned_steps,
                "wait_steps": result.wait_steps,
                "step_timeout_seconds": args.step_timeout_seconds,
                "final_dimension": result.final_dimension,
                "portal_activated": result.portal_activated,
                "entered_nether": result.entered_nether,
                "terminated": evaluation_state.episode_terminated,
                "evaluation_evidence": dict(result.evaluation_evidence),
                "formal_evaluation": {
                    "success": evaluation_result.success,
                    "step_id": evaluation_result.step_id,
                    "milestones": list(evaluation_result.milestones),
                    "blocking_conditions": list(
                        evaluation_result.blocking_conditions
                    ),
                    "failure_type": evaluation_result.failure_type,
                    "failure_step": evaluation_result.failure_step,
                    "last_successful_milestone": (
                        evaluation_result.last_successful_milestone
                    ),
                    "episode_terminated": (
                        evaluation_result.episode_terminated
                    ),
                    "terminated_step": evaluation_result.terminated_step,
                    "terminated_reason": evaluation_result.terminated_reason,
                    "entered_via_episode_portal": (
                        evaluation_result.entered_via_episode_portal
                    ),
                },
                "evaluator_state": {
                    "entered_via_episode_portal_by_agent": dict(
                        evaluation_state.entered_via_episode_portal_by_agent
                    ),
                    "transition_step_by_agent": dict(
                        evaluation_state.transition_step_by_agent
                    ),
                    "pre_transition_position_by_agent": dict(
                        evaluation_state.pre_transition_position_by_agent
                    ),
                    "matched_frame_identity_by_agent": dict(
                        evaluation_state.matched_frame_identity_by_agent
                    ),
                    "attributed_obsidian_offsets": [
                        list(offset)
                        for offset in evaluation_state.attributed_obsidian_offsets
                    ],
                    "external_obsidian_offsets": [
                        list(offset)
                        for offset in evaluation_state.external_obsidian_offsets
                    ],
                    "milestone_events": [
                        event.to_dict() for event in evaluator_events
                    ],
                },
                "blocked_reason": blocked_reason,
            }
        )
        summary["run_dir"] = str(run_dir)
        summary["artifacts"] = {
            "task_instance": "task_instance.json",
            "experiment_config": "experiment_config.json",
            "code_version": "code_version.json",
            "initial_frame": "initial.png",
            "final_frame": "final.png",
            "decision_frames": "decision_frames/",
            "action_events": "events.jsonl",
            "evaluator_events": "evaluator_events.jsonl",
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if status == "passed" else 2
    except Exception as error:
        summary = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        backend.close()
        if live_viewer is not None:
            live_viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
