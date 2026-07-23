#!/usr/bin/env python3
"""Run a local, keyboard-operated MineRL FindCave evidence session.

This is deliberately separate from the Qwen Agent: no model is loaded and no
model decision is used. The player controls a live first-person OpenCV window;
every candidate frame saved with ``C`` remains available for the existing
offline cave-evidence review.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_agent.actions import water_hazard_direction
from mc_agent.env import MineRLEnvAdapter


TARGET_TICK_SECONDS = 1.0 / 20.0
WINDOW_NAME = "MineRL Manual FindCave"
CAMERA_STEP_DEGREES = 15.0
CAMERA_MIN_PITCH_DEGREES = -45.0
CAMERA_MAX_PITCH_DEGREES = 45.0


class ManualControls:
    """Translate local window keys into a safe, one-tick MineRL action."""

    def __init__(self) -> None:
        self.forward = False
        self.back = False
        self.left = False
        self.right = False
        self.sprint = True
        self.pending_jump = False
        self.pending_pitch = 0.0
        self.pending_yaw = 0.0
        self.commanded_pitch = 0.0
        self.capture_requested = False
        self.quit_requested = False
        self.notice = "Ready"

    def handle_key(self, key: int) -> None:
        """Apply one local key event; unknown keys intentionally do nothing."""
        if key in (ord("q"), 27):
            self.quit_requested = True
            self.notice = "Ending manual session"
        elif key == ord("w"):
            self.forward = not self.forward
            if self.forward:
                self.back = False
            self.notice = f"forward={'on' if self.forward else 'off'}"
        elif key == ord("s"):
            self.back = not self.back
            if self.back:
                self.forward = False
            self.notice = f"back={'on' if self.back else 'off'}"
        elif key == ord("a"):
            self.left = not self.left
            if self.left:
                self.right = False
            self.notice = f"left={'on' if self.left else 'off'}"
        elif key == ord("d"):
            self.right = not self.right
            if self.right:
                self.left = False
            self.notice = f"right={'on' if self.right else 'off'}"
        elif key == ord("r"):
            self.sprint = not self.sprint
            self.notice = f"sprint={'on' if self.sprint else 'off'}"
        elif key == ord(" "):
            self.pending_jump = True
            self.notice = "jump"
        elif key == ord("i"):
            self.pending_pitch = -CAMERA_STEP_DEGREES
            self.notice = "look up"
        elif key == ord("k"):
            self.pending_pitch = CAMERA_STEP_DEGREES
            self.notice = "look down"
        elif key == ord("j"):
            self.pending_yaw = -CAMERA_STEP_DEGREES
            self.notice = "look left"
        elif key == ord("l"):
            self.pending_yaw = CAMERA_STEP_DEGREES
            self.notice = "look right"
        elif key == ord("c"):
            self.capture_requested = True
            self.notice = "candidate frame saved"

    def next_action(self, action_space, *, center_water_hazard: bool) -> dict:
        """Return one action with human movement bounded by local safety rules."""
        action = action_space.no_op()
        if center_water_hazard and self.forward:
            self.forward = False
            self.notice = "forward paused: water ahead"
        next_pitch = self.commanded_pitch + self.pending_pitch
        applied_pitch = self.pending_pitch
        if next_pitch < CAMERA_MIN_PITCH_DEGREES:
            applied_pitch = CAMERA_MIN_PITCH_DEGREES - self.commanded_pitch
        elif next_pitch > CAMERA_MAX_PITCH_DEGREES:
            applied_pitch = CAMERA_MAX_PITCH_DEGREES - self.commanded_pitch
        self.commanded_pitch += applied_pitch
        action["camera"] = np.asarray([applied_pitch, self.pending_yaw], dtype=np.float32)
        action["forward"] = int(self.forward)
        action["back"] = int(self.back)
        action["left"] = int(self.left)
        action["right"] = int(self.right)
        action["jump"] = int(self.pending_jump and self.forward)
        action["attack"] = 0
        action["sprint"] = int(self.sprint and self.forward)
        action["ESC"] = 0
        self.pending_jump = False
        self.pending_pitch = 0.0
        self.pending_yaw = 0.0
        return action


def _draw_overlay(pov: np.ndarray, controls: ManualControls, tick: int) -> np.ndarray:
    frame = cv2.cvtColor(pov, cv2.COLOR_RGB2BGR)
    lines = [
        f"tick {tick} | {controls.notice}",
        "W/A/S/D toggle move | I/J/K/L look | Space jump | R sprint",
        "C save candidate | Q or Esc end | forward pauses at center water",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_manual_session(
    *, seed: int | None, ticks: int, mission_ticks: int, output_root: Path
) -> Path:
    """Open one manual session and persist candidate evidence under ``runs/``."""
    if ticks < 1 or mission_ticks < ticks:
        raise ValueError("ticks must be positive and mission_ticks must be at least ticks")
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    candidates_dir = session_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=False)
    events_path = session_dir / "events.jsonl"
    controls = ManualControls()
    completed_ticks = 0
    candidates: list[str] = []
    stopped_reason = "tick_budget"

    with MineRLEnvAdapter(max_episode_steps=mission_ticks) as adapter:
        if seed is not None:
            adapter.seed(seed)
        observation = adapter.reset()
        Image.fromarray(observation["pov"]).save(session_dir / "initial.png")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        try:
            while completed_ticks < ticks and not controls.quit_requested:
                tick_started = time.perf_counter()
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    controls.handle_key(key)
                if controls.capture_requested:
                    filename = f"candidate-tick-{completed_ticks:05d}.png"
                    Image.fromarray(observation["pov"]).save(candidates_dir / filename)
                    candidates.append(filename)
                    controls.capture_requested = False
                if controls.quit_requested:
                    stopped_reason = "user_quit"
                    break
                action = controls.next_action(
                    adapter.action_space,
                    center_water_hazard=water_hazard_direction(observation["pov"])
                    == "center",
                )
                step = adapter.step(action)
                completed_ticks += 1
                observation = step.observation
                events_path.open("a", encoding="utf-8").write(
                    json.dumps(
                        {"tick": completed_ticks, "action": _json_action(action), "done": step.done}
                    )
                    + "\n"
                )
                cv2.imshow(WINDOW_NAME, _draw_overlay(observation["pov"], controls, completed_ticks))
                if step.done:
                    stopped_reason = "environment_done"
                    break
                time.sleep(max(0.0, TARGET_TICK_SECONDS - (time.perf_counter() - tick_started)))
        finally:
            cv2.destroyWindow(WINDOW_NAME)
    Image.fromarray(observation["pov"]).save(session_dir / "final.png")
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "env_id": "MineRLBasaltFindCave-v0",
                "seed": seed,
                "tick_budget": ticks,
                "mission_ticks": mission_ticks,
                "completed_ticks": completed_ticks,
                "stopped_reason": stopped_reason,
                "candidate_frames": candidates,
                "qwen_loaded": False,
                "esc_nonzero": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return session_dir


def _json_action(action: dict) -> dict:
    """Convert NumPy camera values to plain JSON numbers for the event log."""
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in action.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually explore MineRL FindCave")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=18000)
    parser.add_argument("--mission-ticks", type=int, default=18000)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/manual-findcave")
    )
    args = parser.parse_args()
    session_dir = run_manual_session(
        seed=args.seed,
        ticks=args.ticks,
        mission_ticks=args.mission_ticks,
        output_root=args.output_root,
    )
    print(json.dumps({"session_dir": str(session_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
