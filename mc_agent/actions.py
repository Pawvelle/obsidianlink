"""Strict, non-executable macro-action protocol."""

from __future__ import annotations

import json
import math
import queue
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


ALLOWED_ACTIONS = {
    "wait",
    "look",
    "turn",
    "move_forward",
    "retreat",
    "sidestep_left",
    "sidestep_right",
}
ESCAPE_ACTIONS = {"retreat", "sidestep_left", "sidestep_right"}
ALLOWED_KEYS = {
    "action",
    "duration_ticks",
    "camera",
    "attack",
    "jump",
    "sprint",
    "cave_visible",
    "reason",
}
ALLOWED_CAMERA_KEYS = {"pitch", "yaw"}


@dataclass(frozen=True)
class MacroAction:
    action: str = "wait"
    duration_ticks: int = 1
    camera_pitch: float = 0.0
    camera_yaw: float = 0.0
    attack: bool = False
    jump: bool = False
    sprint: bool = False
    cave_visible: bool = False
    reason: str = ""

    @classmethod
    def no_op(cls, reason: str = "") -> "MacroAction":
        return cls(action="wait", duration_ticks=1, reason=reason[:160])

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    action: MacroAction
    accepted: bool
    error: str | None = None


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def limit_macro_action(action: MacroAction) -> MacroAction:
    """Defense-in-depth limiter for actions constructed outside the parser."""
    try:
        if action.action not in ALLOWED_ACTIONS:
            raise ValueError("action outside whitelist")
        if type(action.duration_ticks) is not int:
            raise ValueError("duration must be integer")
        pitch = _clamp(_number(action.camera_pitch, "camera_pitch"), -30, 30)
        yaw = _clamp(_number(action.camera_yaw, "camera_yaw"), -30, 30)
        attack = _boolean(action.attack, "attack")
        jump = _boolean(action.jump, "jump")
        sprint = _boolean(action.sprint, "sprint")
        cave_visible = _boolean(action.cave_visible, "cave_visible")
        if action.action in ESCAPE_ACTIONS:
            pitch = 0.0
            yaw = 0.0
            attack = False
            jump = False
            sprint = False
        if not isinstance(action.reason, str):
            raise ValueError("reason must be string")
        return MacroAction(
            action=action.action,
            duration_ticks=int(_clamp(action.duration_ticks, 1, 40)),
            camera_pitch=pitch,
            camera_yaw=yaw,
            attack=attack,
            jump=jump,
            sprint=sprint,
            cave_visible=cave_visible,
            reason=action.reason[:160],
        )
    except (TypeError, ValueError) as error:
        return MacroAction.no_op(f"limited to no-op: {error}")


def parse_macro_action(raw: str) -> ParseResult:
    """Parse JSON strictly; any structural failure returns a one-tick no-op."""
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("action payload must be one JSON object")
        unknown = set(value) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")

        action_name = value.get("action")
        if action_name not in ALLOWED_ACTIONS:
            raise ValueError("action is missing or outside the whitelist")

        duration = value.get("duration_ticks", 1)
        if type(duration) is not int:
            raise ValueError("duration_ticks must be an integer")
        duration = int(_clamp(duration, 1, 40))

        camera = value.get("camera", {})
        if not isinstance(camera, dict):
            raise ValueError("camera must be an object")
        unknown_camera = set(camera) - ALLOWED_CAMERA_KEYS
        if unknown_camera:
            raise ValueError(f"unknown camera fields: {sorted(unknown_camera)}")
        pitch = _clamp(_number(camera.get("pitch", 0.0), "camera.pitch"), -30, 30)
        yaw = _clamp(_number(camera.get("yaw", 0.0), "camera.yaw"), -30, 30)
        if action_name in {"look", "turn"} and pitch == 0.0 and yaw == 0.0:
            raise ValueError("look and turn require a non-zero camera angle")

        reason = value.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        parsed = MacroAction(
            action=action_name,
            duration_ticks=duration,
            camera_pitch=pitch,
            camera_yaw=yaw,
            attack=_boolean(value.get("attack", False), "attack"),
            jump=_boolean(value.get("jump", False), "jump"),
            sprint=_boolean(value.get("sprint", False), "sprint"),
            cave_visible=_boolean(
                value.get("cave_visible", False),
                "cave_visible",
            ),
            reason=reason[:160],
        )
        return ParseResult(action=limit_macro_action(parsed), accepted=True)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        message = str(error)
        return ParseResult(
            action=MacroAction.no_op(reason=f"rejected: {message}"),
            accepted=False,
            error=message,
        )


def is_cave_candidate(action: MacroAction) -> bool:
    """Require a cave claim to carry minimal visible-evidence words."""
    if not action.cave_visible:
        return False
    words = set(re.findall(r"[a-z]+", action.reason.lower()))
    evidence_groups = (
        ("dark", "black"),
        ("stone", "rock", "rocky"),
        ("opening", "entrance", "mouth"),
        ("left", "center", "right", "ahead"),
    )
    return all(words.intersection(group) for group in evidence_groups)


DARK_OPENING_LUMINANCE_THRESHOLD = 55.0
DARK_OPENING_MIN_REGION_FRACTION = 0.03
DARK_OPENING_GRID_ROWS = 9
DARK_OPENING_GRID_COLS = 12


def has_dark_opening_region(
    pov: np.ndarray,
    *,
    dark_luminance: float = DARK_OPENING_LUMINANCE_THRESHOLD,
    min_region_fraction: float = DARK_OPENING_MIN_REGION_FRACTION,
    grid_rows: int = DARK_OPENING_GRID_ROWS,
    grid_cols: int = DARK_OPENING_GRID_COLS,
) -> bool:
    """Deterministically veto a cave claim whose frame has no dark patch.

    This is a narrow, local, fail-closed check: it only rejects frames that
    are obviously implausible (large bright area, no continuous dark region,
    e.g. a sunlit sandstone wall). Passing this check never confirms a cave by
    itself; it is combined with the existing text-evidence gate
    (``is_cave_candidate``) and still requires human frame review afterward.
    """
    if not isinstance(pov, np.ndarray) or pov.ndim != 3 or pov.shape[2] != 3:
        raise ValueError("pov must be an RGB image")
    height, width, _ = pov.shape
    if height < grid_rows or width < grid_cols:
        raise ValueError("pov is too small for darkness-region detection")
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError("grid_rows and grid_cols must be positive")
    if not 0.0 < min_region_fraction <= 1.0:
        raise ValueError("min_region_fraction must be within (0, 1]")

    world_height = max(1, int(height * 0.85))  # exclude the bottom HUD strip
    world = pov[:world_height].astype(np.float32, copy=False)
    luminance = (
        world[..., 0] * 0.299 + world[..., 1] * 0.587 + world[..., 2] * 0.114
    )

    row_edges = np.linspace(0, luminance.shape[0], grid_rows + 1).astype(int)
    col_edges = np.linspace(0, luminance.shape[1], grid_cols + 1).astype(int)
    dark_cell = np.zeros((grid_rows, grid_cols), dtype=bool)
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = luminance[
                row_edges[row] : row_edges[row + 1],
                col_edges[col] : col_edges[col + 1],
            ]
            if cell.size and float(cell.mean()) <= dark_luminance:
                dark_cell[row, col] = True

    if not dark_cell.any():
        return False

    visited = np.zeros_like(dark_cell)
    largest_component = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if not dark_cell[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component_size = 0
            while stack:
                current_row, current_col = stack.pop()
                component_size += 1
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row, next_col = current_row + delta_row, current_col + delta_col
                    if (
                        0 <= next_row < grid_rows
                        and 0 <= next_col < grid_cols
                        and dark_cell[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            largest_component = max(largest_component, component_size)

    region_fraction = largest_component / (grid_rows * grid_cols)
    return region_fraction >= min_region_fraction


def resolve_cave_direction(reason: str) -> str | None:
    """Map the model's stated left/center/right/ahead word to one band.

    Returns ``None`` when the reason names zero direction words or more than
    one distinct band (e.g. both "left" and "right"): an ambiguous or absent
    direction must fail the frame veto closed rather than guess a band.
    """
    words = set(re.findall(r"[a-z]+", reason.lower()))
    direction_words = {"left": "left", "center": "center", "right": "right", "ahead": "center"}
    matches = {direction_words[word] for word in words if word in direction_words}
    if len(matches) != 1:
        return None
    return next(iter(matches))


def has_directional_dark_opening_region(
    pov: np.ndarray,
    direction: str,
    **kwargs: Any,
) -> bool:
    """Restrict the frame-veto darkness check to the claimed left/center/right band.

    A model claiming "opening on the left" must not be validated by a dark
    patch that only exists on the right (or center); each third of the frame
    is checked independently using the same deterministic
    ``has_dark_opening_region`` logic, applied only to the claimed band.
    """
    if direction not in {"left", "center", "right"}:
        raise ValueError("direction must be left, center, or right")
    if not isinstance(pov, np.ndarray) or pov.ndim != 3 or pov.shape[2] != 3:
        raise ValueError("pov must be an RGB image")
    width = pov.shape[1]
    third = width // 3
    bounds = {
        "left": (0, third),
        "center": (third, 2 * third),
        "right": (2 * third, width),
    }
    start, end = bounds[direction]
    band = pov[:, start:end]
    return has_dark_opening_region(band, **kwargs)


class LatestActionMailbox:
    """Capacity-one mailbox in which a newer action replaces the old one."""

    def __init__(self):
        self._queue: queue.Queue[MacroAction] = queue.Queue(maxsize=1)

    def publish(self, action: MacroAction) -> None:
        try:
            self._queue.put_nowait(action)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(action)

    def take_latest(self) -> MacroAction | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class Watchdog:
    """Immediate, thread-safe stop signal for the environment loop."""

    def __init__(self, max_ticks: int | None = None):
        self.max_ticks = max_ticks
        self.ticks = 0
        self._stop = threading.Event()
        self.reason: str | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set() or (
            self.max_ticks is not None and self.ticks >= self.max_ticks
        )

    def request_stop(self, reason: str = "requested") -> None:
        self.reason = reason
        self._stop.set()

    def after_tick(self) -> None:
        self.ticks += 1
        if self.max_ticks is not None and self.ticks >= self.max_ticks:
            self.reason = self.reason or "max_ticks"


class MacroExecutor:
    """Translate a bounded macro-action into deterministic MineRL ticks."""

    def __init__(self, action_space: Any, watchdog: Watchdog | None = None):
        self.action_space = action_space
        self.watchdog = watchdog
        self.current = MacroAction.no_op("initial")
        self.elapsed_ticks = self.current.duration_ticks

    @property
    def needs_action(self) -> bool:
        return self.elapsed_ticks >= self.current.duration_ticks

    def submit(self, action: MacroAction) -> None:
        self.current = limit_macro_action(action)
        self.elapsed_ticks = 0

    def interrupt(self, reason: str = "interrupt") -> None:
        self.current = MacroAction.no_op(reason)
        self.elapsed_ticks = self.current.duration_ticks

    def next_tick(self) -> dict[str, Any]:
        if self.watchdog is not None and self.watchdog.should_stop:
            self.interrupt(self.watchdog.reason or "watchdog")
            return self._no_op()
        if self.needs_action:
            return self._no_op()

        tick = self._no_op()
        action = self.current
        first_tick = self.elapsed_ticks == 0
        if action.action == "move_forward":
            tick["forward"] = 1
        elif action.action == "retreat":
            tick["back"] = 1
        elif action.action == "sidestep_left":
            tick["left"] = 1
        elif action.action == "sidestep_right":
            tick["right"] = 1
        if first_tick and action.action in {"look", "turn", "move_forward"}:
            tick["camera"] = np.asarray(
                [action.camera_pitch, action.camera_yaw], dtype=np.float32
            )
        tick["attack"] = int(action.attack)
        tick["jump"] = int(action.jump)
        tick["sprint"] = int(action.sprint and action.action == "move_forward")
        tick["ESC"] = 0
        self.elapsed_ticks += 1
        return tick

    def _no_op(self) -> dict[str, Any]:
        tick = self.action_space.no_op()
        tick["ESC"] = 0
        return tick


def safe_camera_recovery(index: int) -> MacroAction:
    """Return an alternating one-tick camera sweep with no interaction keys."""
    if type(index) is not int or index < 0:
        raise ValueError("recovery index must be a non-negative integer")
    yaw = 20.0 if index % 2 == 0 else -20.0
    return limit_macro_action(
        MacroAction(
            action="look",
            duration_ticks=1,
            camera_pitch=0.0,
            camera_yaw=yaw,
            attack=False,
            jump=False,
            sprint=False,
            reason="deterministic safe camera recovery",
        )
    )


def water_hazard_direction(pov: np.ndarray) -> str | None:
    """Return the dominant visible dark-water side, if it is large enough.

    Bright sky-blue pixels and the HUD strip are deliberately excluded. This is
    a narrow fail-safe for an imminent water crossing, not scene classification.
    """
    if not isinstance(pov, np.ndarray) or pov.ndim != 3 or pov.shape[2] != 3:
        raise ValueError("pov must be an RGB image")
    height, width, _ = pov.shape
    if height < 3 or width < 3:
        raise ValueError("pov is too small for water-hazard detection")
    world = pov[: max(1, int(height * 0.85))].astype(np.int16, copy=False)
    red, green, blue = world[..., 0], world[..., 1], world[..., 2]
    water = (
        (blue >= 80)
        & (blue <= 180)
        & (green <= 110)
        & (blue * 100 >= red * 135)
        & (blue * 100 >= green * 115)
    )
    thirds = [
        water[:, index * width // 3 : (index + 1) * width // 3].mean()
        for index in range(3)
    ]
    strongest = max(range(3), key=thirds.__getitem__)
    if thirds[strongest] < 0.05:
        return None
    return ("left", "center", "right")[strongest]


def safe_water_recovery(direction: str, index: int) -> MacroAction:
    """Use bounded displacement to leave a locally detected water hazard."""
    if direction not in {"left", "center", "right"}:
        raise ValueError("water hazard direction must be left, center, or right")
    if type(index) is not int or index < 0:
        raise ValueError("recovery index must be a non-negative integer")
    if direction == "left":
        action = "sidestep_right"
    elif direction == "right":
        action = "sidestep_left"
    else:
        action = "retreat"
    return limit_macro_action(
        MacroAction(
            action=action,
            duration_ticks=6,
            camera_pitch=0.0,
            camera_yaw=0.0,
            attack=False,
            jump=False,
            sprint=False,
            cave_visible=False,
            reason=f"local water hazard on {direction}",
        )
    )


def safe_stuck_recovery(index: int) -> MacroAction:
    """Use a bounded lateral move after verified lack of forward progress."""
    if type(index) is not int or index < 0:
        raise ValueError("recovery index must be a non-negative integer")
    return limit_macro_action(
        MacroAction(
            action="sidestep_right" if index % 2 == 0 else "sidestep_left",
            duration_ticks=6,
            camera_pitch=0.0,
            camera_yaw=0.0,
            attack=False,
            jump=False,
            sprint=False,
            cave_visible=False,
            reason="local low-progress recovery",
        )
    )


TURN_SCAN_STEPS = 3
TURN_SCAN_STEP_DEGREES = 20.0
TURN_SCAN_MAX_TOTAL_DEGREES = TURN_SCAN_STEPS * TURN_SCAN_STEP_DEGREES


def safe_turn_scan_recovery(index: int) -> list[MacroAction]:
    """Return one fixed, capped local turn scan instead of unbounded sidesteps.

    Used after repeated ("consecutive") forward low-progress recoveries, in
    place of continuing to alternate sidesteps indefinitely. The scan is a
    short, deterministic sequence of camera-only, one-tick turns: no movement
    key, no attack, no jump, no sprint, and ESC is never part of a macro
    action. Its cumulative rotation is hard-capped at
    ``TURN_SCAN_MAX_TOTAL_DEGREES`` (``TURN_SCAN_STEPS`` steps of
    ``TURN_SCAN_STEP_DEGREES`` each), independent of how many times this is
    called; callers must submit a fresh observation once the returned
    sequence is fully executed instead of waiting inside the MineRL loop.
    """
    if type(index) is not int or index < 0:
        raise ValueError("recovery index must be a non-negative integer")
    direction = 1.0 if index % 2 == 0 else -1.0
    return [
        limit_macro_action(
            MacroAction(
                action="turn",
                duration_ticks=1,
                camera_pitch=0.0,
                camera_yaw=direction * TURN_SCAN_STEP_DEGREES,
                attack=False,
                jump=False,
                sprint=False,
                cave_visible=False,
                reason=f"bounded local turn scan step {step + 1}/{TURN_SCAN_STEPS}",
            )
        )
        for step in range(TURN_SCAN_STEPS)
    ]


def safe_forward_continuation(remaining_ticks: int) -> MacroAction:
    """One bounded forward macro used by the local continuation safety layer.

    Qwen only chooses direction; while the step loop waits for the next
    (slow) decision, this lets it keep making limited forward progress
    instead of idling. The caller is responsible for enforcing the
    cumulative ``LOCAL_FORWARD_CONTINUATION_MAX_TICKS`` budget across calls;
    this only clamps one macro to the existing 1..40 duration limit via
    ``limit_macro_action`` and keeps it a plain, camera-neutral forward step
    with no attack, jump, sprint, or cave claim.
    """
    if type(remaining_ticks) is not int or remaining_ticks < 1:
        raise ValueError("remaining_ticks must be a positive integer")
    return limit_macro_action(
        MacroAction(
            action="move_forward",
            duration_ticks=min(40, remaining_ticks),
            camera_pitch=0.0,
            camera_yaw=0.0,
            attack=False,
            jump=False,
            sprint=False,
            cave_visible=False,
            reason="bounded local forward continuation while awaiting the next decision",
        )
    )


__all__ = [
    "LatestActionMailbox",
    "MacroAction",
    "MacroExecutor",
    "ParseResult",
    "Watchdog",
    "ESCAPE_ACTIONS",
    "has_dark_opening_region",
    "has_directional_dark_opening_region",
    "is_cave_candidate",
    "limit_macro_action",
    "parse_macro_action",
    "resolve_cave_direction",
    "safe_camera_recovery",
    "safe_forward_continuation",
    "safe_stuck_recovery",
    "safe_turn_scan_recovery",
    "safe_water_recovery",
    "water_hazard_direction",
    "TURN_SCAN_MAX_TOTAL_DEGREES",
    "TURN_SCAN_STEPS",
    "TURN_SCAN_STEP_DEGREES",
]
