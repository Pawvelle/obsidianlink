"""Show the agent POV and a live operation board on the desktop."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation

if TYPE_CHECKING:
    from obsidianlink.agents.episode_log import EpisodeLogger

WINDOW_NAME = "ObsidianLink Agent POV"
PROCESS_WINDOW_NAME = "ObsidianLink Agent Process"

_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)


def annotate_frame(frame: Any, hud: dict[str, str]) -> Any:
    """Return a BGR image with HUD text. Safe if OpenCV/numpy are missing."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("frame must be an HxWx3 RGB image")
    bgr = array[:, :, :3][:, :, ::-1].copy()
    lines = [f"{key}: {value}" for key, value in hud.items() if value]
    return _draw_lines(bgr, lines, overlay_ratio=0.55, max_chars=96)


def format_process_event(event: str, payload: dict[str, Any] | None = None) -> str:
    """One human-readable Chinese line for the process board."""
    data = payload or {}
    if event == "task":
        return f"任务开始  {data.get('task', '')}"
    if event == "planner_output":
        decision = data.get("decision") or {}
        kind = decision.get("type", "")
        name = decision.get("name") or decision.get("query") or kind
        args = decision.get("arguments") or {}
        extra = json.dumps(args, ensure_ascii=False) if args else ""
        subgoal = decision.get("subgoal") or decision.get("active_subgoal_id") or ""
        return (
            f"Planner#{data.get('cycle', '?')}  {kind} {name} {extra}  "
            f"子目标={subgoal}  原因={decision.get('reason', '')}"
        )
    if event == "validation":
        status = "通过" if data.get("accepted") else "拒绝"
        return (
            f"Validator  {status}  {data.get('skill', '')}  "
            f"{data.get('reason', '')}"
        )
    if event == "skill_execution":
        result = data.get("result") or {}
        advanced = result.get("advanced_goal")
        if advanced is False:
            progress = "未推进目标"
        elif advanced is True:
            progress = "推进目标"
        else:
            progress = "结果待观察"
        return (
            f"Skill  {data.get('skill', '')}  "
            f"{'成功' if data.get('success') else '失败'}  "
            f"{progress}  {data.get('message', '')}"
        )
    if event == "memory_update":
        return f"Memory  {data.get('source', '')}  {data.get('query', '')}"
    if event == "result":
        return (
            f"结束  success={data.get('success')}  "
            f"{data.get('reason', '')}  背包={data.get('inventory', {})}"
        )
    return f"{event}  {json.dumps(data, ensure_ascii=False)[:180]}"


def raise_minecraft_windows() -> None:
    """Bring Java/Minecraft windows forward on macOS. No-op elsewhere."""
    script = (
        'tell application "System Events"\n'
        "    repeat with p in (every process whose background only is false)\n"
        "        set n to name of p as text\n"
        "        if n contains \"java\" or n contains \"Minecraft\" "
        'or n contains \"lwjgl\" or n contains \"python\" then\n'
        "            try\n"
        "                set frontmost of p to true\n"
        "            end try\n"
        "        end if\n"
        "    end repeat\n"
        "end tell\n"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return


class LiveDesktopView(Environment):
    """Wrap an environment and mirror POV to a desktop OpenCV window."""

    def __init__(
        self,
        env: Environment,
        *,
        window_name: str = WINDOW_NAME,
        scale: int = 2,
    ) -> None:
        self.env = env
        self.window_name = window_name
        self.scale = max(1, int(scale))
        self.hud: dict[str, str] = {}
        self._window_ready = False
        self._cv2: Any = None
        self._steps_since_raise = 0
        self._last_observation: Observation | None = None

    def set_hud(self, **fields: Any) -> None:
        for key, value in fields.items():
            self.hud[str(key)] = "" if value is None else str(value)

    def refresh(self) -> None:
        if self._last_observation is not None:
            self._show(self._last_observation)

    def reset(self) -> Observation:
        observation = self.env.reset()
        self._show(observation)
        raise_minecraft_windows()
        return observation

    def observe(self) -> Observation:
        return self.env.observe()

    def local_view(self) -> dict[str, Any]:
        getter = getattr(self.env, "local_view", None)
        if callable(getter):
            view = getter()
            return dict(view) if isinstance(view, dict) else {}
        return {}

    def step(self, action: Action) -> Observation:
        self.set_hud(tick=_action_label(action))
        observation = self.env.step(action)
        self._show(observation)
        self._steps_since_raise += 1
        if self._steps_since_raise >= 80:
            raise_minecraft_windows()
            self._steps_since_raise = 0
        return observation

    def close(self) -> None:
        try:
            if self._cv2 is not None:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
        except Exception:
            pass
        self.env.close()

    def _show(self, observation: Observation) -> None:
        self._last_observation = observation
        if observation.frame is None:
            return
        try:
            import cv2
        except ImportError:
            return
        self._cv2 = cv2
        inventory = observation.inventory or {}
        compact = ", ".join(f"{name}={qty}" for name, qty in sorted(inventory.items())[:8])
        self.set_hud(
            selected=observation.selected_item or "none",
            inventory=compact or "(empty)",
        )
        try:
            image = annotate_frame(observation.frame, self.hud)
        except (TypeError, ValueError):
            return
        if self.scale > 1:
            image = cv2.resize(
                image,
                (image.shape[1] * self.scale, image.shape[0] * self.scale),
                interpolation=cv2.INTER_NEAREST,
            )
        if not self._window_ready:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, image.shape[1], image.shape[0])
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            self._window_ready = True
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)


class LiveProcessBoard:
    """Second desktop window that lists planner / validator / skill events."""

    def __init__(
        self,
        *,
        window_name: str = PROCESS_WINDOW_NAME,
        max_lines: int = 22,
        width: int = 1100,
        height: int = 720,
        view: LiveDesktopView | None = None,
    ) -> None:
        self.window_name = window_name
        self.max_lines = max(6, int(max_lines))
        self.width = int(width)
        self.height = int(height)
        self.view = view
        self.lines: list[str] = []
        self._window_ready = False
        self._cv2: Any = None

    def push(self, line: str) -> None:
        text = str(line).strip()
        if not text:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        rendered = f"[{stamp}] {text}"
        print(rendered, flush=True)
        self.lines.append(rendered)
        self.lines = self.lines[-self.max_lines :]
        self._redraw()
        if self.view is not None:
            self.view.refresh()

    def show_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.push(format_process_event(event, payload))

    def close(self) -> None:
        try:
            if self._cv2 is not None:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
        except Exception:
            pass

    def _redraw(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return
        self._cv2 = cv2
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        header = [
            "ObsidianLink 操作过程（全程可视化）",
            "窗口1=Minecraft  窗口2=Agent POV  本窗口=Planner/Validator/Skill/结果",
            "",
        ]
        image = _draw_lines(canvas, header + self.lines, overlay_ratio=0.0, max_chars=140)
        if not self._window_ready:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            self._window_ready = True
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)


class DisplayEpisodeLogger:
    """Write episode files and mirror every event onto the process board."""

    def __init__(self, logger: EpisodeLogger, board: LiveProcessBoard) -> None:
        self.logger = logger
        self.board = board

    @property
    def directory(self):
        return self.logger.directory

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.logger.record(event, payload)
        self.board.show_event(event, payload)
        if self.board.view is not None:
            hud = _hud_from_event(event, payload or {})
            if hud:
                self.board.view.set_hud(**hud)
                self.board.view.refresh()

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.logger.write_summary(summary)
        self.board.push(
            f"摘要已保存  {self.logger.summary_path}  "
            f"success={summary.get('success')}"
        )


def _hud_from_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    if event == "planner_output":
        decision = payload.get("decision") or {}
        kind = decision.get("type", "")
        name = decision.get("name") or decision.get("query") or kind
        return {
            "cycle": str(payload.get("cycle", "")),
            "decision": f"{kind} {name}"[:80],
            "reason": str(decision.get("reason") or "")[:80],
            "subgoal": str(decision.get("subgoal") or decision.get("active_subgoal_id") or "")[:80],
            "status": "planner",
        }
    if event == "validation":
        return {
            "validation": ("OK" if payload.get("accepted") else "REJECT")
            + " "
            + str(payload.get("reason") or "")[:70],
            "status": "validator",
        }
    if event == "skill_execution":
        return {
            "skill": f"{payload.get('skill', '')} {payload.get('message', '')}"[:80],
            "status": "skill",
        }
    if event == "result":
        return {"status": f"done success={payload.get('success')}"}
    return {}


def _action_label(action: Action) -> str:
    parts = [action.type.value]
    if action.dx or action.dz:
        parts.append(f"dx={action.dx} dz={action.dz}")
    if action.yaw or action.pitch:
        parts.append(f"yaw={action.yaw:.0f} pitch={action.pitch:.0f}")
    if action.target:
        parts.append(str(action.target))
    if action.jump:
        parts.append("jump")
    return " ".join(parts)


def _draw_lines(
    bgr: Any,
    lines: list[str],
    *,
    overlay_ratio: float,
    max_chars: int,
) -> Any:
    import numpy as np

    if overlay_ratio > 0:
        try:
            import cv2
        except ImportError:
            return bgr
        overlay = bgr.copy()
        height = 22 * (len(lines) + 1)
        cv2.rectangle(
            overlay,
            (0, 0),
            (bgr.shape[1], min(bgr.shape[0], height)),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, overlay_ratio, bgr, 1.0 - overlay_ratio, 0, bgr)
    clipped = [str(line)[:max_chars] for line in lines]
    drawn = _draw_unicode_lines(bgr, clipped)
    if drawn is not None:
        return drawn
    return _draw_ascii_lines(bgr, clipped)


def _draw_unicode_lines(bgr: Any, lines: list[str]) -> Any | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    font = _load_cjk_font(18)
    if font is None:
        return None
    rgb = bgr[:, :, ::-1].copy()
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    y = 8
    for line in lines:
        draw.text((8, y), line, font=font, fill=(255, 255, 255))
        y += 22
    return __import__("numpy").asarray(image)[:, :, ::-1].copy()


def _draw_ascii_lines(bgr: Any, lines: list[str]) -> Any:
    try:
        import cv2
    except ImportError:
        return bgr
    y = 18
    for line in lines:
        cv2.putText(
            bgr,
            line.encode("ascii", "replace").decode("ascii"),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return bgr


def _load_cjk_font(size: int) -> Any:
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size=size, index=0)
        except OSError:
            continue
    return None


__all__ = [
    "WINDOW_NAME",
    "PROCESS_WINDOW_NAME",
    "DisplayEpisodeLogger",
    "LiveDesktopView",
    "LiveProcessBoard",
    "annotate_frame",
    "format_process_event",
    "raise_minecraft_windows",
]
