"""Show the agent POV in a desktop window while MineRL also runs Minecraft."""

from __future__ import annotations

import subprocess
from typing import Any

from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation

WINDOW_NAME = "ObsidianLink Agent POV"


def annotate_frame(frame: Any, hud: dict[str, str]) -> Any:
    """Return a BGR image with HUD text. Safe if OpenCV/numpy are missing."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("frame must be an HxWx3 RGB image")
    bgr = array[:, :, :3][:, :, ::-1].copy()
    try:
        import cv2
    except ImportError:
        return bgr
    lines = [f"{key}: {value}" for key, value in hud.items() if value]
    overlay = bgr.copy()
    height = 22 * (len(lines) + 1)
    cv2.rectangle(overlay, (0, 0), (bgr.shape[1], min(bgr.shape[0], height)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)
    y = 18
    for line in lines:
        cv2.putText(
            bgr,
            line[:96],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return bgr


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

    def set_hud(self, **fields: Any) -> None:
        for key, value in fields.items():
            self.hud[str(key)] = "" if value is None else str(value)

    def reset(self) -> Observation:
        observation = self.env.reset()
        self._show(observation)
        raise_minecraft_windows()
        return observation

    def observe(self) -> Observation:
        return self.env.observe()

    def step(self, action: Action) -> Observation:
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


__all__ = [
    "WINDOW_NAME",
    "LiveDesktopView",
    "annotate_frame",
    "raise_minecraft_windows",
]
