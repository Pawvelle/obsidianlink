"""Deterministic whitelist-only recovery macro-actions."""

from __future__ import annotations

from .schema import MacroAction, limit_macro_action


class ForwardProbeGate:
    """Require a fresh bounded run of low-change observations before probing."""

    def __init__(self, required_low_change_windows: int = 2):
        if type(required_low_change_windows) is not int or required_low_change_windows < 1:
            raise ValueError("required_low_change_windows must be a positive integer")
        self.required_low_change_windows = required_low_change_windows
        self.consecutive_low_change_windows = 0

    @property
    def eligible(self) -> bool:
        return (
            self.consecutive_low_change_windows
            >= self.required_low_change_windows
        )

    def reset(self) -> None:
        self.consecutive_low_change_windows = 0

    def observe(self, low_change: bool) -> int:
        if type(low_change) is not bool:
            raise ValueError("low_change must be boolean")
        if low_change:
            self.consecutive_low_change_windows += 1
        else:
            self.reset()
        return self.consecutive_low_change_windows

    def consume(self) -> None:
        if not self.eligible:
            raise RuntimeError("forward probe gate is not eligible")
        self.reset()


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


def safe_forward_probe() -> MacroAction:
    """Return one bounded forward tick with every interaction modifier disabled."""
    return limit_macro_action(
        MacroAction(
            action="move_forward",
            duration_ticks=1,
            camera_pitch=0.0,
            camera_yaw=0.0,
            attack=False,
            jump=False,
            sprint=False,
            reason="deterministic bounded forward probe",
        )
    )


def is_safe_forward_probe(action: MacroAction) -> bool:
    """Check the exact execution-layer safety contract for a forward probe."""
    return (
        action.action == "move_forward"
        and action.duration_ticks == 1
        and action.camera_pitch == 0.0
        and action.camera_yaw == 0.0
        and not action.attack
        and not action.jump
        and not action.sprint
    )
