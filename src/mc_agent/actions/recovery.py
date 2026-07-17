"""Deterministic whitelist-only recovery macro-actions."""

from __future__ import annotations

from .schema import MacroAction, limit_macro_action


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
