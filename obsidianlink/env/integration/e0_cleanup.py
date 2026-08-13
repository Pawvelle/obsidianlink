"""Observable E0 cleanup signals.

``close()`` returning is not proof that Java/Minecraft/MineRL processes
were released. This contract records only signals that can be inspected
on the existing backend object without launching MineRL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROCESS_RELEASE_LIMITATION = (
    "close() returning does not prove that Java, Minecraft, or MineRL "
    "processes were released"
)


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _optional_bool(value: object, field_name: str) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{field_name} must be bool or None")


@dataclass(frozen=True)
class E0CleanupStatus:
    close_returned: bool
    backend_marked_closed: bool | None
    environment_reference_cleared: bool | None
    owner_cleared: bool | None
    process_release_proven: bool = False
    limitation: str = PROCESS_RELEASE_LIMITATION

    def __post_init__(self) -> None:
        _require_bool(self.close_returned, "close_returned")
        _require_bool(self.process_release_proven, "process_release_proven")
        _optional_bool(self.backend_marked_closed, "backend_marked_closed")
        _optional_bool(
            self.environment_reference_cleared, "environment_reference_cleared"
        )
        _optional_bool(self.owner_cleared, "owner_cleared")
        if not isinstance(self.limitation, str) or not self.limitation.strip():
            raise ValueError("limitation must be a non-empty string")
        if self.process_release_proven:
            raise ValueError(
                "this runtime cannot claim MineRL/Minecraft process release"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_marked_closed": self.backend_marked_closed,
            "close_returned": self.close_returned,
            "environment_reference_cleared": self.environment_reference_cleared,
            "limitation": self.limitation,
            "owner_cleared": self.owner_cleared,
            "process_release_proven": False,
        }


def inspect_minerl_cleanup(
    backend: object | None,
    *,
    close_returned: bool,
) -> E0CleanupStatus:
    """Read cleanup signals from a MineRL-style backend after close."""

    if backend is None:
        return E0CleanupStatus(
            close_returned=close_returned,
            backend_marked_closed=True,
            environment_reference_cleared=True,
            owner_cleared=True,
        )
    opened = getattr(backend, "_opened", None)
    env = getattr(backend, "_env", None)
    owner = getattr(backend, "_owner_thread", None)
    backend_marked_closed = None if opened is None else opened is False
    environment_cleared = None
    if hasattr(backend, "_env"):
        environment_cleared = env is None
    owner_cleared = None
    if hasattr(backend, "_owner_thread"):
        owner_cleared = owner is None
    return E0CleanupStatus(
        close_returned=close_returned,
        backend_marked_closed=backend_marked_closed,
        environment_reference_cleared=environment_cleared,
        owner_cleared=owner_cleared,
    )
