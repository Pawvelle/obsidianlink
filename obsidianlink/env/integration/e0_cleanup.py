"""Observable E0 cleanup signals and OS-level process-release inspection.

``close()`` returning is not proof that Java/Minecraft/MineRL processes
were released. ``E0CleanupStatus`` records only signals that can be
inspected on the existing backend object without launching MineRL.

``ProcessReleaseStatus`` is the separate OS/process-table observation.
It reuses the same PID-tree inspection already used by startup
reliability: a release claim requires that a MineRL/Minecraft/JVM child
was actually seen, the subprocess exited, and those tracked PIDs are
gone. This module does not launch, kill, or manage processes.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PROCESS_RELEASE_LIMITATION = (
    "close() returning does not prove that Java, Minecraft, or MineRL "
    "processes were released"
)
PROCESS_RELEASE_NOT_OBSERVED = (
    "MineRL/Minecraft/JVM child was not observed in the OS process table"
)
PROCESS_RELEASE_RESIDUAL = (
    "tracked MineRL/Minecraft/JVM or descendant PIDs remained after cleanup"
)
PROCESS_RELEASE_SUBPROCESS_ALIVE = (
    "case subprocess had not exited when process release was inspected"
)
_MINERL_RUNTIME_COMMAND = re.compile(r"(?:^|/)java(?:\s|$)")


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

    def has_explicit_failure(self) -> bool:
        """Return True when an observed cleanup signal is explicitly False.

        ``None`` means the signal is unavailable and is not treated as
        failure. ``process_release_proven`` is never a success condition.
        """

        if self.close_returned is False:
            return True
        return any(
            value is False
            for value in (
                self.backend_marked_closed,
                self.environment_reference_cleared,
                self.owner_cleared,
            )
        )

    def failure_detail(self) -> str | None:
        failed = [
            name
            for name, value in (
                ("close_returned", self.close_returned),
                ("backend_marked_closed", self.backend_marked_closed),
                ("environment_reference_cleared", self.environment_reference_cleared),
                ("owner_cleared", self.owner_cleared),
            )
            if value is False
        ]
        if not failed:
            return None
        return "observable cleanup failed: " + ", ".join(failed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_marked_closed": self.backend_marked_closed,
            "close_returned": self.close_returned,
            "environment_reference_cleared": self.environment_reference_cleared,
            "limitation": self.limitation,
            "owner_cleared": self.owner_cleared,
            "process_release_proven": False,
        }


def snapshot_process_table() -> dict[int, tuple[int, str]]:
    """Return ``{pid: (ppid, command)}`` from the OS process table."""

    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[int, tuple[int, str]] = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            result[int(parts[0])] = (int(parts[1]), parts[2])
        except ValueError:
            continue
    return result


def descendant_pids(root_pid: int, table: Mapping[int, tuple[int, str]]) -> set[int]:
    """Return PIDs in ``table`` that are descendants of ``root_pid``."""

    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in table.items():
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    found.discard(root_pid)
    return found


def is_minerl_runtime_command(command: object) -> bool:
    if not isinstance(command, str) or not command.strip():
        return False
    return _MINERL_RUNTIME_COMMAND.search(command) is not None


def tracked_descendants(root_pid: int, table: Mapping[int, tuple[int, str]] | None = None) -> dict[int, str]:
    snapshot = snapshot_process_table() if table is None else table
    return {pid: snapshot[pid][1] for pid in descendant_pids(root_pid, snapshot) if pid in snapshot}


def merge_tracked_descendants(
    tracked: dict[int, str],
    observed: Mapping[int, str],
) -> None:
    """Keep the strongest JVM identity seen for each PID.

    A later degraded command such as ``(java)`` must not replace a
    previously observed ``java ...`` command line. A later complete
    command may replace a weaker placeholder.
    """

    for pid, command in observed.items():
        previous = tracked.get(pid)
        if previous is not None and is_minerl_runtime_command(previous):
            continue
        tracked[pid] = command


def residual_descendants(
    tracked: Mapping[int, str],
    table: Mapping[int, tuple[int, str]] | None = None,
) -> dict[int, str]:
    """Return tracked PIDs that still exist. Identity text is not required."""

    snapshot = snapshot_process_table() if table is None else table
    return {pid: snapshot[pid][1] for pid in tracked if pid in snapshot}


@dataclass(frozen=True)
class ProcessReleaseStatus:
    """OS-level observation of whether tracked MineRL children terminated."""

    tracked_children: tuple[dict[str, Any], ...]
    residual_children: tuple[dict[str, Any], ...]
    subprocess_exited: bool
    minerl_runtime_observed: bool
    process_release_proven: bool
    limitation: str

    def __post_init__(self) -> None:
        _require_bool(self.subprocess_exited, "subprocess_exited")
        _require_bool(self.minerl_runtime_observed, "minerl_runtime_observed")
        _require_bool(self.process_release_proven, "process_release_proven")
        if not isinstance(self.limitation, str) or not self.limitation.strip():
            raise ValueError("limitation must be a non-empty string")
        if not isinstance(self.tracked_children, tuple) or not isinstance(
            self.residual_children, tuple
        ):
            raise ValueError("tracked_children and residual_children must be tuples")
        if self.process_release_proven:
            if not self.subprocess_exited or not self.minerl_runtime_observed:
                raise ValueError(
                    "process_release_proven requires an exited subprocess and an "
                    "observed MineRL/Minecraft/JVM child"
                )
            if self.residual_children:
                raise ValueError(
                    "process_release_proven requires tracked children to be absent"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "limitation": self.limitation,
            "minerl_runtime_observed": self.minerl_runtime_observed,
            "process_release_proven": self.process_release_proven,
            "residual_children": [dict(item) for item in self.residual_children],
            "subprocess_exited": self.subprocess_exited,
            "tracked_children": [dict(item) for item in self.tracked_children],
        }


def inspect_os_process_release(
    *,
    tracked_children: Sequence[Mapping[str, Any]] = (),
    residual_children: Sequence[Mapping[str, Any]] = (),
    subprocess_exited: bool,
) -> ProcessReleaseStatus:
    """Decide process release from OS PID-tree observations.

    ``env.close()`` is ignored here. Proven only when a Java/Minecraft
    child was seen, the case subprocess exited, and no tracked PID
    remains in the process table.
    """

    _require_bool(subprocess_exited, "subprocess_exited")
    tracked = tuple(dict(item) for item in tracked_children)
    residual = tuple(dict(item) for item in residual_children)
    observed = any(is_minerl_runtime_command(item.get("command")) for item in tracked)
    if not subprocess_exited:
        limitation = PROCESS_RELEASE_SUBPROCESS_ALIVE
        proven = False
    elif residual:
        limitation = PROCESS_RELEASE_RESIDUAL
        proven = False
    elif not observed:
        limitation = PROCESS_RELEASE_NOT_OBSERVED
        proven = False
    else:
        limitation = (
            "OS process table shows tracked MineRL/Minecraft/JVM children "
            "are no longer present after cleanup"
        )
        proven = True
    return ProcessReleaseStatus(
        tracked_children=tracked,
        residual_children=residual,
        subprocess_exited=subprocess_exited,
        minerl_runtime_observed=observed,
        process_release_proven=proven,
        limitation=limitation,
    )


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
