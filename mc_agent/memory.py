"""Bounded relative-orientation memory for short-horizon visual exploration."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from mc_agent.actions import MacroAction

CaveEntryStateName = Literal["idle", "entering", "entered", "aborted", "unverified"]
CAVE_ENTRY_REJECT_REASONS = frozenset(
    {
        "double_confirmation_not_established",
        "completion_already_requested",
        "insufficient_forward_ticks",
        "already_activated",
    }
)


@dataclass(frozen=True)
class OrientationView:
    heading: int
    low_change: bool


@dataclass(frozen=True)
class OrientationState:
    active: bool
    relative_yaw: float
    heading: int
    suggested_yaw: int
    recent_views: tuple[OrientationView, ...]
    unique_headings: int
    revisit_samples: int
    total_samples: int

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrientationMemory:
    def __init__(self, max_recent_views: int = 3, bucket_degrees: int = 20):
        if max_recent_views < 1:
            raise ValueError("max_recent_views must be positive")
        if bucket_degrees < 1 or 360 % bucket_degrees != 0:
            raise ValueError("bucket_degrees must divide 360")
        self.max_recent_views = max_recent_views
        self.bucket_degrees = bucket_degrees
        self._recent: deque[OrientationView] = deque(maxlen=max_recent_views)
        self._visited: set[int] = set()
        self._visit_counts: dict[int, int] = {}
        self._relative_yaw = 0.0
        self._revisit_samples = 0
        self._total_samples = 0

    def reset(self) -> None:
        self._recent.clear()
        self._visited.clear()
        self._visit_counts.clear()
        self._relative_yaw = 0.0
        self._revisit_samples = 0
        self._total_samples = 0

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        return (yaw + 180.0) % 360.0 - 180.0

    def _heading_bucket(self) -> int:
        half = self.bucket_degrees / 2.0
        bucket = (
            math.floor((self._relative_yaw + half) / self.bucket_degrees)
            * self.bucket_degrees
        )
        return int(self._wrap_yaw(float(bucket)))

    def observe_action(self, action: MacroAction) -> OrientationState:
        self._relative_yaw = self._wrap_yaw(
            self._relative_yaw + float(action.camera_yaw)
        )
        return self.snapshot()

    def observe_view(self, low_change: bool) -> OrientationState:
        if type(low_change) is not bool:
            raise ValueError("low_change must be boolean")
        heading = self._heading_bucket()
        self._revisit_samples += int(heading in self._visited)
        self._visited.add(heading)
        self._visit_counts[heading] = self._visit_counts.get(heading, 0) + 1
        self._total_samples += 1
        self._recent.append(OrientationView(heading=heading, low_change=low_change))
        return self.snapshot()

    def snapshot(self) -> OrientationState:
        heading = self._heading_bucket()
        right = int(self._wrap_yaw(heading + self.bucket_degrees))
        left = int(self._wrap_yaw(heading - self.bucket_degrees))
        suggested_yaw = (
            self.bucket_degrees
            if self._visit_counts.get(right, 0) <= self._visit_counts.get(left, 0)
            else -self.bucket_degrees
        )
        return OrientationState(
            active=bool(self._recent),
            relative_yaw=self._relative_yaw,
            heading=heading,
            suggested_yaw=suggested_yaw,
            recent_views=tuple(self._recent),
            unique_headings=len(self._visited),
            revisit_samples=self._revisit_samples,
            total_samples=self._total_samples,
        )


@dataclass(frozen=True)
class CaveTargetState:
    """Short-lived bearing for one locally validated cave observation."""

    active: bool
    direction: str | None
    source_tick: int | None
    remaining_decisions: int
    forward_ticks_after_acquisition: int

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaveEntrySnapshot:
    """Read-only view of the bounded Phase 5 entry phase."""

    state: CaveEntryStateName
    enabled: bool
    activation_tick: int | None
    completion_tick: int | None
    entry_budget_ticks: int
    entry_forward_ticks: int
    cancellation_reason: str | None
    evidence_frame_path: str | None
    pre_entry_luminance: float | None
    post_entry_luminance: float | None
    plausible: bool | None

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaveEntryPhase:
    """Bounded, single-shot "enter the cave" state for Phase 5.

    This is deliberately not a general navigation state machine. It can only
    be activated once per episode, only after the existing
    ``cave_target`` has been double-confirmed by the local frame gate, and
    only while ``cave_completion_requested`` is still ``False``. Activation
    replaces the immediate local ``ESC`` from the reconfirmation with a
    short, locally driven forward block. When that block finishes (or is
    cancelled by an existing safety guard), the surrounding environment
    owner decides what to do based on the terminal state; this class
    never emits ESC by itself.

    The state machine is intentionally small and non-recoverable: once it
    leaves ``"idle"`` it can only go to one of three terminal states:

    - ``"entered"`` — the bounded forward block ran to completion **and**
      the post-entry frame passed the local plausibility check. The
      surrounding owner is expected to fire exactly one local
      ``request_cave_completion()`` and set ``cave_completion_requested``
      to ``True``.
    - ``"aborted"`` — a safety guard tripped before the budget was
      exhausted. The post-entry evidence frame is *not* written (the
      bounded block did not reach its planned endpoint). The surrounding
      owner must not call ``request_cave_completion()`` and must leave
      ``cave_completion_requested`` ``False``.
    - ``"unverified"`` — the bounded forward block ran to completion but
      the post-entry frame did **not** pass the local plausibility check.
      The post-entry evidence frame *is* written and preserved for human
      review, but the surrounding owner must not call
      ``request_cave_completion()`` and must leave
      ``cave_completion_requested`` ``False``. The episode falls
      through to ``tick_budget`` or ``environment_done`` for the
      termination reason.

    All three terminal states are non-recoverable: ``activate()`` raises
    if called after any of them.
    """

    def __init__(self, *, max_budget_ticks: int = 30, enabled: bool = False):
        if type(max_budget_ticks) is not int or max_budget_ticks < 1:
            raise ValueError("max_budget_ticks must be a positive integer")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self.max_budget_ticks = max_budget_ticks
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self._state: CaveEntryStateName = "idle"
        self._activation_tick: int | None = None
        self._completion_tick: int | None = None
        self._entry_forward_ticks = 0
        self._remaining_budget_ticks = 0
        self._cancellation_reason: str | None = None
        self._evidence_frame_path: str | None = None
        self._pre_entry_luminance: float | None = None
        self._post_entry_luminance: float | None = None
        self._plausible: bool | None = None

    @property
    def state(self) -> CaveEntryStateName:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == "entering"

    @property
    def is_terminal(self) -> bool:
        # ``unverified`` is also terminal: the phase ran to the end of its
        # bounded budget but the post-entry frame did not satisfy the
        # local plausibility check, so neither an ESC nor another entry
        # activation is permitted afterwards.
        return self._state in {"entered", "aborted", "unverified"}

    def can_activate(
        self,
        *,
        cave_target_reconfirmations: int,
        forward_ticks_after_acquisition: int,
        cave_completion_requested: bool,
    ) -> bool:
        if not self.enabled:
            return False
        if self._state != "idle":
            return False
        if cave_completion_requested:
            return False
        if cave_target_reconfirmations < 1:
            return False
        if forward_ticks_after_acquisition < 1:
            return False
        return True

    def activation_blocker(
        self,
        *,
        cave_target_reconfirmations: int,
        forward_ticks_after_acquisition: int,
        cave_completion_requested: bool,
    ) -> str | None:
        """Return the deterministic reason ``can_activate`` would say no.

        ``None`` is returned when activation would be accepted. The result is
        intended for event logging only -- the surrounding agent loop must
        still consult ``can_activate`` for the actual decision.
        """
        if not self.enabled:
            return "phase_disabled"
        if self._state != "idle":
            return "already_activated"
        if cave_completion_requested:
            return "completion_already_requested"
        if cave_target_reconfirmations < 1:
            return "double_confirmation_not_established"
        if forward_ticks_after_acquisition < 1:
            return "insufficient_forward_ticks"
        return None

    def activate(self, tick: int) -> CaveEntrySnapshot:
        if self._state != "idle":
            raise RuntimeError(
                f"cave entry phase cannot be re-activated from {self._state!r}"
            )
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        self._state = "entering"
        self._activation_tick = tick
        self._completion_tick = None
        self._entry_forward_ticks = 0
        self._remaining_budget_ticks = self.max_budget_ticks
        self._cancellation_reason = None
        self._evidence_frame_path = None
        self._pre_entry_luminance = None
        self._post_entry_luminance = None
        self._plausible = None
        return self.snapshot()

    def record_pre_entry_luminance(self, value: float) -> None:
        if self._state != "entering":
            return
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("pre_entry_luminance must be a finite number")
        self._pre_entry_luminance = float(value)

    def record_forward_tick(self) -> CaveEntrySnapshot:
        if self._state != "entering":
            return self.snapshot()
        self._entry_forward_ticks += 1
        return self.snapshot()

    def remaining_budget(self) -> int:
        return self._remaining_budget_ticks

    def consume_budget(self, ticks: int) -> int:
        """Reserve ``ticks`` of the local forward budget for the next macro.

        Returns the number of ticks actually reserved (capped by the remaining
        budget). The remaining budget is reduced by the same amount so the
        caller cannot accidentally re-enter with the same budget.
        """
        if type(ticks) is not int or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        if self._state != "entering":
            return 0
        granted = min(ticks, self._remaining_budget_ticks)
        self._remaining_budget_ticks -= granted
        return granted

    def complete(
        self,
        *,
        tick: int,
        evidence_frame_path: str,
        post_entry_luminance: float,
        plausible: bool | None = None,
    ) -> CaveEntrySnapshot:
        """Move the phase from ``entering`` to a terminal state.

        ``plausible`` is optional: when omitted, the phase derives the
        flag from the coarse local luminance rule. Either way, a
        ``plausible=False`` (or an undecidable one) result lands the
        phase in the ``unverified`` terminal state instead of
        ``entered``. Callers must NOT emit the single local ESC tick
        when the returned state is ``unverified``.
        """
        if self._state != "entering":
            raise RuntimeError(
                f"cave entry phase cannot be completed from {self._state!r}"
            )
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        if not isinstance(evidence_frame_path, str) or not evidence_frame_path:
            raise ValueError("evidence_frame_path must be a non-empty string")
        if not isinstance(post_entry_luminance, (int, float)) or not math.isfinite(
            float(post_entry_luminance)
        ):
            raise ValueError("post_entry_luminance must be a finite number")
        self._completion_tick = tick
        self._cancellation_reason = None
        self._evidence_frame_path = evidence_frame_path
        self._post_entry_luminance = float(post_entry_luminance)
        derived = (
            self._derive_plausibility()
            if plausible is None
            else bool(plausible)
        )
        self._plausible = bool(derived) if derived is not None else False
        self._state = "entered" if self._plausible else "unverified"
        return self.snapshot()

    def mark_unverified(
        self,
        *,
        tick: int,
        reason: str,
    ) -> CaveEntrySnapshot:
        """Force the phase into the ``unverified`` terminal state.

        Used when the surrounding environment owner has a hard reason
        (for example an explicit post-frame classifier) to refuse the
        single local ESC tick even though the bounded forward block
        ran to completion. Like :meth:`abort`, the phase can only be
        marked once and never re-entered.
        """
        if self._state != "entering":
            raise RuntimeError(
                f"cave entry phase cannot be marked unverified from {self._state!r}"
            )
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        self._state = "unverified"
        self._completion_tick = tick
        self._cancellation_reason = reason
        self._plausible = False
        return self.snapshot()

    def abort(self, *, reason: str, tick: int) -> CaveEntrySnapshot:
        if self._state != "entering":
            return self.snapshot()
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        self._state = "aborted"
        self._completion_tick = tick
        self._cancellation_reason = reason
        self._plausible = None
        return self.snapshot()

    def _derive_plausibility(self) -> bool:
        """A coarse local gate: did the agent step into a darker area?

        This is a gate, not an annotation. When it returns ``False`` the
        phase is sealed in the ``unverified`` terminal state: the
        surrounding environment owner must not call
        ``request_cave_completion()``, must leave
        ``cave_completion_requested`` ``False``, and the single local
        ``ESC`` tick is suppressed. The post-entry evidence frame is
        still preserved so a human can review the result.

        Two acceptable conditions: the post-frame world is already well
        below a low absolute interior threshold, or it dropped noticeably
        relative to the pre-entry frame. If neither signal is present we
        return ``False``.
        """
        post = self._post_entry_luminance
        pre = self._pre_entry_luminance
        if post is None:
            return False
        absolute_interior_luminance = 50.0
        relative_drop_ratio = 0.7
        if post <= absolute_interior_luminance:
            return True
        if pre is not None and pre > 0.0 and post <= pre * relative_drop_ratio:
            return True
        return False

    def snapshot(self) -> CaveEntrySnapshot:
        return CaveEntrySnapshot(
            state=self._state,
            enabled=self.enabled,
            activation_tick=self._activation_tick,
            completion_tick=self._completion_tick,
            entry_budget_ticks=self.max_budget_ticks,
            entry_forward_ticks=self._entry_forward_ticks,
            cancellation_reason=self._cancellation_reason,
            evidence_frame_path=self._evidence_frame_path,
            pre_entry_luminance=self._pre_entry_luminance,
            post_entry_luminance=self._post_entry_luminance,
            plausible=self._plausible,
        )


class CaveTargetMemory:
    """Keep a verified cave bearing only long enough to re-observe it safely.

    This is deliberately not a map or long-term exploration memory. It exists
    only after the existing text-and-frame cave gate has accepted a source
    frame, expires after a small number of planner decisions, and tracks
    relative camera yaw so an instruction such as "left" remains meaningful
    after a controlled turn.
    """

    _DIRECTION_TO_BEARING = {"left": -20.0, "center": 0.0, "right": 20.0}

    def __init__(self, max_decisions: int = 4):
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.max_decisions = max_decisions
        self.reset()

    def reset(self) -> None:
        self._bearing: float | None = None
        self._source_tick: int | None = None
        self._remaining_decisions = 0
        self._forward_ticks = 0

    def acquire(self, direction: str, source_tick: int) -> CaveTargetState:
        if direction not in self._DIRECTION_TO_BEARING:
            raise ValueError("direction must be left, center, or right")
        if type(source_tick) is not int or source_tick < 0:
            raise ValueError("source_tick must be a non-negative integer")
        self._bearing = self._DIRECTION_TO_BEARING[direction]
        self._source_tick = source_tick
        self._remaining_decisions = self.max_decisions
        self._forward_ticks = 0
        return self.snapshot()

    def observe_action(self, action: MacroAction) -> CaveTargetState:
        if self._bearing is not None:
            self._bearing = OrientationMemory._wrap_yaw(
                self._bearing - float(action.camera_yaw)
            )
        return self.snapshot()

    def observe_forward_tick(self) -> CaveTargetState:
        if self._bearing is not None:
            self._forward_ticks += 1
        return self.snapshot()

    def consume_decision(self) -> CaveTargetState:
        if self._bearing is not None:
            self._remaining_decisions -= 1
            if self._remaining_decisions <= 0:
                self.reset()
        return self.snapshot()

    def snapshot(self) -> CaveTargetState:
        if self._bearing is None:
            return CaveTargetState(False, None, None, 0, 0)
        if self._bearing <= -10.0:
            direction = "left"
        elif self._bearing >= 10.0:
            direction = "right"
        else:
            direction = "center"
        return CaveTargetState(
            active=True,
            direction=direction,
            source_tick=self._source_tick,
            remaining_decisions=self._remaining_decisions,
            forward_ticks_after_acquisition=self._forward_ticks,
        )


@dataclass(frozen=True)
class FrameChange:
    mean_absolute_difference: float
    changed_pixel_fraction: float
    low_change: bool

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrameChangeDetector:
    """Compare sparse grayscale POV features while ignoring the HUD strip."""

    def __init__(
        self,
        mean_difference_threshold: float = 0.005,
        changed_fraction_threshold: float = 0.01,
        pixel_difference_threshold: float = 20.0,
    ):
        if not 0.0 <= mean_difference_threshold <= 1.0:
            raise ValueError("mean_difference_threshold must be within [0, 1]")
        if not 0.0 <= changed_fraction_threshold <= 1.0:
            raise ValueError("changed_fraction_threshold must be within [0, 1]")
        if not 0.0 <= pixel_difference_threshold <= 255.0:
            raise ValueError("pixel_difference_threshold must be within [0, 255]")
        self.mean_difference_threshold = mean_difference_threshold
        self.changed_fraction_threshold = changed_fraction_threshold
        self.pixel_difference_threshold = pixel_difference_threshold
        self._reference: np.ndarray | None = None

    def reset(self, pov: np.ndarray) -> None:
        self._reference = self._feature(pov)

    def compare_and_update(self, pov: np.ndarray) -> FrameChange:
        if self._reference is None:
            raise RuntimeError("frame change detector must be reset first")
        current = self._feature(pov)
        difference = np.abs(current - self._reference)
        mean_difference = float(difference.mean() / 255.0)
        changed_fraction = float(
            np.count_nonzero(difference >= self.pixel_difference_threshold)
            / difference.size
        )
        result = FrameChange(
            mean_absolute_difference=mean_difference,
            changed_pixel_fraction=changed_fraction,
            low_change=(
                mean_difference < self.mean_difference_threshold
                and changed_fraction < self.changed_fraction_threshold
            ),
        )
        self._reference = current
        return result

    @staticmethod
    def _feature(pov: np.ndarray) -> np.ndarray:
        if not isinstance(pov, np.ndarray) or pov.dtype != np.uint8:
            raise ValueError("pov must be a uint8 numpy array")
        if pov.shape != (360, 640, 3):
            raise ValueError(f"unexpected pov shape: {pov.shape}")
        sampled = pov[:300:8, ::8].astype(np.float32)
        return (
            sampled[..., 0] * 0.299
            + sampled[..., 1] * 0.587
            + sampled[..., 2] * 0.114
        )


__all__ = [
    "FrameChange",
    "FrameChangeDetector",
    "CaveTargetMemory",
    "CaveTargetState",
    "CaveEntryPhase",
    "CaveEntrySnapshot",
    "CaveEntryStateName",
    "OrientationMemory",
    "OrientationState",
    "OrientationView",
]
