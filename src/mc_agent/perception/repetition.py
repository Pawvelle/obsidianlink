"""Bounded state for measuring and discouraging immediate action repetition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mc_agent.actions import MacroAction


@dataclass(frozen=True)
class RepetitionState:
    active: bool
    last_action: str | None
    consecutive_count: int
    current_was_repeat: bool

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class RepetitionDetector:
    """Track consecutive accepted macro-action names.

    The feedback state becomes active after the first action so the next planner
    request can penalize selecting the same action name again. Measurement counts
    an actual repetition only when two consecutive accepted names match.
    """

    def __init__(self) -> None:
        self._last_action: str | None = None
        self._consecutive_count = 0
        self._current_was_repeat = False

    def reset(self) -> None:
        self._last_action = None
        self._consecutive_count = 0
        self._current_was_repeat = False

    def observe(self, action: MacroAction) -> RepetitionState:
        repeated = action.action == self._last_action
        if repeated:
            self._consecutive_count += 1
        else:
            self._last_action = action.action
            self._consecutive_count = 1
        self._current_was_repeat = repeated
        return self.snapshot()

    def snapshot(self) -> RepetitionState:
        return RepetitionState(
            active=self._last_action is not None,
            last_action=self._last_action,
            consecutive_count=self._consecutive_count,
            current_was_repeat=self._current_was_repeat,
        )
