"""P1 E3 public selected-item observation contract.

This MineRL-independent module validates only an already-public payload:
identity fields plus ``selected_item``. It is a temporary P1 calibration
surface, not the future v2 canonical Observation and not the legacy
``obsidianlink.core.types.Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.env.validation.cases.lifecycle import initial_state_exists


PUBLIC_SELECTED_ITEM_ALLOWED_KEYS = frozenset(
    {"agent_id", "episode_id", "selected_item", "step_id"}
)
PUBLIC_SELECTED_ITEM_LEAK_KEYS = frozenset(
    {
        "bucket_fluid",
        "dimension",
        "equipped_items",
        "evaluator_grid",
        "fluid",
        "fluid_truth",
        "frame",
        "info",
        "inventory",
        "messages",
        "portal_dimension",
        "portal_grid",
        "portal_grid_origin",
        "portal_transition",
        "portal_truth",
        "pov",
        "rgb",
        "server_truth",
        "block_truth",
        "grid_anchor",
        "evaluator_dimension",
        "flow_state",
        "observed_block",
        "server_fluid_truth",
        "truth_snapshot",
        "visible_inventory",
        "workflow_stage",
    }
)

SELECTED_ITEM_OK = "selected_item_ok"
SELECTED_ITEM_MISSING = "selected_item_missing"
SELECTED_ITEM_NONE = "selected_item_none"
SELECTED_ITEM_TYPE_INVALID = "selected_item_type_invalid"
SELECTED_ITEM_EMPTY = "selected_item_empty"
SELECTED_ITEM_LEAK = "selected_item_leak"
SELECTED_ITEM_OUTCOMES = frozenset(
    {
        SELECTED_ITEM_OK,
        SELECTED_ITEM_MISSING,
        SELECTED_ITEM_NONE,
        SELECTED_ITEM_TYPE_INVALID,
        SELECTED_ITEM_EMPTY,
        SELECTED_ITEM_LEAK,
    }
)


def validate_selected_item(value: object, field_name: str = "selected_item") -> str:
    """Return a stripped non-empty item identifier or raise fail-closed."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


@dataclass(frozen=True)
class PublicSelectedItemObservation:
    """Minimal temporary P1 selected-item calibration observation."""

    episode_id: str
    agent_id: str
    step_id: int
    selected_item: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", validate_selected_item(self.episode_id, "episode_id")
        )
        object.__setattr__(
            self, "agent_id", validate_selected_item(self.agent_id, "agent_id")
        )
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(
            self, "selected_item", validate_selected_item(self.selected_item)
        )

    def as_public_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "selected_item": self.selected_item,
            "step_id": self.step_id,
        }


@dataclass(frozen=True)
class SelectedItemInspection:
    """Pure E3 structural and isolation inspection result."""

    outcome: str
    present: bool
    error: str | None = None
    selected_item: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in SELECTED_ITEM_OUTCOMES:
            raise ValueError(f"unknown selected-item outcome: {self.outcome!r}")
        if type(self.present) is not bool:
            raise ValueError("present must be bool")
        if self.error is not None:
            validate_selected_item(self.error, "error")
        if self.outcome == SELECTED_ITEM_OK:
            if not self.present or self.error is not None:
                raise ValueError("selected_item_ok requires a present valid item")
            object.__setattr__(
                self, "selected_item", validate_selected_item(self.selected_item)
            )
        elif self.selected_item is not None:
            raise ValueError("invalid selected-item inspections must not expose an item")

    @property
    def valid(self) -> bool:
        return self.outcome == SELECTED_ITEM_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "outcome": self.outcome,
            "present": self.present,
            "selected_item": self.selected_item,
        }


def inspect_selected_item(value: object) -> SelectedItemInspection:
    """Fail closed unless the observed selected item is a non-empty string."""

    if value is None:
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_NONE,
            present=False,
            error="selected_item field is None",
        )
    if not isinstance(value, str):
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_TYPE_INVALID,
            present=True,
            error="selected_item must be a string",
        )
    if not value.strip():
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_EMPTY,
            present=True,
            error="selected_item must be a non-empty string",
        )
    return SelectedItemInspection(
        outcome=SELECTED_ITEM_OK,
        present=True,
        selected_item=value.strip(),
    )


def _payload_mapping(value: object) -> Mapping[object, object] | None:
    if isinstance(value, PublicSelectedItemObservation):
        return value.as_public_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _field_names(keys: list[object]) -> str:
    return ", ".join(sorted(repr(key) for key in keys))


def inspect_public_selected_item(
    reset_result: object, *, episode_id: str
) -> SelectedItemInspection:
    """Validate exact E3 identity, initial step, item, and field isolation."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_MISSING,
            present=False,
            error="episode_id must be a non-empty string",
        )
    episode_id = episode_id.strip()
    if not isinstance(reset_result, Mapping) or not reset_result:
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_MISSING,
            present=False,
            error="public selected-item observation is missing",
        )
    if not initial_state_exists(reset_result, episode_id=episode_id):
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_MISSING,
            present=False,
            error="reset did not return a usable public selected-item state",
        )

    last_ok: SelectedItemInspection | None = None
    for outer_agent_id, value in reset_result.items():
        if not isinstance(outer_agent_id, str) or not outer_agent_id.strip() or value is None:
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_MISSING,
                present=False,
                error="public selected-item observation is missing",
            )
        payload = _payload_mapping(value)
        if payload is None:
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_MISSING,
                present=False,
                error="selected-item payload must be a Mapping",
            )
        leaked = [key for key in payload if key in PUBLIC_SELECTED_ITEM_LEAK_KEYS]
        unknown = [
            key
            for key in payload
            if key not in PUBLIC_SELECTED_ITEM_ALLOWED_KEYS
            and key not in PUBLIC_SELECTED_ITEM_LEAK_KEYS
        ]
        if leaked or unknown:
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_LEAK,
                present="selected_item" in payload and payload.get("selected_item") is not None,
                error="public selected-item payload leaked non-E3 fields: "
                + _field_names(leaked + unknown),
            )
        if "selected_item" not in payload:
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_MISSING,
                present=False,
                error="selected_item field is missing",
            )
        if not {"agent_id", "episode_id", "step_id"}.issubset(payload):
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_MISSING,
                present=payload["selected_item"] is not None,
                error="public selected-item identity fields are missing",
            )
        if (
            payload["agent_id"] != outer_agent_id
            or payload["episode_id"] != episode_id
            or type(payload["step_id"]) is not int
            or payload["step_id"] != 0
        ):
            return SelectedItemInspection(
                outcome=SELECTED_ITEM_MISSING,
                present=payload["selected_item"] is not None,
                error="public selected-item identity or initial step is invalid",
            )
        inspection = inspect_selected_item(payload["selected_item"])
        if not inspection.valid:
            return inspection
        last_ok = inspection

    if last_ok is None:
        return SelectedItemInspection(
            outcome=SELECTED_ITEM_MISSING,
            present=False,
            error="public selected-item observation is missing",
        )
    return last_ok
