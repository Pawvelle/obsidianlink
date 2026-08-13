"""P1 E2 public inventory observation contract.

This module is MineRL-independent and validates only an already-public
inventory payload: identity fields plus positive, exact-Python-int item
quantities. RGB, selected item, later P1 fields, and evaluator-only truth
must not appear on the payload.

This temporary calibration type is neither the future v2 canonical
Observation nor the legacy ``obsidianlink.core.types.Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.env.validation.cases.lifecycle import initial_state_exists


PUBLIC_INVENTORY_ALLOWED_KEYS = frozenset(
    {"agent_id", "episode_id", "inventory", "step_id"}
)
PUBLIC_INVENTORY_LEAK_KEYS = frozenset(
    {
        "equipped_items",
        "frame",
        "info",
        "messages",
        "portal_dimension",
        "portal_grid",
        "portal_grid_origin",
        "portal_transition",
        "pov",
        "rgb",
        "selected_item",
        "workflow_stage",
    }
)

INVENTORY_OK = "inventory_ok"
INVENTORY_MISSING = "inventory_missing"
INVENTORY_NONE = "inventory_none"
INVENTORY_TYPE_INVALID = "inventory_type_invalid"
INVENTORY_ITEM_INVALID = "inventory_item_invalid"
INVENTORY_QUANTITY_INVALID = "inventory_quantity_invalid"
INVENTORY_LEAK = "inventory_leak"
INVENTORY_OUTCOMES = frozenset(
    {
        INVENTORY_OK,
        INVENTORY_MISSING,
        INVENTORY_NONE,
        INVENTORY_TYPE_INVALID,
        INVENTORY_ITEM_INVALID,
        INVENTORY_QUANTITY_INVALID,
        INVENTORY_LEAK,
    }
)


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validated_inventory(inventory: object) -> dict[str, int]:
    if not isinstance(inventory, Mapping):
        raise TypeError("inventory must be a Mapping")
    detached: dict[str, int] = {}
    for item, quantity in inventory.items():
        if not isinstance(item, str) or not item.strip():
            raise ValueError("inventory item names must be non-empty strings")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError(
                "inventory quantities must be exact Python ints greater than zero"
            )
        detached[item] = quantity
    return detached


@dataclass(frozen=True)
class PublicInventoryObservation:
    """Minimal temporary P1 public inventory observation."""

    episode_id: str
    agent_id: str
    step_id: int
    inventory: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", _require_identifier(self.episode_id, "episode_id")
        )
        object.__setattr__(
            self, "agent_id", _require_identifier(self.agent_id, "agent_id")
        )
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "inventory", _validated_inventory(self.inventory))

    def as_public_dict(self) -> dict[str, object]:
        """Return a detached plain public mapping."""

        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "inventory": dict(self.inventory),
            "step_id": self.step_id,
        }


@dataclass(frozen=True)
class InventoryInspection:
    """Pure E2 structural and isolation inspection result."""

    outcome: str
    present: bool
    error: str | None = None
    inventory: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in INVENTORY_OUTCOMES:
            raise ValueError(f"unknown inventory outcome: {self.outcome!r}")
        if type(self.present) is not bool:
            raise ValueError("present must be bool")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        if self.outcome == INVENTORY_OK:
            if not self.present or self.error is not None:
                raise ValueError("inventory_ok requires a present valid inventory")
            object.__setattr__(self, "inventory", _validated_inventory(self.inventory))
        elif self.inventory is not None:
            raise ValueError("invalid inventory inspections must not expose inventory")

    @property
    def valid(self) -> bool:
        return self.outcome == INVENTORY_OK

    def as_dict(self) -> dict[str, Any]:
        """Return a detached plain inspection snapshot."""

        return {
            "error": self.error,
            "inventory": None if self.inventory is None else dict(self.inventory),
            "outcome": self.outcome,
            "present": self.present,
        }


def inspect_inventory(inventory: object) -> InventoryInspection:
    """Fail closed unless ``inventory`` has valid public item semantics."""

    if inventory is None:
        return InventoryInspection(
            outcome=INVENTORY_NONE,
            present=False,
            error="inventory field is None",
        )
    if not isinstance(inventory, Mapping):
        return InventoryInspection(
            outcome=INVENTORY_TYPE_INVALID,
            present=True,
            error="inventory must be a Mapping",
        )
    detached: dict[str, int] = {}
    for item, quantity in inventory.items():
        if not isinstance(item, str) or not item.strip():
            return InventoryInspection(
                outcome=INVENTORY_ITEM_INVALID,
                present=True,
                error="inventory item names must be non-empty strings",
            )
        if type(quantity) is not int or quantity <= 0:
            return InventoryInspection(
                outcome=INVENTORY_QUANTITY_INVALID,
                present=True,
                error=(
                    "inventory quantities must be exact Python ints "
                    "greater than zero"
                ),
            )
        detached[item] = quantity
    return InventoryInspection(
        outcome=INVENTORY_OK,
        present=True,
        inventory=detached,
    )


def _payload_mapping(value: object) -> Mapping[object, object] | None:
    if isinstance(value, PublicInventoryObservation):
        return value.as_public_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _field_names(keys: list[object]) -> str:
    return ", ".join(sorted((repr(key) for key in keys)))


def inspect_public_inventory(
    reset_result: object, *, episode_id: str
) -> InventoryInspection:
    """Inspect a reset-result mapping for E2 inventory validity.

    Every agent payload must have exact public identity, initial step zero,
    and a structurally valid inventory. Any non-E2 field fails as leakage.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        return InventoryInspection(
            outcome=INVENTORY_MISSING,
            present=False,
            error="episode_id must be a non-empty string",
        )
    episode_id = episode_id.strip()
    if not isinstance(reset_result, Mapping) or not reset_result:
        return InventoryInspection(
            outcome=INVENTORY_MISSING,
            present=False,
            error="public inventory observation is missing",
        )
    if not initial_state_exists(reset_result, episode_id=episode_id):
        return InventoryInspection(
            outcome=INVENTORY_MISSING,
            present=False,
            error="reset did not return a usable public inventory state",
        )

    last_ok: InventoryInspection | None = None
    for outer_agent_id, value in reset_result.items():
        if (
            not isinstance(outer_agent_id, str)
            or not outer_agent_id.strip()
            or value is None
        ):
            return InventoryInspection(
                outcome=INVENTORY_MISSING,
                present=False,
                error="public inventory observation is missing",
            )
        payload = _payload_mapping(value)
        if payload is None:
            return InventoryInspection(
                outcome=INVENTORY_MISSING,
                present=False,
                error="inventory payload must be a Mapping",
            )
        leaked = [key for key in payload if key in PUBLIC_INVENTORY_LEAK_KEYS]
        unknown = [
            key
            for key in payload
            if key not in PUBLIC_INVENTORY_ALLOWED_KEYS
            and key not in PUBLIC_INVENTORY_LEAK_KEYS
        ]
        if leaked or unknown:
            return InventoryInspection(
                outcome=INVENTORY_LEAK,
                present="inventory" in payload,
                error="public inventory payload leaked non-inventory fields: "
                + _field_names(leaked + unknown),
            )
        if "inventory" not in payload:
            return InventoryInspection(
                outcome=INVENTORY_MISSING,
                present=False,
                error="inventory field is missing",
            )
        required_identity = {"agent_id", "episode_id", "step_id"}
        if not required_identity.issubset(payload):
            return InventoryInspection(
                outcome=INVENTORY_MISSING,
                present=True,
                error="public inventory identity fields are missing",
            )
        if (
            payload["agent_id"] != outer_agent_id
            or payload["episode_id"] != episode_id
            or type(payload["step_id"]) is not int
            or payload["step_id"] != 0
        ):
            return InventoryInspection(
                outcome=INVENTORY_MISSING,
                present=True,
                error="public inventory identity or initial step is invalid",
            )
        inspection = inspect_inventory(payload["inventory"])
        if not inspection.valid:
            return inspection
        last_ok = inspection

    if last_ok is None:
        return InventoryInspection(
            outcome=INVENTORY_MISSING,
            present=False,
            error="public inventory observation is missing",
        )
    return last_ok
