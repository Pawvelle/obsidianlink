"""P1 E1 public RGB observation contract.

This module never imports MineRL. It validates an already-public RGB
payload: identity fields plus an HxWx3 uint8 image array. Inventory,
selected item, and evaluator-only world truth are not part of E1 and
must not appear on the public payload.

This is not a v2 canonical Observation type and not the legacy
``obsidianlink.core.types.Observation`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.env.validation.cases.lifecycle import initial_state_exists


PUBLIC_RGB_ALLOWED_KEYS = frozenset({"agent_id", "episode_id", "rgb", "step_id"})
PUBLIC_RGB_LEAK_KEYS = frozenset(
    {
        "bucket_fluid",
        "equipped_items",
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
        "pov",
        "selected_item",
        "visible_inventory",
        "workflow_stage",
    }
)
RGB_OK = "rgb_ok"
RGB_MISSING = "rgb_missing"
RGB_NONE = "rgb_none"
RGB_SHAPE_INVALID = "rgb_shape_invalid"
RGB_DTYPE_INVALID = "rgb_dtype_invalid"
RGB_LEAK = "rgb_leak"
RGB_OUTCOMES = frozenset(
    {
        RGB_OK,
        RGB_MISSING,
        RGB_NONE,
        RGB_SHAPE_INVALID,
        RGB_DTYPE_INVALID,
        RGB_LEAK,
    }
)


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class PublicRGBObservation:
    """Minimal P1 public RGB observation.

    ``rgb`` is the Agent-visible image array. Callers must not attach
    inventory, selected item, or evaluator-only fields.
    """

    episode_id: str
    agent_id: str
    step_id: int
    rgb: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _require_identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _require_identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")

    def as_public_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "rgb": self.rgb,
            "step_id": self.step_id,
        }


@dataclass(frozen=True)
class RGBInspection:
    outcome: str
    present: bool
    error: str | None = None
    height: int | None = None
    width: int | None = None
    channels: int | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in RGB_OUTCOMES:
            raise ValueError(f"unknown RGB outcome: {self.outcome!r}")
        if type(self.present) is not bool:
            raise ValueError("present must be bool")
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be None or a non-empty string")
        if self.outcome == RGB_OK:
            if not self.present or self.error is not None:
                raise ValueError("rgb_ok requires a present valid RGB array")
            if (
                type(self.height) is not int
                or type(self.width) is not int
                or type(self.channels) is not int
                or self.height < 1
                or self.width < 1
                or self.channels != 3
                or self.dtype != "uint8"
            ):
                raise ValueError("rgb_ok requires HxWx3 uint8 metadata")

    @property
    def valid(self) -> bool:
        return self.outcome == RGB_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "dtype": self.dtype,
            "error": self.error,
            "height": self.height,
            "outcome": self.outcome,
            "present": self.present,
            "width": self.width,
        }


def _dtype_name(value: object) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    name = getattr(dtype, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    text = str(dtype).strip()
    return text or None


def _payload_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, PublicRGBObservation):
        return value.as_public_dict()
    if isinstance(value, Mapping):
        return value
    rgb = getattr(value, "rgb", None)
    has_rgb = hasattr(value, "rgb")
    if not has_rgb:
        return None
    payload: dict[str, object] = {"rgb": rgb}
    episode_id = getattr(value, "episode_id", None)
    agent_id = getattr(value, "agent_id", None)
    step_id = getattr(value, "step_id", None)
    if episode_id is not None:
        payload["episode_id"] = episode_id
    if agent_id is not None:
        payload["agent_id"] = agent_id
    if step_id is not None:
        payload["step_id"] = step_id
    return payload


def inspect_rgb_array(rgb: object) -> RGBInspection:
    """Fail closed unless ``rgb`` is an HxWx3 uint8 image array."""

    if rgb is None:
        return RGBInspection(
            outcome=RGB_NONE,
            present=False,
            error="RGB/POV field is None",
        )
    shape = getattr(rgb, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        return RGBInspection(
            outcome=RGB_SHAPE_INVALID,
            present=True,
            error="RGB must be an HxWx3 image array",
        )
    try:
        height = int(shape[0])
        width = int(shape[1])
        channels = int(shape[2])
    except (TypeError, ValueError):
        return RGBInspection(
            outcome=RGB_SHAPE_INVALID,
            present=True,
            error="RGB shape must contain integer H, W, and channel sizes",
        )
    if height < 1 or width < 1 or channels != 3:
        return RGBInspection(
            outcome=RGB_SHAPE_INVALID,
            present=True,
            error=(
                "RGB shape must be H×W×3 with H>0 and W>0; "
                f"got {shape!r}"
            ),
            height=height if height >= 0 else None,
            width=width if width >= 0 else None,
            channels=channels if channels >= 0 else None,
            dtype=_dtype_name(rgb),
        )
    dtype_name = _dtype_name(rgb)
    if dtype_name != "uint8":
        return RGBInspection(
            outcome=RGB_DTYPE_INVALID,
            present=True,
            error=(
                "RGB dtype must be uint8 to match current MineRL POV output; "
                f"got {dtype_name!r}"
            ),
            height=height,
            width=width,
            channels=channels,
            dtype=dtype_name,
        )
    return RGBInspection(
        outcome=RGB_OK,
        present=True,
        height=height,
        width=width,
        channels=channels,
        dtype="uint8",
    )


def inspect_public_rgb(reset_result: object, *, episode_id: str) -> RGBInspection:
    """Inspect a public reset mapping for E1 RGB validity.

    Presence of an initial state mapping is checked by the runner first.
    This helper then requires every agent payload to expose a public RGB
    field and nothing from inventory, selected item, or evaluator truth.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        return RGBInspection(
            outcome=RGB_MISSING,
            present=False,
            error="episode_id must be a non-empty string",
        )
    if not isinstance(reset_result, Mapping) or not reset_result:
        return RGBInspection(
            outcome=RGB_MISSING,
            present=False,
            error="public RGB observation is missing",
        )
    if not initial_state_exists(reset_result, episode_id=episode_id):
        return RGBInspection(
            outcome=RGB_MISSING,
            present=False,
            error="reset did not return a usable public RGB state",
        )

    last_ok: RGBInspection | None = None
    for agent_id, value in reset_result.items():
        if not isinstance(agent_id, str) or not agent_id.strip() or value is None:
            return RGBInspection(
                outcome=RGB_MISSING,
                present=False,
                error="public RGB observation is missing",
            )
        payload = _payload_mapping(value)
        if payload is None:
            return RGBInspection(
                outcome=RGB_MISSING,
                present=False,
                error="RGB/POV field is missing",
            )
        leaked = sorted(key for key in payload if key in PUBLIC_RGB_LEAK_KEYS)
        extra = sorted(
            key
            for key in payload
            if key not in PUBLIC_RGB_ALLOWED_KEYS and key not in PUBLIC_RGB_LEAK_KEYS
        )
        if leaked or extra:
            names = leaked + extra
            return RGBInspection(
                outcome=RGB_LEAK,
                present="rgb" in payload,
                error="public RGB payload leaked non-RGB fields: " + ", ".join(names),
            )
        if "rgb" not in payload:
            return RGBInspection(
                outcome=RGB_MISSING,
                present=False,
                error="RGB/POV field is missing",
            )
        rgb = payload["rgb"]
        if rgb is None:
            return RGBInspection(
                outcome=RGB_NONE,
                present=False,
                error="RGB/POV field is None",
            )
        inspection = inspect_rgb_array(rgb)
        if not inspection.valid:
            return inspection
        last_ok = inspection
    if last_ok is None:
        return RGBInspection(
            outcome=RGB_MISSING,
            present=False,
            error="public RGB observation is missing",
        )
    return last_ok
