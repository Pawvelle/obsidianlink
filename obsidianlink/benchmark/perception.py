"""Perception report types shared by Diagnostic tasks.

Two report flavours are defined here:

* :class:`PerceptionReport` — the Phase 2A *inventory* report
  (``inventory`` + ``selected_item``). The D1 inventory pilot uses
  this.
* :class:`PresenceReport` — the *presence* report (``visible``
  boolean). D1-01 Lava and D1-02 Water (and the historical
  Phase 2C presence family) use this. The report only asks
  "is the target visible in the frame".

The report is what the :class:`Evaluator` compares against the
ground truth. For Phase 2A the ground truth is the agent-visible
observation the Agent just acted on (inventory pilot). For Phase
2C, the ground truth is the **hidden** truth attached to the
``Task`` (the controlled-scene env knows what block it placed
in the world; the Agent never sees that).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerceptionReport:
    """Structured perception output an Agent can emit on a given step.

    ``inventory`` is the ``{item_name: count}`` summary the Agent
    believes it currently holds. ``selected_item`` is the hotbar item
    the Agent believes is selected. Either field can be ``None`` (or an
    empty dict) when the Agent is unsure or the underlying observation
    is empty.
    """

    inventory: dict[str, int] | None = None
    selected_item: str | None = None

    def is_well_formed(self) -> bool:
        """A report is well-formed iff ``inventory`` is a dict (possibly empty).

        ``selected_item`` is free-form: the Agent may report ``None``
        for either "no hotbar item visible" or "I don't know".
        """
        return isinstance(self.inventory, dict)


def parse_perception_report(response: str) -> PerceptionReport | None:
    """Extract a :class:`PerceptionReport` from a model response string.

    The model contract (Phase 1) is ``str -> str``. For Diagnostic
    tasks we extend the response with an optional ``report`` field::

        {"action": "WAIT",
         "report": {"inventory": {"dirt": 4}, "selected_item": "dirt"}}

    Returns ``None`` when:

    * the response is not parseable JSON,
    * the JSON is not a dict,
    * the dict has no ``report`` key,
    * the ``report`` value is not a dict.

    Callers should treat ``None`` as "no perception report emitted" and
    the :class:`Evaluator` decides whether that means failure.
    """
    if not isinstance(response, str) or not response.strip():
        return None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("report")
    if not isinstance(raw, dict):
        return None

    raw_inv = raw.get("inventory", {})
    inventory: dict[str, int] | None
    if isinstance(raw_inv, dict):
        cleaned: dict[str, int] = {}
        for name, qty in raw_inv.items():
            try:
                cleaned[str(name)] = int(qty)
            except (TypeError, ValueError):
                continue
        inventory = cleaned
    else:
        inventory = None

    raw_sel = raw.get("selected_item", None)
    if raw_sel is None:
        selected_item: str | None = None
    else:
        selected_item = str(raw_sel)

    return PerceptionReport(
        inventory=inventory,
        selected_item=selected_item,
    )


# ---------------------------------------------------------------------------
# Presence report (D1-01 lava, D1-02 water; historical Phase 2C family)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PresenceReport:
    """A boolean visibility report.

    The Agent emits ``{"visible": true}`` or ``{"visible": false}``
    in response to a D1 presence prompt (e.g. "is LAVA visible?").

    ``visible`` is ``None`` when the report could not be parsed
    (which is an ``output_protocol_error`` per the Phase 2C
    failure-mode contract — *not* a perception error).
    """

    visible: bool | None = None

    def is_well_formed(self) -> bool:
        """A well-formed presence report has a boolean ``visible``."""
        return isinstance(self.visible, bool)


def parse_presence_report(response: str) -> PresenceReport | None:
    """Extract a :class:`PresenceReport` from a model response string.

    Accepts responses of the form::

        {"visible": true}
        {"visible": false}
        {"action": "WAIT", "visible": true}        # tolerated
        {"action": "WAIT", "report": {"visible": true}}  # tolerated

    Returns ``None`` when the response is not parseable JSON or
    the JSON is not a dict. The returned :class:`PresenceReport`
    has ``visible=None`` if the dict has no usable boolean
    ``visible`` key — that is the protocol-level signal the
    :class:`D1PresenceEvaluator` maps to ``output_protocol_error``.
    """
    if not isinstance(response, str) or not response.strip():
        return None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    # The Agent prompt asks for a top-level ``visible`` key. We
    # also tolerate ``report.visible`` for symmetry with the
    # inventory D1 format; the evaluator only cares about the
    # final boolean.
    raw = data.get("visible")
    if raw is None and isinstance(data.get("report"), dict):
        raw = data["report"].get("visible")
    if not isinstance(raw, bool):
        return PresenceReport(visible=None)
    return PresenceReport(visible=raw)


__all__ = [
    "PerceptionReport",
    "PresenceReport",
    "parse_perception_report",
    "parse_presence_report",
]
