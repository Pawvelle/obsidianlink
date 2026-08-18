"""Episode result. Keep the metric set small until a research question needs more.

The first-version metric set is fixed by the Development Plan:

* ``success``
* ``steps`` (environment steps)
* ``model_calls``
* ``invalid_actions``
* ``elapsed_time``

``evidence`` is a free-form bag for evaluator-specific signals (e.g.
the perception report the Agent emitted, the ground-truth snapshot the
Evaluator compared against, per-field match details). It is NOT a
replacement for the primary metric set; it exists so individual D / L
tasks can attach their own diagnostic breadcrumbs without bloating the
shared metric surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Result:
    task_id: str
    success: bool
    steps: int
    model_calls: int
    invalid_actions: int
    elapsed_time: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
