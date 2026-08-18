"""A Task describes a problem. It must not contain a solver or scripted route.

A Task carries the **hidden ground truth** for tasks whose truth
comes from a controlled scene (e.g. D1 Lava Presence: the env
places a lava block, and the evaluator needs to know that fact
to grade the Agent's report). The ground truth is task-level,
not env-level, because the same env can be reused across many
tasks (different targets, different positions) — the truth
belongs to the *question being asked*, not the scene.

For tasks whose ground truth is the agent-visible observation
itself (the original D1 inventory pilot, for example),
``ground_truth`` stays at its default ``None`` and the
evaluator derives truth from the observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    goal: str
    max_steps: int
    #: Hidden ground truth for tasks that need a separate
    #: evaluator-only channel. Must NOT be placed into the
    #: agent-visible observation or prompt. ``None`` means
    #: "the evaluator must derive truth from the observation
    #: it receives" (the D1 inventory pilot pattern).
    ground_truth: Any = None
