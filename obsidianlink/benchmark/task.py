"""A Task describes a problem. It must not contain a solver or scripted route."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    goal: str
    max_steps: int
