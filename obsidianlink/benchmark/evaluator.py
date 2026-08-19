"""Evaluator interface.

Evaluator-only information must never enter agent observations, prompts, or memory.
This module does not implement a truth framework.

The ABC exposes the minimum the BenchmarkRunner MUST forward after an
episode finishes, plus optional channels specific to a task family:

* ``report``     — the latest structured output the Agent emitted that
  the task cares about (e.g. a :class:`PerceptionReport` for D1).
* ``observation`` — the *agent-visible* observation the Agent most
  recently acted on. For Phase 2A this is also the
  evaluator-only ground truth (D1 has no server-side secret).
* ``final_observation`` — the observation *after* the last
  ``env.step()``. D2 uses this when a visual check of the
  post-action frame is useful; D1 ignores it.
* ``hidden_state`` — evaluator-only world snapshot (e.g. D3-01
  yaw from MineRL monitors). Must never have been shown to the
  Agent.
* ``ground_truth`` — task-level hidden truth (D1 presence bool,
  D2-01 initial direction, D3-01 spawn-yaw condition).

The primary metric set (success / steps / model_calls /
invalid_actions / elapsed_time) is fixed by the Development Plan; the
Evaluator is free to attach per-task ``evidence`` to the :class:`Result`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        report: Any = None,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        final_observation: Any = None,
        hidden_state: Any = None,
    ) -> Result:
        raise NotImplementedError
