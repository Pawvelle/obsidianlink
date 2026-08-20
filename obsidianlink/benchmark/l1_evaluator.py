"""Formal L1 Evaluator.

Judges only from evaluator-only truth: never ``ObservationFromGrid``,
never Agent-visible ``Observation`` fields, never Agent/Oracle self-report.

Truth channels (live-verified on MineRL 1.0.2 / MCP-Reborn, Malmo 0.37.0,
see ``obsidianlink/env/l1_scene.py`` and the 2026-08-19 feasibility spike):

* ``reward`` from ``RewardForTouchingBlockType(nether_portal)`` — gym
  step() reward, captured into ``Environment.hidden_state`` by
  :class:`obsidianlink.env.minerl.MineRLEnvironment`. A real
  ``nether_portal`` block only exists once a frame has been correctly
  ignited, so touching it is *portal_activated* / *portal_contacted*
  evidence.
* ``biome_id`` from ``ObservationFromCurrentLocation`` (``location_stats``
  / gym ``info``). Minecraft 1.16.5's legacy biome id for the Nether is
  ``8`` ("hell"). A confirmed ``biome_id == 8`` sample *after* portal
  activation is the primary *nether_entered* truth.

``portal_constructed`` (frame-complete-before-ignition) has no cheap,
reliable non-``ObservationFromGrid`` truth on this stack, so it is left
``"unknown"`` unless the caller supplies external evidence (e.g. Oracle
scripted construction bookkeeping is *not* evaluator truth and must not
be used to set this field to True).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import EVALUATOR_FAILURE, Result
from obsidianlink.benchmark.task import Task
from obsidianlink.env.environment import observation_field_names

NETHER_BIOME_ID = 8

_LEAKED_FIELD_NAMES = (
    "hidden_state",
    "reward",
    "biome_id",
    "portal_activated",
    "nether_entered",
    "success",
    "xpos",
    "ypos",
    "zpos",
    "yaw",
    "pitch",
)


def leaked_evaluator_fields(observation: Any) -> list[str]:
    """Any evaluator-only name reachable on the Agent-visible Observation."""
    if observation is None:
        return []
    leaked = [name for name in _LEAKED_FIELD_NAMES if hasattr(observation, name)]
    extra = [
        name
        for name in getattr(observation, "__dict__", {})
        if name not in observation_field_names()
    ]
    return sorted(set(leaked + extra))


def portal_activated_from_rewards(reward_samples: Iterable[float | None]) -> bool:
    """True if any sample shows the ``nether_portal`` touch reward firing."""
    return any((r or 0.0) > 0.0 for r in reward_samples)


@dataclass(frozen=True)
class BiomeSample:
    reward: float | None
    biome_id: float | None


def resolve_nether_entered(
    samples: Iterable[BiomeSample], *, baseline_biome_id: float | None
) -> dict[str, Any]:
    """Nether-entry truth: activation evidence, then a strict biome match.

    Returns a dict (not just a bool) so the caller can see *why* — a
    weaker "biome changed but not to id 8" signal is recorded as evidence,
    never silently upgraded to success.
    """
    samples = list(samples)
    activated = portal_activated_from_rewards(s.reward for s in samples)
    activated_index = None
    for i, s in enumerate(samples):
        if (s.reward or 0.0) > 0.0:
            activated_index = i
            break

    strict_match = False
    weak_change = False
    after = samples[activated_index:] if activated_index is not None else []
    for s in after:
        if s.biome_id is None:
            continue
        if int(s.biome_id) == NETHER_BIOME_ID:
            strict_match = True
        elif baseline_biome_id is not None and int(s.biome_id) != int(baseline_biome_id):
            weak_change = True

    return {
        "portal_activated": activated,
        "activated_at_sample": activated_index,
        "baseline_biome_id": baseline_biome_id,
        "nether_biome_strict_match": strict_match,
        "biome_changed_weak": weak_change,
        "nether_entered": bool(activated and strict_match),
    }


@dataclass
class L1Milestones:
    """Accumulated evaluator-only evidence for one L1 episode."""

    baseline_biome_id: float | None = None
    samples: list[BiomeSample] = field(default_factory=list)
    steps_observed: int = 0

    def observe(self, hidden_state: Mapping[str, Any] | None) -> None:
        if not isinstance(hidden_state, Mapping):
            return
        reward = hidden_state.get("reward")
        biome_id = hidden_state.get("biome_id")
        if self.steps_observed == 0 and biome_id is not None and self.baseline_biome_id is None:
            self.baseline_biome_id = biome_id
        self.samples.append(BiomeSample(reward=reward, biome_id=biome_id))
        self.steps_observed += 1

    def resolve(self) -> dict[str, Any]:
        return resolve_nether_entered(self.samples, baseline_biome_id=self.baseline_biome_id)


class L1Evaluator(Evaluator):
    """Stateful across an episode: call :meth:`observe_step` every tick,
    then :meth:`evaluate` once at the end. Truth is evaluator-only and is
    never derived from Agent/Oracle self-report or ``ObservationFromGrid``.
    """

    def __init__(self) -> None:
        self.milestones = L1Milestones()

    def reset(self) -> None:
        self.milestones = L1Milestones()

    def observe_step(self, hidden_state: Mapping[str, Any] | None) -> None:
        self.milestones.observe(hidden_state)

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        hidden_state: Any = None,
        used_vision: bool | None = None,
        fallback_reason: str | None = None,
        vision_calls: int = 0,
    ) -> Result:
        if hidden_state is not None:
            self.observe_step(hidden_state)

        leaked = leaked_evaluator_fields(observation)
        resolution = self.milestones.resolve()

        evidence: dict[str, Any] = {
            "milestones": {
                "portal_constructed": "unknown",
                "portal_activated": resolution["portal_activated"],
                "portal_contacted": resolution["portal_activated"],
                "nether_entered": resolution["nether_entered"],
            },
            "evaluator_truth": {
                "activation_source": "RewardForTouchingBlockType(nether_portal)",
                "nether_entry_source": (
                    f"biome_id == {NETHER_BIOME_ID} (Nether) via "
                    "ObservationFromCurrentLocation, after activation"
                ),
                "baseline_biome_id": resolution["baseline_biome_id"],
                "nether_biome_strict_match": resolution["nether_biome_strict_match"],
                "biome_changed_weak": resolution["biome_changed_weak"],
                "activated_at_sample": resolution["activated_at_sample"],
                "not_used": "ObservationFromGrid",
            },
            "steps_observed": self.milestones.steps_observed,
        }

        if leaked:
            evidence["leaked_fields"] = leaked
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence={**evidence, "failure_class": EVALUATOR_FAILURE, "reason": "evaluation_error"},
            )

        success = bool(resolution["nether_entered"])
        reason = "ok" if success else "nether_entry_not_confirmed"
        return Result(
            task_id=task.task_id,
            success=success,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
            evidence={**evidence, "reason": reason},
        )


__all__ = [
    "NETHER_BIOME_ID",
    "BiomeSample",
    "L1Evaluator",
    "L1Milestones",
    "leaked_evaluator_fields",
    "portal_activated_from_rewards",
    "resolve_nether_entered",
]
