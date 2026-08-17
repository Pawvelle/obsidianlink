"""Benchmark run record and minimal driver for v2.

This module is the smallest thing that closes the v2 P2 kernel loop:

  TaskIdentity -> (backend, agent, evaluator) -> BenchmarkRunRecord

It reuses every other v2 kernel primitive in
:mod:`obsidianlink.benchmark` and :mod:`obsidianlink.core.types` and
deliberately does **not** introduce a registry, plugin system, or
abstract base class hierarchy.

P1-specific types (``EnvironmentValidationResult``,
``EnvironmentValidationCase``, the ``p1_suite`` orchestrator) are
out of scope here. P1 remains its own calibration pipeline.

Evidence separation is enforced at two points:

* :class:`obsidianlink.benchmark.evidence.EvidenceRecord` carries an
  :class:`obsidianlink.benchmark.evidence.EvidenceChannel` that names
  which surface the payload belongs to.
* The driver writes agent-visible observation and action payloads to
  ``AGENT_VISIBLE`` records and evaluator state to
  ``EVALUATOR_ONLY`` records. The agent's
  :meth:`act(observation) -> MacroAction` only ever sees
  agent-visible data; evaluator-only payload never appears in
  :class:`obsidianlink.core.types.Observation`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from obsidianlink.benchmark.evaluator import EvaluatorVerdict
from obsidianlink.benchmark.evidence import (
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    VerificationLevel,
)
from obsidianlink.benchmark.metrics import MetricName, MetricRecord
from obsidianlink.benchmark.runner import RUNNER_STATUSES
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    ExecutionMode,
    LayoutType,
    TaskIdentity,
)


SCHEMA_VERSION = "p2.benchmark.run_record.v1"
MAX_STEP_LIMIT = 100_000


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """Aggregate v2 evidence for one benchmark episode.

    This is the single on-disk artifact produced by the v2 kernel
    driver. It is the only type the kernel writes; everything else
    (observation, action, evaluator verdict) is reachable through
    this record.
    """

    task: TaskIdentity
    runner_status: str
    verdict: EvaluatorVerdict
    evidence: tuple[EvidenceRecord, ...]
    metrics: tuple[MetricRecord, ...]
    verification_level: VerificationLevel
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskIdentity):
            raise ValueError("task must be TaskIdentity")
        if self.runner_status not in RUNNER_STATUSES:
            raise ValueError(f"runner_status must be one of {sorted(RUNNER_STATUSES)}")
        if not isinstance(self.verdict, EvaluatorVerdict):
            raise ValueError("verdict must be EvaluatorVerdict")
        if not isinstance(self.evidence, tuple):
            raise ValueError("evidence must be a tuple of EvidenceRecord")
        if not all(isinstance(item, EvidenceRecord) for item in self.evidence):
            raise ValueError("evidence must contain only EvidenceRecord instances")
        if not isinstance(self.metrics, tuple):
            raise ValueError("metrics must be a tuple of MetricRecord")
        if not all(isinstance(item, MetricRecord) for item in self.metrics):
            raise ValueError("metrics must contain only MetricRecord instances")
        if not isinstance(self.verification_level, VerificationLevel):
            raise ValueError("verification_level must be VerificationLevel")
        _require_identifier(self.schema_version, "schema_version")
        if self.verdict.success and self.runner_status != "completed":
            raise ValueError("success verdict requires runner_status == 'completed'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence": [
                {
                    "agent_id": record.identity.agent_id,
                    "channel": record.channel.value,
                    "episode_id": record.identity.episode_id,
                    "event_type": record.event_type,
                    "payload": _mapping_to_json(record.payload),
                    "step_id": record.identity.step_id,
                }
                for record in self.evidence
            ],
            "metrics": [
                {
                    "name": metric.name.value,
                    "numerator": metric.numerator,
                    "denominator": metric.denominator,
                    "value": metric.value,
                }
                for metric in self.metrics
            ],
            "runner_status": self.runner_status,
            "schema_version": self.schema_version,
            "task": self.task.as_dict(),
            "verification_level": self.verification_level.value,
            "verdict": {
                "agent_id": self.verdict.identity.agent_id,
                "episode_id": self.verdict.identity.episode_id,
                "evidence_complete": self.verdict.evidence_complete,
                "outcome": self.verdict.outcome,
                "step_id": self.verdict.identity.step_id,
                "success": self.verdict.success,
            },
        }


def write_run_record(record: BenchmarkRunRecord, path: Path) -> Path:
    """Persist a :class:`BenchmarkRunRecord` as JSON.

    The path is created if needed. Parent directory must exist.
    """

    if not isinstance(record, BenchmarkRunRecord):
        raise ValueError("record must be BenchmarkRunRecord")
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_run_record(path: Path) -> BenchmarkRunRecord:
    """Load a :class:`BenchmarkRunRecord` previously written by :func:`write_run_record`."""

    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _record_from_dict(payload)


def _record_from_dict(payload: Mapping[str, Any]) -> BenchmarkRunRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("run record payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run record schema_version: {schema_version!r}"
        )
    task_payload = payload.get("task")
    if not isinstance(task_payload, Mapping):
        raise ValueError("task must be a mapping")
    task = TaskIdentity(
        task_instance_id=task_payload["task_instance_id"],
        suite=BenchmarkSuite(task_payload["suite"]),
        mode=ExecutionMode(task_payload["mode"]),
        level=str(task_payload["level"]),
        layout=LayoutType(task_payload["layout"]),
        family=str(task_payload.get("family", "nether_portal_construction")),
    )
    verdict_payload = payload.get("verdict")
    if not isinstance(verdict_payload, Mapping):
        raise ValueError("verdict must be a mapping")
    identity = EvidenceIdentity(
        episode_id=verdict_payload["episode_id"],
        step_id=int(verdict_payload["step_id"]),
        agent_id=verdict_payload.get("agent_id"),
    )
    verdict = EvaluatorVerdict(
        identity=identity,
        success=bool(verdict_payload["success"]),
        outcome=str(verdict_payload["outcome"]),
        evidence_complete=bool(verdict_payload["evidence_complete"]),
    )
    evidence_records: list[EvidenceRecord] = []
    for raw in payload.get("evidence", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("evidence entries must be mappings")
        rec_identity = EvidenceIdentity(
            episode_id=raw["episode_id"],
            step_id=int(raw["step_id"]),
            agent_id=raw.get("agent_id"),
        )
        evidence_records.append(
            EvidenceRecord(
                identity=rec_identity,
                channel=EvidenceChannel(str(raw["channel"])),
                event_type=str(raw["event_type"]),
                payload=raw.get("payload", {}),
            )
        )
    metric_records: list[MetricRecord] = []
    for raw in payload.get("metrics", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("metric entries must be mappings")
        metric_records.append(
            MetricRecord(
                name=MetricName(raw["name"]),
                value=float(raw["value"]),
                numerator=raw.get("numerator"),
                denominator=raw.get("denominator"),
            )
        )
    return BenchmarkRunRecord(
        task=task,
        runner_status=str(payload.get("runner_status")),
        verdict=verdict,
        evidence=tuple(evidence_records),
        metrics=tuple(metric_records),
        verification_level=VerificationLevel(str(payload["verification_level"])),
        schema_version=schema_version,
    )


def _mapping_to_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _value_to_json(item) for key, item in value.items()}


def _value_to_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return _mapping_to_json(value)
    if isinstance(value, Mapping):
        return _mapping_to_json(value)
    if isinstance(value, (list, tuple)):
        return [_value_to_json(item) for item in value]
    return value


BackendT = Any
AgentT = Any
EvaluatorT = Any
MetricsHook = Callable[[int, Mapping[str, Any]], tuple[MetricRecord, ...]]


def run_benchmark(
    *,
    episode_id: str,
    task: TaskIdentity,
    backend: BackendT,
    agent: AgentT,
    evaluator: EvaluatorT,
    max_steps: int = 1,
    initial_evaluator_state: Mapping[str, Any] | None = None,
    metrics_hook: MetricsHook | None = None,
) -> BenchmarkRunRecord:
    """Run one v2 benchmark episode end-to-end without any LLM call.

    Parameters
    ----------
    episode_id
        Episode identity shared by every :class:`EvidenceRecord`.
    task
        The frozen :class:`TaskIdentity`. The driver does not read
        any field of ``task`` other than to record it.
    backend
        Object that exposes ``open()``,
        ``reset() -> Mapping[agent_id, observation]``,
        ``step(actions) -> step_result`` (with ``.observations``), and
        ``close()``. The :class:`obsidianlink.env.fake.FakeEnvironmentBackend`
        satisfies this contract for offline stubs; any MineRL/Malmo
        adapter that wraps
        :class:`obsidianlink.env.minerl_backend.MineRLEnvironmentBackend`
        does the same for live runs.
    agent
        Object that exposes ``agent_id`` and ``act(observation) -> action``.
        The :class:`obsidianlink.agents.Agent` Protocol describes this.
    evaluator
        Object that exposes ``evaluate(state) -> EvaluatorVerdict``. The
        :class:`obsidianlink.benchmark.Evaluator` Protocol describes
        this. The driver passes the backend's evaluator-only state
        here; the agent never sees it.
    max_steps
        Hard cap on environment steps. Defaults to one for the
        contract smoke test.
    initial_evaluator_state
        Optional mapping that the driver forwards to ``evaluator.evaluate``.
        Evaluator-only payload must never appear in any observation
        passed to the agent.
    metrics_hook
        Optional callable invoked at the end of the episode to derive
        :class:`MetricRecord` entries from the recorded evidence.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    if not isinstance(task, TaskIdentity):
        raise ValueError("task must be TaskIdentity")
    if not isinstance(max_steps, int) or max_steps < 1 or max_steps > MAX_STEP_LIMIT:
        raise ValueError(f"max_steps must be in [1, {MAX_STEP_LIMIT}]")
    agent_id = getattr(agent, "agent_id", None)
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent must expose a non-empty agent_id")

    evidence: list[EvidenceRecord] = []
    runner_status = "failed"
    verdict: EvaluatorVerdict | None = None
    try:
        backend.open()
        observations = backend.reset()
        if not isinstance(observations, Mapping) or not observations:
            raise RuntimeError("backend.reset must return a non-empty mapping")
        if agent_id not in observations:
            raise RuntimeError(
                f"backend.reset did not return observation for agent_id={agent_id!r}"
            )
        for step_id in range(max_steps):
            observation = observations[agent_id]
            evidence.append(
                EvidenceRecord(
                    identity=EvidenceIdentity(
                        episode_id=episode_id, step_id=step_id, agent_id=agent_id
                    ),
                    channel=EvidenceChannel.AGENT_VISIBLE,
                    event_type="observation",
                    payload=_agent_visible_observation_payload(observation),
                )
            )
            action = agent.act(observation)
            action_type = getattr(action, "action_type", None)
            action_target = getattr(action, "target", None)
            if not isinstance(action_type, str):
                raise RuntimeError("agent.act must return an object with action_type")
            evidence.append(
                EvidenceRecord(
                    identity=EvidenceIdentity(
                        episode_id=episode_id, step_id=step_id, agent_id=agent_id
                    ),
                    channel=EvidenceChannel.AGENT_VISIBLE,
                    event_type="action",
                    payload={"action_type": action_type, "target": action_target},
                )
            )
            step_result = backend.step({agent_id: action})
            next_observations = getattr(step_result, "observations", None)
            if not isinstance(next_observations, Mapping) or not next_observations:
                raise RuntimeError("backend.step result must expose observations mapping")
            observations = next_observations
            if getattr(step_result, "terminated", False) or getattr(
                step_result, "truncated", False
            ):
                # Episode ended naturally; do not call backend.step() again.
                break
        runner_status = "completed"
    except Exception as error:
        evidence.append(
            EvidenceRecord(
                identity=EvidenceIdentity(
                    episode_id=episode_id, step_id=0, agent_id=agent_id
                ),
                channel=EvidenceChannel.EVALUATOR_ONLY,
                event_type="runner_error",
                payload={"error_type": type(error).__name__, "error": str(error)},
            )
        )
    finally:
        try:
            backend.close()
        except Exception as error:  # noqa: BLE001 - close errors are recorded, not raised
            evidence.append(
                EvidenceRecord(
                    identity=EvidenceIdentity(
                        episode_id=episode_id, step_id=0, agent_id=agent_id
                    ),
                    channel=EvidenceChannel.EVALUATOR_ONLY,
                    event_type="close_error",
                    payload={"error_type": type(error).__name__, "error": str(error)},
                )
            )
            if runner_status == "completed":
                # close failure must not let the kernel claim success; the
                # evaluator is forced to reflect the failed close via
                # runner_status='failed' and the recorded close_error evidence.
                runner_status = "failed"

    state = dict(initial_evaluator_state or {})
    state["evidence"] = tuple(evidence)
    state["runner_status"] = runner_status
    verdict = evaluator.evaluate(state)
    if not isinstance(verdict, EvaluatorVerdict):
        raise RuntimeError("evaluator.evaluate must return EvaluatorVerdict")
    evidence.append(
        EvidenceRecord(
            identity=verdict.identity,
            channel=EvidenceChannel.EVALUATOR_ONLY,
            event_type="verdict",
            payload={
                "outcome": verdict.outcome,
                "evidence_complete": verdict.evidence_complete,
            },
        )
    )

    if metrics_hook is not None:
        observation_steps = sum(
            1 for record in evidence if record.event_type == "observation"
        )
        metrics = tuple(metrics_hook(observation_steps, state))
    else:
        metrics = ()
    for metric in metrics:
        if not isinstance(metric, MetricRecord):
            raise RuntimeError("metrics_hook must yield MetricRecord instances")

    return BenchmarkRunRecord(
        task=task,
        runner_status=runner_status,
        verdict=verdict,
        evidence=tuple(evidence),
        metrics=metrics,
        verification_level=VerificationLevel.UNIT_VERIFIED,
    )


def _agent_visible_observation_payload(observation: Any) -> dict[str, Any]:
    """Project an observation to the agent-visible surface only.

    Reads only the public :class:`obsidianlink.core.types.Observation`
    fields. Any evaluator-only payload attached to the observation
    object (for example, a non-``None`` ``frame`` slot) is dropped.
    """

    visible_inventory = getattr(observation, "visible_inventory", None)
    return {
        "episode_id": getattr(observation, "episode_id", None),
        "agent_id": getattr(observation, "agent_id", None),
        "step_id": getattr(observation, "step_id", None),
        "timestamp": getattr(observation, "timestamp", None),
        "selected_item": getattr(observation, "selected_item", None),
        "visible_inventory": (
            dict(visible_inventory) if visible_inventory is not None else None
        ),
        "messages": list(getattr(observation, "messages", ())),
        "workflow_stage": getattr(observation, "workflow_stage", None),
    }


__all__ = [
    "BenchmarkRunRecord",
    "MAX_STEP_LIMIT",
    "SCHEMA_VERSION",
    "load_run_record",
    "run_benchmark",
    "write_run_record",
]
