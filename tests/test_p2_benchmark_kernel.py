"""P2 Benchmark Kernel smoke test.

Proves the v2 kernel can:

1. load a minimal ``TaskIdentity`` (no model, no MineRL);
2. drive a stub ``Agent`` against an in-memory stub ``Backend``;
3. accept a stub ``Evaluator`` and produce an ``EvaluatorVerdict``;
4. persist a ``BenchmarkRunRecord`` to disk;
5. keep evaluator-only payload off the agent-visible surface.

No LLM, no MineRL, no Gradle, no paid API call.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import TestCase

from obsidianlink.benchmark import (
    BenchmarkRunRecord,
    BenchmarkSuite,
    EvaluatorVerdict,
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    ExecutionMode,
    LayoutType,
    MetricName,
    MetricRecord,
    TaskIdentity,
    VerificationLevel,
    load_run_record,
    run_benchmark,
    write_run_record,
)
from obsidianlink.core.types import BackendStep, MacroAction, Observation


EPISODE_ID = "p2-benchmark-kernel-smoke-001"


class StubAgent:
    """Minimal agent that always waits. Exposes ``agent_id`` for the driver."""

    agent_id = "agent_1"

    def __init__(self, action_log: list[MacroAction]) -> None:
        self._log = action_log

    def act(self, observation: Observation) -> MacroAction:
        self._log.append(MacroAction.wait())
        return MacroAction.wait()


class StubBackend:
    """In-memory backend that satisfies the driver contract without Minecraft.

    ``reset`` publishes a single agent observation. ``step`` is a no-op
    that returns the same observation back. ``close`` is recorded.
    """

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.resets = 0
        self.steps = 0

    def open(self) -> None:
        self.opened = True

    def reset(self) -> Mapping[str, Observation]:
        self.resets += 1
        return {
            "agent_1": Observation(
                episode_id=EPISODE_ID,
                agent_id="agent_1",
                step_id=0,
                timestamp=0.0,
                frame=None,
                visible_inventory={"dirt": 1},
                selected_item="dirt",
                messages=(),
                workflow_stage="init",
            )
        }

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        self.steps += 1
        return BackendStep(
            episode_id=EPISODE_ID,
            step_id=self.steps,
            observations={
                "agent_1": Observation(
                    episode_id=EPISODE_ID,
                    agent_id="agent_1",
                    step_id=self.steps,
                    timestamp=float(self.steps),
                    frame=None,
                    visible_inventory={"dirt": 1},
                    selected_item="dirt",
                    messages=(),
                    workflow_stage="step",
                )
            },
            rewards={"agent_1": 0.0},
            terminated=False,
            truncated=False,
            info={},
        )

    def close(self) -> None:
        self.closed = True


class StubEvaluator:
    """Evaluator that always succeeds and reports a known outcome.

    The driver passes an ``initial_evaluator_state`` mapping plus the
    recorded evidence to ``evaluate``. The evaluator surfaces a private
    secret on the EVALUATOR_ONLY channel; the driver must keep it out
    of agent-visible records.
    """

    secret_truth = "evaluator-only-secret-do-not-leak"

    def evaluate(self, state: Mapping[str, Any]) -> EvaluatorVerdict:
        # the evaluator reads its own evaluator-only state and produces a verdict
        return EvaluatorVerdict(
            identity=EvidenceIdentity(
                episode_id=EPISODE_ID,
                step_id=int(state.get("step_id", 0)),
                agent_id=None,
            ),
            success=True,
            outcome="kernel_smoke_ok",
            evidence_complete=True,
        )


class BenchmarkKernelSmokeTest(TestCase):
    def setUp(self) -> None:
        self.task = TaskIdentity(
            task_instance_id="p2-kernel-smoke",
            suite=BenchmarkSuite.DIAGNOSTIC,
            mode=ExecutionMode.SINGLE,
            level="D1",
            layout=LayoutType.CONTROLLED,
        )
        self.backend = StubBackend()
        self.action_log: list[MacroAction] = []
        self.agent = StubAgent(self.action_log)
        self.evaluator = StubEvaluator()

    def test_kernel_runs_full_episode_without_llm(self) -> None:
        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=self.agent,
            evaluator=self.evaluator,
            max_steps=2,
        )

        self.assertIsInstance(record, BenchmarkRunRecord)
        self.assertEqual(record.task, self.task)
        self.assertEqual(record.runner_status, "completed")
        self.assertTrue(record.verdict.success)
        self.assertEqual(record.verdict.outcome, "kernel_smoke_ok")
        self.assertEqual(record.verification_level, VerificationLevel.UNIT_VERIFIED)
        # 2 observation records + 2 action records + 1 verdict record = 5
        self.assertEqual(len(record.evidence), 5)

        # Backend lifecycle
        self.assertTrue(self.backend.opened)
        self.assertTrue(self.backend.closed)
        self.assertEqual(self.backend.resets, 1)
        self.assertEqual(self.backend.steps, 2)
        self.assertEqual(len(self.action_log), 2)

    def test_agent_visible_observation_payload_never_contains_evaluator_secret(self) -> None:
        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=self.agent,
            evaluator=self.evaluator,
            max_steps=1,
            initial_evaluator_state={"truth": StubEvaluator.secret_truth},
        )
        for evidence in record.evidence:
            # Evidence payloads are MappingProxyType; use repr to scan all values
            self.assertNotIn(StubEvaluator.secret_truth, repr(dict(evidence.payload)))
        # agent-visible channels are observation + action only
        channels = {ev.channel for ev in record.evidence}
        self.assertIn(EvidenceChannel.AGENT_VISIBLE, channels)
        self.assertIn(EvidenceChannel.EVALUATOR_ONLY, channels)
        for ev in record.evidence:
            if ev.channel is EvidenceChannel.AGENT_VISIBLE:
                self.assertIn(ev.event_type, {"observation", "action"})

    def test_observation_payload_excludes_evaluator_only_fields(self) -> None:
        # Inject a backend that hides its evaluator-only truth in the
        # observation; the driver must strip it via the projection.
        class SneakyBackend(StubBackend):
            def reset(self):  # type: ignore[override]
                obs = super().reset()["agent_1"]
                object.__setattr__(obs, "frame", {"truth": "evaluator-only-hidden"})
                return {"agent_1": obs}

        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=SneakyBackend(),
            agent=self.agent,
            evaluator=self.evaluator,
            max_steps=1,
        )
        for ev in record.evidence:
            if ev.channel is EvidenceChannel.AGENT_VISIBLE and ev.event_type == "observation":
                # The driver must not surface the backend's raw ``frame`` payload.
                # Only the whitelisted agent-visible fields should appear.
                self.assertNotIn("frame", ev.payload)
                # And the hidden evaluator-only marker must not appear anywhere
                # in the projected payload.
                self.assertNotIn("evaluator-only-hidden", repr(dict(ev.payload)))

    def test_run_record_round_trips_through_disk(self) -> None:
        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=self.agent,
            evaluator=self.evaluator,
            max_steps=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_record.json"
            write_run_record(record, path)
            loaded = load_run_record(path)
            self.assertEqual(loaded.task, record.task)
            self.assertEqual(loaded.verdict, record.verdict)
            self.assertEqual(loaded.runner_status, record.runner_status)
            self.assertEqual(len(loaded.evidence), len(record.evidence))
            self.assertEqual(loaded.schema_version, record.schema_version)
            self.assertEqual(loaded.verification_level, record.verification_level)

    def test_metrics_hook_can_attach_diagnostic_metrics(self) -> None:
        def hook(step_id: int, state: Mapping[str, Any]) -> tuple[MetricRecord, ...]:
            return (
                MetricRecord(
                    name=MetricName.ENVIRONMENT_STEPS,
                    value=float(step_id),
                ),
            )

        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=self.agent,
            evaluator=self.evaluator,
            max_steps=3,
            metrics_hook=hook,
        )
        self.assertEqual(len(record.metrics), 1)
        self.assertEqual(record.metrics[0].name, MetricName.ENVIRONMENT_STEPS)
        self.assertEqual(record.metrics[0].value, 3.0)

    def test_task_identity_validation_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            TaskIdentity(
                task_instance_id="",
                suite=BenchmarkSuite.DIAGNOSTIC,
                mode=ExecutionMode.SINGLE,
                level="D1",
                layout=LayoutType.CONTROLLED,
            )
        with self.assertRaises(ValueError):
            TaskIdentity(
                task_instance_id="ok",
                suite=BenchmarkSuite.DIAGNOSTIC,
                mode=ExecutionMode.SINGLE,
                level="L1",  # wrong level for DIAGNOSTIC suite
                layout=LayoutType.CONTROLLED,
            )

    def test_evaluator_only_state_passed_to_evaluator_not_to_agent(self) -> None:
        seen_evaluator_states: list[Mapping[str, Any]] = []

        class CapturingEvaluator(StubEvaluator):
            def evaluate(self, state: Mapping[str, Any]) -> EvaluatorVerdict:
                seen_evaluator_states.append(dict(state))
                return super().evaluate(state)

        seen_agent_observations: list[Observation] = []

        class SpyAgent(StubAgent):
            def act(self, observation: Observation) -> MacroAction:
                seen_agent_observations.append(observation)
                return super().act(observation)

        run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=SpyAgent(self.action_log),
            evaluator=CapturingEvaluator(),
            max_steps=1,
            initial_evaluator_state={"server_truth": "hidden"},
        )
        # evaluator received the private state
        self.assertEqual(len(seen_evaluator_states), 1)
        self.assertIn("server_truth", seen_evaluator_states[0])
        # agent never received it
        for obs in seen_agent_observations:
            self.assertNotIn("server_truth", obs.frame if obs.frame else {})

    def test_rejects_non_macro_action(self) -> None:
        class BadAgent(StubAgent):
            def act(self, observation: Observation) -> object:  # type: ignore[override]
                return "not-a-macro-action"

        class HonestEvaluator(StubEvaluator):
            def evaluate(self, state: Mapping[str, Any]) -> EvaluatorVerdict:
                # Reflect the runner's actual status; do not claim success
                # if the runner did not complete.
                success = state.get("runner_status") == "completed"
                return EvaluatorVerdict(
                    identity=EvidenceIdentity(
                        episode_id=EPISODE_ID,
                        step_id=0,
                        agent_id=None,
                    ),
                    success=success,
                    outcome="runner_failed" if not success else "ok",
                    evidence_complete=True,
                )

        record = run_benchmark(
            episode_id=EPISODE_ID,
            task=self.task,
            backend=self.backend,
            agent=BadAgent(self.action_log),
            evaluator=HonestEvaluator(),
            max_steps=1,
        )
        self.assertEqual(record.runner_status, "failed")
        self.assertFalse(record.verdict.success)
        # runner_error is recorded on the evaluator-only channel
        error_records = [
            ev for ev in record.evidence if ev.event_type == "runner_error"
        ]
        self.assertEqual(len(error_records), 1)
        self.assertEqual(error_records[0].channel, EvidenceChannel.EVALUATOR_ONLY)


if __name__ == "__main__":
    unittest.main()
