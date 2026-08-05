"""Offline tests for the R2 backend capability manifest.

These tests prove, in code, that:

* :class:`obsidianlink.env.capabilities.BackendCapabilities` is an
  immutable, type-strict description of what a backend supports and
  that its public snapshot is JSON-serializable with stable key
  order.
* :func:`obsidianlink.env.capabilities.assert_casting_c1_capabilities`
  fails closed with a stable ordered ``missing`` tuple when the
  manifest is incomplete.
* :func:`obsidianlink.env.capabilities.assert_backend_can_start_task`
  enforces the casting-c1 gate at the workflow boundary and is a
  no-op for other workflows.
* :class:`obsidianlink.env.fake.FakeEnvironmentBackend` and
  :class:`obsidianlink.env.minerl_backend.MineRLEnvironmentBackend`
  both run the gate from their ``reset`` entry point. When the gate
  fires, the underlying runtime is never touched: no task is set,
  no step is incremented, no observation is produced, and (for the
  real backend) the env factory is never called.
* Agent-visible observations never leak evaluator-only
  target-cell, target-block, fluid, or portal-grid truth.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime when checking the
``MineRLEnvironmentBackend`` capability — the manifest is a static
declaration.
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from obsidianlink.core.interfaces import EnvironmentBackend
from obsidianlink.core.types import MacroAction
from obsidianlink.env.capabilities import (
    CAPABILITY_IDS,
    BackendCapabilities,
    CapabilityMismatchError,
    assert_backend_can_start_task,
    assert_casting_c1_capabilities,
    missing_for_casting_c1,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from tests.helpers import casting_c1_task, sample_task


class BackendCapabilitiesImmutabilityTests(unittest.TestCase):
    def test_capability_id_tuple_is_canonical_and_ordered(self) -> None:
        expected = (
            "select_water_bucket",
            "select_lava_bucket",
            "use_water_bucket",
            "use_lava_bucket",
            "public_inventory",
            "selected_item",
            "target_block_truth",
            "fluid_truth",
        )
        self.assertEqual(CAPABILITY_IDS, expected)
        # Ensure the tuple is hashable and immutable.
        with self.assertRaises(TypeError):
            CAPABILITY_IDS[0] = "renamed"  # type: ignore[index]

    def test_backend_capabilities_is_frozen(self) -> None:
        caps = BackendCapabilities.full()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            caps.can_select_water_bucket = False  # type: ignore[misc]

    def test_field_types_must_be_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "can_select_water_bucket"):
            BackendCapabilities(can_select_water_bucket=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "exposes_fluid_truth"):
            BackendCapabilities(exposes_fluid_truth="yes")  # type: ignore[arg-type]

    def test_missing_returns_stable_canonical_order(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_select_water_bucket=False,
            can_use_lava_bucket=False,
        )
        self.assertEqual(
            caps.missing(),
            ("select_water_bucket", "use_lava_bucket"),
        )
        # Build a second instance with the same fields in different
        # order to prove the result is order-stable.
        caps_reordered = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_lava_bucket=False,
            can_select_water_bucket=False,
        )
        self.assertEqual(caps.missing(), caps_reordered.missing())


class BackendCapabilitiesAsDictTests(unittest.TestCase):
    """``as_dict()`` must be JSON-serializable with canonical key order.

    The casting manifest is meant to be embedded in evidence and run
    reports. If a caller cannot ``json.dumps`` the result, the
    manifest is not actually serializable, which violates the R2
    contract.
    """

    def test_as_dict_is_json_serializable(self) -> None:
        caps = BackendCapabilities.full()
        payload = json.dumps(caps.as_dict())
        round_tripped = json.loads(payload)
        self.assertEqual(
            round_tripped,
            {cap_id: True for cap_id in CAPABILITY_IDS},
        )

    def test_as_dict_round_trips_for_partial_manifest(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_water_bucket=False,
            exposes_target_block_truth=False,
            exposes_fluid_truth=False,
        )
        round_tripped = json.loads(json.dumps(caps.as_dict()))
        self.assertEqual(round_tripped["use_water_bucket"], False)
        self.assertEqual(round_tripped["target_block_truth"], False)
        self.assertEqual(round_tripped["fluid_truth"], False)
        # The remaining fields stay True.
        self.assertEqual(round_tripped["public_inventory"], True)
        # Every capability id is present in the snapshot.
        self.assertEqual(set(round_tripped), set(CAPABILITY_IDS))

    def test_as_dict_key_order_is_canonical(self) -> None:
        full = BackendCapabilities.full()
        self.assertEqual(list(full.as_dict()), list(CAPABILITY_IDS))
        partial = dataclasses.replace(
            BackendCapabilities.full(),
            exposes_fluid_truth=False,
            can_select_water_bucket=False,
        )
        self.assertEqual(list(partial.as_dict()), list(CAPABILITY_IDS))

    def test_as_dict_values_are_strict_bools(self) -> None:
        for value in BackendCapabilities.full().as_dict().values():
            self.assertIs(type(value), bool)

    def test_as_dict_returns_fresh_dict_each_call(self) -> None:
        caps = BackendCapabilities.full()
        first = caps.as_dict()
        second = caps.as_dict()
        # Distinct objects, equal contents.
        self.assertIsNot(first, second)
        self.assertEqual(first, second)
        # Mutating the snapshot must not affect the frozen caps.
        first["select_water_bucket"] = False
        first.pop("fluid_truth")
        self.assertTrue(caps.can_select_water_bucket)
        self.assertTrue(caps.exposes_fluid_truth)
        # The second snapshot is unaffected.
        self.assertEqual(second, {cap_id: True for cap_id in CAPABILITY_IDS})

    def test_as_dict_is_mutated_independently_from_caps(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            exposes_selected_item=False,
        )
        snapshot = caps.as_dict()
        snapshot["selected_item"] = True
        snapshot["use_lava_bucket"] = False
        # caps is frozen and unaffected.
        self.assertFalse(caps.exposes_selected_item)
        self.assertTrue(caps.can_use_lava_bucket)


class FullManifestPassesGateTests(unittest.TestCase):
    def test_full_manifest_passes_casting_c1_gate(self) -> None:
        caps = BackendCapabilities.full()
        self.assertTrue(caps.supports_casting_c1)
        self.assertEqual(caps.missing_for_casting_c1(), ())
        self.assertEqual(missing_for_casting_c1(caps), ())
        # The gate returns ``None`` when the manifest is complete.
        self.assertIsNone(assert_casting_c1_capabilities(caps))

    def test_full_manifest_passes_gate_with_task_id(self) -> None:
        caps = BackendCapabilities.full()
        self.assertIsNone(
            assert_casting_c1_capabilities(
                caps, task_id="casting_c1_fixed_seed_0"
            )
        )


class MissingCapabilitiesAreReportedTests(unittest.TestCase):
    def _caps_without(self, *missing_fields: str) -> BackendCapabilities:
        return dataclasses.replace(
            BackendCapabilities.full(), **{field: False for field in missing_fields}
        )

    def test_missing_single_capability_is_reported(self) -> None:
        caps = self._caps_without("can_use_water_bucket")
        self.assertFalse(caps.supports_casting_c1)
        self.assertEqual(caps.missing_for_casting_c1(), ("use_water_bucket",))
        with self.assertRaises(CapabilityMismatchError) as ctx:
            assert_casting_c1_capabilities(caps, task_id="casting_c1_fixed_seed_0")
        self.assertEqual(ctx.exception.missing, ("use_water_bucket",))
        self.assertEqual(ctx.exception.task_id, "casting_c1_fixed_seed_0")
        self.assertIn("use_water_bucket", str(ctx.exception))
        self.assertIn("casting_c1_fixed_seed_0", str(ctx.exception))

    def test_missing_multiple_capabilities_are_reported_in_canonical_order(self) -> None:
        caps = self._caps_without(
            "can_use_lava_bucket",
            "exposes_selected_item",
            "exposes_fluid_truth",
            "can_select_water_bucket",
        )
        self.assertEqual(
            caps.missing_for_casting_c1(),
            (
                "select_water_bucket",
                "use_lava_bucket",
                "selected_item",
                "fluid_truth",
            ),
        )

    def test_capability_mismatch_error_normalises_unknown_and_reorder(self) -> None:
        caps = self._caps_without(
            "exposes_fluid_truth", "can_use_lava_bucket", "can_select_water_bucket"
        )
        with self.assertRaises(CapabilityMismatchError) as ctx:
            assert_casting_c1_capabilities(caps)
        # Canonical order regardless of which fields the caller set.
        self.assertEqual(
            ctx.exception.missing,
            (
                "select_water_bucket",
                "use_lava_bucket",
                "fluid_truth",
            ),
        )

    def test_capability_mismatch_error_rejects_empty_or_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing capability ids"):
            CapabilityMismatchError(())
        with self.assertRaisesRegex(ValueError, "missing capability ids"):
            CapabilityMismatchError(("select_water_bucket", ""))  # type: ignore[arg-type]


class BackendStartGateTests(unittest.TestCase):
    """``assert_backend_can_start_task`` is the workflow-aware wrapper.

    The function must enforce the casting-c1 manifest only for the
    ``casting_c1_fixed`` workflow. Other workflows must be a
    no-op so the legacy Route A0 contract keeps working.
    """

    def test_casting_c1_task_runs_the_gate(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_water_bucket=False,
        )
        backend = FakeEnvironmentBackend(capabilities=caps)
        with self.assertRaises(CapabilityMismatchError) as ctx:
            assert_backend_can_start_task(backend, casting_c1_task())
        self.assertEqual(ctx.exception.missing, ("use_water_bucket",))
        self.assertEqual(ctx.exception.task_id, "casting_c1_fixed_seed_0")

    def test_non_casting_task_is_a_no_op_for_missing_caps(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_water_bucket=False,
            exposes_target_block_truth=False,
        )
        backend = FakeEnvironmentBackend(capabilities=caps)
        # The non-casting workflow must not be constrained by the
        # casting manifest; the gate returns ``None`` silently.
        self.assertIsNone(assert_backend_can_start_task(backend, sample_task()))

    def test_full_cap_backend_passes_for_casting_c1(self) -> None:
        backend = FakeEnvironmentBackend()  # default BackendCapabilities.full()
        self.assertIsNone(
            assert_backend_can_start_task(backend, casting_c1_task())
        )

    def test_gate_rejects_non_capabilities_return(self) -> None:
        class _BadBackend:
            def capabilities(self) -> None:
                return None

        with self.assertRaisesRegex(TypeError, "BackendCapabilities"):
            assert_backend_can_start_task(_BadBackend(), casting_c1_task())


class FakeBackendManifestTests(unittest.TestCase):
    def test_default_fake_backend_claims_full_capabilities(self) -> None:
        backend = FakeEnvironmentBackend()
        caps = backend.capabilities()
        self.assertIsInstance(caps, BackendCapabilities)
        self.assertTrue(caps.supports_casting_c1)

    def test_fake_backend_can_pretend_to_lack_capabilities(self) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_select_water_bucket=False,
            can_use_lava_bucket=False,
            exposes_fluid_truth=False,
        )
        backend = FakeEnvironmentBackend.with_capabilities(caps)
        self.assertIs(backend.capabilities(), caps)
        self.assertEqual(
            backend.capabilities().missing_for_casting_c1(),
            ("select_water_bucket", "use_lava_bucket", "fluid_truth"),
        )
        with self.assertRaisesRegex(
            CapabilityMismatchError, "select_water_bucket"
        ):
            assert_casting_c1_capabilities(
                backend.capabilities(),
                task_id=casting_c1_task().task_id,
            )

    def test_fake_backend_rejects_non_capabilities_constructor_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "BackendCapabilities"):
            FakeEnvironmentBackend(capabilities={"can_select_water_bucket": True})  # type: ignore[arg-type]

    def test_fake_backend_protocol_duck_typing_includes_capabilities(self) -> None:
        backend = FakeEnvironmentBackend()
        # ``EnvironmentBackend`` is a ``@runtime_checkable`` protocol;
        # adding ``capabilities`` keeps the fake backend a structural
        # member of the contract.
        self.assertIsInstance(backend, EnvironmentBackend)


class FakeBackendResetGateTests(unittest.TestCase):
    """``FakeEnvironmentBackend.reset`` must enforce the gate.

    The reset path is the canonical entry point for every episode,
    so the pre-episode gate must run there. An incomplete manifest
    must fail closed *before* ``_task`` is set, *before* the
    evaluation baseline is constructed, and *before* any
    observation is produced.
    """

    def test_full_cap_fake_backend_resets_casting_task(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            observations = backend.reset(casting_c1_task())
            self.assertIn("agent_1", observations)
            self.assertEqual(backend._task.workflow, "casting_c1_fixed")
            # Baseline evaluation state was created.
            self.assertIsNotNone(backend._evaluation_state)
        finally:
            backend.close()

    def test_non_casting_workflow_skips_the_casting_gate(self) -> None:
        # A non-casting workflow must not be over-constrained by the
        # casting manifest. Even with a manifest that would fail the
        # casting-c1 gate, ``sample_task`` (workflow ``route_a_a0``)
        # must reset cleanly.
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_water_bucket=False,
            exposes_target_block_truth=False,
        )
        backend = FakeEnvironmentBackend(capabilities=caps)
        backend.open()
        try:
            observations = backend.reset(sample_task())
            self.assertIn("agent_1", observations)
            self.assertEqual(backend._task.workflow, "route_a_a0")
        finally:
            backend.close()

    def test_missing_cap_fake_backend_casting_reset_rejected_before_state_mutation(
        self,
    ) -> None:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            can_use_water_bucket=False,
            exposes_target_block_truth=False,
        )
        backend = FakeEnvironmentBackend(capabilities=caps)
        backend.open()
        # Spy on the private observation builder. If the gate
        # behaves, the spy must never fire because reset() raises
        # before ``_observations()`` is called.
        original_observations = backend._observations
        observations_calls: list[int] = []

        def spy_observations() -> object:
            observations_calls.append(1)
            return original_observations()

        backend._observations = spy_observations  # type: ignore[method-assign]
        try:
            with self.assertRaises(CapabilityMismatchError) as ctx:
                backend.reset(casting_c1_task())
            # The fake runtime produced no observation.
            self.assertEqual(observations_calls, [])
            # No task was set, no step was incremented.
            self.assertIsNone(backend._task)
            self.assertEqual(backend._step_id, 0)
            # No evaluation baseline was constructed.
            self.assertIsNone(backend._evaluation_state)
            # The exception carries canonical missing + task_id.
            self.assertEqual(
                ctx.exception.missing,
                ("use_water_bucket", "target_block_truth"),
            )
            self.assertEqual(
                ctx.exception.task_id, "casting_c1_fixed_seed_0"
            )
            # The exception message is helpful for callers.
            self.assertIn("use_water_bucket", str(ctx.exception))
            self.assertIn("target_block_truth", str(ctx.exception))
        finally:
            backend.close()

    def test_missing_cap_step_path_also_stays_untouched(self) -> None:
        # Even if a caller bypasses the explicit gate call, the
        # ``step`` method must not be entered with a stale / missing
        # state. We assert this by attempting ``step`` after a
        # failed ``reset`` and confirming it raises ``RuntimeError``
        # because the task is still unset (the standard
        # ``_require_task`` contract).
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            exposes_fluid_truth=False,
        )
        backend = FakeEnvironmentBackend(capabilities=caps)
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError):
                backend.reset(casting_c1_task())
            with self.assertRaisesRegex(RuntimeError, "not been reset"):
                backend.step({"agent_1": MacroAction.wait()})
        finally:
            backend.close()


class FakeBackendObservationIsolationTests(unittest.TestCase):
    """Agent-visible observations must not leak evaluator-only truth."""

    def test_fake_observation_excludes_evaluator_only_truth(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            observations = backend.reset(casting_c1_task())
            for observation in observations.values():
                frame = observation.frame
                if isinstance(frame, dict):
                    for forbidden in (
                        "target_block",
                        "fluid_truth",
                        "portal_grid",
                        "target_cell",
                        "scenario_parameters",
                    ):
                        self.assertNotIn(forbidden, frame)
                self.assertIsNone(getattr(observation, "target_block_truth", None))
                self.assertIsNone(getattr(observation, "fluid_truth", None))
                self.assertIsNone(getattr(observation, "target_cell", None))
        finally:
            backend.close()

    def test_fake_observation_frame_remains_public_dict_after_step(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            step = backend.step({"agent_1": MacroAction.wait()})
            for observation in step.observations.values():
                frame = observation.frame
                if isinstance(frame, dict):
                    for forbidden in (
                        "target_block",
                        "fluid_truth",
                        "portal_grid",
                        "target_cell",
                        "scenario_parameters",
                    ):
                        self.assertNotIn(forbidden, frame)
        finally:
            backend.close()


class MineRLBackendManifestTests(unittest.TestCase):
    """The current MineRL backend must honestly report what it can
    serve. R2 is precisely about *exposing* the gaps so that R3 / R4
    know which work is still outstanding.
    """

    def test_minerl_backend_does_not_claim_unimplemented_capabilities(
        self,
    ) -> None:
        caps = MineRLEnvironmentBackend.casting_c1_capabilities()
        self.assertIsInstance(caps, BackendCapabilities)
        for field in (
            "can_select_water_bucket",
            "can_select_lava_bucket",
            "can_use_water_bucket",
            "can_use_lava_bucket",
            "exposes_selected_item",
            "exposes_target_block_truth",
            "exposes_fluid_truth",
        ):
            self.assertFalse(
                getattr(caps, field),
                f"{field} must stay False until the capability is wired in",
            )
        # ``exposes_public_inventory`` is currently served by
        # ``_public_observations``; pin it as True so the contract
        # is explicit.
        self.assertTrue(caps.exposes_public_inventory)
        # And the gate must therefore fail closed for the real
        # MineRL backend today.
        with self.assertRaises(CapabilityMismatchError) as ctx:
            assert_casting_c1_capabilities(caps, task_id="casting_c1_fixed_seed_0")
        for expected in (
            "select_water_bucket",
            "select_lava_bucket",
            "use_water_bucket",
            "use_lava_bucket",
            "selected_item",
            "target_block_truth",
            "fluid_truth",
        ):
            self.assertIn(expected, ctx.exception.missing)

    def test_minerl_backend_instance_capabilities_match_static(self) -> None:
        backend = MineRLEnvironmentBackend(reset_warmup_steps=0)
        # Capabilities are static, so the instance method must agree
        # with the static helper. The backend has not been opened, so
        # no MineRL session is required to answer this question.
        self.assertEqual(
            backend.capabilities(),
            MineRLEnvironmentBackend.casting_c1_capabilities(),
        )


class MineRLResetGateTests(unittest.TestCase):
    """The real MineRL backend must fail closed for casting tasks
    *before* the env factory is called and *before* any state is
    mutated. The MineRL runtime is never touched for a casting-c1
    task it cannot serve today.
    """

    def test_minerl_casting_reset_rejected_before_env_creation(self) -> None:
        factory_calls: list[str] = []

        def tracking_factory(task: object) -> object:
            factory_calls.append("env_factory_called")  # type: ignore[arg-type]
            # If the gate is wired correctly, this assertion never
            # fires because the gate rejects the reset before the
            # factory is invoked.
            raise AssertionError(
                "env_factory must not be called when the gate fails"
            )

        backend = MineRLEnvironmentBackend(
            env_factory=tracking_factory,  # type: ignore[arg-type]
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError) as ctx:
                backend.reset(casting_c1_task())
            # Env factory was never called.
            self.assertEqual(factory_calls, [])
            # State was not mutated.
            self.assertIsNone(backend._task)
            self.assertEqual(backend._step_id, 0)
            # No underlying MineRL environment was created.
            self.assertIsNone(backend._env)
            # Canonical missing + task_id.
            self.assertEqual(
                ctx.exception.missing,
                (
                    "select_water_bucket",
                    "select_lava_bucket",
                    "use_water_bucket",
                    "use_lava_bucket",
                    "selected_item",
                    "target_block_truth",
                    "fluid_truth",
                ),
            )
            self.assertEqual(
                ctx.exception.task_id, "casting_c1_fixed_seed_0"
            )
        finally:
            backend.close()

    def test_minerl_non_casting_workflow_does_not_invoke_casting_gate(self) -> None:
        # A non-casting workflow must not be blocked by the casting
        # gate. We use the legacy ``route_a_a0`` workflow via
        # ``sample_task()`` and a dummy env factory that records the
        # call; the env factory *should* be called because the
        # casting gate is a no-op for ``route_a_a0``. We stop short
        # of actually resetting a real MineRL environment (which
        # would violate the R2 no-real-MineRL rule) by patching
        # ``_validate_raw_observation`` to a no-op — the test only
        # needs to prove the gate is bypassed and the factory is
        # reached.
        factory_calls: list[str] = []

        class _DummyEnv:
            def seed(self, value: int) -> None:
                return None

            def reset(self) -> dict[str, object]:
                return {
                    "pov": _zero_pov(),
                    "inventory": {},
                }

            def step(self, action: object):
                return (
                    {
                        "pov": _zero_pov(),
                        "inventory": {},
                    },
                    0.0,
                    False,
                    {},
                )

            def close(self) -> None:
                return None

            class action_space:
                @staticmethod
                def no_op() -> dict[str, int]:
                    return {}

                @staticmethod
                def contains(value: object) -> bool:
                    return True

        def factory(task: object) -> object:
            factory_calls.append("ok")
            return _DummyEnv()

        backend = MineRLEnvironmentBackend(
            env_factory=factory,  # type: ignore[arg-type]
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            observations = backend.reset(sample_task())
            self.assertEqual(factory_calls, ["ok"])
            self.assertIn("agent_1", observations)
        finally:
            backend.close()


def _zero_pov() -> object:
    import numpy as np

    return np.zeros((360, 640, 3), dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
