"""Offline tests for the R6 C1 live-smoke runner wiring."""

from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import TaskInstance
from obsidianlink.core.types import MacroAction
from obsidianlink.drivers.casting_c1 import CastingPlanStep, build_casting_action_plan
from obsidianlink.runners.casting_c1_live_smoke import (
    C1ReactiveStubEnv,
    C1SmokePreflightError,
    EVALUATOR_ONLY_TOKENS,
    FROZEN_TARGET_CELL,
    OfflineC1StubEnvFactory,
    REQUIRED_EVIDENCE_FILES,
    build_default_c1_plan,
    build_offline_stub_env_factory,
    load_frozen_c1_task,
    preflight_c1_live_smoke,
    run_casting_c1_live_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"
TASK_PATH = ROOT / "benchmark/instances/active/casting_c1_fixed.json"
FORMAL_RUNS_DIR = (ROOT / "runs").resolve()


def _task_payload(**updates: Any) -> dict[str, Any]:
    payload = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    payload.update(updates)
    return payload


def _factory_tracker() -> OfflineC1StubEnvFactory:
    return build_offline_stub_env_factory()


class C1LiveSmokeRunnerPositiveTests(unittest.TestCase):
    def test_positive_path_produces_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "c1_smoke_positive"
            factory = _factory_tracker()
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=factory,
            )
            self.assertEqual(len(factory.created_envs), 1)
            self.assertTrue(result.driver_completed)
            self.assertTrue(result.evaluator_success)
            self.assertTrue(result.evidence_complete)
            self.assertEqual(result.close_status, "closed")
            self.assertTrue(result.overall_success)
            self.assertEqual(result.evaluator_outcome, "success")
            self.assertIsNone(result.failure_reason)

            for filename in REQUIRED_EVIDENCE_FILES:
                path = output_dir / filename
                self.assertTrue(path.is_file(), msg=filename)
                self.assertGreater(path.stat().st_size, 0, msg=filename)

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["workflow"], "casting_c1_fixed")
            self.assertEqual(summary["target_cell"], list(FROZEN_TARGET_CELL))
            self.assertTrue(summary["driver_completed"])
            self.assertTrue(summary["evaluator_success"])
            self.assertTrue(summary["evidence_complete"])
            self.assertEqual(summary["close_status"], "closed")

            code_version = json.loads((output_dir / "code_version.json").read_text())
            self.assertIn("commit", code_version)
            self.assertIn("dirty", code_version)
            self.assertNotIn("dirty_paths", code_version)

            with Image.open(output_dir / "initial.png") as image:
                self.assertEqual(image.size, (640, 360))
            with Image.open(output_dir / "final.png") as image:
                self.assertEqual(image.size, (640, 360))

            events = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue(events)
            for event in events:
                self.assertIn("episode_id", event)
                self.assertIn("step_id", event)
                payload = json.dumps(event)
                for token in EVALUATOR_ONLY_TOKENS:
                    self.assertNotIn(token, payload)

            self.assertFalse(FORMAL_RUNS_DIR in output_dir.parents)


class C1LiveSmokePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "preflight"
        self.factory = _factory_tracker()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_preflight(self, **kwargs: Any) -> None:
        defaults = {
            "output_dir": self.output_dir,
            "env_factory": self.factory,
        }
        defaults.update(kwargs)
        preflight_c1_live_smoke(**defaults)

    def _assert_preflight_fails(self, **kwargs: Any) -> None:
        with self.assertRaises(C1SmokePreflightError):
            self._run_preflight(**kwargs)
        self.assertEqual(self.factory.created_envs, ())

    def test_rejects_output_under_formal_runs(self) -> None:
        with self.assertRaises(C1SmokePreflightError):
            preflight_c1_live_smoke(
                output_dir=FORMAL_RUNS_DIR / "c1_smoke",
                env_factory=self.factory,
            )
        self.assertEqual(self.factory.created_envs, ())

    def test_rejects_existing_output_dir(self) -> None:
        self.output_dir.mkdir()
        (self.output_dir / "summary.json").write_text(
            "user-owned sentinel", encoding="utf-8"
        )
        self._assert_preflight_fails()
        self.assertEqual(
            (self.output_dir / "summary.json").read_text(encoding="utf-8"),
            "user-owned sentinel",
        )

    def test_rejects_arbitrary_callable_factory(self) -> None:
        self._assert_preflight_fails(env_factory=lambda task: object())

    def test_rejects_non_absolute_output_dir(self) -> None:
        self._assert_preflight_fails(output_dir="relative/output")

    def test_rejects_live_request(self) -> None:
        self._assert_preflight_fails(request_live=True)

    def test_rejects_live_run_allowed_override(self) -> None:
        self._assert_preflight_fails(allow_live_run_override=True)

    def test_rejects_missing_env_factory(self) -> None:
        with self.assertRaises(C1SmokePreflightError):
            preflight_c1_live_smoke(
                output_dir=self.output_dir,
                env_factory=None,
            )

    def test_rejects_unknown_execution_mode(self) -> None:
        with self.assertRaises(C1SmokePreflightError):
            run_casting_c1_live_smoke(
                output_dir=self.output_dir,
                execution_mode="live_minerl",
                env_factory=self.factory,
            )
        self.assertEqual(self.factory.created_envs, ())

    def test_rejects_wrong_workflow(self) -> None:
        bad_task = TaskInstance.from_dict(_task_payload(workflow="casting_c3_fixed"))
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_agent(self) -> None:
        payload = _task_payload()
        payload["agent_ids"] = ["agent_2"]
        payload["spawn_positions"] = {"agent_2": payload["spawn_positions"]["agent_1"]}
        payload["initial_inventories"] = {
            "agent_2": payload["initial_inventories"]["agent_1"]
        }
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_family(self) -> None:
        payload = _task_payload()
        payload["scenario_parameters"] = dict(payload["scenario_parameters"])
        payload["scenario_parameters"]["task_family"] = "ruined"
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_mode(self) -> None:
        payload = _task_payload()
        payload["scenario_parameters"] = dict(payload["scenario_parameters"])
        payload["scenario_parameters"]["agent_mode"] = "multi"
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_layout(self) -> None:
        payload = _task_payload()
        payload["scenario_parameters"] = dict(payload["scenario_parameters"])
        payload["scenario_parameters"]["layout_type"] = "randomized"
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_target_cell(self) -> None:
        payload = _task_payload()
        payload["scenario_parameters"] = dict(payload["scenario_parameters"])
        payload["scenario_parameters"]["target_cell"] = [0, 4, 1]
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_wrong_taxonomy(self) -> None:
        payload = _task_payload()
        payload["scenario_parameters"] = dict(payload["scenario_parameters"])
        payload["scenario_parameters"]["task_level"] = "C2"
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_missing_inventory(self) -> None:
        payload = _task_payload()
        payload["initial_inventories"] = {
            "agent_1": {
                "water_bucket": 1,
                "lava_bucket": 1,
                "cobblestone": 0,
            }
        }
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_extra_obsidian_inventory(self) -> None:
        payload = _task_payload()
        payload["initial_inventories"] = {
            "agent_1": dict(payload["initial_inventories"]["agent_1"])
        }
        payload["initial_inventories"]["agent_1"]["obsidian"] = 64
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_modified_limits(self) -> None:
        payload = _task_payload()
        payload["limits"] = dict(payload["limits"])
        payload["limits"]["max_environment_steps"] = 999
        payload["limits"]["max_model_calls"] = 99
        bad_task = TaskInstance.from_dict(payload)
        self._assert_preflight_fails(task=bad_task)

    def test_rejects_truncated_plan(self) -> None:
        truncated = build_casting_action_plan()[:-1]
        self._assert_preflight_fails(plan=truncated)

    def test_rejects_reordered_plan(self) -> None:
        plan = list(build_casting_action_plan())
        plan[0], plan[1] = plan[1], plan[0]
        self._assert_preflight_fails(plan=tuple(plan))

    def test_rejects_custom_plan_step(self) -> None:
        plan = list(build_casting_action_plan())
        plan[3] = CastingPlanStep(
            label="support.block_1.custom",
            phase=plan[3].phase,
            action=plan[3].action,
            relevant_action=plan[3].relevant_action,
        )
        self._assert_preflight_fails(plan=tuple(plan))


class C1LiveSmokeEvaluatorFailureTests(unittest.TestCase):
    def test_evaluator_failure_when_stub_never_produces_obsidian(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "c1_smoke_failure"
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=build_offline_stub_env_factory(
                    produce_obsidian=False
                ),
            )
            self.assertTrue(result.driver_completed)
            self.assertFalse(result.evaluator_success)
            self.assertFalse(result.overall_success)
            self.assertIsNotNone(result.failure_reason)
            self.assertNotEqual(result.evaluator_outcome, "success")
            self.assertTrue(result.evidence_complete)
            self.assertEqual(result.close_status, "closed")

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertFalse(summary["evaluator_success"])
            self.assertIsNotNone(summary["failure_reason"])


class C1LiveSmokeEvidenceTests(unittest.TestCase):
    def test_public_events_and_summary_exclude_evaluator_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "c1_smoke_evidence"
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=build_offline_stub_env_factory(),
            )
            self.assertTrue(result.evidence_complete)
            for filename in ("events.jsonl", "summary.json"):
                payload = (output_dir / filename).read_text(encoding="utf-8")
                for token in EVALUATOR_ONLY_TOKENS:
                    self.assertNotIn(token, payload, msg=filename)

            evaluator_events = [
                json.loads(line)
                for line in (output_dir / "evaluator_events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue(evaluator_events)
            for event in evaluator_events:
                self.assertIn("episode_id", event)
                self.assertIn("step_id", event)
                self.assertIn("outcome", event)


class C1LiveSmokeCloseTests(unittest.TestCase):
    def test_close_is_always_attempted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "c1_smoke_close"
            factory = build_offline_stub_env_factory()
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=factory,
            )
            self.assertEqual(len(factory.created_envs), 1)
            self.assertTrue(factory.created_envs[0].closed)
            self.assertEqual(result.close_status, "closed")


class C1LiveSmokeCatalogInvariantTests(unittest.TestCase):
    def test_catalog_invariants_remain_frozen(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        self.assertEqual(catalog.active_compatibility_id, "casting_c3_fixed")
        c5 = next(
            entry
            for entry in catalog.entries
            if entry.compatibility_id == "casting_s_c5_fixed"
        )
        self.assertEqual(c5.implementation_status, "contract_only")
        self.assertFalse(c5.live_run_allowed)


class C1LiveSmokeHelperTests(unittest.TestCase):
    def test_public_runner_has_no_backend_injection_parameter(self) -> None:
        parameters = inspect.signature(run_casting_c1_live_smoke).parameters
        self.assertNotIn("backend", parameters)

    def test_load_frozen_task_and_default_plan(self) -> None:
        task = load_frozen_c1_task()
        self.assertEqual(task.workflow, "casting_c1_fixed")
        self.assertEqual(task.agent_ids, ("agent_1",))
        plan = build_default_c1_plan()
        self.assertEqual(plan, build_casting_action_plan())

    def test_reactive_stub_produces_obsidian_sequence(self) -> None:
        task = load_frozen_c1_task()
        env = C1ReactiveStubEnv(task)
        env.reset()
        env.step({"hotbar.2": 1, "use": 1})
        env.step({"hotbar.1": 1, "use": 1})
        raw, _, _, _ = env.step({"forward": 0, "back": 0})
        grid = np.asarray(raw["portal_grid"], dtype=np.int32)
        spawn = task.spawn_positions["agent_1"]
        grid_target = tuple(
            FROZEN_TARGET_CELL[axis] - spawn[axis] for axis in range(3)
        )
        index = (
            (grid_target[1] - (-1)) * 8 * 7
            + (grid_target[2] - 0) * 8
            + (grid_target[0] - (-3))
        )
        self.assertEqual(int(grid.reshape(-1)[index]), 5)  # obsidian id

    def test_successful_finalization_leaves_no_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output_dir = parent / "atomic-output"
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=build_offline_stub_env_factory(),
            )
            self.assertTrue(result.evidence_complete)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(
                list(parent.glob(".atomic-output.staging-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
