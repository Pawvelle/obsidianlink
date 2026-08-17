from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, Mapping

from obsidianlink.benchmark.evaluator import EvaluatorVerdict
from obsidianlink.benchmark.evidence import EvidenceIdentity
from obsidianlink.env.validation import (
    E0_LIFECYCLE_CASE,
    EnvironmentValidationRecorder,
    EnvironmentValidationResult,
    EnvironmentValidationRunner,
    P1_VALIDATION_CASES,
    p1_validation_manifest,
)
from obsidianlink.env.validation.cases.lifecycle import initial_state_exists
from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
)
from obsidianlink.env.validation.result import UNIT_VERIFIED, VALIDATION_OUTCOMES


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PACKAGE = ROOT / "obsidianlink/env/validation"
BANNED_PREFIXES = (
    "obsidianlink.drivers",
    "obsidianlink.runners",
    "obsidianlink.evaluation",
    "obsidianlink.agents",
    "obsidianlink.benchmark.evaluator",
    "obsidianlink.env.fake",
    "obsidianlink.env.minerl_backend",
    "obsidianlink.env.integration",
    "minerl",
)
EPISODE_ID = "e0-offline-episode"


@dataclass
class _State:
    episode_id: str
    step_id: int
    frame: Any = "ignored-rgb"
    visible_inventory: Mapping[str, int] | None = None
    selected_item: str | None = None


_UNSET = object()


class _LifecycleStub:
    def __init__(
        self,
        *,
        reset_result: object = _UNSET,
        reset_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.reset_calls = 0
        self.close_calls = 0
        self._reset_result = reset_result
        self._reset_error = reset_error
        self._close_error = close_error

    def reset(self) -> object:
        self.reset_calls += 1
        if self._reset_error is not None:
            raise self._reset_error
        if self._reset_result is not _UNSET:
            return self._reset_result
        return {
            "agent_1": _State(episode_id=EPISODE_ID, step_id=0),
        }

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class _ExplodingMapping(dict):
    def items(self):  # type: ignore[override]
        raise RuntimeError("unexpected state inspection failure")

    def __iter__(self):
        raise RuntimeError("unexpected state inspection failure")


def _imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts if source.name == "__init__.py" else relative.parts[:-1]
    package = ".".join(package_parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(relative_name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _run(
    stub: _LifecycleStub | None = None,
    *,
    factory_error: Exception | None = None,
    case: EnvironmentValidationCase = E0_LIFECYCLE_CASE,
) -> tuple[EnvironmentValidationResult, _LifecycleStub | None]:
    holder: dict[str, _LifecycleStub | None] = {"backend": stub}

    def factory() -> object:
        if factory_error is not None:
            raise factory_error
        backend = holder["backend"]
        if backend is None:
            backend = _LifecycleStub()
            holder["backend"] = backend
        return backend

    result = EnvironmentValidationRunner().run(
        case,
        factory,
        episode_id=EPISODE_ID,
    )
    return result, holder["backend"]


class E0LifecycleValidationTests(unittest.TestCase):
    def test_successful_lifecycle_is_unit_verified_only(self) -> None:
        result, stub = _run()
        assert stub is not None
        self.assertEqual(stub.reset_calls, 1)
        self.assertEqual(stub.close_calls, 1)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "lifecycle_ok")
        self.assertTrue(result.created)
        self.assertTrue(result.reset_completed)
        self.assertTrue(result.initial_state_present)
        self.assertTrue(result.closed)
        self.assertEqual(result.check_id, EnvironmentValidationId.E0)
        self.assertEqual(result.name, "reset_close")
        self.assertEqual(result.episode_id, EPISODE_ID)
        self.assertEqual(result.step_id, 0)
        self.assertEqual(result.verification_level, UNIT_VERIFIED)
        self.assertFalse(result.real_execution_performed)
        self.assertFalse(result.integration_verified)
        self.assertTrue(result.calibration_only)
        self.assertIsNone(result.error)
        self.assertIsNone(result.close_error)

    def test_reset_failure_fails_closed_and_still_closes(self) -> None:
        stub = _LifecycleStub(reset_error=RuntimeError("reset exploded"))
        result, _ = _run(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertTrue(result.created)
        self.assertFalse(result.reset_completed)
        self.assertFalse(result.initial_state_present)
        self.assertTrue(result.closed)
        self.assertEqual(stub.close_calls, 1)
        self.assertIn("reset exploded", result.error or "")

    def test_missing_initial_state_fails_closed_and_still_closes(self) -> None:
        for reset_result in (None, {}, {"agent_1": None}):
            stub = _LifecycleStub(reset_result=reset_result)
            result, _ = _run(stub)
            with self.subTest(reset_result=reset_result):
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "initial_state_missing")
                self.assertTrue(result.reset_completed)
                self.assertFalse(result.initial_state_present)
                self.assertTrue(result.closed)
                self.assertEqual(stub.close_calls, 1)

    def test_invalid_initial_state_identity_fails_closed(self) -> None:
        stub = _LifecycleStub(
            reset_result={"agent_1": _State(episode_id="other", step_id=0)}
        )
        result, _ = _run(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "initial_state_missing")
        self.assertTrue(result.closed)

    def test_close_failure_does_not_claim_clean_lifecycle(self) -> None:
        stub = _LifecycleStub(close_error=OSError("close failed"))
        result, _ = _run(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "close_failed")
        self.assertTrue(result.reset_completed)
        self.assertTrue(result.initial_state_present)
        self.assertFalse(result.closed)
        self.assertIsNotNone(result.close_error)
        self.assertIn("close failed", result.close_error or "")

    def test_reset_and_close_failure_records_cleanup_error(self) -> None:
        stub = _LifecycleStub(
            reset_error=RuntimeError("reset exploded"),
            close_error=OSError("close failed"),
        )
        result, _ = _run(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertFalse(result.closed)
        self.assertIn("reset exploded", result.error or "")
        self.assertIn("close failed", result.close_error or "")
        self.assertEqual(stub.close_calls, 1)

    def test_unexpected_runtime_failure_is_deterministic_and_closes(self) -> None:
        stub = _LifecycleStub(reset_result=_ExplodingMapping({"agent_1": object()}))
        result, _ = _run(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")
        self.assertTrue(result.created)
        self.assertTrue(result.reset_completed)
        self.assertTrue(result.closed)
        self.assertEqual(stub.close_calls, 1)
        self.assertIn("unexpected state inspection failure", result.error or "")
        snapshot = result.as_dict()
        self.assertEqual(snapshot, json.loads(json.dumps(snapshot)))

    def test_create_failure_does_not_claim_success_or_close(self) -> None:
        result, stub = _run(factory_error=RuntimeError("cannot create backend"))
        self.assertIsNone(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "create_failed")
        self.assertFalse(result.created)
        self.assertFalse(result.closed)
        self.assertIn("cannot create backend", result.error or "")

    def test_e0_does_not_inspect_rgb_inventory_or_selected_item(self) -> None:
        payload = {
            "agent_1": {
                "episode_id": EPISODE_ID,
                "step_id": 0,
                "frame": {"pixels": "not-inspected"},
                "visible_inventory": {"lava_bucket": 1},
                "selected_item": "lava_bucket",
            }
        }
        self.assertTrue(initial_state_exists(payload, episode_id=EPISODE_ID))
        stub = _LifecycleStub(reset_result=payload)
        result, _ = _run(stub)
        self.assertTrue(result.success)
        self.assertNotIn("lava_bucket", json.dumps(result.as_dict()))
        self.assertNotIn("pixels", json.dumps(result.as_dict()))

    def test_result_is_independent_from_evaluator_verdict(self) -> None:
        result, _ = _run()
        self.assertFalse(issubclass(EnvironmentValidationResult, EvaluatorVerdict))
        self.assertNotIsInstance(result, EvaluatorVerdict)
        self.assertFalse(hasattr(result, "evidence_complete"))
        self.assertFalse(hasattr(result, "identity"))
        verdict = EvaluatorVerdict(
            identity=EvidenceIdentity(EPISODE_ID, 0, "agent_1"),
            success=True,
            outcome="unused",
            evidence_complete=True,
        )
        self.assertNotEqual(type(result), type(verdict))
        with self.assertRaises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_result_rejects_integration_claims(self) -> None:
        kwargs = dict(
            check_id=EnvironmentValidationId.E0,
            name="reset_close",
            episode_id=EPISODE_ID,
            step_id=0,
            success=True,
            outcome="lifecycle_ok",
            created=True,
            reset_completed=True,
            initial_state_present=True,
            closed=True,
        )
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            EnvironmentValidationResult(**kwargs, integration_verified=True)
        with self.assertRaisesRegex(ValueError, "real execution"):
            EnvironmentValidationResult(**kwargs, real_execution_performed=True)
        with self.assertRaisesRegex(ValueError, "unit_verified"):
            EnvironmentValidationResult(**kwargs, verification_level="integration_verified")
        self.assertIn("lifecycle_ok", VALIDATION_OUTCOMES)

    def test_recorder_writes_deterministic_offline_evidence(self) -> None:
        result, _ = _run()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e0" / "validation_result.json"
            written = EnvironmentValidationRecorder().record(result, path)
            self.assertEqual(written, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, result.as_dict())
            self.assertEqual(payload["verification_level"], "unit_verified")
            self.assertFalse(payload["integration_verified"])
            self.assertFalse(payload["real_execution_performed"])
            self.assertTrue(payload["calibration_only"])

    def test_offline_success_does_not_change_p1_manifest(self) -> None:
        result, _ = _run()
        self.assertTrue(result.success)
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[0]["check_id"], "E0")
        self.assertEqual(manifest[0]["name"], "reset_close")
        self.assertTrue(all(item["status"] == "not_run" for item in manifest))
        self.assertEqual(
            [item["check_id"] for item in manifest],
            [f"E{index}" for index in range(13)],
        )

    def test_all_checklist_cases_have_success_outcomes(self) -> None:
        from obsidianlink.env.validation.runner import _success_outcome

        for case in P1_VALIDATION_CASES:
            self.assertIsNotNone(_success_outcome(case), case.check_id.value)

    def test_e12_without_truth_surfaces_fails_closed(self) -> None:
        e12 = P1_VALIDATION_CASES[12]
        self.assertEqual(e12.check_id, EnvironmentValidationId.E12)
        result, stub = _run(case=e12)
        self.assertIsNotNone(stub)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")
        self.assertTrue(result.created)
        self.assertIn("E12 backend", result.error or "")

    def test_e0_case_matches_manifest_reset_close(self) -> None:
        self.assertIs(E0_LIFECYCLE_CASE.check_id, EnvironmentValidationId.E0)
        self.assertEqual(E0_LIFECYCLE_CASE.name, "reset_close")
        self.assertFalse(E0_LIFECYCLE_CASE.requires_server_truth)
        self.assertTrue(E0_LIFECYCLE_CASE.calibration_only)


class E0ValidationIsolationTests(unittest.TestCase):
    def test_validation_package_does_not_import_forbidden_modules(self) -> None:
        sources = tuple(sorted(VALIDATION_PACKAGE.rglob("*.py")))
        self.assertTrue(sources)
        for source in sources:
            for module in _imported_modules(source):
                for prefix in BANNED_PREFIXES:
                    self.assertFalse(
                        _matches_prefix(module, prefix),
                        f"{source.relative_to(ROOT)} imports forbidden module {module}",
                    )

    def test_validation_package_does_not_import_evaluator_verdict(self) -> None:
        forbidden_names = {"EvaluatorVerdict", "FakeEnvironmentBackend"}
        for source in VALIDATION_PACKAGE.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[-1] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.update(alias.name for alias in node.names)
            self.assertTrue(
                imported.isdisjoint(forbidden_names),
                f"{source.relative_to(ROOT)} imports {sorted(imported & forbidden_names)}",
            )


if __name__ == "__main__":
    unittest.main()
