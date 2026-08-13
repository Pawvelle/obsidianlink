from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from obsidianlink.env.integration import (
    AUTHORIZED_LIVE_E1_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E1,
    E1AuthorizationError,
    E1MineRLRunRecord,
    MineRLE1RGBAdapter,
    preflight_authorized_e1,
    run_authorized_e1_minerl,
)
from obsidianlink.env.integration.e0_cleanup import inspect_minerl_cleanup
from obsidianlink.env.integration.e1_adapter import public_rgb_observation
from obsidianlink.env.integration.e1_run import reset_authorized_e1_process_guards_for_tests
from obsidianlink.env.validation import (
    E1_RGB_CASE,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.result import UNIT_VERIFIED


ROOT = Path(__file__).resolve().parents[1]
EPISODE_ID = "e1-bridge-episode"


def _top_level_imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts if source.name == "__init__.py" else relative.parts[:-1]
    package = ".".join(package_parts)
    for node in tree.body:
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


def _valid_rgb() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


class _RecordingMineRLBackend:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[object] = []
        self._opened = False
        self._env: object | None = None
        self._task = None
        self._owner_thread: int | None = None
        self.open_error: Exception | None = None
        self.reset_error: Exception | None = None
        self.close_error: Exception | None = None
        self.reset_result: object | None = None

    def open(self) -> None:
        self.calls.append("open")
        if self.open_error is not None:
            raise self.open_error
        self._opened = True
        self._owner_thread = 1

    def reset(self, task: object) -> object:
        self.calls.append(("reset", type(task).__name__, getattr(task, "task_id", None)))
        if self.reset_error is not None:
            raise self.reset_error
        self._task = task
        self._env = object()
        if self.reset_result is not None:
            return self.reset_result
        return {
            "agent_1": SimpleNamespace(
                episode_id=getattr(task, "task_id", EPISODE_ID),
                step_id=0,
                agent_id="agent_1",
                frame=_valid_rgb(),
                visible_inventory={"dirt": 1},
                selected_item="dirt",
                workflow_stage="route_a_a0",
                portal_grid="evaluator-only",
            )
        }

    def close(self) -> None:
        self.calls.append("close")
        if self.close_error is not None:
            raise self.close_error
        self._env = None
        self._task = None
        self._owner_thread = None
        self._opened = False


def _backend_cls(
    *,
    reset_result: object | None = None,
    reset_error: Exception | None = None,
) -> type[_RecordingMineRLBackend]:
    class _Configured(_RecordingMineRLBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.reset_result = reset_result
            self.reset_error = reset_error

    _Configured.__name__ = "RecordingMineRLBackend"
    return _Configured


class PublicRGBAdapterTests(unittest.TestCase):
    def test_adapter_emits_rgb_without_inventory_or_truth(self) -> None:
        adapter = MineRLE1RGBAdapter(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(),
        )
        state = adapter.reset()
        adapter.close()
        payload = state["agent_1"]
        self.assertEqual(set(payload), {"agent_id", "episode_id", "rgb", "step_id"})
        self.assertEqual(payload["episode_id"], EPISODE_ID)
        self.assertEqual(payload["rgb"].shape, (360, 640, 3))
        self.assertEqual(payload["rgb"].dtype, np.uint8)
        self.assertNotIn("visible_inventory", payload)
        self.assertNotIn("selected_item", payload)
        self.assertNotIn("workflow_stage", payload)
        self.assertNotIn("portal_grid", payload)
        self.assertNotIn("frame", payload)
        self.assertNotIn("inventory", payload)

    def test_public_projection_reads_pov_and_drops_leaks(self) -> None:
        raw = {
            "agent_1": {
                "episode_id": EPISODE_ID,
                "agent_id": "agent_1",
                "step_id": 0,
                "pov": _valid_rgb(),
                "inventory": {"dirt": 1},
                "selected_item": "dirt",
                "portal_grid": [0],
            }
        }
        projected = public_rgb_observation(raw, episode_id=EPISODE_ID)
        self.assertEqual(set(projected["agent_1"]), {"agent_id", "episode_id", "rgb", "step_id"})
        self.assertIs(projected["agent_1"]["rgb"], raw["agent_1"]["pov"])

    def test_runner_accepts_adapter_public_rgb(self) -> None:
        factory = MineRLE1RGBAdapter.lifecycle_factory(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(),
        )
        result = EnvironmentValidationRunner().run(
            E1_RGB_CASE,
            factory,
            episode_id=EPISODE_ID,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "rgb_ok")
        self.assertFalse(result.integration_verified)
        self.assertFalse(result.real_execution_performed)


class ImportSafetyTests(unittest.TestCase):
    def test_e1_modules_do_not_import_minerl_at_module_level(self) -> None:
        for relative in (
            "obsidianlink/env/integration/__init__.py",
            "obsidianlink/env/integration/e1_adapter.py",
            "obsidianlink/env/integration/e1_run.py",
            "obsidianlink/env/validation/rgb.py",
            "obsidianlink/env/validation/cases/rgb.py",
        ):
            source = ROOT / relative
            imported = _top_level_imported_modules(source)
            for module in imported:
                self.assertFalse(
                    _matches_prefix(module, "obsidianlink.env.minerl_backend"),
                    f"{relative} top-level imports {module}",
                )
                self.assertFalse(
                    _matches_prefix(module, "minerl"),
                    f"{relative} top-level imports {module}",
                )

    def test_importing_e1_run_does_not_bind_minerl_backend(self) -> None:
        source = inspect.getsource(run_authorized_e1_minerl)
        self.assertIn("_production_backend_cls", source)
        production_source = inspect.getsource(
            __import__(
                "obsidianlink.env.integration.e1_run",
                fromlist=["_production_backend_cls"],
            )._production_backend_cls
        )
        self.assertIn("from obsidianlink.env.minerl_backend import", production_source)


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_e1_process_guards_for_tests()

    def tearDown(self) -> None:
        reset_authorized_e1_process_guards_for_tests()

    def test_entrypoint_refuses_without_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with patch(
                "obsidianlink.env.integration.e1_run._production_backend_cls"
            ) as production:
                with self.assertRaisesRegex(E1AuthorizationError, "execution_mode"):
                    run_authorized_e1_minerl(
                        execution_mode="offline",
                        authorized_live_run=AUTHORIZED_LIVE_E1_RUN_VALUE,
                        output_dir=output_dir,
                    )
                production.assert_not_called()

    def test_preflight_accepts_flags_without_starting_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e1_rgb_observation"
            runs_root.mkdir()
            output_dir = runs_root / "run-1"
            with patch(
                "obsidianlink.env.integration.e1_run.FORMAL_E1_RUNS_ROOT",
                runs_root.resolve(),
            ):
                with patch(
                    "obsidianlink.env.integration.e1_run._production_backend_cls"
                ) as production:
                    payload = preflight_authorized_e1(
                        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E1,
                        authorized_live_run=AUTHORIZED_LIVE_E1_RUN_VALUE,
                        output_dir=output_dir,
                    )
                    production.assert_not_called()
            self.assertFalse(payload["integration_verified"])
            self.assertFalse(payload["real_execution_performed"])
            self.assertEqual(payload["verification_level"], UNIT_VERIFIED)

    def test_authorized_stub_run_stays_unit_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e1_rgb_observation"
            runs_root.mkdir()
            output_dir = runs_root / "run-stub"
            with patch(
                "obsidianlink.env.integration.e1_run.FORMAL_E1_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.e1_run._production_backend_cls",
                return_value=_backend_cls(),
            ):
                record = run_authorized_e1_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E1,
                    authorized_live_run=AUTHORIZED_LIVE_E1_RUN_VALUE,
                    output_dir=output_dir,
                    episode_id=EPISODE_ID,
                )
                self.assertIsInstance(record, E1MineRLRunRecord)
                self.assertTrue(record.success)
                self.assertEqual(record.outcome, "rgb_ok")
                self.assertTrue(record.authorization_accepted)
                self.assertFalse(record.real_execution_performed)
                self.assertFalse(record.integration_verified)
                self.assertTrue(record.rgb_present)
                self.assertEqual(record.rgb_height, 360)
                self.assertEqual(record.rgb_width, 640)
                self.assertEqual(record.rgb_channels, 3)
                self.assertEqual(record.rgb_dtype, "uint8")
                payload = json.loads(
                    (output_dir / "e1_rgb.json").read_text(encoding="utf-8")
                )
                self.assertFalse(payload["integration_verified"])
                self.assertFalse(payload["real_execution_performed"])
                self.assertEqual(payload["check_id"], "E1")
                self.assertNotIn("pixels", json.dumps(payload))
                authorization = json.loads(
                    (output_dir / "authorization.json").read_text(encoding="utf-8")
                )
                self.assertFalse(authorization["model_api_authorized"])
                self.assertFalse(authorization["gradle_authorized"])
                manifest = p1_validation_manifest()
                self.assertTrue(all(item["status"] == "not_run" for item in manifest))

    def test_record_rejects_integration_claim(self) -> None:
        cleanup = inspect_minerl_cleanup(None, close_returned=True)
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            E1MineRLRunRecord(
                check_id="E1",
                name="rgb_observation",
                episode_id=EPISODE_ID,
                step_id=0,
                backend_identity="RecordingMineRLBackend",
                opened=True,
                created=True,
                reset_completed=True,
                initial_state_present=True,
                closed=True,
                success=True,
                outcome="rgb_ok",
                cleanup=cleanup,
                rgb_present=True,
                rgb_height=360,
                rgb_width=640,
                rgb_channels=3,
                rgb_dtype="uint8",
                integration_verified=True,
            )


if __name__ == "__main__":
    unittest.main()
