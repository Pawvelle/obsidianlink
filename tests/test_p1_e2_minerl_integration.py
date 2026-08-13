from __future__ import annotations

import ast
import importlib.util
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

from obsidianlink.env.integration import MineRLE2InventoryAdapter
from obsidianlink.env.integration.e2_adapter import public_inventory_observation
from obsidianlink.env.integration.e2_config import (
    E2_AGENT_ID,
    E2_CALIBRATION_INVENTORY,
    build_e2_compatibility_task,
)
from obsidianlink.env.validation import (
    E2_INVENTORY_CASE,
    EnvironmentValidationRunner,
    P1_VALIDATION_CASES,
    p1_validation_manifest,
)
from obsidianlink.env.validation.result import UNIT_VERIFIED


ROOT = Path(__file__).resolve().parents[1]
EPISODE_ID = "e2-adapter-offline-episode"


def _observation(
    visible_inventory: object,
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "agent_id": E2_AGENT_ID,
        "episode_id": EPISODE_ID,
        "step_id": 0,
        "visible_inventory": visible_inventory,
    }
    value.update(extra)
    return {E2_AGENT_ID: value}


class _RecordingBackend:
    def __init__(self, reset_result: object, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[object] = []
        self.reset_result = reset_result
        self.reset_task: object | None = None
        self._opened = False
        self._env: object | None = None
        self._owner_thread: int | None = None

    def open(self) -> None:
        self.calls.append("open")
        self._opened = True
        self._owner_thread = 1

    def reset(self, task: object) -> object:
        self.calls.append("reset")
        self.reset_task = task
        self._env = object()
        return self.reset_result

    def step(self, action: object) -> None:
        self.calls.append("step")
        raise AssertionError("E2 adapter must not execute actions")

    def close(self) -> None:
        self.calls.append("close")
        self._env = None
        self._owner_thread = None
        self._opened = False


def _backend_cls(reset_result: object) -> type[_RecordingBackend]:
    class _Configured(_RecordingBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(reset_result, **kwargs)

    _Configured.__name__ = "RecordingMineRLBackend"
    return _Configured


def _run_adapter(
    reset_result: object,
) -> tuple[object, MineRLE2InventoryAdapter]:
    holder: dict[str, MineRLE2InventoryAdapter | None] = {"adapter": None}
    factory = MineRLE2InventoryAdapter.lifecycle_factory(
        episode_id=EPISODE_ID,
        backend_cls=_backend_cls(reset_result),
    )

    def capturing_factory() -> MineRLE2InventoryAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    result = EnvironmentValidationRunner().run(
        E2_INVENTORY_CASE,
        capturing_factory,
        episode_id=EPISODE_ID,
        expected_inventory=E2_CALIBRATION_INVENTORY,
    )
    adapter = holder["adapter"]
    assert adapter is not None
    return result, adapter


def _top_level_imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package = ".".join(relative.parts[:-1])
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


class E2CalibrationConfigTests(unittest.TestCase):
    def test_calibration_inventory_is_nonempty_multi_item_and_distinct(self) -> None:
        inventory = dict(E2_CALIBRATION_INVENTORY)
        self.assertEqual(
            inventory,
            {"dirt": 7, "obsidian": 4, "flint_and_steel": 1},
        )
        self.assertGreaterEqual(len(inventory), 3)
        self.assertEqual(len(set(inventory.values())), len(inventory))
        self.assertTrue(all(type(value) is int and value > 0 for value in inventory.values()))

    def test_compatibility_task_uses_dedicated_e2_inventory(self) -> None:
        task = build_e2_compatibility_task(EPISODE_ID)
        self.assertEqual(task.task_id, EPISODE_ID)
        self.assertEqual(
            dict(task.initial_inventories[E2_AGENT_ID]),
            dict(E2_CALIBRATION_INVENTORY),
        )
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E2")
        self.assertEqual(
            task.scenario_parameters["p1_validation_name"],
            "inventory_observation",
        )
        self.assertIs(task.scenario_parameters["compatibility_only"], True)
        self.assertIs(task.scenario_parameters["not_a_benchmark_task"], True)

    def test_legacy_task_does_not_escape_integration_public_api(self) -> None:
        import obsidianlink.env.integration as integration

        self.assertNotIn("TaskInstance", integration.__all__)
        self.assertNotIn("build_e2_compatibility_task", integration.__all__)
        self.assertFalse(hasattr(integration, "TaskInstance"))


class PublicInventoryAdapterTests(unittest.TestCase):
    def test_visible_inventory_is_projected_with_identity_and_quantities(self) -> None:
        raw_inventory = {"dirt": 7, "obsidian": 4, "flint_and_steel": 1}
        raw = {
            E2_AGENT_ID: SimpleNamespace(
                agent_id=E2_AGENT_ID,
                episode_id=EPISODE_ID,
                step_id=0,
                visible_inventory=raw_inventory,
                frame="rgb-must-be-dropped",
                selected_item="obsidian",
                workflow_stage="route_a_a0",
                portal_grid="evaluator-only",
                info={"private": True},
            )
        }
        projected = public_inventory_observation(raw, episode_id=EPISODE_ID)
        payload = projected[E2_AGENT_ID]
        self.assertEqual(
            set(payload),
            {"agent_id", "episode_id", "inventory", "step_id"},
        )
        self.assertEqual(payload["inventory"], raw_inventory)
        self.assertIsNot(payload["inventory"], raw_inventory)
        self.assertEqual(payload["agent_id"], E2_AGENT_ID)
        self.assertEqual(payload["episode_id"], EPISODE_ID)
        self.assertEqual(payload["step_id"], 0)

    def test_mapping_visible_inventory_is_preferred_over_inventory_fallback(self) -> None:
        raw = _observation(
            {"dirt": 2},
            inventory=dict(E2_CALIBRATION_INVENTORY),
            rgb="drop",
            selected_item="drop",
            workflow_stage="drop",
            portal_grid="drop",
            portal_grid_origin="drop",
            portal_dimension="drop",
            portal_transition="drop",
            messages=("drop",),
            equipped_items={"drop": 1},
            position=(1, 2, 3),
            info={"drop": True},
        )
        projected = public_inventory_observation(raw, episode_id=EPISODE_ID)
        payload = projected[E2_AGENT_ID]
        self.assertEqual(payload["inventory"], {"dirt": 2})
        self.assertEqual(
            set(payload),
            {"agent_id", "episode_id", "inventory", "step_id"},
        )

    def test_mapping_inventory_fallback_is_narrow_and_preserves_invalid_value(self) -> None:
        raw = {
            E2_AGENT_ID: {
                "agent_id": E2_AGENT_ID,
                "episode_id": EPISODE_ID,
                "step_id": 0,
                "inventory": {"obsidian": "4"},
            }
        }
        projected = public_inventory_observation(raw, episode_id=EPISODE_ID)
        self.assertEqual(
            projected[E2_AGENT_ID]["inventory"],
            {"obsidian": "4"},
        )

    def test_identity_fallback_matches_existing_adapter_semantics(self) -> None:
        projected = public_inventory_observation(
            {E2_AGENT_ID: {"visible_inventory": {"dirt": 2}}},
            episode_id=EPISODE_ID,
        )
        self.assertEqual(
            projected[E2_AGENT_ID],
            {
                "agent_id": E2_AGENT_ID,
                "episode_id": EPISODE_ID,
                "inventory": {"dirt": 2},
                "step_id": 0,
            },
        )


class E2AdapterRunnerTests(unittest.TestCase):
    def test_backend_observed_inventory_is_not_replaced_by_config(self) -> None:
        result, adapter = _run_adapter(_observation({"dirt": 2}))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertEqual(result.observed_inventory, {"dirt": 2})
        self.assertEqual(
            result.expected_inventory,
            dict(E2_CALIBRATION_INVENTORY),
        )
        backend = adapter._backend
        assert isinstance(backend, _RecordingBackend)
        task = backend.reset_task
        self.assertEqual(
            dict(task.initial_inventories[E2_AGENT_ID]),  # type: ignore[union-attr]
            dict(E2_CALIBRATION_INVENTORY),
        )
        self.assertEqual(backend.calls, ["open", "reset", "close"])

    def test_exact_backend_inventory_matches(self) -> None:
        result, adapter = _run_adapter(
            _observation(dict(E2_CALIBRATION_INVENTORY))
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "inventory_ok")
        self.assertEqual(result.verification_level, UNIT_VERIFIED)
        self.assertFalse(result.real_execution_performed)
        self.assertFalse(result.integration_verified)
        backend = adapter._backend
        assert isinstance(backend, _RecordingBackend)
        self.assertNotIn("step", backend.calls)

    def test_empty_backend_inventory_is_mismatch(self) -> None:
        result, _ = _run_adapter(_observation({}))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertEqual(result.observed_inventory, {})

    def test_extra_backend_item_is_mismatch(self) -> None:
        observed = {**dict(E2_CALIBRATION_INVENTORY), "cobblestone": 3}
        result, _ = _run_adapter(_observation(observed))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_mismatch")

    def test_invalid_quantity_is_not_normalized(self) -> None:
        result, _ = _run_adapter(_observation({"obsidian": "4"}))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_quantity_invalid")

    def test_adapter_exposes_no_step_and_preserves_cleanup_semantics(self) -> None:
        adapter = MineRLE2InventoryAdapter(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(_observation({"dirt": 2})),
        )
        self.assertFalse(hasattr(adapter, "step"))
        adapter.reset()
        adapter.close()
        cleanup = adapter.cleanup_status()
        self.assertTrue(cleanup.close_returned)
        self.assertTrue(cleanup.backend_marked_closed)
        self.assertTrue(cleanup.environment_reference_cleared)
        self.assertTrue(cleanup.owner_cleared)
        self.assertFalse(cleanup.process_release_proven)


class E2AdapterIsolationTests(unittest.TestCase):
    def test_e2_imports_do_not_bind_minerl_or_backend_at_module_level(self) -> None:
        for relative in (
            "obsidianlink/env/integration/e2_adapter.py",
            "obsidianlink/env/integration/e2_config.py",
        ):
            imports = _top_level_imported_modules(ROOT / relative)
            self.assertFalse(
                any(
                    module == "minerl"
                    or module.startswith("minerl.")
                    or module == "obsidianlink.env.minerl_backend"
                    or module.startswith("obsidianlink.env.minerl_backend.")
                    for module in imports
                )
            )

    def test_factory_construction_does_not_resolve_production_backend(self) -> None:
        with patch.object(
            MineRLE2InventoryAdapter,
            "_resolve_backend_cls",
        ) as resolver:
            factory = MineRLE2InventoryAdapter.lifecycle_factory(
                episode_id=EPISODE_ID
            )
            adapter = factory()
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)

    def test_reset_signature_is_minimal(self) -> None:
        parameters = inspect.signature(MineRLE2InventoryAdapter.reset).parameters
        self.assertEqual(tuple(parameters), ("self",))

    def test_validation_dependency_direction_remains_integration_to_validation(self) -> None:
        source = ROOT / "obsidianlink/env/validation/inventory.py"
        imports = _top_level_imported_modules(source)
        self.assertFalse(
            any(
                module == "obsidianlink.env.integration"
                or module.startswith("obsidianlink.env.integration.")
                or module == "obsidianlink.env.minerl_backend"
                or module.startswith("obsidianlink.env.minerl_backend.")
                for module in imports
            )
        )

    def test_manifest_remains_not_run_and_e3_remains_unimplemented(self) -> None:
        e2 = p1_validation_manifest()[2]
        self.assertEqual(e2["status"], "not_run")
        self.assertFalse(e2["requires_server_truth"])

        called = False

        def forbidden_factory() -> object:
            nonlocal called
            called = True
            raise AssertionError("E3 backend must not be created")

        e3_result = EnvironmentValidationRunner().run(
            P1_VALIDATION_CASES[3],
            forbidden_factory,
            episode_id=EPISODE_ID,
        )
        self.assertFalse(e3_result.success)
        self.assertIn("unimplemented", e3_result.error or "")
        self.assertFalse(called)

    def test_no_e2_evidence_directory_was_created(self) -> None:
        self.assertFalse((ROOT / "runs/p1_e2_inventory_observation").exists())


if __name__ == "__main__":
    unittest.main()
