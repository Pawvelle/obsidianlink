from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from obsidianlink.env.integration import MineRLE3SelectedItemAdapter
from obsidianlink.env.integration.e3_adapter import public_selected_item_observation
from obsidianlink.env.integration.e3_config import (
    E3_AGENT_ID,
    E3_CALIBRATION_INVENTORY,
    E3_EXPECTED_SELECTED_ITEM,
    build_e3_compatibility_task,
)
from obsidianlink.env.validation import E3_SELECTED_ITEM_CASE, EnvironmentValidationRunner


ROOT = Path(__file__).resolve().parents[1]
EPISODE_ID = "e3-adapter-episode"


class _Backend:
    def __init__(self, reset_result: object, **kwargs: Any) -> None:
        self.reset_result = reset_result
        self.kwargs = kwargs
        self.calls: list[object] = []
        self.reset_task = None
        self._opened = False
        self._env = None
        self._owner_thread = None

    def open(self) -> None:
        self.calls.append("open")
        self._opened = True

    def reset(self, task: object) -> object:
        self.calls.append("reset")
        self.reset_task = task
        self._env = object()
        return self.reset_result

    def close(self) -> None:
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None

    def step(self, action: object) -> None:
        raise AssertionError("E3 adapter must not step")


def _backend_cls(reset_result: object) -> type[_Backend]:
    class Configured(_Backend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(reset_result, **kwargs)
    Configured.__name__ = "RecordingMineRLBackend"
    return Configured


def _raw(item: object = E3_EXPECTED_SELECTED_ITEM, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "agent_id": E3_AGENT_ID,
        "episode_id": EPISODE_ID,
        "step_id": 0,
        "selected_item": item,
        "visible_inventory": {E3_EXPECTED_SELECTED_ITEM: 1},
        "frame": "drop",
        "messages": ("drop",),
        "workflow_stage": "drop",
        "portal_grid": "drop",
        "equipped_items": {"mainhand": {"type": "must-not-read"}},
    }
    payload.update(extra)
    return {E3_AGENT_ID: SimpleNamespace(**payload)}


def _run(reset_result: object):
    adapter_holder = {}
    factory = MineRLE3SelectedItemAdapter.lifecycle_factory(
        episode_id=EPISODE_ID, backend_cls=_backend_cls(reset_result)
    )
    def capture():
        adapter = factory()
        adapter_holder["adapter"] = adapter
        return adapter
    result = EnvironmentValidationRunner().run(
        E3_SELECTED_ITEM_CASE, capture, episode_id=EPISODE_ID,
        expected_selected_item=E3_EXPECTED_SELECTED_ITEM,
    )
    return result, adapter_holder["adapter"]


class E3ConfigAndAdapterTests(unittest.TestCase):
    def test_config_is_dedicated_single_item_compatibility_only(self) -> None:
        self.assertEqual(dict(E3_CALIBRATION_INVENTORY), {"flint_and_steel": 1})
        task = build_e3_compatibility_task(EPISODE_ID)
        self.assertEqual(dict(task.initial_inventories[E3_AGENT_ID]), dict(E3_CALIBRATION_INVENTORY))
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E3")
        self.assertTrue(task.scenario_parameters["compatibility_only"])
        self.assertTrue(task.scenario_parameters["not_a_benchmark_task"])

    def test_projects_only_backend_public_selected_item(self) -> None:
        projected = public_selected_item_observation(_raw(), episode_id=EPISODE_ID)
        self.assertEqual(projected[E3_AGENT_ID], {
            "agent_id": E3_AGENT_ID,
            "episode_id": EPISODE_ID,
            "selected_item": E3_EXPECTED_SELECTED_ITEM,
            "step_id": 0,
        })

    def test_visible_inventory_and_raw_equipped_items_cannot_derive_item(self) -> None:
        raw = {
            E3_AGENT_ID: {
                "agent_id": E3_AGENT_ID,
                "episode_id": EPISODE_ID,
                "step_id": 0,
                "visible_inventory": {E3_EXPECTED_SELECTED_ITEM: 1},
                "equipped_items": {"mainhand": {"type": E3_EXPECTED_SELECTED_ITEM}},
            }
        }
        projected = public_selected_item_observation(raw, episode_id=EPISODE_ID)
        self.assertNotIn("selected_item", projected[E3_AGENT_ID])
        result, _ = _run(raw)
        self.assertEqual(result.outcome, "selected_item_missing")

    def test_wrong_backend_item_remains_wrong_despite_expected_config(self) -> None:
        result, adapter = _run(_raw("obsidian"))
        self.assertEqual(result.outcome, "selected_item_mismatch")
        self.assertEqual(result.observed_selected_item, "obsidian")
        self.assertEqual(result.expected_selected_item, E3_EXPECTED_SELECTED_ITEM)
        self.assertEqual(adapter._backend.calls, ["open", "reset", "close"])

    def test_backend_none_remains_none(self) -> None:
        result, _ = _run(_raw(None))
        self.assertEqual(result.outcome, "selected_item_none")
        self.assertIsNone(result.observed_selected_item)

    def test_arbitrary_backend_fields_do_not_leak(self) -> None:
        projected = public_selected_item_observation(
            {E3_AGENT_ID: {"selected_item": E3_EXPECTED_SELECTED_ITEM, "truth": object(), "inventory": object()}},
            episode_id=EPISODE_ID,
        )
        self.assertEqual(set(projected[E3_AGENT_ID]), {"agent_id", "episode_id", "selected_item", "step_id"})

    def test_identity_mismatch_is_preserved_not_rewritten(self) -> None:
        result, _ = _run(_raw(E3_EXPECTED_SELECTED_ITEM, episode_id="wrong"))
        self.assertEqual(result.outcome, "initial_state_missing")

    def test_import_is_lazy_and_reset_signature_minimal(self) -> None:
        paths = (
            ROOT / "obsidianlink/env/integration/e3_adapter.py",
            ROOT / "obsidianlink/env/integration/e3_config.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            top = "\n".join(ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
            self.assertNotIn("obsidianlink.env.minerl_backend", top)
            self.assertNotIn("import minerl", top)
        with patch.object(MineRLE3SelectedItemAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE3SelectedItemAdapter.lifecycle_factory(episode_id=EPISODE_ID)()
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)
        self.assertEqual(tuple(inspect.signature(MineRLE3SelectedItemAdapter.reset).parameters), ("self",))


if __name__ == "__main__":
    unittest.main()
