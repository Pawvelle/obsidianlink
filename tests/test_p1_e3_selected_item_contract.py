from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

from obsidianlink.env.validation import (
    PublicSelectedItemObservation,
    inspect_public_selected_item,
    inspect_selected_item,
    p1_validation_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
EPISODE_ID = "e3-contract-episode"
AGENT_ID = "agent_1"
_UNSET = object()


def _state(item: object = _UNSET, **extra: object) -> dict[str, dict[str, object]]:
    payload: dict[str, object] = {
        "agent_id": AGENT_ID,
        "episode_id": EPISODE_ID,
        "step_id": 0,
    }
    if item is not _UNSET:
        payload["selected_item"] = item
    payload.update(extra)
    return {AGENT_ID: payload}


class E3SelectedItemContractTests(unittest.TestCase):
    def test_valid_selected_item_is_accepted(self) -> None:
        result = inspect_public_selected_item(
            _state("flint_and_steel"), episode_id=EPISODE_ID
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.selected_item, "flint_and_steel")

    def test_missing_none_empty_and_wrong_type_fail_closed(self) -> None:
        cases = (
            (_UNSET, "selected_item_missing"),
            (None, "selected_item_none"),
            ("", "selected_item_empty"),
            ("   ", "selected_item_empty"),
            (4, "selected_item_type_invalid"),
            (True, "selected_item_type_invalid"),
        )
        for value, outcome in cases:
            with self.subTest(value=value):
                result = inspect_public_selected_item(
                    _state(value), episode_id=EPISODE_ID
                )
                self.assertFalse(result.valid)
                self.assertEqual(result.outcome, outcome)

    def test_identity_and_initial_step_fail_closed(self) -> None:
        cases = (
            {"episode_id": "wrong"},
            {"agent_id": "agent_2"},
            {"step_id": 1},
            {"step_id": True},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                result = inspect_public_selected_item(
                    _state("flint_and_steel", **changes), episode_id=EPISODE_ID
                )
                self.assertEqual(result.outcome, "selected_item_missing")

    def test_identity_fields_are_required(self) -> None:
        for field in ("agent_id", "episode_id", "step_id"):
            payload = _state("flint_and_steel")[AGENT_ID]
            del payload[field]
            with self.subTest(field=field):
                result = inspect_public_selected_item(
                    {AGENT_ID: payload}, episode_id=EPISODE_ID
                )
                self.assertEqual(result.outcome, "selected_item_missing")

    def test_every_non_e3_or_unknown_public_field_is_rejected(self) -> None:
        leaks = (
            "inventory",
            "visible_inventory",
            "rgb",
            "frame",
            "pov",
            "messages",
            "workflow_stage",
            "portal_grid",
            "fluid_truth",
            "portal_truth",
            "dimension",
            "equipped_items",
            "info",
            "arbitrary_unknown",
        )
        for field in leaks:
            with self.subTest(field=field):
                result = inspect_public_selected_item(
                    _state("flint_and_steel", **{field: object()}),
                    episode_id=EPISODE_ID,
                )
                self.assertEqual(result.outcome, "selected_item_leak")
                self.assertIn(field, result.error or "")

    def test_temporary_public_type_has_only_four_fields(self) -> None:
        observation = PublicSelectedItemObservation(
            episode_id=EPISODE_ID,
            agent_id=AGENT_ID,
            step_id=0,
            selected_item="flint_and_steel",
        )
        self.assertEqual(
            set(observation.as_public_dict()),
            {"agent_id", "episode_id", "selected_item", "step_id"},
        )
        for field in ("inventory", "rgb", "frame", "equipped_items", "messages"):
            self.assertFalse(hasattr(observation, field))

    def test_none_is_invalid_for_calibration_even_though_backend_allows_none(self) -> None:
        self.assertEqual(inspect_selected_item(None).outcome, "selected_item_none")

    def test_manifest_remains_calibration_only_not_run(self) -> None:
        e3 = p1_validation_manifest()[3]
        self.assertEqual(e3["check_id"], "E3")
        self.assertEqual(e3["name"], "selected_item")
        self.assertFalse(e3["requires_server_truth"])
        self.assertTrue(e3["calibration_only"])
        self.assertEqual(e3["status"], "not_run")

    def test_contract_module_has_no_minerl_or_integration_import(self) -> None:
        path = ROOT / "obsidianlink/env/validation/selected_item.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        package = "obsidianlink.env.validation"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    imports.append(importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    ))
                elif node.module:
                    imports.append(node.module)
        banned = ("minerl", "obsidianlink.env.integration", "obsidianlink.env.minerl_backend")
        self.assertFalse(any(any(module == p or module.startswith(p + ".") for p in banned) for module in imports))


if __name__ == "__main__":
    unittest.main()
