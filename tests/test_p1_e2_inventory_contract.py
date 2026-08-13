from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from typing import Mapping

from obsidianlink.env.validation import (
    InventoryInspection,
    PublicInventoryObservation,
    inspect_inventory,
    inspect_public_inventory,
    p1_validation_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_MODULE = ROOT / "obsidianlink/env/validation/inventory.py"
EPISODE_ID = "e2-offline-episode"
AGENT_ID = "agent_1"
_UNSET = object()


def _public_state(
    inventory: object = _UNSET,
    *,
    episode_id: object = EPISODE_ID,
    agent_id: object = AGENT_ID,
    step_id: object = 0,
    **extra: object,
) -> dict[str, dict[str, object]]:
    payload: dict[str, object] = {
        "agent_id": agent_id,
        "episode_id": episode_id,
        "step_id": step_id,
    }
    if inventory is not _UNSET:
        payload["inventory"] = inventory
    payload.update(extra)
    return {AGENT_ID: payload}


def _inspection(inventory: object = _UNSET, **kwargs: object) -> InventoryInspection:
    return inspect_public_inventory(
        _public_state(inventory, **kwargs), episode_id=EPISODE_ID
    )


def _imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package = ".".join(relative.parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


class E2InventoryContractTests(unittest.TestCase):
    def test_empty_inventory_remains_structurally_valid(self) -> None:
        inspection = inspect_inventory({})
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.outcome, "inventory_ok")
        self.assertEqual(inspection.inventory, {})

    def test_valid_multi_item_inventory_succeeds(self) -> None:
        inventory = {"dirt": 7, "obsidian": 4, "flint_and_steel": 1}
        inspection = _inspection(inventory)
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.outcome, "inventory_ok")
        self.assertTrue(inspection.present)
        self.assertEqual(inspection.inventory, inventory)

    def test_missing_inventory_fails_closed(self) -> None:
        inspection = _inspection()
        self.assertFalse(inspection.valid)
        self.assertEqual(inspection.outcome, "inventory_missing")
        self.assertFalse(inspection.present)

    def test_none_inventory_fails_closed(self) -> None:
        inspection = _inspection(None)
        self.assertEqual(inspection.outcome, "inventory_none")
        self.assertFalse(inspection.present)

    def test_non_mapping_inventory_fails_closed(self) -> None:
        for inventory in (["obsidian"], "obsidian", 4):
            with self.subTest(inventory=inventory):
                inspection = _inspection(inventory)
                self.assertEqual(inspection.outcome, "inventory_type_invalid")
                self.assertTrue(inspection.present)

    def test_empty_item_name_fails_closed(self) -> None:
        for item in ("", "   "):
            with self.subTest(item=repr(item)):
                self.assertEqual(
                    _inspection({item: 1}).outcome, "inventory_item_invalid"
                )

    def test_non_string_item_name_fails_closed(self) -> None:
        self.assertEqual(
            _inspection({1: 4}).outcome,  # type: ignore[dict-item]
            "inventory_item_invalid",
        )

    def test_string_quantity_is_not_coerced(self) -> None:
        self.assertEqual(
            _inspection({"obsidian": "4"}).outcome,
            "inventory_quantity_invalid",
        )

    def test_float_quantity_is_not_coerced(self) -> None:
        self.assertEqual(
            _inspection({"obsidian": 4.0}).outcome,
            "inventory_quantity_invalid",
        )

    def test_bool_quantity_is_not_an_int(self) -> None:
        self.assertEqual(
            _inspection({"obsidian": True}).outcome,
            "inventory_quantity_invalid",
        )

    def test_zero_quantity_fails_closed(self) -> None:
        self.assertEqual(
            _inspection({"obsidian": 0}).outcome,
            "inventory_quantity_invalid",
        )

    def test_negative_quantity_fails_closed(self) -> None:
        self.assertEqual(
            _inspection({"obsidian": -1}).outcome,
            "inventory_quantity_invalid",
        )

    def test_other_p1_and_evaluator_fields_are_leaks(self) -> None:
        leaks = {
            "rgb": object(),
            "selected_item": "obsidian",
            "portal_grid": [["obsidian"]],
        }
        for field, value in leaks.items():
            with self.subTest(field=field):
                inspection = _inspection({"obsidian": 4}, **{field: value})
                self.assertEqual(inspection.outcome, "inventory_leak")
                self.assertTrue(inspection.present)
                self.assertIn(field, inspection.error or "")

    def test_unknown_extra_field_is_also_a_leak(self) -> None:
        inspection = _inspection({"obsidian": 4}, future_field="not allowed")
        self.assertEqual(inspection.outcome, "inventory_leak")

    def test_wrong_episode_identity_fails_closed(self) -> None:
        inspection = _inspection({"obsidian": 4}, episode_id="other-episode")
        self.assertEqual(inspection.outcome, "inventory_missing")
        self.assertFalse(inspection.valid)

    def test_nonzero_initial_step_fails_closed(self) -> None:
        inspection = _inspection({"obsidian": 4}, step_id=1)
        self.assertEqual(inspection.outcome, "inventory_missing")
        self.assertFalse(inspection.valid)

    def test_agent_identity_must_match_reset_mapping_key(self) -> None:
        inspection = _inspection({"obsidian": 4}, agent_id="agent_2")
        self.assertEqual(inspection.outcome, "inventory_missing")

    def test_identity_fields_are_required(self) -> None:
        for field in ("agent_id", "episode_id", "step_id"):
            payload = _public_state({"obsidian": 4})[AGENT_ID]
            del payload[field]
            with self.subTest(field=field):
                inspection = inspect_public_inventory(
                    {AGENT_ID: payload}, episode_id=EPISODE_ID
                )
                self.assertEqual(inspection.outcome, "inventory_missing")

    def test_public_type_has_only_e2_fields(self) -> None:
        observation = PublicInventoryObservation(
            episode_id=EPISODE_ID,
            agent_id=AGENT_ID,
            step_id=0,
            inventory={"obsidian": 4},
        )
        self.assertEqual(
            set(observation.as_public_dict()),
            {"agent_id", "episode_id", "inventory", "step_id"},
        )
        for field in (
            "rgb",
            "selected_item",
            "workflow_stage",
            "portal_grid",
            "info",
        ):
            with self.subTest(field=field):
                self.assertFalse(hasattr(observation, field))
        self.assertTrue(
            inspect_public_inventory(
                {AGENT_ID: observation}, episode_id=EPISODE_ID
            ).valid
        )

    def test_plain_mappings_are_detached(self) -> None:
        source = {"obsidian": 4}
        observation = PublicInventoryObservation(
            episode_id=EPISODE_ID,
            agent_id=AGENT_ID,
            step_id=0,
            inventory=source,
        )
        source["obsidian"] = 99
        public_one = observation.as_public_dict()
        public_inventory = public_one["inventory"]
        assert isinstance(public_inventory, dict)
        public_inventory["obsidian"] = 88
        public_two = observation.as_public_dict()
        self.assertEqual(public_two["inventory"], {"obsidian": 4})

        inspected_source = {"dirt": 7}
        inspection = inspect_inventory(inspected_source)
        inspected_source["dirt"] = 77
        snapshot_one = inspection.as_dict()
        snapshot_inventory = snapshot_one["inventory"]
        assert isinstance(snapshot_inventory, dict)
        snapshot_inventory["dirt"] = 66
        self.assertEqual(inspection.as_dict()["inventory"], {"dirt": 7})

    def test_constructor_rejects_invalid_inventory_semantics(self) -> None:
        invalid: tuple[Mapping[object, object] | object, ...] = (
            None,
            {"": 1},
            {1: 1},
            {"obsidian": "4"},
            {"obsidian": 4.0},
            {"obsidian": True},
            {"obsidian": 0},
        )
        for inventory in invalid:
            with self.subTest(inventory=inventory):
                with self.assertRaises((TypeError, ValueError)):
                    PublicInventoryObservation(
                        episode_id=EPISODE_ID,
                        agent_id=AGENT_ID,
                        step_id=0,
                        inventory=inventory,  # type: ignore[arg-type]
                    )

    def test_e2_manifest_remains_calibration_only_not_run_without_truth(self) -> None:
        manifest = p1_validation_manifest()
        e2 = manifest[2]
        self.assertEqual(e2["check_id"], "E2")
        self.assertEqual(e2["name"], "inventory_observation")
        self.assertFalse(e2["requires_server_truth"])
        self.assertTrue(e2["calibration_only"])
        self.assertEqual(e2["status"], "not_run")

    def test_inventory_module_has_no_disallowed_architecture_imports(self) -> None:
        banned = (
            "minerl",
            "obsidianlink.agents",
            "obsidianlink.baselines",
            "obsidianlink.benchmark",
            "obsidianlink.drivers",
            "obsidianlink.env.integration",
            "obsidianlink.env.minerl_backend",
            "obsidianlink.evaluation",
        )
        imports = _imported_modules(INVENTORY_MODULE)
        for module in imports:
            with self.subTest(module=module):
                self.assertFalse(
                    any(
                        module == prefix or module.startswith(f"{prefix}.")
                        for prefix in banned
                    )
                )


if __name__ == "__main__":
    unittest.main()
