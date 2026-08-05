from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from obsidianlink.core.task_catalog import (
    TaskCatalog,
    TaskTaxonomy,
    load_task_catalog,
    validate_catalog_references,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"


class TaskTaxonomyTests(unittest.TestCase):
    def test_canonical_name_is_derived(self) -> None:
        taxonomy = TaskTaxonomy("casting", "single", "C2", "fixed")
        self.assertEqual(taxonomy.canonical_name, "casting_s_c2_fixed")

    def test_family_and_level_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            TaskTaxonomy("casting", "single", "R1", "fixed")

    def test_taxonomy_is_frozen(self) -> None:
        taxonomy = TaskTaxonomy("casting", "single", "C1", "fixed")
        with self.assertRaises(FrozenInstanceError):
            taxonomy.task_level = "C2"  # type: ignore[misc]


class TaskCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_task_catalog(CATALOG_PATH)

    def test_catalog_loads_and_references_resolve(self) -> None:
        validate_catalog_references(self.catalog, ROOT)
        self.assertEqual(self.catalog.schema_version, "0.1")
        self.assertEqual(self.catalog.catalog_version, "2026-08-05")
        self.assertEqual(len(self.catalog.entries), 7)

    def test_active_entry_is_casting_c2_compatibility_task(self) -> None:
        active = self.catalog.active_entry
        self.assertEqual(active.compatibility_id, "casting_c3_fixed")
        self.assertEqual(active.canonical_name, "casting_s_c2_fixed")
        self.assertIsNotNone(active.taxonomy)
        self.assertEqual(active.taxonomy.task_level, "C2")  # type: ignore[union-attr]
        self.assertFalse(active.live_run_allowed)

    def test_all_task_instances_are_cataloged(self) -> None:
        instance_paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "benchmark/instances").rglob("*.json")
        }
        cataloged_paths = {entry.instance_path for entry in self.catalog.entries}
        self.assertEqual(cataloged_paths, instance_paths)

    def test_all_experiment_configs_are_cataloged(self) -> None:
        experiment_paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs/experiments").rglob("*.json")
        }
        cataloged_paths = {
            path for entry in self.catalog.entries for path in entry.experiment_paths
        }
        self.assertEqual(cataloged_paths, experiment_paths)

    def test_calibration_entries_are_not_benchmark_visible(self) -> None:
        calibration_entries = [
            entry for entry in self.catalog.entries if entry.kind == "calibration"
        ]
        self.assertEqual(len(calibration_entries), 2)
        for entry in calibration_entries:
            self.assertFalse(entry.benchmark_visible)
            self.assertIsNone(entry.taxonomy)
            self.assertEqual(entry.implementation_status, "legacy_regression")

    def test_catalog_round_trip_is_json_serializable(self) -> None:
        snapshot = self.catalog.as_dict()
        self.assertEqual(TaskCatalog.from_dict(snapshot), self.catalog)
        json.dumps(snapshot)

    def test_unknown_catalog_field_fails_closed(self) -> None:
        snapshot = self.catalog.as_dict()
        snapshot["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            TaskCatalog.from_dict(snapshot)

    def test_duplicate_compatibility_id_fails_closed(self) -> None:
        snapshot = self.catalog.as_dict()
        entries = snapshot["entries"]
        self.assertIsInstance(entries, list)
        entries[1]["compatibility_id"] = entries[0]["compatibility_id"]
        with self.assertRaisesRegex(ValueError, "compatibility_id.*unique"):
            TaskCatalog.from_dict(snapshot)

    def test_unsafe_instance_path_fails_closed(self) -> None:
        snapshot = self.catalog.as_dict()
        entries = snapshot["entries"]
        self.assertIsInstance(entries, list)
        entries[0]["instance_path"] = "../secret.json"
        with self.assertRaisesRegex(ValueError, "safe relative"):
            TaskCatalog.from_dict(snapshot)


if __name__ == "__main__":
    unittest.main()
