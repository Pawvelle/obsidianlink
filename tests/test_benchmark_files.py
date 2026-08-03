from __future__ import annotations

import json
import unittest
from pathlib import Path

from obsidianlink.core.types import TaskInstance


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkFileTests(unittest.TestCase):
    def test_development_instance_matches_core_contract(self) -> None:
        path = ROOT / "benchmark/instances/route_a_a0_development.json"
        task = TaskInstance.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(task.task_id, "route_a_a0_development_seed_0")
        self.assertEqual(task.route, "obsidian_mining")
        self.assertEqual(task.agent_ids, ("agent_1",))

    def test_phase_zero_config_points_to_existing_task(self) -> None:
        path = ROOT / "configs/experiments/phase0_fake_a0.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["backend"], "fake")
        self.assertTrue((ROOT / config["task_instance"]).is_file())
        self.assertIsNone(config["planner"])

    def test_phase_three_scripted_config_freezes_a0_instance(self) -> None:
        path = ROOT / "configs/experiments/phase3_scripted_a0.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        task_path = ROOT / config["task_instance"]
        task = TaskInstance.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
        self.assertEqual(config["planner"], "scripted_a0")
        self.assertIsNone(config["failure_injection"])
        self.assertEqual(config["max_placement_retries"], 0)
        self.assertEqual(config["step_timeout_seconds"], 30.0)
        self.assertEqual(task.task_id, "route_a_a0_phase3_seed_0")
        self.assertEqual(task.initial_inventories["agent_1"]["obsidian"], 14)

    def test_phase_three_vlm_configs_share_the_frozen_a0_task(self) -> None:
        for filename in (
            "phase3_single_workflow_a0.json",
            "phase3_single_direct_a0.json",
        ):
            with self.subTest(filename=filename):
                path = ROOT / "configs/experiments" / filename
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["task_instance"],
                    "benchmark/instances/route_a_a0_phase3.json",
                )
                self.assertTrue((ROOT / config["model_lock"]).is_file())
                self.assertEqual(config["evaluator"], "portal_v0")

    def test_phase_four_a1_instance_freezes_nearby_obsidian_contract(self) -> None:
        path = ROOT / "benchmark/instances/route_a_a1_phase4.json"
        task = TaskInstance.from_dict(json.loads(path.read_text(encoding="utf-8")))
        inventory = task.initial_inventories["agent_1"]
        scenario = task.scenario_parameters

        self.assertEqual(task.task_id, "route_a_a1_phase4_seed_0")
        self.assertEqual(task.workflow, "route_a_a1")
        self.assertEqual(task.difficulty, 2)
        self.assertNotIn("obsidian", inventory)
        self.assertEqual(inventory["diamond_pickaxe"], 1)
        self.assertEqual(scenario["variant"], "nearby_obsidian")
        self.assertEqual(scenario["obsidian_required"], 14)
        self.assertEqual(scenario["obsidian_deposit"]["minimum_blocks"], 14)
        self.assertLessEqual(
            scenario["obsidian_deposit"]["max_distance_blocks"], 8
        )
        self.assertIn("obsidian_quota_collected", task.milestones)
        with self.assertRaises(TypeError):
            scenario["obsidian_deposit"]["minimum_blocks"] = 13

    def test_phase_four_a1_scripted_config_freezes_mining_slice(self) -> None:
        path = ROOT / "configs/experiments/phase4_scripted_a1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        task_path = ROOT / config["task_instance"]
        task = TaskInstance.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
        self.assertEqual(config["planner"], "scripted_a1")
        self.assertEqual(config["obsidian_quota_required"], 14)
        self.assertGreaterEqual(config["max_no_progress_retries"], 1)
        self.assertGreaterEqual(config["max_cell_retry_attempts"], 1)
        self.assertEqual(task.task_id, "route_a_a1_phase4_seed_0")
        self.assertEqual(task.workflow, "route_a_a1")
        self.assertNotIn("obsidian", task.initial_inventories["agent_1"])
        self.assertEqual(
            task.scenario_parameters["obsidian_required"], 14
        )


if __name__ == "__main__":
    unittest.main()
