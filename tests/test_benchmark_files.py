from __future__ import annotations

import json
import unittest
from pathlib import Path

from obsidianlink.core.types import TaskInstance


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkFileTests(unittest.TestCase):
    def test_active_casting_c1_contract_is_explicitly_offline_only(self) -> None:
        task_path = (
            ROOT / "benchmark/instances/active/casting_c1_fixed.json"
        )
        config_path = (
            ROOT / "configs/experiments/active/casting_c1_contract.json"
        )
        task = TaskInstance.from_dict(
            json.loads(task_path.read_text(encoding="utf-8"))
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(task.task_id, "casting_c1_fixed_seed_0")
        self.assertEqual(task.route, "lava_casting")
        self.assertEqual(task.workflow, "casting_c1_fixed")
        self.assertEqual(task.difficulty, 1)
        self.assertEqual(
            task.scenario_parameters["implementation_status"],
            "contract_only",
        )
        self.assertEqual(task.scenario_parameters["task_family"], "casting")
        self.assertEqual(task.scenario_parameters["agent_mode"], "single")
        self.assertEqual(task.scenario_parameters["task_level"], "C1")
        self.assertEqual(task.scenario_parameters["layout_type"], "fixed")
        self.assertEqual(
            task.scenario_parameters["compatibility_task_name"],
            "casting_s_c1_fixed",
        )
        self.assertFalse(task.scenario_parameters["allow_live_run"])
        self.assertIn("first_obsidian_cast", task.milestones)
        self.assertEqual(
            config["task_instance"],
            "benchmark/instances/active/casting_c1_fixed.json",
        )
        self.assertEqual(config["status"], "contract_only")
        self.assertFalse(config["allow_live_run"])
        self.assertEqual(config["max_real_runs"], 0)

    def test_active_casting_c3_contract_is_offline_verified_and_fixed(self) -> None:
        task_path = ROOT / "benchmark/instances/active/casting_c3_fixed.json"
        config_path = ROOT / "configs/experiments/active/casting_c3_contract.json"
        task = TaskInstance.from_dict(
            json.loads(task_path.read_text(encoding="utf-8"))
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(task.task_id, "casting_c3_fixed_seed_0")
        self.assertEqual(task.workflow, "casting_c3_fixed")
        self.assertEqual(task.difficulty, 2)
        self.assertEqual(
            task.scenario_parameters["target_cells"],
            ((2, 4, 3), (3, 4, 3), (4, 4, 3)),
        )
        self.assertEqual(
            task.scenario_parameters["implementation_status"],
            "offline_fake_verified",
        )
        self.assertEqual(task.scenario_parameters["task_family"], "casting")
        self.assertEqual(task.scenario_parameters["agent_mode"], "single")
        self.assertEqual(task.scenario_parameters["task_level"], "C2")
        self.assertEqual(task.scenario_parameters["layout_type"], "fixed")
        self.assertEqual(
            task.scenario_parameters["compatibility_task_name"],
            "casting_s_c2_fixed",
        )
        self.assertFalse(task.scenario_parameters["allow_live_run"])
        self.assertEqual(config["status"], "offline_fake_verified")
        self.assertEqual(config["backend"], "fake")
        self.assertEqual(config["planner"], "deterministic_casting_c3")
        self.assertEqual(config["evaluator"], "continuous_casting_v1")
        self.assertFalse(config["allow_live_run"])
        self.assertEqual(config["max_real_runs"], 0)

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

if __name__ == "__main__":
    unittest.main()
