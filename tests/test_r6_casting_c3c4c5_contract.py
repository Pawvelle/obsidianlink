"""Contract tests for the R6 Casting-S-C3 / C4 / C5 contract-freeze stage.

These tests only verify the *contract*: that the new task instances, the
experiment configs, the catalog entries, the frame geometry and the
information-isolation contract are all consistent with B0 taxonomy and
the existing Frozen ``FrameCandidate`` / ``EvaluationState`` field
semantics. They do **not** exercise a driver, an evaluator or a real
MineRL run; that work is the next sub-task.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from obsidianlink.core.task_catalog import (
    load_task_catalog,
    validate_catalog_references,
)
from obsidianlink.core.types import TaskInstance


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"

CONTRACT_INSTANCE_PATHS = {
    "C3": ROOT / "benchmark/instances/casting/single/casting_s_c3_fixed.json",
    "C4": ROOT / "benchmark/instances/casting/single/casting_s_c4_fixed.json",
    "C5": ROOT / "benchmark/instances/casting/single/casting_s_c5_fixed.json",
}
CONTRACT_CONFIG_PATHS = {
    "C3": ROOT / "configs/experiments/active/casting_s_c3_contract.json",
    "C4": ROOT / "configs/experiments/active/casting_s_c4_contract.json",
    "C5": ROOT / "configs/experiments/active/casting_s_c5_contract.json",
}
EXPECTED_LEVELS = {"C3": "C3", "C4": "C4", "C5": "C5"}
EXPECTED_CANONICAL = {
    "C3": "casting_s_c3_fixed",
    "C4": "casting_s_c4_fixed",
    "C5": "casting_s_c5_fixed",
}

# Runtime observations and evaluator results are hidden. Static geometry,
# designated agents, actions and items are public task rules and therefore do
# not belong in this list.
EVALUATOR_ONLY_RUNTIME_FIELDS = (
    "baseline_grid",
    "current_grid",
    "relevant_action_steps_by_cell",
    "obsidian_transition_steps",
    "first_activation_step",
    "latched_frame_identity",
    "latched_activation_offsets",
    "entered_via_episode_portal_by_agent",
    "matched_frame_identity_by_agent",
    "first_nether_step_by_agent",
    "pre_transition_position_by_agent",
    "transition_step_by_agent",
    "agents_in_nether",
    "attribution_failed_candidates",
    "external_obsidian_offsets",
    "episode_obsidian_offsets",
    "outcome",
    "success",
    "failure_type",
)

# Existing policy/driver code must not consume the unfiltered scenario mapping.
# R6 runtime does not exist yet; when it is introduced it must use a dedicated
# public-context builder rather than weakening this guard.
AGENT_RUNTIME_SOURCES = (
    "obsidianlink/agents",
    "obsidianlink/workflows",
    "obsidianlink/drivers",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


def _agent_runtime_python_sources() -> tuple[Path, ...]:
    sources: list[Path] = []
    for relative in AGENT_RUNTIME_SOURCES:
        path = ROOT / relative
        if path.is_file():
            sources.append(path)
        else:
            sources.extend(sorted(path.rglob("*.py")))
    return tuple(sources)


class ContractInstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_task_catalog(CATALOG_PATH)
        validate_catalog_references(self.catalog, ROOT)

    def test_all_three_contract_instances_exist(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                self.assertTrue(
                    path.is_file(),
                    f"contract instance missing: {path}",
                )

    def test_all_three_contract_configs_exist(self) -> None:
        for level, path in CONTRACT_CONFIG_PATHS.items():
            with self.subTest(level=level):
                self.assertTrue(
                    path.is_file(),
                    f"contract config missing: {path}",
                )

    def test_contract_instances_have_correct_taxonomy(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                payload = _load_json(path)
                instance = TaskInstance.from_dict(payload)
                params = instance.scenario_parameters
                self.assertEqual(params["task_family"], "casting")
                self.assertEqual(params["agent_mode"], "single")
                self.assertEqual(params["task_level"], EXPECTED_LEVELS[level])
                self.assertEqual(params["layout_type"], "fixed")
                self.assertEqual(
                    params["compatibility_task_name"],
                    EXPECTED_CANONICAL[level],
                )
                self.assertEqual(
                    params["implementation_status"], "contract_only"
                )
                self.assertFalse(params["allow_live_run"])
                self.assertTrue(params["requires_explicit_live_run_approval"])
                self.assertFalse(params["allow_minecraft_commands"])
                self.assertFalse(params["allow_evaluator_world_mutation"])
                self.assertEqual(instance.agent_ids, ("agent_1",))
                self.assertEqual(instance.world_seed, 0)
                self.assertEqual(instance.workflow, EXPECTED_CANONICAL[level])
                self.assertEqual(
                    instance.task_id, f"{EXPECTED_CANONICAL[level]}_seed_0"
                )

    def test_contract_configs_match_instances(self) -> None:
        for level, path in CONTRACT_CONFIG_PATHS.items():
            with self.subTest(level=level):
                config = _load_json(path)
                instance = _load_json(CONTRACT_INSTANCE_PATHS[level])
                self.assertEqual(config["status"], "contract_only")
                self.assertEqual(config["backend"], "not_implemented")
                self.assertEqual(config["planner"], "not_implemented")
                expected_evaluator = (
                    "obsidianlink.evaluation.casting_nether_entry_evaluator."
                    "FrozenNetherEntryEvaluator"
                    if level == "C5"
                    else "not_implemented"
                )
                self.assertEqual(config["evaluator"], expected_evaluator)
                self.assertFalse(config["allow_live_run"])
                self.assertTrue(config["requires_gradle_approval"])
                self.assertTrue(config["requires_minerl_run_approval"])
                self.assertEqual(config["max_real_runs"], 0)
                expected_task_path = (
                    "benchmark/instances/casting/single/"
                    f"{EXPECTED_CANONICAL[level]}.json"
                )
                self.assertEqual(config["task_instance"], expected_task_path)
                self.assertEqual(
                    instance["task_id"],
                    f"{EXPECTED_CANONICAL[level]}_seed_0",
                )

    def test_all_contracts_continue_the_water_lava_casting_route(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                payload = _load_json(path)
                inventory = payload["initial_inventories"]["agent_1"]
                params = payload["scenario_parameters"]
                self.assertNotIn("obsidian", inventory)
                self.assertEqual(inventory["water_bucket"], 14)
                self.assertEqual(inventory["lava_bucket"], 14)
                self.assertEqual(inventory["cobblestone"], 28)
                self.assertIn(
                    "vanilla_water_lava_block_update",
                    params["mechanics_required"],
                )
                self.assertIn("water", payload["instruction"])
                self.assertIn("lava", payload["instruction"])
                attribution = params["evaluator_contract"]["frame_attribution"]
                self.assertEqual(
                    attribution["required_mechanism"],
                    "vanilla_water_lava_block_update",
                )
                self.assertFalse(attribution["direct_obsidian_placement_allowed"])

    def test_public_and_evaluator_contract_namespaces_are_explicit(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                params = _load_json(path)["scenario_parameters"]
                self.assertIsInstance(params["public_task_spec"], dict)
                self.assertIsInstance(params["evaluator_contract"], dict)
                self.assertNotEqual(
                    params["public_task_spec"], params["evaluator_contract"]
                )


class FrameGeometryContractTests(unittest.TestCase):
    """The C3/C4/C5 contracts freeze the 4x5 frame geometry verbatim.

    The frozen coordinates must agree with
    ``obsidianlink.evaluation.frame_geometry._candidate_frame_cells``
    so the future R6 evaluator can read the same offsets.
    """

    def setUp(self) -> None:
        from obsidianlink.evaluation import frame_geometry as fg

        self.fg = fg
        self._frame = fg._candidate_frame_cells(
            fg.PLANE_Z, 0, 0, 1, 4, 5
        )
        self._expected_offsets = {tuple(c.as_tuple()) for c in self._frame[0]}

    def test_c3_frame_offsets_match_geometry_module(self) -> None:
        payload = _load_json(CONTRACT_INSTANCE_PATHS["C3"])
        frame = payload["scenario_parameters"]["public_task_spec"]["frame_plan"]
        offsets = {tuple(o) for o in frame["fixed_offsets"]}
        self.assertEqual(offsets, self._expected_offsets)
        self.assertEqual(frame["minecraft_minimum_required_block_count"], 10)
        self.assertEqual(frame["benchmark_required_full_ring_block_count"], 14)
        self.assertTrue(frame["require_full_ring"])
        self.assertEqual(frame["required_corner_count"], 4)
        self.assertEqual(len(self._frame[0]), 14)
        self.assertEqual(
            self.fg.FrameCandidate.required_count.fget(
                SimpleNamespace(width=4, height=5)
            ),
            10,
        )

    def test_c4_frame_offsets_match_geometry_module(self) -> None:
        payload = _load_json(CONTRACT_INSTANCE_PATHS["C4"])
        public = payload["scenario_parameters"]["public_task_spec"]
        offsets = {tuple(o) for o in public["frame_plan"]["fixed_offsets"]}
        self.assertEqual(offsets, self._expected_offsets)
        ignition = public["ignition_plan"]
        self.assertTrue(ignition["required"])
        self.assertEqual(ignition["action"], "use_item")
        self.assertEqual(ignition["item"], "flint_and_steel")
        self.assertEqual(ignition["target_policy"], "exact")
        self.assertEqual(tuple(ignition["target_offset"]), (1, 1, 1))
        # The ignition offset must lie in the frame interior.
        self.assertIn(
            tuple(ignition["target_offset"]),
            {tuple(c.as_tuple()) for c in self._frame[1]},
        )
        activation = payload["scenario_parameters"]["evaluator_contract"][
            "activation_attribution"
        ]
        self.assertTrue(activation["require_exact_public_target"])
        self.assertTrue(activation["require_latched_frame_identity_match"])

    def test_c5_nether_entry_contract_is_strict(self) -> None:
        payload = _load_json(CONTRACT_INSTANCE_PATHS["C5"])
        params = payload["scenario_parameters"]
        nether = params["public_task_spec"]["nether_entry_goal"]
        self.assertTrue(nether["required"])
        self.assertEqual(list(nether["designated_agent_ids"]), ["agent_1"])
        self.assertEqual(nether["source_dimension"], "minecraft:overworld")
        self.assertEqual(nether["target_dimension"], "minecraft:the_nether")
        ignition = params["public_task_spec"]["ignition_plan"]
        self.assertTrue(ignition["required"])
        attribution = params["evaluator_contract"]["nether_entry_attribution"]
        self.assertTrue(attribution["require_entered_via_episode_portal"])
        self.assertTrue(attribution["require_matched_frame_identity"])
        self.assertTrue(attribution["require_pre_transition_position"])
        self.assertTrue(attribution["require_transition_step"])
        self.assertTrue(attribution["fail_closed_on_missing_truth"])
        self.assertEqual(
            attribution["unknown_attribution_outcome"],
            "nether_entry_portal_unknown",
        )
        # The C5 frame geometry must still be the C3/C4 frozen one.
        offsets = {
            tuple(o)
            for o in params["public_task_spec"]["frame_plan"]["fixed_offsets"]
        }
        self.assertEqual(offsets, self._expected_offsets)

    def test_c3_must_not_pretend_to_ignite_or_enter_nether(self) -> None:
        payload = _load_json(CONTRACT_INSTANCE_PATHS["C3"])
        public = payload["scenario_parameters"]["public_task_spec"]
        self.assertFalse(public["ignition_plan"]["required"])
        self.assertFalse(public["nether_entry_goal"]["required"])
        self.assertEqual(
            list(public["nether_entry_goal"]["designated_agent_ids"]),
            [],
        )

    def test_frame_baseline_policy_is_stronger_than_no_complete_frame(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                attribution = _load_json(path)["scenario_parameters"][
                    "evaluator_contract"
                ]["frame_attribution"]
                self.assertEqual(
                    attribution["baseline_policy"],
                    "all_benchmark_frame_cells_non_obsidian",
                )
                self.assertTrue(attribution["fail_closed_on_missing_truth"])


class CatalogContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_task_catalog(CATALOG_PATH)
        validate_catalog_references(self.catalog, ROOT)

    def test_c3c4c5_entries_are_quarantined_legacy_regressions(self) -> None:
        for level, canonical in EXPECTED_CANONICAL.items():
            with self.subTest(level=level):
                entry = self.catalog.entry_for_compatibility_id(canonical)
                self.assertEqual(entry.kind, "legacy")
                self.assertFalse(entry.benchmark_visible)
                self.assertFalse(entry.live_run_allowed)
                self.assertEqual(entry.implementation_status, "legacy_regression")
                self.assertEqual(entry.verification_level, "unit_verified")
                self.assertEqual(entry.canonical_name, canonical)
                self.assertIsNotNone(entry.taxonomy)
                assert entry.taxonomy is not None
                self.assertEqual(entry.taxonomy.task_level, EXPECTED_LEVELS[level])
                self.assertEqual(
                    entry.instance_path,
                    f"benchmark/instances/casting/single/{canonical}.json",
                )
                self.assertEqual(
                    tuple(entry.experiment_paths),
                    (
                        # Historical config files drop the ``_fixed``
                        # suffix; the R6 contract-only configs follow
                        # the same pattern (e.g.
                        # ``casting_s_c3_contract.json``).
                        f"configs/experiments/active/{canonical.replace('_fixed', '')}_contract.json",
                    ),
                )

    def test_p1_is_active_without_a_benchmark_task(self) -> None:
        self.assertEqual(
            self.catalog.active_phase, "P1-REAL-MINERL-ENVIRONMENT-VALIDATION"
        )
        self.assertIsNone(self.catalog.active_benchmark_task_id)
        self.assertIsNone(self.catalog.active_entry)

    def test_route_a_a0_remains_calibration(self) -> None:
        for compatibility_id in (
            "route_a_a0_development",
            "route_a_a0_phase3",
        ):
            with self.subTest(compatibility_id=compatibility_id):
                entry = self.catalog.entry_for_compatibility_id(compatibility_id)
                self.assertEqual(entry.kind, "calibration")
                self.assertFalse(entry.benchmark_visible)
                self.assertIsNone(entry.taxonomy)

    def test_canonical_names_match_taxonomy(self) -> None:
        for entry in self.catalog.entries:
            if entry.taxonomy is None:
                continue
            with self.subTest(canonical_name=entry.canonical_name):
                self.assertEqual(
                    entry.canonical_name,
                    entry.taxonomy.canonical_name,
                )

    def test_legacy_instance_paths_remain_compatible(self) -> None:
        for entry in self.catalog.entries:
            if entry.kind != "legacy":
                continue
            with self.subTest(instance_path=entry.instance_path):
                parts = entry.instance_path.split("/")
                self.assertGreaterEqual(len(parts), 4)
                self.assertEqual(parts[0], "benchmark")
                self.assertEqual(parts[1], "instances")
                self.assertIn(parts[2], {"casting", "active"})


class InformationIsolationContractTests(unittest.TestCase):
    """Freeze the public/hidden contract without claiming an R6 runtime.

    Geometry and goals are intentionally public. Actual world state,
    transition evidence and evaluator outcomes remain hidden. Existing Agent
    code may consume ``TaskInstance.instruction`` but must not read the raw
    scenario mapping; a future R6 driver must introduce a typed public-context
    builder and corresponding end-to-end isolation tests.
    """

    def test_instruction_contains_enough_public_geometry_to_execute(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                payload = _load_json(path)
                instruction = payload["instruction"]
                for public_fact in (
                    "task-origin marker",
                    "plane_z",
                    "[0,0,1]",
                    "width 4",
                    "height 5",
                    "14 obsidian cells",
                ):
                    self.assertIn(public_fact, instruction)

    def test_runtime_truth_keys_are_absent_from_agent_visible_contract(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                payload = _load_json(path)
                public_keys = _all_mapping_keys(
                    payload["scenario_parameters"]["public_task_spec"]
                )
                instruction = payload["instruction"]
                for field_name in EVALUATOR_ONLY_RUNTIME_FIELDS:
                    self.assertNotIn(field_name, public_keys)
                    self.assertNotIn(field_name, instruction)

    def test_existing_agent_runtime_does_not_read_raw_scenario_parameters(self) -> None:
        sources = _agent_runtime_python_sources()
        self.assertTrue(sources)
        for source_path in sources:
            relative = source_path.relative_to(ROOT)
            with self.subTest(source=str(relative)):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute):
                        self.assertNotEqual(
                            node.attr,
                            "scenario_parameters",
                            f"{relative} reads the unfiltered task scenario",
                        )
                    if isinstance(node, ast.Constant) and isinstance(
                        node.value, str
                    ):
                        self.assertNotEqual(
                            node.value,
                            "evaluator_contract",
                            f"{relative} references evaluator-only config",
                        )

    def test_evaluator_contract_contains_policy_not_runtime_truth_values(self) -> None:
        for level, path in CONTRACT_INSTANCE_PATHS.items():
            with self.subTest(level=level):
                evaluator_contract = _load_json(path)["scenario_parameters"][
                    "evaluator_contract"
                ]
                contract_keys = _all_mapping_keys(evaluator_contract)
                for runtime_field in EVALUATOR_ONLY_RUNTIME_FIELDS:
                    self.assertNotIn(
                        runtime_field,
                        contract_keys,
                        f"{level} instance must not contain episode truth values",
                    )


class DocumentationLinkTests(unittest.TestCase):
    """Active v2 docs must quarantine the historical C3/C4/C5 scope."""

    def test_core_docs_reference_v2_scope_and_legacy_archive(self) -> None:
        expected_doc_substrings = {
            "README.md": "legacy quarantine",
            "ROADMAP.md": "P1 — Real Environment Validation",
            "PROJECT_STATUS.md": "P1-REAL-MINERL-ENVIRONMENT-VALIDATION",
            "BENCHMARK_SPEC.md": "L4 Open-World Construction",
            "DATASET_CARD.md": "unit_verified",
        }
        for relative, needle in expected_doc_substrings.items():
            with self.subTest(doc=relative, needle=needle):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn(needle, text)

    def test_root_docs_do_not_claim_c3c4c5_as_active_v2_tasks(self) -> None:
        for relative in (
            "README.md",
            "PROJECT_STATUS.md",
            "ROADMAP.md",
            "BENCHMARK_SPEC.md",
            "DATASET_CARD.md",
        ):
            with self.subTest(doc=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotIn("Casting-S-C3 / C4 / C5", text)
                self.assertNotIn("active_compatibility_id", text)


if __name__ == "__main__":
    unittest.main()
