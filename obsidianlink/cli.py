from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from obsidianlink.benchmark.evidence import VerificationLevel
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    ExecutionMode,
    LayoutType,
    TaskIdentity,
)
from obsidianlink.core.task_catalog import load_task_catalog, validate_catalog_references
from obsidianlink.env.validation import P1_VALIDATION_CASES, p1_validation_manifest


ROOT = Path(__file__).resolve().parents[1]
TASK_CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"


def _offline_contract_check() -> dict[str, object]:
    """Validate the v2/P1 offline contract without executing a solver."""

    catalog = load_task_catalog(TASK_CATALOG_PATH)
    validate_catalog_references(catalog, ROOT)
    if catalog.benchmark_entries:
        raise RuntimeError("P1 scope freeze must not expose benchmark task entries")
    if len(P1_VALIDATION_CASES) != 13:
        raise RuntimeError("P1 validation checklist must contain E0 through E12")
    expected_ids = tuple(f"E{index}" for index in range(13))
    actual_ids = tuple(case.check_id.value for case in P1_VALIDATION_CASES)
    if actual_ids != expected_ids:
        raise RuntimeError("P1 validation checklist order must be E0 through E12")

    # Pure taxonomy construction proves the v2 identity layer imports cleanly.
    example = TaskIdentity(
        task_instance_id="taxonomy_example_not_registered",
        suite=BenchmarkSuite.END_TO_END,
        mode=ExecutionMode.SINGLE,
        level="L1",
        layout=LayoutType.CONTROLLED,
    )

    return {
        "status": "ok",
        "phase": catalog.active_phase,
        "active_task": catalog.active_phase,
        "active_benchmark_task_id": catalog.active_benchmark_task_id,
        "verification_level": VerificationLevel.UNIT_VERIFIED.value,
        "task_catalog_version": catalog.catalog_version,
        "task_catalog_entries": len(catalog.entries),
        "benchmark_visible_entries": len(catalog.benchmark_entries),
        "legacy_entries": sum(entry.kind == "legacy" for entry in catalog.entries),
        "calibration_entries": sum(
            entry.kind == "calibration" for entry in catalog.entries
        ),
        "live_run_allowed": False,
        "v2_taxonomy_example": example.as_dict(),
        "p1_validation": {
            "contract_ready": True,
            "real_execution_performed": False,
            "integration_verified": False,
            "cases": list(p1_validation_manifest()),
        },
        "note": (
            "Offline v2 schema, registry, kernel boundary, and P1 E0-E12 manifest "
            "validation only. FakeBackend and deterministic legacy drivers were not "
            "executed. No MineRL/Minecraft, Gradle, or paid model API call was made. "
            "E10 offline contract is unit_verified; E10 real conversion reviewed "
            "success is YES and is not integration_verified. E11 offline contract "
            "is unit_verified; E11 geometry real verified is YES and is not "
            "integration_verified. The clean canonical runtime excludes old "
            "marshal, paused-executor, diagnostic, and E12 patches. A separate "
            "logging-only E11 diagnostic proved normal client packet delivery and "
            "server-side FlintAndSteelItem/canLightPortal/placePortalBlocks, but "
            "those events occur after replay stop and the frozen evaluator still "
            "observed fire and 0/6 nether_portal with truth_missing_count=0; "
            "outcome portal_activation_not_observed. "
            "E11 real reviewed success is NO and is not integration_verified. "
            "E12 and end-to-end portal "
            "construction remain unverified in the real "
            "environment."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obsidianlink",
        description="ObsidianLink v2 benchmark development utilities.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate v2 scope and P1 offline contracts without starting MineRL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check:
        build_parser().print_help()
        return 0
    print(json.dumps(_offline_contract_check(), ensure_ascii=False, sort_keys=True))
    return 0
