from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.core.task_catalog import (  # noqa: E402
    load_task_catalog,
    validate_catalog_references,
)


def _check_project_files() -> dict[str, bool]:
    required = (
        "PROJECT_STATUS.md",
        "AGENTS.md",
        "README.md",
        "ROADMAP.md",
        "BENCHMARK_SPEC.md",
        "DATASET_CARD.md",
        "docs/benchmark/TASK_TAXONOMY.md",
        "docs/architecture/TASK_REGISTRY.md",
        "docs/tasks/casting/casting_c1_fixed.md",
        "docs/tasks/casting/casting_c3_fixed.md",
        "docs/tasks/casting/casting_s_c3_fixed.md",
        "docs/tasks/casting/casting_s_c4_fixed.md",
        "docs/tasks/casting/casting_s_c5_fixed.md",
        "pyproject.toml",
        "environment.yml",
        "model.lock.json",
        "obsidianlink/__init__.py",
        "obsidianlink/core/task_catalog.py",
        "benchmark/catalog/tasks.json",
        "benchmark/schemas/task_catalog.schema.json",
        "benchmark/schemas/task_instance.schema.json",
        "benchmark/instances/active/casting_c1_fixed.json",
        "benchmark/instances/active/casting_c3_fixed.json",
        "benchmark/instances/casting/single/casting_s_c3_fixed.json",
        "benchmark/instances/casting/single/casting_s_c4_fixed.json",
        "benchmark/instances/casting/single/casting_s_c5_fixed.json",
        "configs/experiments/active/casting_c1_contract.json",
        "configs/experiments/active/casting_c3_contract.json",
        "configs/experiments/active/casting_s_c3_contract.json",
        "configs/experiments/active/casting_s_c4_contract.json",
        "configs/experiments/active/casting_s_c5_contract.json",
        "docs/runbooks/FIRST_OBSIDIAN_BLOCK.md",
    )
    return {path: (ROOT / path).is_file() for path in required}


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "gym", "torch", "transformers"):
        module = importlib.import_module(module_name)
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _java_version() -> dict[str, str]:
    java = Path(sys.executable).parent / "java"
    if not java.is_file():
        raise FileNotFoundError(
            f"expected the Conda Java executable next to Python: {java}"
        )
    completed = subprocess.run(
        [java, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = (completed.stderr or completed.stdout).splitlines()[0]
    return {
        "executable": str(java),
        "version": first_line,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also import the pinned MineRL model runtime dependencies",
    )
    args = parser.parse_args()

    files = _check_project_files()
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    validate_catalog_references(catalog, ROOT)
    active_entry = catalog.active_entry
    if active_entry.taxonomy is None:  # guarded by TaskCatalog
        raise RuntimeError("active task catalog entry must have taxonomy")
    taxonomy = active_entry.taxonomy.as_dict()
    taxonomy["compatibility_task_name"] = active_entry.canonical_name
    result: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "project_files": files,
        "phase": "r6_c1_player_relative_truth_grid_anchor_offline_fix_complete",
        "active_task": active_entry.compatibility_id,
        "task_taxonomy": taxonomy,
        "task_catalog_version": catalog.catalog_version,
        "task_catalog_entries": len(catalog.entries),
        "live_run_allowed": False,
    }
    if args.runtime:
        result["runtime_versions"] = _runtime_versions()
        result["java"] = _java_version()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not all(files.values()):
        return 1
    if sys.version_info[:2] != (3, 10):
        print(
            "warning: the validated project runtime is Python 3.10; "
            "Phase 0 standard-library checks may still run here.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
