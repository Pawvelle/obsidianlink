from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDA_ENVIRONMENT = "mc-agent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.core.task_catalog import (  # noqa: E402
    load_task_catalog,
    validate_catalog_references,
)
from obsidianlink.env.validation import p1_validation_manifest  # noqa: E402


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
        "docs/architecture/V2_ARCHITECTURE.md",
        "docs/architecture/P1_ENVIRONMENT_VALIDATION.md",
        "docs/legacy/v1/PROJECT_STATUS_V1.md",
        "docs/legacy/v1/BENCHMARK_SPEC_V1.md",
        "pyproject.toml",
        "environment.yml",
        "model.lock.json",
        "obsidianlink/benchmark/task.py",
        "obsidianlink/benchmark/runner.py",
        "obsidianlink/benchmark/evaluator.py",
        "obsidianlink/benchmark/evidence.py",
        "obsidianlink/env/validation/contract.py",
        "obsidianlink/core/task_catalog.py",
        "benchmark/catalog/tasks.json",
        "benchmark/schemas/task_catalog.schema.json",
        "benchmark/schemas/task_instance.schema.json",
        "benchmark/schemas/v2_task_identity.schema.json",
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
        raise FileNotFoundError(f"expected the Conda Java executable next to Python: {java}")
    completed = subprocess.run(
        [java, "-version"], check=True, capture_output=True, text=True
    )
    first_line = (completed.stderr or completed.stdout).splitlines()[0]
    return {"executable": str(java), "version": first_line}


def _conda_environment() -> dict[str, object]:
    conda_prefix_value = os.environ.get("CONDA_PREFIX")
    conda_prefix = Path(conda_prefix_value).resolve() if conda_prefix_value else None
    python_executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    prefix_name = prefix.name
    active_name = os.environ.get("CONDA_DEFAULT_ENV")
    matches = (
        prefix_name == EXPECTED_CONDA_ENVIRONMENT
        and python_executable.parent == prefix / "bin"
        and (conda_prefix is None or conda_prefix == prefix)
        and (active_name is None or active_name == EXPECTED_CONDA_ENVIRONMENT)
    )
    return {
        "expected": EXPECTED_CONDA_ENVIRONMENT,
        "active": active_name,
        "prefix": str(prefix),
        "python_executable": str(python_executable),
        "matches_expected": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also import pinned runtime dependencies; never starts MineRL",
    )
    args = parser.parse_args()

    files = _check_project_files()
    conda_environment = _conda_environment()
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    validate_catalog_references(catalog, ROOT)
    result: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "conda_environment": conda_environment,
        "project_files": files,
        "phase": catalog.active_phase,
        "active_task": catalog.active_phase,
        "active_benchmark_task_id": catalog.active_benchmark_task_id,
        "verification_level": "unit_verified",
        "task_catalog_version": catalog.catalog_version,
        "task_catalog_entries": len(catalog.entries),
        "benchmark_visible_entries": len(catalog.benchmark_entries),
        "p1_validation_contract_ready": True,
        "p1_real_environment_executed": False,
        "p1_integration_verified": False,
        "p1_cases": list(p1_validation_manifest()),
        "live_run_allowed": False,
    }
    if args.runtime:
        result["runtime_versions"] = _runtime_versions()
        result["java"] = _java_version()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not all(files.values()) or not conda_environment["matches_expected"]:
        if not conda_environment["matches_expected"]:
            print(
                "error: run this repository with the Conda environment "
                f"{EXPECTED_CONDA_ENVIRONMENT!r}; use "
                "`conda run -n mc-agent python scripts/check_environment.py`.",
                file=sys.stderr,
            )
        return 1
    if sys.version_info[:2] != (3, 10):
        print(
            "warning: the validated project runtime is Python 3.10; "
            "standard-library checks may still run here.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
