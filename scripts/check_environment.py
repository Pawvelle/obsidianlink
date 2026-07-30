from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _check_project_files() -> dict[str, bool]:
    required = (
        "README.md",
        "ROADMAP.md",
        "BENCHMARK_SPEC.md",
        "DATASET_CARD.md",
        "pyproject.toml",
        "environment.yml",
        "model.lock.json",
        "obsidianlink/__init__.py",
        "benchmark/schemas/task_instance.schema.json",
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
    result: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "project_files": files,
        "phase": "phase_0_clean_core",
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
