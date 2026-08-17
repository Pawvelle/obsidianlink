"""Explicit already-built JAR selection for the P1 E0--E12 suite.

This is not a runtime manager. Each check_id maps to one frozen SHA-256.
The helper copies that already-built ``mcprec-6.13.jar`` into the deployed
slot only after hashing. It never runs Gradle or launches MineRL.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping

DEPLOYED_JAR_NAME = "mcprec-6.13.jar"
RUNTIME_CANONICAL = "canonical"
RUNTIME_E11_COMPLETION_BARRIER = "e11_completion_barrier"
RUNTIME_E12_PORTAL_FIXTURE = "e12_portal_fixture"
# Frozen identities: canonical from scripts/build_p1_e11_completion_barrier_runtime.py
# and docs/architecture/P1_CANONICAL_RUNTIME.md; E11/E12 from the existing
# deployed-JAR records in e11_run.py / e12_run.py.
CANONICAL_JAR_SHA256 = "684c20ec533897b44e9f2f73340f66ab41a6f61e7c9ae7e0f1db6fae7430751e"
E11_COMPLETION_BARRIER_JAR_SHA256 = (
    "6b5705e49220f5af33b5b0d06f7c162afef501a849d54cf57b242933bfd3ef72"
)
E12_PORTAL_FIXTURE_JAR_SHA256 = (
    "f459c36b7aaacd7e5f98ff9bbe001f1d54e77b73740537c24d5c5540290d36f4"
)

RUNTIME_SHA256 = {
    RUNTIME_CANONICAL: CANONICAL_JAR_SHA256,
    RUNTIME_E11_COMPLETION_BARRIER: E11_COMPLETION_BARRIER_JAR_SHA256,
    RUNTIME_E12_PORTAL_FIXTURE: E12_PORTAL_FIXTURE_JAR_SHA256,
}


class P1SuiteRuntimeError(ValueError):
    """Raised when the required already-built JAR is missing or mismatched."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_runtime(check_id: str) -> tuple[str, str]:
    """Return ``(runtime_name, sha256)`` for one P1 validation case."""

    if check_id == "E11":
        name = RUNTIME_E11_COMPLETION_BARRIER
    elif check_id == "E12":
        name = RUNTIME_E12_PORTAL_FIXTURE
    elif check_id in {f"E{index}" for index in range(11)}:
        name = RUNTIME_CANONICAL
    else:
        raise P1SuiteRuntimeError(f"unknown P1 suite check_id: {check_id!r}")
    return name, RUNTIME_SHA256[name]


def deployed_mcp_libs() -> Path:
    from obsidianlink.env.integration.native_runtime import discover_minerl_path

    return (discover_minerl_path() / "MCP-Reborn" / "build" / "libs").resolve()


def _candidate_jars(mcp_libs: Path, sha256: str) -> tuple[Path, ...]:
    prefix = sha256[:8]
    ordered: list[Path] = []
    active = mcp_libs / DEPLOYED_JAR_NAME
    if active.is_file():
        ordered.append(active)
    backups = mcp_libs / "backups"
    if backups.is_dir():
        for path in sorted(backups.glob(f"{DEPLOYED_JAR_NAME}*")):
            if path.is_file() and prefix in path.name:
                ordered.append(path)
    return tuple(ordered)


def find_verified_jar(mcp_libs: Path, sha256: str) -> Path:
    for path in _candidate_jars(mcp_libs, sha256):
        if _sha256_file(path) == sha256:
            return path
    raise P1SuiteRuntimeError(
        f"required already-built JAR sha256 {sha256} was not found or did not match"
    )


def activate_required_runtime(
    check_id: str,
    *,
    mcp_libs: Path | None = None,
) -> dict[str, Any]:
    """Make the required already-built JAR active. Never builds."""

    name, expected = required_runtime(check_id)
    libs = (mcp_libs or deployed_mcp_libs()).resolve()
    active = libs / DEPLOYED_JAR_NAME
    source = find_verified_jar(libs, expected)
    already_active = source.resolve() == active.resolve()
    if not already_active:
        if active.is_file():
            current = _sha256_file(active)
            backups = libs / "backups"
            backups.mkdir(parents=True, exist_ok=True)
            backup = backups / f"{DEPLOYED_JAR_NAME}.p1-suite-{current[:8]}"
            if not backup.exists():
                shutil.copy2(active, backup)
        temporary = active.with_name(f"{DEPLOYED_JAR_NAME}.p1-suite-tmp")
        shutil.copy2(source, temporary)
        copied = _sha256_file(temporary)
        if copied != expected:
            temporary.unlink(missing_ok=True)
            raise P1SuiteRuntimeError(
                f"copied JAR sha256 {copied} does not match required {expected}"
            )
        temporary.replace(active)
    if _sha256_file(active) != expected:
        raise P1SuiteRuntimeError("active JAR sha256 mismatch")
    return {
        "already_active": already_active,
        "check_id": check_id,
        "gradle_invoked": False,
        "jar_path": str(active.resolve()),
        "runtime": name,
        "sha256": expected,
        "source_path": str(source.resolve()),
        "verified": True,
    }


def runtime_record(check_id: str) -> dict[str, Any]:
    """Offline identity record. Does not inspect or copy JARs."""

    name, sha256 = required_runtime(check_id)
    return {
        "check_id": check_id,
        "gradle_invoked": False,
        "runtime": name,
        "sha256": sha256,
        "verified": False,
    }


def failed_runtime_record(check_id: str, error: BaseException) -> dict[str, Any]:
    record = runtime_record(check_id)
    record["error"] = str(error)
    record["verified"] = False
    return record


def is_verified_runtime(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("verified") is True
        and value.get("gradle_invoked") is False
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and isinstance(value.get("runtime"), str)
        and isinstance(value.get("check_id"), str)
    )
