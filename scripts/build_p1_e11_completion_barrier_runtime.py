#!/usr/bin/env python3
"""Stage and optionally build the canonical E11 server-completion barrier runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "minerl" / "p1-e11-action-completion-barrier.patch"
CANONICAL_JAR_SHA256 = "684c20ec533897b44e9f2f73340f66ab41a6f61e7c9ae7e0f1db6fae7430751e"
IGNORED = (".gradle", "build", "run", "logs", "crash-reports", "hs_err_pid*.log")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_patch(root: Path) -> None:
    result = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "--no-backup-if-mismatch", "-i", str(PATCH)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def stage_completion_barrier_runtime(canonical_root: Path, output_root: Path) -> dict[str, object]:
    canonical_root = canonical_root.resolve()
    output_root = output_root.resolve()
    baseline = canonical_root / "build/libs/mcprec-6.13.jar"
    if not baseline.is_file() or sha256(baseline) != CANONICAL_JAR_SHA256:
        raise RuntimeError("canonical baseline JAR identity mismatch")
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output root already exists: {output_root}")
    shutil.copytree(canonical_root, output_root, ignore=shutil.ignore_patterns(*IGNORED))
    apply_patch(output_root)
    source = "\n".join(
        (output_root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/main/java/com/minerl/multiagent/env/EnvServer.java",
            "src/main/java/net/minecraft/client/ReplaySender.java",
            "src/main/java/net/minecraft/network/play/ServerPlayNetHandler.java",
        )
    )
    required = (
        "E11_ACTION_COMPLETION_MONITOR",
        "awaitE11FlintAndSteelCompletionBarrier",
        "isE11FlintAndSteelCompletionBarrierPending",
        "completeE11FlintAndSteelCompletionBarrier",
        "EnvServer.isE11FlintAndSteelCompletionBarrierPending()",
    )
    if any(marker not in source for marker in required):
        raise RuntimeError("completion barrier source is incomplete")
    forbidden = ("Blocks.NETHER_PORTAL", "PortalSize", "FlintAndSteelItem", "Entity.java", "ServerPlayerEntity.java")
    if any(marker in PATCH.read_text(encoding="utf-8") for marker in forbidden):
        raise RuntimeError("completion barrier patch exceeds its allowed execution boundary")
    return {
        "baseline_jar_sha256": CANONICAL_JAR_SHA256,
        "canonical_root": str(canonical_root),
        "completion_barrier_patch": {"name": PATCH.name, "sha256": sha256(PATCH)},
        "output_root": str(output_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = stage_completion_barrier_runtime(args.canonical_root, args.output_root)
    if args.build:
        subprocess.run(
            ["./gradlew", "--no-daemon", "shadowJar", "-x", "jaxb"],
            cwd=args.output_root.resolve(),
            check=True,
        )
        jar = args.output_root.resolve() / "build/libs/mcprec-6.13.jar"
        if not jar.is_file():
            raise RuntimeError("completion barrier JAR is missing")
        manifest["runtime_jar_sha256"] = sha256(jar)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
