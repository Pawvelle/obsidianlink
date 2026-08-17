#!/usr/bin/env python3
"""Stage and optionally build the canonical P1 MineRL runtime.

The input must be the frozen, generated MCP-Reborn source tree.  This script
does not use MineRL's historical ``mcp_patch.diff`` and deliberately excludes
all E11 diagnostic, marshal, paused-executor, and E12 patches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "minerl"

CANONICAL_PATCHES = (
    "obsidianlink-envserver.patch",
    "p1-canonical-audio-source.patch",
    "e10-drawing-decorator.patch",
    "e11-drawing-decorator-obsidian.patch",
    "p1-env-integrated-server-unpaused.patch",
)

FROZEN_SOURCE_SHA256 = {
    "launchClient.sh": "7e15699c0d0aea517f87680eb5d760d02519d9744285fa0d348f799e2ed77183",
    "src/main/java/com/minerl/multiagent/env/EnvServer.java":
        "ea83babfbc764900ccda163d361ca9e9301b4c6103aa93fbe792b020742208e7",
    "src/main/java/net/minecraft/client/audio/SoundEngine.java":
        "ad3f9050335397e9ec6b9cf4acf3ae2fce28b64aa8e8fcab01426931bca59f36",
    "src/main/java/net/minecraft/server/integrated/IntegratedServer.java":
        "0a271b097d53ac6258b44054c9b252597232d38aba1cc714bb28fefd78a95563",
}

IGNORED_STAGE_NAMES = {
    ".gradle",
    "build",
    "crash-reports",
    "logs",
    "run",
    "saves",
}

FORBIDDEN_PRODUCTION_MARKERS = (
    "ObsidianLinkE11Task",
    "executeObsidianLinkE11Task",
    "marshalFlintAndSteelUseToServerThread",
    "queueFlintAndSteelUseToServerThread",
    "awaitPendingFlintAndSteelMarshal",
    "[E11-DIAG]",
    "[E11-MARSHAL]",
    "portal_transition",
    "PortalTransition",
    "entered_via_portal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_source(source_root: Path) -> None:
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"frozen source file is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen source hash mismatch for {relative}: {actual}"
            )


def _stage_source(source_root: Path, output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output root already exists: {output_root}")
    shutil.copytree(
        source_root,
        output_root,
        ignore=shutil.ignore_patterns(
            *IGNORED_STAGE_NAMES,
            "hs_err_pid*.log",
            "usercache.json",
            "options.txt",
        ),
    )


def _apply_patch(output_root: Path, patch_name: str) -> None:
    patch_path = PATCH_ROOT / patch_name
    completed = subprocess.run(
        [
            "patch",
            "-p1",
            "--forward",
            "--batch",
            "--no-backup-if-mismatch",
            "-i",
            str(patch_path),
        ],
        cwd=output_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"canonical patch failed: {patch_name}\n"
            f"{completed.stdout}{completed.stderr}"
        )


def validate_canonical_source(output_root: Path) -> None:
    java_root = output_root / "src/main/java"
    env_server = (
        java_root / "com/minerl/multiagent/env/EnvServer.java"
    ).read_text(encoding="utf-8")
    integrated_server = (
        java_root / "net/minecraft/server/integrated/IntegratedServer.java"
    ).read_text(encoding="utf-8")
    sound_engine = (
        java_root / "net/minecraft/client/audio/SoundEngine.java"
    ).read_text(encoding="utf-8")
    launcher = (output_root / "launchClient.sh").read_text(encoding="utf-8")

    combined = "\n".join(
        (env_server, integrated_server, sound_engine, launcher)
    )
    present = [
        marker for marker in FORBIDDEN_PRODUCTION_MARKERS if marker in combined
    ]
    if present:
        raise RuntimeError(
            "forbidden production markers present: " + ", ".join(present)
        )
    required = (
        "Blocks.LAVA.getDefaultState()",
        "Blocks.OBSIDIAN.getDefaultState()",
        "must not pre-place portal or fire",
        "this.mc.gameSettings.envPort == 0",
        "obsidianlink.disableClientAudio",
    )
    missing = [marker for marker in required if marker not in combined]
    if missing:
        raise RuntimeError(
            "canonical production markers missing: " + ", ".join(missing)
        )
    if "Blocks.NETHER_PORTAL" in env_server:
        raise RuntimeError("EnvServer must not create nether_portal blocks")


def stage_canonical_runtime(
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    validate_frozen_source(source_root)
    _stage_source(source_root, output_root)
    for patch_name in CANONICAL_PATCHES:
        _apply_patch(output_root, patch_name)
    validate_canonical_source(output_root)
    return {
        "canonical_patches": [
            {
                "name": name,
                "sha256": _sha256(PATCH_ROOT / name),
            }
            for name in CANONICAL_PATCHES
        ],
        "excluded_patch_families": [
            "e11-diagnostic",
            "e11-marshal",
            "e11-paused-executor",
            "e12",
            "mcp_patch.diff",
        ],
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "output_root": str(output_root),
        "source_root": str(source_root),
    }


def _build(output_root: Path) -> None:
    subprocess.run(
        ["./gradlew", "--no-daemon", "shadowJar"],
        cwd=output_root,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = stage_canonical_runtime(args.source_root, args.output_root)
    if args.build:
        _build(args.output_root.resolve())
        jar = args.output_root.resolve() / "build/libs/mcprec-6.13.jar"
        if not jar.is_file():
            raise RuntimeError(f"canonical runtime JAR is missing: {jar}")
        manifest["runtime_jar_sha256"] = _sha256(jar)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
