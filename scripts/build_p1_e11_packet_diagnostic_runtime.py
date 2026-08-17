#!/usr/bin/env python3
"""Stage and optionally build the one-use E11 packet-chain logging runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BUILDER = ROOT / "scripts" / "build_p1_canonical_runtime.py"
DIAGNOSTIC_PATCH = "e11-packet-chain-diagnostic.patch"


def _canonical_builder():
    spec = importlib.util.spec_from_file_location("build_p1_canonical_runtime", CANONICAL_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("canonical runtime builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage_packet_diagnostic_runtime(canonical_runtime_root: Path, output_root: Path) -> dict[str, object]:
    builder = _canonical_builder()
    canonical_runtime_root = canonical_runtime_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output root already exists: {output_root}")
    launcher = canonical_runtime_root / "launchClient.sh"
    if not launcher.is_file() or builder._sha256(launcher) != builder.FROZEN_SOURCE_SHA256["launchClient.sh"]:
        raise RuntimeError("canonical runtime launcher identity mismatch")
    integrated = canonical_runtime_root / "src/main/java/net/minecraft/server/integrated/IntegratedServer.java"
    env_server = canonical_runtime_root / "src/main/java/com/minerl/multiagent/env/EnvServer.java"
    if not integrated.is_file() or not env_server.is_file():
        raise RuntimeError("canonical runtime sources are incomplete")
    if "this.mc.gameSettings.envPort == 0" not in integrated.read_text(encoding="utf-8"):
        raise RuntimeError("canonical runtime lacks the unpaused environment server patch")
    if "Blocks.OBSIDIAN.getDefaultState()" not in env_server.read_text(encoding="utf-8"):
        raise RuntimeError("canonical runtime lacks the E11 obsidian fixture")
    shutil.copytree(
        canonical_runtime_root,
        output_root,
        ignore=shutil.ignore_patterns(".gradle", "build", "run", "logs", "crash-reports", "hs_err_pid*.log"),
    )
    builder._apply_patch(output_root.resolve(), DIAGNOSTIC_PATCH)
    patch = ROOT / "patches" / "minerl" / DIAGNOSTIC_PATCH
    text = patch.read_text(encoding="utf-8")
    required = (
        "client_send packet=CPlayerTryUseItemOnBlockPacket",
        "server_received handler=processTryUseItemOnBlock",
        "server_player_interaction flint_enter",
        "flint_onItemUse light_fire",
        "fire_onBlockAdded canLightPortal=",
        "portal_place_enter",
    )
    if any(marker not in text for marker in required):
        raise RuntimeError("packet diagnostic patch is incomplete")
    added_lines = [line for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if any("Blocks.NETHER_PORTAL" in line for line in added_lines) or not any(
        "setBlockState(p_242967_2_, blockstate, 18);" in line
        for line in text.splitlines()
    ):
        raise RuntimeError("packet diagnostic patch must not alter portal placement semantics")
    manifest: dict[str, object] = {
        "canonical_runtime_root": str(canonical_runtime_root),
    }
    manifest.update(
        {
            "diagnostic_only": True,
            "diagnostic_patch": {
                "name": DIAGNOSTIC_PATCH,
                "sha256": builder._sha256(patch),
            },
            "diagnostic_log_markers": list(required),
        }
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-runtime-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = stage_packet_diagnostic_runtime(args.canonical_runtime_root, args.output_root)
    if args.build:
        builder = _canonical_builder()
        builder._build(args.output_root.resolve())
        jar = args.output_root.resolve() / "build/libs/mcprec-6.13.jar"
        if not jar.is_file():
            raise RuntimeError(f"diagnostic JAR is missing: {jar}")
        manifest["runtime_jar_sha256"] = builder._sha256(jar)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        args.manifest.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
