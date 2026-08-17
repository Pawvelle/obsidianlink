#!/usr/bin/env python3
"""Fail closed unless a canonical P1 JAR has the expected semantic diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Sequence


ALLOWED_CHANGED_ENTRIES = {
    "net/minecraft/server/integrated/IntegratedServer.class",
    "version.properties",
}

REQUIRED_IDENTICAL_ENTRIES = {
    "net/minecraft/block/AbstractFireBlock.class",
    "net/minecraft/block/NetherPortalBlock.class",
    "net/minecraft/item/FlintAndSteelItem.class",
    "net/minecraft/block/PortalSize.class",
    "net/minecraft/entity/Entity.class",
    "net/minecraft/entity/player/ServerPlayerEntity.class",
    "com/minerl/multiagent/env/EnvServer.class",
    "net/minecraft/client/audio/SoundEngine.class",
}

FORBIDDEN_CHANGED_PREFIXES = (
    "net/minecraft/entity/Entity",
    "net/minecraft/entity/player/ServerPlayerEntity",
)

FORBIDDEN_CANDIDATE_MARKERS = (
    b"ObsidianLinkE11Task",
    b"executeObsidianLinkE11Task",
    b"E11-MARSHAL",
    b"E11-DIAG",
    b"portal_transition",
    b"entered_via_portal",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_runtime_jars(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    with zipfile.ZipFile(baseline_path) as baseline_zip, zipfile.ZipFile(
        candidate_path
    ) as candidate_zip:
        baseline_names = set(baseline_zip.namelist())
        candidate_names = set(candidate_zip.namelist())
        all_names = baseline_names | candidate_names
        changed = sorted(
            name
            for name in all_names
            if name not in baseline_names
            or name not in candidate_names
            or baseline_zip.read(name) != candidate_zip.read(name)
        )
        semantic_changed = sorted(
            name
            for name in changed
            if name == "version.properties"
            or name.startswith("com/minerl/")
            or name.startswith("net/minecraft/")
        )
        nonsemantic_changed = sorted(set(changed) - set(semantic_changed))
        unexpected = sorted(
            set(semantic_changed) - ALLOWED_CHANGED_ENTRIES
        )
        forbidden_changes = sorted(
            name
            for name in changed
            if name.startswith(FORBIDDEN_CHANGED_PREFIXES)
        )
        identical = {
            name: (
                name in baseline_names
                and name in candidate_names
                and baseline_zip.read(name) == candidate_zip.read(name)
            )
            for name in sorted(REQUIRED_IDENTICAL_ENTRIES)
        }
        marker_hits: dict[str, list[str]] = {}
        for name in sorted(candidate_names):
            if not name.endswith(".class"):
                continue
            data = candidate_zip.read(name)
            hits = [
                marker.decode("ascii")
                for marker in FORBIDDEN_CANDIDATE_MARKERS
                if marker in data
            ]
            if hits:
                marker_hits[name] = hits

    clean = (
        not unexpected
        and not forbidden_changes
        and all(identical.values())
        and not marker_hits
        and "net/minecraft/server/integrated/IntegratedServer.class" in changed
    )
    result = {
        "allowed_changed_entries": sorted(ALLOWED_CHANGED_ENTRIES),
        "baseline_jar_sha256": _sha256(baseline_path.read_bytes()),
        "candidate_jar_sha256": _sha256(candidate_path.read_bytes()),
        "changed_semantic_entries": semantic_changed,
        "nonsemantic_packaging_diff_count": len(nonsemantic_changed),
        "nonsemantic_packaging_diff_sha256": _sha256(
            "\n".join(nonsemantic_changed).encode("utf-8")
        ),
        "forbidden_candidate_markers": marker_hits,
        "forbidden_entity_or_server_player_changes": forbidden_changes,
        "required_entries_identical": identical,
        "semantic_diff_clean": clean,
        "unexpected_changed_entries": unexpected,
    }
    if not clean:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_runtime_jars(
        args.baseline.resolve(),
        args.candidate.resolve(),
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
