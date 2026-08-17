#!/usr/bin/env python3
"""Fail closed unless an E11 packet diagnostic JAR changes logging targets only."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Sequence


EXPECTED_CHANGED_ENTRIES = {
    "net/minecraft/client/multiplayer/PlayerController.class",
    "net/minecraft/network/play/ServerPlayNetHandler.class",
    # javac regenerates this switch-map helper when the enclosing handler's
    # source line table changes; it contains no new interaction logic.
    "net/minecraft/network/play/ServerPlayNetHandler$1.class",
    "net/minecraft/server/management/PlayerInteractionManager.class",
    "net/minecraft/item/FlintAndSteelItem.class",
    "net/minecraft/block/AbstractFireBlock.class",
    "net/minecraft/block/PortalSize.class",
    "version.properties",
}

REQUIRED_LOG_MARKERS = {
    "net/minecraft/client/multiplayer/PlayerController.class": b"client_send packet=CPlayerTryUseItemOnBlockPacket",
    "net/minecraft/network/play/ServerPlayNetHandler.class": b"server_received handler=processTryUseItemOnBlock",
    "net/minecraft/server/management/PlayerInteractionManager.class": b"server_player_interaction flint_enter",
    "net/minecraft/item/FlintAndSteelItem.class": b"flint_onItemUse light_fire",
    "net/minecraft/block/AbstractFireBlock.class": b"fire_onBlockAdded canLightPortal=",
    "net/minecraft/block/PortalSize.class": b"portal_place_enter",
}

FORBIDDEN_MARKERS = (
    b"ObsidianLinkE11Task",
    b"executeObsidianLinkE11Task",
    b"E11-MARSHAL",
    b"portal_transition",
    b"entered_via_portal",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_packet_diagnostic_jars(baseline: Path, candidate: Path) -> dict[str, object]:
    with zipfile.ZipFile(baseline) as base_zip, zipfile.ZipFile(candidate) as candidate_zip:
        baseline_info = {info.filename: info for info in base_zip.infolist()}
        candidate_info = {info.filename: info for info in candidate_zip.infolist()}
        names = set(baseline_info) | set(candidate_info)
        changed = sorted(
            name
            for name in names
            if name not in baseline_info
            or name not in candidate_info
            or (baseline_info[name].CRC, baseline_info[name].file_size)
            != (candidate_info[name].CRC, candidate_info[name].file_size)
        )
        semantic_changed = sorted(
            name
            for name in changed
            if name == "version.properties" or name.startswith(("com/minerl/", "net/minecraft/"))
        )
        unexpected = sorted(set(semantic_changed) - EXPECTED_CHANGED_ENTRIES)
        missing = sorted(EXPECTED_CHANGED_ENTRIES - set(semantic_changed))
        marker_hits = {
            name: [marker.decode("ascii") for marker in FORBIDDEN_MARKERS if marker in candidate_zip.read(name)]
            for name in semantic_changed
            if name.endswith(".class") and any(marker in candidate_zip.read(name) for marker in FORBIDDEN_MARKERS)
        }
        missing_logs = sorted(
            name for name, marker in REQUIRED_LOG_MARKERS.items() if marker not in candidate_zip.read(name)
        )
    result = {
        "baseline_jar_sha256": _sha256(baseline),
        "candidate_jar_sha256": _sha256(candidate),
        "expected_changed_entries": sorted(EXPECTED_CHANGED_ENTRIES),
        "changed_semantic_entries": semantic_changed,
        "unexpected_changed_entries": unexpected,
        "missing_expected_changed_entries": missing,
        "missing_required_log_markers": missing_logs,
        "forbidden_marker_hits": marker_hits,
        "semantic_diff_clean": not unexpected and not missing and not missing_logs and not marker_hits,
    }
    if not result["semantic_diff_clean"]:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_packet_diagnostic_jars(args.baseline.resolve(), args.candidate.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
