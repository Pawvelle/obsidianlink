#!/usr/bin/env python3
"""CLI entry point for offline C1 live-smoke validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from obsidianlink.runners.casting_c1_live_smoke import (
    EXECUTION_MODE_OFFLINE_STUB,
    build_offline_stub_env_factory,
    run_casting_c1_live_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline C1 live-smoke runner with a reactive stub "
            "env_factory. This command never starts MineRL or Minecraft."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[EXECUTION_MODE_OFFLINE_STUB],
        help="execution mode (offline_stub only)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="absolute output directory outside formal runs/",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live:
        print("error: live MineRL execution is not supported", file=sys.stderr)
        return 1
    if "--live" in sys.argv:
        print("error: live MineRL execution is not supported", file=sys.stderr)
        return 1

    result = run_casting_c1_live_smoke(
        output_dir=args.output_dir,
        execution_mode=args.mode,
        env_factory=build_offline_stub_env_factory(),
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    if (
        result.evaluator_success
        and result.evidence_complete
        and result.driver_completed
        and result.close_status == "closed"
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
