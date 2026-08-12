#!/usr/bin/env python3
"""Authorized one-shot live C1 MineRL smoke CLI.

Requires explicit:

  --mode authorized_live_c1
  --authorized-live-run casting_c1_fixed

This script never enables catalog ``live_run_allowed``. It does not accept
caller-supplied env factories or backends. Gradle and model APIs are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED_MINERL = ROOT / "vendor" / "minerl"
for import_root in (ROOT, VENDORED_MINERL):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from obsidianlink.runners.casting_c1_live import (
    AUTHORIZED_LIVE_RUN_VALUE,
    DEFAULT_WALL_CLOCK_SECONDS,
    EXECUTION_MODE_AUTHORIZED_LIVE_C1,
    allocate_live_run_dir,
    collect_runtime_preflight,
    preflight_authorized_c1_live,
    run_casting_c1_authorized_live,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one authorized casting_c1_fixed MineRL/Minecraft smoke. "
            "Catalog live_run_allowed remains false."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[EXECUTION_MODE_AUTHORIZED_LIVE_C1],
        help="must be authorized_live_c1",
    )
    parser.add_argument(
        "--authorized-live-run",
        required=True,
        choices=[AUTHORIZED_LIVE_RUN_VALUE],
        help="must be casting_c1_fixed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "optional absolute directory under runs/casting_c1_fixed/; "
            "default allocates a fresh timestamped directory"
        ),
    )
    parser.add_argument(
        "--wall-clock-seconds",
        type=int,
        default=DEFAULT_WALL_CLOCK_SECONDS,
        help="hard external wall-clock timeout (default 900)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="read-only runtime preflight; do not start MineRL",
    )
    parser.add_argument(
        "--allow-gradle",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--request-model",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.allow_gradle:
        print("error: Gradle is not authorized", file=sys.stderr)
        return 2
    if args.request_model:
        print("error: model API is not authorized", file=sys.stderr)
        return 2

    if args.preflight_only:
        payload = collect_runtime_preflight(dry_run=True)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 1 if payload.get("gradle_needed") else 0

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else allocate_live_run_dir()
    )
    # Validate authorization before starting Minecraft.
    preflight_authorized_c1_live(
        output_dir=output_dir,
        execution_mode=args.mode,
        authorized_live_run=args.authorized_live_run,
        allow_gradle=False,
        request_model=False,
        wall_clock_seconds=args.wall_clock_seconds,
    )

    result = run_casting_c1_authorized_live(
        output_dir=output_dir,
        execution_mode=args.mode,
        authorized_live_run=args.authorized_live_run,
        allow_gradle=False,
        request_model=False,
        wall_clock_seconds=args.wall_clock_seconds,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if result.overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
