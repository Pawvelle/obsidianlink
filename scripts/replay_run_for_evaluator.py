"""Compare the Phase 2 evaluator verdict with the historical Scripted-A0 run.

This script does NOT touch MineRL. It reads ``events.jsonl`` and
``summary.json`` from a historical run and reports whether the new
Phase 2 evaluator can be replayed against the recorded evidence.

The Phase 1 historical run is a real MineRL capture that intentionally
collapses several block names to ``other`` and does not include a
per-step grid. Phase 2 cannot rebuild a frame identity from those
artifacts alone, so this script does **not** fabricate a successful
``EvaluationState`` to make the auto evaluator agree with the manual
review. Instead, it reports ``status=insufficient_evidence`` and lists
the specific facts that the new evaluator requires but the historical
artefacts do not contain.

Usage:

    /opt/anaconda3/envs/mc-agent/bin/python scripts/replay_run_for_evaluator.py \\
        --run-dir runs/history/phase1-scripted-a0/20260730-214356
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_action(
    events: list[dict[str, Any]],
    action_type: str,
    target: str,
) -> int:
    return sum(
        1
        for event in events
        if event.get("action_type") == action_type
        and event.get("target") == target
        and event.get("translation_accepted", True)
    )


def _first_step_for(
    events: list[dict[str, Any]],
    predicate,
) -> int | None:
    for event in events:
        if predicate(event):
            return int(event.get("step_id", 0))
    return None


def replay(run_dir: Path) -> dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    if not events_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"missing events.jsonl or summary.json in {run_dir}"
        )
    events = _load_jsonl(events_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = summary.get("evaluation_evidence", {})

    # ------------------------------------------------------------------
    # Recorded counts (these *are* trustworthy because they come from
    # the MineRL stats / bridge). They let us confirm the run
    # *happened* the way the Phase 1 manual review described, but they
    # are NOT enough to drive the Phase 2 evaluator.
    # ------------------------------------------------------------------
    obsidian_placements = _count_action(events, "place_block", "obsidian")
    dirt_placements = _count_action(events, "place_block", "dirt")
    flint_uses = _count_action(events, "use_item", "flint_and_steel")
    first_obsidian_step = _first_step_for(
        events,
        lambda e: e.get("action_type") == "place_block"
        and e.get("target") == "obsidian",
    )
    last_obsidian_step = (
        _first_step_for(
            list(reversed(events)),
            lambda e: e.get("action_type") == "place_block"
            and e.get("target") == "obsidian",
        )
    )
    first_flint_step = _first_step_for(
        events,
        lambda e: e.get("action_type") == "use_item"
        and e.get("target") == "flint_and_steel",
    )
    last_step = int(events[-1].get("step_id", 0)) if events else 0
    summary_entered_nether = bool(summary.get("entered_nether"))
    final_dimension = evidence.get("dimension")
    max_obsidian_added = evidence.get("max_obsidian_added")
    portal_activated_latched = evidence.get("portal_activated_latched")

    # ------------------------------------------------------------------
    # Phase 2 requires more than the historical artefacts provide.
    # Specifically, the new evaluator needs:
    #
    # 1. Per-step portal grid snapshots (not just an aggregated
    #    grid_changes list) so the latched frame identity can be
    #    reconstructed and the attribution queue can be matched to
    #    observed obsidian deltas.
    # 2. A per-action obsidian attribution stream: the historical
    #    events record which step issued a place_block(obsidian)
    #    but not which grid cell that action produced. The new
    #    evaluator refuses to count unattributed obsidian toward
    #    ``portal_built_by_episode``.
    # 3. A precise Nether transition step for each agent, plus the
    #    agent's pre-transition Overworld position, so the
    #    evaluator can confirm the agent stepped through the
    #    latched portal rather than an external teleport.
    # 4. An explicit termination signal so the evaluator can
    #    classify failures without inventing them.
    # 5. ``portal_grid_changes`` must preserve ``obsidian`` and
    #    ``fire`` block names. The historical bridge collapses
    #    them to ``other`` and that is not enough.
    # ------------------------------------------------------------------
    missing_evidence: list[str] = []
    if "portal_grid" not in evidence and "portal_grid_changes" not in evidence:
        missing_evidence.append("per-step portal_grid snapshots")
    if first_obsidian_step is None or last_obsidian_step is None:
        missing_evidence.append("obsidian placement step range")
    if first_flint_step is None:
        missing_evidence.append("flint_and_steel activation step")
    if final_dimension != "minecraft:the_nether":
        missing_evidence.append("final dimension == minecraft:the_nether")
    if max_obsidian_added != obsidian_placements:
        missing_evidence.append(
            "max_obsidian_added must equal the number of obsidian "
            "place_block events"
        )
    # The bridge normalises obsidian / fire to ``other``; the new
    # detector refuses to consider a cell whose truth is missing.
    changes = evidence.get("portal_grid_changes", [])
    obsidian_normalised = any(
        c.get("after") == "obsidian" for c in changes
    )
    if not obsidian_normalised:
        missing_evidence.append(
            "portal_grid_changes preserves obsidian / fire block "
            "names (currently normalised to 'other')"
        )
    # Phase 2 attribution: per-action obsidian cell coordinates.
    missing_evidence.append(
        "per-action obsidian cell coordinates so each "
        "place_block(obsidian) can be attributed to its target cell "
        "(the historical events only record the action step, not "
        "the resulting cell)"
    )
    # Phase 2 Nether-entry correlation: the agent's pre-transition
    # Overworld position is required.
    missing_evidence.append(
        "agent pre-transition Overworld position for Nether-entry "
        "portal correlation (events.jsonl only records the final "
        "dimension, not when the transition happened or where the "
        "agent was standing)"
    )
    # Explicit termination signal.
    if "terminated_step" not in evidence and "status" not in summary:
        missing_evidence.append(
            "explicit episode termination signal "
            "(terminated_step / terminated_reason)"
        )

    return {
        "run_dir": str(run_dir),
        "status": "insufficient_evidence",
        "phase2_evaluator_replay_supported": False,
        "agreed_with_manual_review": False,
        "rationale": (
            "Phase 2 evaluator refuses to attribute episode-built "
            "frames to unattributed obsidian, refuses to confirm "
            "Nether entry without a pre-transition Overworld "
            "position, and refuses to classify a failure without an "
            "explicit termination signal. The historical events / "
            "summary do not provide per-step grid snapshots, "
            "obsidian / fire block names, or pre-transition "
            "positions, so the new evaluator cannot honestly verify "
            "this run."
        ),
        "recorded_facts": {
            "obsidian_placements": obsidian_placements,
            "dirt_placements": dirt_placements,
            "flint_and_steel_uses": flint_uses,
            "first_obsidian_step": first_obsidian_step,
            "last_obsidian_step": last_obsidian_step,
            "first_flint_step": first_flint_step,
            "last_step": last_step,
            "evidence_max_obsidian_added": max_obsidian_added,
            "evidence_portal_activated_latched": portal_activated_latched,
            "evidence_final_dimension": final_dimension,
            "summary_entered_nether": summary_entered_nether,
        },
        "missing_evidence": missing_evidence,
        "phase1_manual_review_unaffected": True,
        "phase1_manual_review_path": "manual_review.md",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/history/phase1-scripted-a0/20260730-214356"),
    )
    args = parser.parse_args(argv)
    result = replay(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
