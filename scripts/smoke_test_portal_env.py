from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
VENDORED_MINERL = ROOT / "vendor/minerl"
for import_root in (ROOT, VENDORED_MINERL):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import PortalA0EnvSpec


TASK_PATH = ROOT / "benchmark/instances/route_a_a0_development.json"


def _task() -> TaskInstance:
    return TaskInstance.from_dict(
        json.loads(TASK_PATH.read_text(encoding="utf-8"))
    )


def _spec_result() -> dict[str, Any]:
    specification = PortalA0EnvSpec(max_episode_steps=500)
    xml = specification.to_xml()
    return {
        "status": "passed",
        "mode": "spec",
        "env_name": specification.name,
        "xml_length": len(xml),
        "observation_keys": sorted(specification.observation_space.spaces),
        "action_keys": sorted(specification.action_space.spaces),
        "has_portal_grid": "portal_grid" in specification.observation_space.spaces,
        "has_portal_items": all(
            item in xml for item in ("obsidian", "flint_and_steel")
        ),
        "real_minerl_started": False,
    }


def _exercise_actions() -> Iterable[MacroAction]:
    return (
        MacroAction("look", parameters={"pitch": 30.0}),
        MacroAction("look", parameters={"pitch": 15.0}),
        MacroAction("equip_item", target="obsidian"),
        MacroAction.wait(),
        MacroAction("place_block", target="obsidian"),
        MacroAction.wait(),
        MacroAction("equip_item", target="flint_and_steel"),
        MacroAction.wait(),
        MacroAction("use_item", target="flint_and_steel"),
        MacroAction.wait(),
        MacroAction("equip_item", target="obsidian"),
        MacroAction.wait(),
        MacroAction("mine_target", target="aimed_block"),
        MacroAction.wait(),
    )


def _real_result(steps: int, exercise_actions: bool) -> dict[str, Any]:
    backend = MineRLEnvironmentBackend()
    backend.open()
    try:
        observations = backend.reset(_task())
        initial_observation = observations["agent_1"]
        completed = 0
        terminated = False
        action_results: list[dict[str, Any]] = []
        actions = (
            list(_exercise_actions())
            if exercise_actions
            else [MacroAction.wait() for _ in range(steps)]
        )
        for action in actions:
            step = backend.step({"agent_1": action})
            completed += 1
            terminated = step.terminated
            action_results.append(
                {
                    "action_type": action.action_type,
                    "target": action.target,
                    "translation_accepted": step.info["translation_accepted"],
                    "translation_error": step.info["translation_error"],
                    "visible_inventory": dict(
                        step.observations["agent_1"].visible_inventory or {}
                    ),
                }
            )
            if terminated:
                break
        requested = len(actions)
        if completed != requested or terminated:
            raise RuntimeError(
                f"real smoke ended early: completed={completed}, "
                f"requested={requested}, terminated={terminated}"
            )
        state = backend.get_evaluation_state()
        evidence = dict(state.evidence)
        requested_position = list(_task().spawn_positions["agent_1"])
        observed_position = evidence.get("position", {})
        observed_xyz = [
            observed_position.get("xpos"),
            observed_position.get("ypos"),
            observed_position.get("zpos"),
        ]
        fixed_spawn_matches = all(
            actual is not None and abs(float(actual) - expected) <= 1.0
            for actual, expected in zip(observed_xyz, requested_position)
        )
        evaluator_transport_ready = bool(
            evidence.get("portal_grid_payload_present")
        )
        dimension_truth_ready = evidence.get("dimension") in {
            "minecraft:overworld",
            "minecraft:the_nether",
            "minecraft:the_end",
        }
        backend_ready = (
            evaluator_transport_ready
            and dimension_truth_ready
            and fixed_spawn_matches
        )
        return {
            "status": "passed" if backend_ready else "blocked",
            "mode": "real",
            "env_name": "ObsidianLinkPortalA0-v0",
            "steps_requested": requested,
            "steps_completed": completed,
            "terminated": terminated,
            "frame_shape": list(initial_observation.frame.shape),
            "visible_inventory": dict(
                initial_observation.visible_inventory or {}
            ),
            "action_results": action_results,
            "capabilities": {
                "reset_step_close": True,
                "pov": True,
                "inventory": True,
                "low_level_action_transport": all(
                    item["translation_accepted"] for item in action_results
                ),
                "requested_spawn": requested_position,
                "observed_spawn": observed_xyz,
                "fixed_spawn_matches": fixed_spawn_matches,
                "portal_grid_transport": evaluator_transport_ready,
                "dimension_truth": dimension_truth_ready,
            },
            "evaluation_evidence": evidence,
            "blocked_reason": (
                None
                if backend_ready
                else "MineRL EnvServer capability contract is incomplete"
            ),
            "real_minerl_started": True,
        }
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("spec", "real"), default="spec")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--exercise-actions",
        action="store_true",
        help="exercise bounded look/equip/place/use/mine action transport",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/phase1-portal-env-smoke",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 20:
        raise ValueError("--steps must be between 1 and 20")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / f"{timestamp}-{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=False)
    exit_code = 0
    try:
        result = (
            _spec_result()
            if args.mode == "spec"
            else _real_result(args.steps, args.exercise_actions)
        )
        if result["status"] == "blocked":
            exit_code = 2
    except Exception as error:
        result = {
            "status": "failed",
            "mode": args.mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "real_minerl_started": args.mode == "real",
        }
        exit_code = 1
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), **result}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
