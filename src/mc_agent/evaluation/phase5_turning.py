"""Single-variable Phase-5 A/B evaluation for turning-loop feedback."""

from __future__ import annotations

import json
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mc_agent.env import MineRLEnvAdapter
from mc_agent.planner import QwenPlannerWorker

from .phase4 import ROOT, _run_episode
from .phase5 import DEFAULT_SEEDS, _aggregate


CONTROL_ARM = "A-change-feedback-control"
TREATMENT_ARM = "B-turning-loop-feedback"


def _aggregate_turning(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate(results)
    rotation_only = sum(result["rotation_only_decisions"] for result in results)
    accepted = aggregate["accepted_decisions"]
    aggregate.update(
        {
            "rotation_only_decisions": rotation_only,
            "rotation_only_decision_rate": (
                rotation_only / accepted if accepted else None
            ),
            "turning_loop_activations": sum(
                result["turning_loop_activations"] for result in results
            ),
            "turning_loop_observations": sum(
                result["turning_loop_observations"] for result in results
            ),
            "forward_decisions": sum(
                result["forward_decisions"] for result in results
            ),
        }
    )
    return aggregate


def run_phase5_turning_loop_ab(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    ticks: int = 800,
    observation_interval: int = 40,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be a non-empty tuple of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if ticks < 1 or observation_interval < 1:
        raise ValueError("ticks and observation_interval must be positive")

    output_root = output_root or ROOT / "artifacts" / "phase5" / "turning-loop-ab"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        (session_dir / arm).mkdir()
    config = {
        "phase": 5,
        "experiment": "turning_loop_feedback",
        "seeds": list(seeds),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "retained_baseline": "validated frame-change feedback enabled in both arms",
        "control": "measure turning loop in shadow mode; no turning-loop prompt state",
        "treatment": "same detector; append prompt state only while loop is active",
        "detector": {
            "window_size": 3,
            "cumulative_abs_yaw_threshold": 30.0,
            "rotation_only": "non-zero-yaw look/turn without move_forward",
        },
        "disabled_variables": [
            "repeated_action_penalty",
            "recovery_macro",
            "short_term_map",
            "hierarchical_prompt",
        ],
    }
    rendered_config = json.dumps(config, indent=2) + "\n"
    (session_dir / "config.json").write_text(rendered_config, encoding="utf-8")
    (session_dir / "stdout.log").write_text("", encoding="utf-8")
    (session_dir / "stderr.log").write_text("", encoding="utf-8")

    stop_all = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_all.set())
    planner = QwenPlannerWorker()
    results: dict[str, list[dict[str, Any]]] = {
        CONTROL_ARM: [],
        TREATMENT_ARM: [],
    }
    execution_order: list[dict[str, Any]] = []

    try:
        planner.start()
        try:
            if not planner.ready.wait(30):
                raise TimeoutError("Qwen planner did not load within 30 seconds")
            if planner.error:
                raise RuntimeError(planner.error)
            with MineRLEnvAdapter() as adapter:
                for pair_index, seed in enumerate(seeds):
                    arm_order = (
                        (CONTROL_ARM, TREATMENT_ARM)
                        if pair_index % 2 == 0
                        else (TREATMENT_ARM, CONTROL_ARM)
                    )
                    for arm in arm_order:
                        if stop_all.is_set():
                            break
                        execution_order.append({"seed": seed, "arm": arm})
                        result = _run_episode(
                            adapter,
                            planner,
                            session_dir / arm,
                            pair_index + 1,
                            ticks,
                            observation_interval,
                            stop_all,
                            episode_id_override=f"seed-{seed}",
                            phase=5,
                            seed=seed,
                            measure_frame_change=True,
                            frame_change_feedback=True,
                            measure_turning_loop=True,
                            turning_loop_feedback=arm == TREATMENT_ARM,
                        )
                        results[arm].append(result)
                    if stop_all.is_set():
                        break
        finally:
            planner.stop()

        control = _aggregate_turning(results[CONTROL_ARM])
        treatment = _aggregate_turning(results[TREATMENT_ARM])
        complete = all(
            len(results[arm]) == len(seeds)
            and all(result["accepted"] for result in results[arm])
            for arm in (CONTROL_ARM, TREATMENT_ARM)
        )
        control_rate = control["rotation_only_decision_rate"]
        treatment_rate = treatment["rotation_only_decision_rate"]
        relative_reduction = (
            (control_rate - treatment_rate) / control_rate
            if control_rate not in (None, 0.0) and treatment_rate is not None
            else 0.0
        )
        decision_cost_ratio = (
            treatment["decision_compute_seconds"]
            / control["decision_compute_seconds"]
            if control["decision_compute_seconds"]
            else None
        )
        invalid_not_worse = (
            treatment["ineffective_decision_rate"]
            <= control["ineffective_decision_rate"]
        )
        advance_recommended = bool(
            complete
            and control_rate is not None
            and treatment_rate is not None
            and relative_reduction >= 0.10
            and invalid_not_worse
            and decision_cost_ratio is not None
            and decision_cost_ratio <= 1.25
        )
        summary = {
            "accepted": complete
            and control["esc_nonzero_ticks"] == 0
            and treatment["esc_nonzero_ticks"] == 0
            and control["stale_decisions"] == 0
            and treatment["stale_decisions"] == 0,
            "advance_recommended": advance_recommended,
            "session_dir": str(session_dir),
            "experiment": "turning_loop_feedback",
            "seeds": list(seeds),
            "ticks_per_episode": ticks,
            "observation_interval": observation_interval,
            "execution_order": execution_order,
            "planner_load_seconds": planner.load_seconds,
            "planner_peak_mps_driver_bytes": planner.peak_mps_driver_bytes,
            "planner_error": planner.error,
            "control": {"aggregate": control, "episodes": results[CONTROL_ARM]},
            "treatment": {
                "aggregate": treatment,
                "episodes": results[TREATMENT_ARM],
            },
            "comparison": {
                "rotation_only_relative_reduction": relative_reduction,
                "decision_compute_cost_ratio": decision_cost_ratio,
                "ineffective_decision_not_worse": invalid_not_worse,
                "control_turning_prompt_changed": False,
                "treatment_only_variable": "turning-loop feedback",
            },
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        (session_dir / "summary.json").write_text(rendered, encoding="utf-8")
        (session_dir / "stdout.log").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not summary["accepted"]:
            raise RuntimeError("Phase-5 turning-loop A/B data integrity gate failed")
        return summary
    except BaseException as error:
        (session_dir / "stderr.log").write_text(repr(error) + "\n", encoding="utf-8")
        raise
