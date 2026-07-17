"""Single-variable Phase-5 A/B evaluation for visual change feedback."""

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


CONTROL_ARM = "A-control-shadow"
TREATMENT_ARM = "B-change-feedback"
DEFAULT_SEEDS = (5101, 5102, 5103)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_ticks = sum(result["completed_ticks"] for result in results)
    total_planner_decisions = sum(result["planner_decisions"] for result in results)
    total_accepted_decisions = sum(
        result["accepted_decisions"] for result in results
    )
    total_ineffective_decisions = sum(
        result["ineffective_decisions"] for result in results
    )
    total_action_windows = sum(result["action_windows"] for result in results)
    ineffective_windows = sum(
        result["ineffective_action_windows"] for result in results
    )
    total_change_samples = sum(result["frame_change_samples"] for result in results)
    low_change_samples = sum(result["low_change_samples"] for result in results)
    weighted_latency = sum(
        (result["decision_latency_mean"] or 0.0) * result["planner_decisions"]
        for result in results
    )
    return {
        "episodes": len(results),
        "episodes_accepted": sum(bool(result["accepted"]) for result in results),
        "ticks": total_ticks,
        "planner_decisions": total_planner_decisions,
        "accepted_decisions": total_accepted_decisions,
        "effective_decisions": total_accepted_decisions - total_ineffective_decisions,
        "ineffective_decisions": total_ineffective_decisions,
        "ineffective_decision_rate": (
            total_ineffective_decisions / total_accepted_decisions
            if total_accepted_decisions
            else None
        ),
        "decision_compute_seconds": weighted_latency,
        "decision_latency_mean": (
            weighted_latency / total_planner_decisions
            if total_planner_decisions
            else None
        ),
        "no_op_tick_rate": (
            sum(result["no_op_ticks"] for result in results) / total_ticks
            if total_ticks
            else None
        ),
        "frame_change_samples": total_change_samples,
        "low_change_samples": low_change_samples,
        "changed_samples": total_change_samples - low_change_samples,
        "low_change_rate": (
            low_change_samples / total_change_samples
            if total_change_samples
            else None
        ),
        "action_windows": total_action_windows,
        "ineffective_action_windows": ineffective_windows,
        "ineffective_action_window_rate": (
            ineffective_windows / total_action_windows
            if total_action_windows
            else None
        ),
        "frame_change_compute_max_seconds": max(
            result["frame_change_compute_max_seconds"] or 0.0 for result in results
        ),
        "esc_nonzero_ticks": sum(result["esc_nonzero_ticks"] for result in results),
        "stale_decisions": sum(result["stale_decisions"] for result in results),
    }


def run_phase5_frame_change_ab(
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

    output_root = output_root or ROOT / "artifacts" / "phase5" / "frame-change-ab"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        (session_dir / arm).mkdir()
    config = {
        "phase": 5,
        "experiment": "frame_change_feedback",
        "seeds": list(seeds),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "control": "measure frame change in shadow mode; planner prompt unchanged",
        "treatment": "same detector and thresholds; feed change result to planner",
        "detector": {
            "mean_difference_threshold": 0.005,
            "changed_fraction_threshold": 0.01,
            "pixel_difference_threshold": 20.0,
            "sample": "rows 0..299, stride 8, grayscale",
        },
        "disabled_variables": [
            "turning_loop_detection",
            "repeated_action_penalty",
            "recovery_macro",
            "short_term_map",
            "hierarchical_prompt",
        ],
    }
    (session_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
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
                            frame_change_feedback=arm == TREATMENT_ARM,
                        )
                        results[arm].append(result)
                    if stop_all.is_set():
                        break
        finally:
            planner.stop()

        control = _aggregate(results[CONTROL_ARM])
        treatment = _aggregate(results[TREATMENT_ARM])
        complete = all(
            len(results[arm]) == len(seeds)
            and all(result["accepted"] for result in results[arm])
            for arm in (CONTROL_ARM, TREATMENT_ARM)
        )
        expected_samples = len(seeds) * max(
            0, (ticks - 1) // observation_interval
        )
        detector_valid = all(
            aggregate["frame_change_samples"] == expected_samples
            for aggregate in (control, treatment)
        )
        control_rate = control["ineffective_decision_rate"]
        treatment_rate = treatment["ineffective_decision_rate"]
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
        advance_recommended = bool(
            complete
            and detector_valid
            and treatment_rate is not None
            and control_rate is not None
            and treatment_rate < control_rate
            and decision_cost_ratio is not None
            and decision_cost_ratio <= 1.25
        )
        summary = {
            "accepted": complete
            and detector_valid
            and control["esc_nonzero_ticks"] == 0
            and treatment["esc_nonzero_ticks"] == 0
            and control["stale_decisions"] == 0
            and treatment["stale_decisions"] == 0,
            "advance_recommended": advance_recommended,
            "session_dir": str(session_dir),
            "experiment": "frame_change_feedback",
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
                "ineffective_decision_relative_reduction": relative_reduction,
                "decision_compute_cost_ratio": decision_cost_ratio,
                "control_prompt_changed": False,
                "treatment_only_variable": "visual change feedback",
            },
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        (session_dir / "summary.json").write_text(rendered, encoding="utf-8")
        (session_dir / "stdout.log").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not summary["accepted"]:
            raise RuntimeError("Phase-5 frame-change A/B data integrity gate failed")
        return summary
    except BaseException as error:
        (session_dir / "stderr.log").write_text(repr(error) + "\n", encoding="utf-8")
        raise
