"""Single-variable Phase-5 A/B evaluation for a safe recovery macro-action."""

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
TREATMENT_ARM = "B-safe-recovery"


def _aggregate_recovery(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate(results)
    accepted = aggregate["accepted_decisions"]
    executed_ineffective = sum(
        result["executed_ineffective_decisions"] for result in results
    )
    followups = sum(result["recovery_followup_decisions"] for result in results)
    effective_followups = sum(
        result["recovery_followup_effective_decisions"] for result in results
    )
    aggregate.update(
        {
            "executed_effective_decisions": accepted - executed_ineffective,
            "executed_ineffective_decisions": executed_ineffective,
            "executed_ineffective_decision_rate": (
                executed_ineffective / accepted if accepted else None
            ),
            "recovery_opportunities": sum(
                result["recovery_opportunities"] for result in results
            ),
            "recovery_actions_applied": sum(
                result["recovery_actions_applied"] for result in results
            ),
            "recovery_followup_decisions": followups,
            "recovery_followup_effective_decisions": effective_followups,
            "recovery_followup_effective_rate": (
                effective_followups / followups if followups else None
            ),
            "forward_decisions": sum(
                result["forward_decisions"] for result in results
            ),
            "executed_forward_decisions": sum(
                result["executed_forward_decisions"] for result in results
            ),
            "forward_ticks": sum(result["forward_ticks"] for result in results),
        }
    )
    return aggregate


def run_phase5_recovery_ab(
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

    output_root = output_root or ROOT / "artifacts" / "phase5" / "recovery-ab"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        (session_dir / arm).mkdir()
    config = {
        "phase": 5,
        "experiment": "safe_recovery_macro",
        "seeds": list(seeds),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "retained_baseline": (
            "validated frame-change feedback plus asynchronous decision acknowledgement"
        ),
        "control": "measure accepted semantic no-op recovery opportunities; execute original",
        "treatment": "replace only accepted semantic no-op with safe camera recovery",
        "recovery_macro": {
            "action": "look",
            "duration_ticks": 1,
            "camera_pitch": 0,
            "camera_yaw": "alternate +20/-20",
            "attack": False,
            "jump": False,
            "sprint": False,
            "trigger": "accepted wait or zero-angle look/turn",
        },
        "disabled_variables": [
            "turning_loop_feedback",
            "repeated_action_penalty",
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
                            measure_recovery=True,
                            apply_recovery=arm == TREATMENT_ARM,
                        )
                        results[arm].append(result)
                    if stop_all.is_set():
                        break
        finally:
            planner.stop()

        control = _aggregate_recovery(results[CONTROL_ARM])
        treatment = _aggregate_recovery(results[TREATMENT_ARM])
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
        control_rate = control["executed_ineffective_decision_rate"]
        treatment_rate = treatment["executed_ineffective_decision_rate"]
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
        recovery_exercised = treatment["recovery_actions_applied"] > 0
        all_opportunities_recovered = (
            treatment["recovery_actions_applied"]
            == treatment["recovery_opportunities"]
        )
        model_invalid_not_worse = (
            treatment["ineffective_decision_rate"]
            <= control["ineffective_decision_rate"]
        )
        no_op_not_worse = treatment["no_op_tick_rate"] <= control["no_op_tick_rate"]
        followup_effective = bool(
            treatment["recovery_followup_decisions"] > 0
            and treatment["recovery_followup_effective_rate"] is not None
            and treatment["recovery_followup_effective_rate"] >= 0.50
        )
        advance_recommended = bool(
            complete
            and detector_valid
            and recovery_exercised
            and all_opportunities_recovered
            and relative_reduction >= 0.20
            and followup_effective
            and no_op_not_worse
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
            "experiment": "safe_recovery_macro",
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
                "executed_ineffective_relative_reduction": relative_reduction,
                "decision_compute_cost_ratio": decision_cost_ratio,
                "model_ineffective_decision_not_worse": model_invalid_not_worse,
                "recovery_followup_effective_gate": followup_effective,
                "no_op_tick_rate_not_worse": no_op_not_worse,
                "treatment_recovery_exercised": recovery_exercised,
                "all_treatment_opportunities_recovered": all_opportunities_recovered,
                "control_execution_changed": False,
                "treatment_only_variable": "safe recovery macro",
            },
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        (session_dir / "summary.json").write_text(rendered, encoding="utf-8")
        (session_dir / "stdout.log").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not summary["accepted"]:
            raise RuntimeError("Phase-5 recovery A/B data integrity gate failed")
        return summary
    except BaseException as error:
        (session_dir / "stderr.log").write_text(repr(error) + "\n", encoding="utf-8")
        raise
