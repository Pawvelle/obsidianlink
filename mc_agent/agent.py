"""Replayable asynchronous Qwen/MineRL agent loop."""

from __future__ import annotations

import json
import signal
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from PIL import Image

from mc_agent.actions import (
    MacroAction,
    MacroExecutor,
    Watchdog,
    is_cave_candidate,
    safe_camera_recovery,
)
from mc_agent.env import MineRLEnvAdapter
from mc_agent.logger import EpisodeLogger
from mc_agent.memory import FrameChangeDetector
from mc_agent.qwen import PlannerDecision, QwenPlannerWorker


ROOT = Path(__file__).resolve().parents[1]
TARGET_TICKS_PER_SECOND = 20.0
TARGET_TICK_SECONDS = 1.0 / TARGET_TICKS_PER_SECOND


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _tick_sleep_seconds(tick_started: float, now: float) -> float:
    """Return clock pacing only; planner state never participates in this delay."""
    return max(0.0, TARGET_TICK_SECONDS - (now - tick_started))


def _action_signature(decision: PlannerDecision) -> str:
    action = decision.action
    return json.dumps(
        {
            "action": action.action,
            "duration_ticks": action.duration_ticks,
            "pitch": action.camera_pitch,
            "yaw": action.camera_yaw,
            "sprint": action.sprint,
        },
        sort_keys=True,
    )


def _action_context(action: MacroAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "duration_ticks": action.duration_ticks,
        "camera": {"pitch": action.camera_pitch, "yaw": action.camera_yaw},
        "attack": action.attack,
        "jump": action.jump,
        "sprint": action.sprint,
        "cave_visible": action.cave_visible,
    }


def _macro_action_has_effect(action: MacroAction) -> bool:
    if action.action == "move_forward":
        return True
    if action.action in {"look", "turn"} and (
        action.camera_pitch != 0.0 or action.camera_yaw != 0.0
    ):
        return True
    return action.attack or action.jump


def _select_executed_action(
    action: MacroAction,
    recovery_index: int,
) -> tuple[MacroAction, bool]:
    """Replace only an accepted semantic no-op with the validated camera recovery."""
    if not _macro_action_has_effect(action):
        return safe_camera_recovery(recovery_index), True
    return action, False


def _episode_passes_gate(
    *,
    completed_ticks: int,
    tick_budget: int,
    early_done: bool,
    effective_decisions: int,
    model_forward_decisions: int,
    forward_ticks: int,
    esc_nonzero: int,
    planner_error: str | None,
) -> bool:
    """Require observable model-driven progress, not merely valid JSON."""
    return (
        completed_ticks == tick_budget
        and not early_done
        and effective_decisions >= 1
        and model_forward_decisions >= 1
        and forward_ticks >= 1
        and esc_nonzero == 0
        and planner_error is None
    )


def _run_episode(
    adapter: MineRLEnvAdapter,
    planner: QwenPlannerWorker,
    session_dir: Path,
    episode_index: int,
    tick_budget: int,
    observation_interval: int,
    stop_all: threading.Event,
    *,
    episode_id_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    episode_id = episode_id_override or f"episode-{episode_index:02d}"
    run_dir = session_dir / episode_id
    logger = EpisodeLogger(
        run_dir,
        {
            "episode_id": episode_id,
            "env_id": "MineRLBasaltFindCave-v0",
            "seed": seed,
            "tick_budget": tick_budget,
            "observation_interval": observation_interval,
            "target_ticks_per_second": TARGET_TICKS_PER_SECOND,
            "planner": "Qwen3-VL-2B-Instruct MPS/FP16, asynchronous",
            "visual_change_feedback": True,
            "safe_camera_recovery": True,
            "acceptance_requires_model_forward": True,
            "esc_policy": "always disabled; manual termination policy not enabled",
        },
    )
    frames_dir = run_dir / "decision_frames"
    frames_dir.mkdir()
    watchdog = Watchdog(max_ticks=tick_budget)
    executor = MacroExecutor(adapter.action_space, watchdog=watchdog)
    change_detector = FrameChangeDetector()
    process = psutil.Process()

    minimum_available = psutil.virtual_memory().available
    peak_rss = process.memory_info().rss
    step_latencies: list[float] = []
    decision_latencies: list[float] = []
    change_compute_latencies: list[float] = []
    action_signatures: list[str] = []
    recovery_action_signatures: list[str] = []
    frame_changes = []
    paced_sleep_seconds = 0.0
    no_op_ticks = 0
    forward_ticks = 0
    decisions = 0
    accepted_decisions = 0
    model_forward_decisions = 0
    rejected_decisions = 0
    ineffective_decisions = 0
    executed_ineffective_decisions = 0
    stale_decisions = 0
    completed_ticks = 0
    reward_sum = 0.0
    early_done = False
    esc_nonzero = 0
    low_change_samples = 0
    action_windows = 0
    ineffective_action_windows = 0
    progress_action_ticks_since_observation = 0
    recovery_opportunities = 0
    recovery_actions_applied = 0
    recovery_followup_pending = False
    recovery_followup_decisions = 0
    recovery_followup_effective_decisions = 0
    cave_candidate_decisions = 0
    raw_cave_visible_decisions = 0
    cave_candidate_observation_ticks: list[int] = []
    cave_candidate_evidence: list[str] = []
    previous_action_context: dict[str, Any] | None = None
    barrier_seconds = 0.0
    started = time.perf_counter()

    try:
        # This is the only planner wait in an episode. It happens before reset,
        # while the MineRL step loop is not running.
        barrier_seconds = planner.begin_episode(episode_id)
        logger.event(
            "planner_episode_barrier",
            episode_id=episode_id,
            wait_seconds=barrier_seconds,
            idle=planner.idle.is_set(),
        )
        if seed is not None:
            adapter.seed(seed)
        observation = adapter.reset()
        change_detector.reset(observation["pov"])
        Image.fromarray(observation["pov"]).save(run_dir / "initial.png")
        logger.event("reset", pov_shape=list(observation["pov"].shape), seed=seed)
        Image.fromarray(observation["pov"]).save(frames_dir / "tick-0000.png")
        planner.submit(episode_id, 0, observation["pov"], None)
        logger.event(
            "observation_published",
            tick=0,
            frame="decision_frames/tick-0000.png",
            visual_change=None,
        )

        while completed_ticks < tick_budget:
            tick_started = time.perf_counter()
            pending_ack: PlannerDecision | None = None
            if stop_all.is_set():
                watchdog.request_stop("SIGINT")
            if planner.error:
                watchdog.request_stop("planner_error")
            if watchdog.should_stop and watchdog.reason not in (None, "max_ticks"):
                logger.event("interrupted", tick=completed_ticks, reason=watchdog.reason)
                break

            decision = planner.decisions.take_latest()
            if decision is not None:
                pending_ack = decision
                if decision.episode_id != episode_id:
                    stale_decisions += 1
                    logger.event(
                        "stale_decision_discarded",
                        decision_episode=decision.episode_id,
                        observation_tick=decision.observation_tick,
                    )
                else:
                    decisions += 1
                    decision_latencies.append(decision.latency_seconds)
                    action_to_execute = decision.action
                    recovery_applied = False
                    cave_candidate_validated = False
                    if decision.accepted:
                        accepted_decisions += 1
                        has_effect = _macro_action_has_effect(decision.action)
                        ineffective_decisions += int(not has_effect)
                        if has_effect:
                            action_signatures.append(_action_signature(decision))
                        model_forward_decisions += int(
                            decision.action.action == "move_forward"
                        )
                        raw_cave_visible_decisions += int(
                            decision.action.cave_visible
                        )
                        cave_candidate_validated = is_cave_candidate(
                            decision.action
                        )
                        if cave_candidate_validated:
                            cave_candidate_decisions += 1
                            cave_candidate_observation_ticks.append(
                                decision.observation_tick
                            )
                            cave_candidate_evidence.append(
                                "decision_frames/"
                                f"tick-{decision.observation_tick:04d}.png"
                            )
                        if recovery_followup_pending:
                            recovery_followup_decisions += 1
                            recovery_followup_effective_decisions += int(has_effect)
                            recovery_followup_pending = False
                        if not has_effect:
                            recovery_opportunities += 1
                            action_to_execute, recovery_applied = _select_executed_action(
                                decision.action,
                                recovery_actions_applied,
                            )
                            recovery_actions_applied += 1
                            recovery_followup_pending = True
                            recovery_action_signatures.append(
                                json.dumps(
                                    action_to_execute.to_log_dict(),
                                    sort_keys=True,
                                )
                            )
                        executed_has_effect = _macro_action_has_effect(action_to_execute)
                        executed_ineffective_decisions += int(not executed_has_effect)
                        previous_action_context = _action_context(action_to_execute)
                    else:
                        rejected_decisions += 1
                        has_effect = False
                        executed_has_effect = False
                    executor.submit(action_to_execute)
                    logger.event(
                        "planner_decision",
                        applied_tick=completed_ticks,
                        observation_tick=decision.observation_tick,
                        staleness_ticks=completed_ticks - decision.observation_tick,
                        latency_seconds=decision.latency_seconds,
                        accepted=decision.accepted,
                        error=decision.error,
                        has_effect=has_effect,
                        executed_has_effect=executed_has_effect,
                        recovery_opportunity=decision.accepted and not has_effect,
                        recovery_applied=recovery_applied,
                        cave_visible=decision.action.cave_visible,
                        cave_candidate_validated=cave_candidate_validated,
                        raw=decision.raw,
                        parsed=decision.action.to_log_dict(),
                        executed=action_to_execute.to_log_dict(),
                    )

            if completed_ticks > 0 and completed_ticks % observation_interval == 0:
                frame_name = f"tick-{completed_ticks:04d}.png"
                Image.fromarray(observation["pov"]).save(frames_dir / frame_name)
                change_started = time.perf_counter()
                change = change_detector.compare_and_update(observation["pov"])
                change_compute_latencies.append(time.perf_counter() - change_started)
                frame_changes.append(change)
                low_change_samples += int(change.low_change)
                if progress_action_ticks_since_observation > 0:
                    action_windows += 1
                    ineffective_action_windows += int(change.low_change)
                visual_change = change.to_log_dict()
                planner.submit(
                    episode_id,
                    completed_ticks,
                    observation["pov"],
                    previous_action_context,
                    visual_change,
                )
                logger.event(
                    "observation_published",
                    tick=completed_ticks,
                    frame=f"decision_frames/{frame_name}",
                    visual_change=visual_change,
                    progress_action_ticks=progress_action_ticks_since_observation,
                )
                progress_action_ticks_since_observation = 0

            tick_action = executor.next_tick()
            camera_changed = bool(
                float(tick_action["camera"][0]) or float(tick_action["camera"][1])
            )
            action_changed = bool(
                camera_changed
                or tick_action["attack"]
                or tick_action["forward"]
                or tick_action["jump"]
                or tick_action["sprint"]
            )
            no_op_ticks += int(not action_changed)
            forward_ticks += int(bool(tick_action["forward"]))
            progress_action_ticks_since_observation += int(action_changed)
            esc_nonzero += int(bool(tick_action["ESC"]))
            step_started = time.perf_counter()
            step = adapter.step(tick_action)
            step_elapsed = time.perf_counter() - step_started
            step_latencies.append(step_elapsed)
            completed_ticks += 1
            watchdog.after_tick()
            reward_sum += step.reward
            observation = step.observation

            # The acknowledgement only releases the planner worker. MineRL has
            # already stepped, so inference never blocks this loop.
            if pending_ack is not None:
                planner.acknowledge_decision(
                    pending_ack.episode_id,
                    pending_ack.observation_tick,
                )
                logger.event(
                    "planner_decision_acknowledged",
                    episode_id=pending_ack.episode_id,
                    observation_tick=pending_ack.observation_tick,
                    after_tick=completed_ticks,
                )

            peak_rss = max(peak_rss, process.memory_info().rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            logger.event(
                "tick",
                tick=completed_ticks,
                action={
                    "ESC": tick_action["ESC"],
                    "attack": tick_action["attack"],
                    "camera": tick_action["camera"],
                    "forward": tick_action["forward"],
                    "jump": tick_action["jump"],
                    "sprint": tick_action["sprint"],
                },
                reward=step.reward,
                done=step.done,
                step_seconds=step_elapsed,
            )
            if step.done:
                early_done = completed_ticks < tick_budget
                logger.event("done", tick=completed_ticks, early=early_done)
                break

            if completed_ticks < tick_budget:
                sleep_seconds = _tick_sleep_seconds(tick_started, time.perf_counter())
                if sleep_seconds > 0:
                    sleep_started = time.perf_counter()
                    stop_all.wait(sleep_seconds)
                    paced_sleep_seconds += time.perf_counter() - sleep_started

        Image.fromarray(observation["pov"]).save(run_dir / "final.png")
    except BaseException as error:
        logger.event("error", error=repr(error), tick=completed_ticks)
        logger.finish(
            {
                "accepted": False,
                "completed_ticks": completed_ticks,
                "error": repr(error),
                "manual_review": {"required": True, "status": "pending"},
            }
        )
        raise

    elapsed = time.perf_counter() - started
    effective_decisions = accepted_decisions - ineffective_decisions
    accepted = _episode_passes_gate(
        completed_ticks=completed_ticks,
        tick_budget=tick_budget,
        early_done=early_done,
        effective_decisions=effective_decisions,
        model_forward_decisions=model_forward_decisions,
        forward_ticks=forward_ticks,
        esc_nonzero=esc_nonzero,
        planner_error=planner.error,
    )
    metrics = {
        "accepted": accepted,
        "episode_id": episode_id,
        "completed_ticks": completed_ticks,
        "tick_budget": tick_budget,
        "early_done": early_done,
        "reward_sum": reward_sum,
        "elapsed_seconds": elapsed,
        "ticks_per_second": completed_ticks / elapsed if elapsed else 0,
        "planner_decisions": decisions,
        "accepted_decisions": accepted_decisions,
        "rejected_decisions": rejected_decisions,
        "effective_decisions": effective_decisions,
        "model_forward_decisions": model_forward_decisions,
        "ineffective_decisions": ineffective_decisions,
        "ineffective_decision_rate": (
            ineffective_decisions / accepted_decisions if accepted_decisions else None
        ),
        "stale_decisions": stale_decisions,
        "episode_barrier_seconds": barrier_seconds,
        "unique_action_signatures": sorted(set(action_signatures)),
        "decision_latency_mean": (
            statistics.mean(decision_latencies) if decision_latencies else None
        ),
        "decision_latency_max": max(decision_latencies) if decision_latencies else None,
        "step_latency_p95": _percentile(step_latencies, 0.95),
        "step_latency_max": max(step_latencies),
        "target_ticks_per_second": TARGET_TICKS_PER_SECOND,
        "paced_sleep_seconds": paced_sleep_seconds,
        "no_op_ticks": no_op_ticks,
        "no_op_tick_rate": no_op_ticks / completed_ticks if completed_ticks else None,
        "forward_ticks": forward_ticks,
        "frame_change_samples": len(frame_changes),
        "low_change_samples": low_change_samples,
        "low_change_rate": low_change_samples / len(frame_changes) if frame_changes else None,
        "mean_frame_difference": (
            statistics.mean(change.mean_absolute_difference for change in frame_changes)
            if frame_changes
            else None
        ),
        "frame_difference_p95": (
            _percentile(
                [change.mean_absolute_difference for change in frame_changes],
                0.95,
            )
            if frame_changes
            else None
        ),
        "action_windows": action_windows,
        "ineffective_action_windows": ineffective_action_windows,
        "ineffective_action_window_rate": (
            ineffective_action_windows / action_windows if action_windows else None
        ),
        "frame_change_compute_mean_seconds": (
            statistics.mean(change_compute_latencies)
            if change_compute_latencies
            else None
        ),
        "frame_change_compute_max_seconds": (
            max(change_compute_latencies) if change_compute_latencies else None
        ),
        "recovery_opportunities": recovery_opportunities,
        "recovery_actions_applied": recovery_actions_applied,
        "executed_effective_decisions": (
            accepted_decisions - executed_ineffective_decisions
        ),
        "executed_ineffective_decisions": executed_ineffective_decisions,
        "executed_ineffective_decision_rate": (
            executed_ineffective_decisions / accepted_decisions
            if accepted_decisions
            else None
        ),
        "recovery_followup_decisions": recovery_followup_decisions,
        "recovery_followup_effective_decisions": recovery_followup_effective_decisions,
        "recovery_followup_effective_rate": (
            recovery_followup_effective_decisions / recovery_followup_decisions
            if recovery_followup_decisions
            else None
        ),
        "recovery_followup_pending_at_end": recovery_followup_pending,
        "recovery_action_signatures": sorted(set(recovery_action_signatures)),
        "cave_candidate_decisions": cave_candidate_decisions,
        "raw_cave_visible_decisions": raw_cave_visible_decisions,
        "cave_candidate_observation_ticks": cave_candidate_observation_ticks,
        "cave_candidate_evidence": cave_candidate_evidence,
        "peak_process_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "esc_nonzero_ticks": esc_nonzero,
        "termination_reason": (
            "tick_budget" if completed_ticks == tick_budget else watchdog.reason
        ),
        "manual_review": {
            "required": True,
            "status": "pending",
            "cave_found": None,
            "evidence": [
                "initial.png",
                "final.png",
                "decision_frames/",
                *cave_candidate_evidence,
            ],
            "note": (
                "Cave candidates require frame review; tick-budget completion is not "
                "FindCave task success."
            ),
        },
    }
    logger.finish(metrics)
    return {"run_dir": str(run_dir), **metrics}


def run_agent(
    episodes: int = 5,
    ticks: int = 240,
    observation_interval: int = 40,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if episodes < 1 or ticks < 1 or observation_interval < 1:
        raise ValueError("episodes, ticks, and observation_interval must be positive")
    output_root = output_root or ROOT / "runs" / "episodes"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    stop_all = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_all.set())
    planner = QwenPlannerWorker()
    results: list[dict[str, Any]] = []
    planner.start()
    try:
        if not planner.ready.wait(30):
            raise TimeoutError("Qwen planner did not load within 30 seconds")
        if planner.error:
            raise RuntimeError(planner.error)
        with MineRLEnvAdapter() as adapter:
            for episode_index in range(1, episodes + 1):
                if stop_all.is_set():
                    break
                results.append(
                    _run_episode(
                        adapter,
                        planner,
                        session_dir,
                        episode_index,
                        ticks,
                        observation_interval,
                        stop_all,
                    )
                )
    finally:
        planner.stop()

    signatures = sorted(
        {
            signature
            for result in results
            for signature in result.get("unique_action_signatures", [])
        }
    )
    total_decisions = sum(result.get("planner_decisions", 0) for result in results)
    total_accepted_decisions = sum(
        result.get("accepted_decisions", 0) for result in results
    )
    total_effective_decisions = sum(
        result.get("effective_decisions", 0) for result in results
    )
    total_model_forward_decisions = sum(
        result.get("model_forward_decisions", 0) for result in results
    )
    total_forward_ticks = sum(result.get("forward_ticks", 0) for result in results)
    total_cave_candidate_decisions = sum(
        result.get("cave_candidate_decisions", 0) for result in results
    )
    cave_candidate_episodes = [
        result["episode_id"]
        for result in results
        if result.get("cave_candidate_decisions", 0) > 0
    ]
    summary = {
        "accepted": (
            len(results) == episodes
            and all(result["accepted"] for result in results)
            and total_effective_decisions >= episodes
            and total_model_forward_decisions >= episodes
            and total_forward_ticks >= episodes
            and len(signatures) >= 2
        ),
        "session_dir": str(session_dir),
        "episodes_requested": episodes,
        "episodes_completed": len(results),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "planner_load_seconds": planner.load_seconds,
        "planner_peak_mps_driver_bytes": planner.peak_mps_driver_bytes,
        "planner_error": planner.error,
        "total_planner_decisions": total_decisions,
        "total_accepted_decisions": total_accepted_decisions,
        "total_effective_decisions": total_effective_decisions,
        "total_model_forward_decisions": total_model_forward_decisions,
        "total_forward_ticks": total_forward_ticks,
        "total_cave_candidate_decisions": total_cave_candidate_decisions,
        "cave_candidate_episodes": cave_candidate_episodes,
        "unique_action_signatures": signatures,
        "model_changed_action": len(signatures) >= 2,
        "episodes": results,
        "manual_review": {
            "required": True,
            "status": "pending",
            "note": (
                "BASALT has no reliable task-success reward; review cave candidate "
                "frames before marking FindCave complete."
            ),
        },
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["accepted"]:
        raise RuntimeError(
            "Agent run did not pass its forward-progress/action-change gate"
        )
    return summary
