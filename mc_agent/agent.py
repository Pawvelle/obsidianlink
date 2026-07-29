"""Replayable asynchronous Qwen/MineRL agent loop."""

from __future__ import annotations

import json
import signal
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image

from mc_agent.actions import (
    MacroAction,
    MacroExecutor,
    Watchdog,
    has_directional_stone_bounded_dark_opening_region,
    is_cave_candidate,
    resolve_dark_opening_direction,
    resolve_cave_direction,
    safe_camera_recovery,
    safe_forward_continuation,
    safe_stuck_recovery,
    safe_turn_scan_recovery,
    safe_water_recovery,
    water_hazard_direction,
)
from mc_agent.env import MineRLEnvAdapter
from mc_agent.logger import EpisodeLogger
from mc_agent.memory import CaveEntryPhase, CaveTargetMemory, FrameChangeDetector
from mc_agent.qwen import PlannerDecision, QwenPlannerWorker


ROOT = Path(__file__).resolve().parents[1]
TARGET_TICKS_PER_SECOND = 20.0
TARGET_TICK_SECONDS = 1.0 / TARGET_TICKS_PER_SECOND
CONSECUTIVE_FORWARD_STALL_TURN_SCAN_THRESHOLD = 2
LOCAL_FORWARD_CONTINUATION_MAX_TICKS = 120
CAMERA_PITCH_GUARD_MIN_DEGREES = -15.0
CAMERA_PITCH_GUARD_MAX_DEGREES = 30.0
CAMERA_PITCH_GUARD_CORRECTION_DEGREES = 15.0
CAVE_COMPLETION_MIN_FORWARD_TICKS = 12
CAVE_ENTRY_PHASE_MAX_TICKS = 30
CAVE_ENTRY_PHASE_MACRO_TICKS = 20
CAVE_ENTRY_INTERIOR_ABSOLUTE_LUMINANCE = 50.0
CAVE_ENTRY_PRE_FRAME_WORLD_HEIGHT = 300
PUBLISHED_FRAME_CACHE_MAX_ENTRIES = 16
FORWARD_CONTINUATION_CANCEL_REASONS = (
    "planner_decision",
    "water_hazard",
    "low_progress",
    "turn_scan",
    "cave_completion",
    "cave_entry_complete",
    "cave_entry_interrupted",
    "environment_done",
    "watchdog",
    "max_ticks",
)


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
    if action.action in {
        "move_forward",
        "retreat",
        "sidestep_left",
        "sidestep_right",
    }:
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


def _guard_camera_pitch(
    action: MacroAction, commanded_pitch_degrees: float
) -> tuple[MacroAction, bool]:
    """Keep cumulative relative camera pitch inside a ground-visible band.

    MineRL camera values are relative deltas. A sequence of otherwise valid
    negative model pitch commands can therefore accumulate until the player
    only sees the sky. The loop tracks its own emitted pitch deltas (reset to
    zero for every episode) and applies a single bounded correction whenever
    the next action would cross either edge of the safe band. The action type,
    yaw, and all interaction flags are preserved; this remains an allowlisted,
    clamped macro action rather than a model-generated command.
    """
    target_pitch = commanded_pitch_degrees + action.camera_pitch
    if CAMERA_PITCH_GUARD_MIN_DEGREES <= target_pitch <= CAMERA_PITCH_GUARD_MAX_DEGREES:
        return action, False

    if target_pitch < CAMERA_PITCH_GUARD_MIN_DEGREES:
        correction = min(
            CAMERA_PITCH_GUARD_CORRECTION_DEGREES,
            CAMERA_PITCH_GUARD_MAX_DEGREES - commanded_pitch_degrees,
        )
    else:
        correction = max(
            -CAMERA_PITCH_GUARD_CORRECTION_DEGREES,
            CAMERA_PITCH_GUARD_MIN_DEGREES - commanded_pitch_degrees,
        )
    return (
        replace(
            action,
            camera_pitch=correction,
            reason="local camera pitch guard correction",
        ),
        True,
    )


def _should_publish_macro_completion_observation(
    *, action_completed: bool, completed_action: str, planner_idle: bool
) -> bool:
    """Publish a real post-movement frame while the worker can consume it.

    Turns and looks deliberately keep periodic visual-change feedback: an
    immediate one-tick follow-up can ask the planner to reassess nearly the
    same view and create a camera loop. This does not participate in pacing.
    """
    return (
        action_completed
        and completed_action
        in {"move_forward", "retreat", "sidestep_left", "sidestep_right"}
        and planner_idle
    )


def _world_strip_mean_luminance(
    pov: np.ndarray, *, world_height: int = CAVE_ENTRY_PRE_FRAME_WORLD_HEIGHT
) -> float:
    """Return the mean luminance of the world strip below the HUD.

    Mirrors the deterministic helper used by the existing cave gates: the
    bottom HUD strip is excluded, and the rest is collapsed to one floating
    point value in ``[0, 255]``. Used only as a coarse annotation for the
    Phase 5 entry evidence frames; it is not a scene classifier.
    """
    if not isinstance(pov, np.ndarray) or pov.ndim != 3 or pov.shape[2] != 3:
        raise ValueError("pov must be an RGB image")
    if world_height < 1 or world_height > pov.shape[0]:
        raise ValueError("world_height must be within the pov height")
    world = pov[:world_height].astype(np.float32, copy=False)
    luminance = (
        world[..., 0] * 0.299 + world[..., 1] * 0.587 + world[..., 2] * 0.114
    )
    return float(luminance.mean())


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
    cave_completion_requested: bool = False,
) -> bool:
    """Require observable model-driven progress, not merely valid JSON."""
    if cave_completion_requested:
        return (
            effective_decisions >= 1
            and model_forward_decisions >= 1
            and forward_ticks >= CAVE_COMPLETION_MIN_FORWARD_TICKS
            and esc_nonzero == 1
            and planner_error is None
        )
    return (
        completed_ticks == tick_budget
        and not early_done
        and effective_decisions >= 1
        and model_forward_decisions >= 1
        and forward_ticks >= 1
        and esc_nonzero == 0
        and planner_error is None
    )


class _PublishedFrameCache:
    """Bounded (episode_id, observation_tick) -> copied POV cache.

    A cave-candidate frame veto must judge the exact frame the model saw at
    ``decision.observation_tick``, not whichever frame happens to be newest
    when the (much later) asynchronous decision is applied. This cache keeps
    only the most recent ``max_entries`` published frames -- just enough to
    span the inference delay -- and always stores a copy, never a mutable
    reference to a MineRL buffer. It is created fresh per episode and is not
    a long-term map or memory.
    """

    def __init__(self, max_entries: int = PUBLISHED_FRAME_CACHE_MAX_ENTRIES):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()

    def put(self, episode_id: str, tick: int, pov: np.ndarray) -> None:
        key = (episode_id, tick)
        self._entries[key] = np.array(pov, copy=True)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, episode_id: str, tick: int) -> np.ndarray | None:
        return self._entries.get((episode_id, tick))

    def __len__(self) -> int:
        return len(self._entries)


def _forward_continuation_is_eligible(
    *, decision_accepted: bool, decision_action: str, executed_action: str
) -> bool:
    """Only an accepted, unmodified move_forward decision may start continuation.

    If the decision was rejected, was not move_forward, or was overridden by
    a safety layer (water hazard or ineffective-action recovery), the local
    continuation safety layer must not run.
    """
    return (
        decision_accepted
        and decision_action == "move_forward"
        and executed_action == "move_forward"
    )


def _forward_continuation_next_duration(remaining_ticks: int) -> int:
    """Return one bounded macro length that never exceeds the remaining budget."""
    if remaining_ticks < 1:
        raise ValueError("remaining_ticks must be a positive integer")
    return min(40, remaining_ticks)


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
    watch: bool = False,
    mission_max_ticks: int | None = None,
    model_lock_path: Path | None = None,
    cave_entry_phase_enabled: bool = False,
    cave_entry_phase_max_ticks: int = CAVE_ENTRY_PHASE_MAX_TICKS,
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
            "mission_max_ticks": mission_max_ticks,
            "observation_interval": observation_interval,
            "target_ticks_per_second": TARGET_TICKS_PER_SECOND,
            "planner": "Qwen3-VL-2B-Instruct MPS/FP16, asynchronous",
            "model_lock_path": str(model_lock_path) if model_lock_path else "model.lock.json",
            "live_view": watch,
            "visual_change_feedback": True,
            "macro_completion_observations": True,
            "safe_camera_recovery": True,
            "camera_pitch_guard": {
                "minimum_degrees": CAMERA_PITCH_GUARD_MIN_DEGREES,
                "maximum_degrees": CAMERA_PITCH_GUARD_MAX_DEGREES,
                "correction_degrees": CAMERA_PITCH_GUARD_CORRECTION_DEGREES,
            },
            "local_water_hazard_guard": True,
            "local_low_progress_guard": True,
            "acceptance_requires_model_forward": True,
            "cave_target_memory": {
                "max_decisions": 4,
                "completion_min_forward_ticks": CAVE_COMPLETION_MIN_FORWARD_TICKS,
            },
            "cave_entry_phase": {
                "enabled": cave_entry_phase_enabled,
                "max_ticks": cave_entry_phase_max_ticks,
                "macro_ticks": CAVE_ENTRY_PHASE_MACRO_TICKS,
                "interior_absolute_luminance": CAVE_ENTRY_INTERIOR_ABSOLUTE_LUMINANCE,
            },
            "esc_policy": "local single tick only after double-confirmed cave evidence",
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
    periodic_observations = 1
    macro_completion_observations = 0
    paced_sleep_seconds = 0.0
    no_op_ticks = 0
    forward_ticks = 0
    retreat_ticks = 0
    sidestep_left_ticks = 0
    sidestep_right_ticks = 0
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
    termination_reason: str | None = None
    terminal_info: dict[str, Any] | None = None
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
    water_hazard_overrides = 0
    water_hazard_directions: list[str] = []
    stuck_recovery_actions = 0
    cave_candidate_decisions = 0
    raw_cave_visible_decisions = 0
    cave_candidate_frame_vetoes = 0
    cave_candidate_frame_missing = 0
    cave_candidate_observation_ticks: list[int] = []
    cave_candidate_evidence: list[str] = []
    cave_target_acquisitions = 0
    cave_target_reconfirmations = 0
    cave_completion_requested = False
    cave_completion_evidence: list[str] = []
    consecutive_forward_stall_periods = 0
    turn_scan_recoveries = 0
    turn_scan_completion_observations = 0
    camera_pitch_guard_overrides = 0
    commanded_camera_pitch_degrees = 0.0
    pending_turn_scan_actions: list[MacroAction] = []
    awaiting_turn_scan_observation = False
    frame_cache = _PublishedFrameCache()
    cave_target = CaveTargetMemory()
    cave_entry_phase = CaveEntryPhase(
        max_budget_ticks=cave_entry_phase_max_ticks,
        enabled=cave_entry_phase_enabled,
    )
    entry_evidence_dir = run_dir / "entry_evidence"
    cave_entry_phase_activated = False
    cave_entry_decisions_during_phase = 0
    cave_entry_decisions_suppressed = 0
    cave_entry_pre_luminance: float | None = None
    cave_entry_post_luminance: float | None = None
    cave_entry_evidence_path: str | None = None
    cave_entry_completion_tick: int | None = None
    cave_entry_cancellation_reason: str | None = None
    cave_entry_local_forward_actions = 0
    forward_continuation_eligible = False
    forward_continuation_session_active = False
    # True while the executor is currently running a macro that the local
    # continuation layer itself submitted (as opposed to the model's own
    # decision macro or a safety override). This -- not the remaining
    # budget -- is what "an active continuation session" means: even after
    # the budget reaches zero, the last submitted macro can still be
    # executing for up to 40 more ticks, and must still be treated as
    # cancellable until it either finishes on its own or is pre-empted.
    forward_continuation_macro_pending = False
    forward_continuation_remaining_ticks = 0
    # Only real ticks handed to MineRL with forward=1 while
    # forward_continuation_macro_pending was true are counted here; this is
    # always a subset of forward_ticks, never a pre-allocated budget.
    forward_continuation_ticks = 0
    forward_continuation_sessions_started = 0
    forward_continuation_cancellations = {
        reason: 0 for reason in FORWARD_CONTINUATION_CANCEL_REASONS
    }
    previous_action_context: dict[str, Any] | None = None
    barrier_seconds = 0.0
    started = time.perf_counter()

    def _cancel_forward_continuation(reason: str, *, tick: int) -> None:
        nonlocal forward_continuation_eligible, forward_continuation_remaining_ticks
        nonlocal forward_continuation_session_active, forward_continuation_macro_pending
        if not forward_continuation_eligible:
            return
        forward_continuation_eligible = False
        forward_continuation_session_active = False
        forward_continuation_macro_pending = False
        forward_continuation_remaining_ticks = 0
        forward_continuation_cancellations[reason] += 1
        logger.event("forward_continuation_cancelled", tick=tick, reason=reason)

    def _submit_action(action: MacroAction) -> None:
        """Submit a local-safe action and keep any target bearing relative."""
        executor.submit(action)
        cave_target.observe_action(action)

    def _submit_observation(
        tick: int,
        pov: np.ndarray,
        previous_action: dict[str, Any] | None,
        visual_change: dict[str, Any] | None = None,
    ) -> None:
        planner.submit(
            episode_id,
            tick,
            pov,
            previous_action,
            visual_change,
            cave_target.snapshot().to_log_dict(),
        )

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
        if watch:
            adapter.render()
        change_detector.reset(observation["pov"])
        Image.fromarray(observation["pov"]).save(run_dir / "initial.png")
        logger.event("reset", pov_shape=list(observation["pov"].shape), seed=seed)
        Image.fromarray(observation["pov"]).save(frames_dir / "tick-0000.png")
        frame_cache.put(episode_id, 0, observation["pov"])
        _submit_observation(0, observation["pov"], None)
        logger.event(
            "observation_published",
            tick=0,
            frame="decision_frames/tick-0000.png",
            visual_change=None,
        )

        while completed_ticks < tick_budget:
            tick_started = time.perf_counter()
            pending_ack: PlannerDecision | None = None
            decision_applied_this_tick = False
            if stop_all.is_set():
                watchdog.request_stop("SIGINT")
            if planner.error:
                watchdog.request_stop("planner_error")
            if watchdog.should_stop and watchdog.reason not in (None, "max_ticks"):
                _cancel_forward_continuation("watchdog", tick=completed_ticks)
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
                    decision_applied_this_tick = True
                    decisions += 1
                    decision_latencies.append(decision.latency_seconds)
                    # A real planner decision always takes priority over the
                    # local forward-continuation safety layer; abandon any
                    # active session rather than resume it out of context.
                    _cancel_forward_continuation(
                        "planner_decision", tick=completed_ticks
                    )
                    target_before_decision = cave_target.snapshot()
                    action_to_execute = decision.action
                    recovery_applied = False
                    camera_pitch_guard_applied = False
                    cave_candidate_validated = False
                    cave_text_evidence_complete = False
                    cave_frame_plausible: bool | None = None
                    candidate_gate_direction: str | None = None
                    candidate_direction_source: str | None = None
                    candidate_gate_frame_tick: int | None = None
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
                        # Text evidence alone can be a model hallucination (e.g.
                        # a sunlit sandstone wall described as a "dark stone
                        # opening"). A candidate must also survive a local,
                        # deterministic frame veto -- applied to the exact frame
                        # published at decision.observation_tick, restricted to
                        # the claimed left/center/right band -- before it is
                        # counted. This still never auto-confirms a cave, it
                        # only avoids counting obviously implausible frames.
                        cave_text_evidence_complete = is_cave_candidate(
                            decision.action
                        )
                        if cave_text_evidence_complete:
                            candidate_gate_direction = resolve_cave_direction(
                                decision.action.reason
                            )
                            evidence_frame = frame_cache.get(
                                episode_id, decision.observation_tick
                            )
                            if evidence_frame is None:
                                cave_candidate_frame_missing += 1
                                cave_frame_plausible = False
                                logger.event(
                                    "cave_candidate_frame_missing",
                                    tick=completed_ticks,
                                    observation_tick=decision.observation_tick,
                                    reason=decision.action.reason,
                                )
                            elif candidate_gate_direction is None:
                                # Ambiguous or unstated direction: fail
                                # closed instead of accepting a dark patch
                                # anywhere in the frame.
                                cave_frame_plausible = False
                            else:
                                candidate_gate_frame_tick = decision.observation_tick
                                cave_frame_plausible = (
                                    has_directional_stone_bounded_dark_opening_region(
                                        evidence_frame, candidate_gate_direction
                                    )
                                )
                                candidate_direction_source = "model_reason"
                                if not cave_frame_plausible:
                                    local_direction = resolve_dark_opening_direction(
                                        evidence_frame
                                    )
                                    if local_direction is not None:
                                        candidate_gate_direction = local_direction
                                        candidate_direction_source = "local_dark_region"
                                        cave_frame_plausible = (
                                            has_directional_stone_bounded_dark_opening_region(
                                                evidence_frame, local_direction
                                            )
                                        )
                                if not cave_frame_plausible:
                                    cave_candidate_frame_vetoes += 1
                                    logger.event(
                                        "cave_candidate_frame_vetoed",
                                        tick=completed_ticks,
                                        observation_tick=decision.observation_tick,
                                        direction=candidate_gate_direction,
                                        reason=decision.action.reason,
                                    )
                            cave_candidate_validated = bool(cave_frame_plausible)
                        if cave_candidate_validated:
                            cave_candidate_decisions += 1
                            cave_candidate_observation_ticks.append(
                                decision.observation_tick
                            )
                            cave_candidate_evidence.append(
                                "decision_frames/"
                                f"tick-{decision.observation_tick:04d}.png"
                            )
                            if (
                                target_before_decision.active
                                and target_before_decision.direction
                                == candidate_gate_direction
                                and target_before_decision.forward_ticks_after_acquisition
                                >= CAVE_COMPLETION_MIN_FORWARD_TICKS
                            ):
                                cave_target_reconfirmations += 1
                                cave_completion_evidence.append(
                                    "decision_frames/"
                                    f"tick-{decision.observation_tick:04d}.png"
                                )
                                if cave_entry_phase.can_activate(
                                    cave_target_reconfirmations=cave_target_reconfirmations,
                                    forward_ticks_after_acquisition=(
                                        target_before_decision.forward_ticks_after_acquisition
                                    ),
                                    cave_completion_requested=cave_completion_requested,
                                ):
                                    # Phase 5: replace the immediate local ESC
                                    # with a bounded, locally driven forward
                                    # block. The environment owner will emit
                                    # exactly one ESC tick when that block
                                    # finishes or is aborted.
                                    cave_entry_phase_activated = True
                                    entry_snapshot = cave_entry_phase.activate(
                                        decision.observation_tick
                                    )
                                    # Stop any macro the executor is already
                                    # running (typically a leftover forward-
                                    # continuation block) so the very next
                                    # tick is the first one of the entry
                                    # phase's own forward budget. Without this
                                    # interrupt the leftover macro would keep
                                    # running forward and inflate the entry
                                    # forward-tick counter.
                                    executor.interrupt(
                                        "cave entry phase activated"
                                    )
                                    cave_entry_pre_luminance = (
                                        _world_strip_mean_luminance(
                                            observation["pov"]
                                        )
                                    )
                                    cave_entry_phase.record_pre_entry_luminance(
                                        cave_entry_pre_luminance
                                    )
                                    entry_evidence_dir.mkdir(exist_ok=True)
                                    logger.event(
                                        "cave_entry_phase_activated",
                                        tick=completed_ticks,
                                        observation_tick=decision.observation_tick,
                                        target=target_before_decision.to_log_dict(),
                                        snapshot=entry_snapshot.to_log_dict(),
                                    )
                                else:
                                    blocker = cave_entry_phase.activation_blocker(
                                        cave_target_reconfirmations=cave_target_reconfirmations,
                                        forward_ticks_after_acquisition=(
                                            target_before_decision.forward_ticks_after_acquisition
                                        ),
                                        cave_completion_requested=cave_completion_requested,
                                    )
                                    # Only fall back to the Phase 4 immediate
                                    # local ESC when the entry phase has not
                                    # already taken over. If the entry phase
                                    # is in flight (or terminal) it will
                                    # issue its own single ESC tick; calling
                                    # ``request_cave_completion`` again here
                                    # would raise.
                                    if not cave_entry_phase.is_active and not cave_entry_phase.is_terminal:
                                        cave_completion_requested = True
                                    logger.event(
                                        "cave_entry_phase_blocked",
                                        tick=completed_ticks,
                                        observation_tick=decision.observation_tick,
                                        blocker=blocker,
                                    )
                            elif candidate_gate_direction is not None:
                                cave_target.acquire(
                                    candidate_gate_direction,
                                    decision.observation_tick,
                                )
                                cave_target_acquisitions += 1
                                logger.event(
                                    "cave_target_acquired",
                                    tick=completed_ticks,
                                    observation_tick=decision.observation_tick,
                                    target=cave_target.snapshot().to_log_dict(),
                                )
                        elif target_before_decision.active:
                            cave_target.consume_decision()
                        # Phase 5: while the bounded entry phase is active the
                        # model only steers the cave gate. The environment
                        # owner runs the local forward block, so any model
                        # decision is recorded but not executed -- a turn,
                        # jump, or new move_forward here would destroy the
                        # evidence frame we are trying to capture.
                        if cave_entry_phase.is_active and not cave_completion_requested:
                            cave_entry_decisions_during_phase += 1
                            cave_entry_decisions_suppressed += 1
                            action_to_execute = MacroAction.no_op(
                                "suppressed during cave entry phase"
                            )
                            recovery_applied = False
                            water_override = False
                            water_direction = None
                            has_effect = False
                            executed_has_effect = False
                            camera_pitch_guard_applied = False
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
                        water_direction = water_hazard_direction(observation["pov"])
                        water_override = (
                            action_to_execute.action == "move_forward"
                            and water_direction is not None
                        )
                        if water_override:
                            action_to_execute = safe_water_recovery(
                                water_direction,
                                water_hazard_overrides,
                            )
                            water_hazard_overrides += 1
                            water_hazard_directions.append(water_direction)
                        (
                            action_to_execute,
                            camera_pitch_guard_applied,
                        ) = _guard_camera_pitch(
                            action_to_execute, commanded_camera_pitch_degrees
                        )
                        camera_pitch_guard_overrides += int(camera_pitch_guard_applied)
                        executed_has_effect = _macro_action_has_effect(action_to_execute)
                        executed_ineffective_decisions += int(not executed_has_effect)
                        previous_action_context = _action_context(action_to_execute)
                        # Only an accepted move_forward decision that survived
                        # untouched (no water-hazard or ineffective-action
                        # override) may open a bounded local continuation
                        # window; Qwen still chooses direction, the local
                        # safety layer only avoids idling while it waits.
                        # Phase 5: the entry phase owns the local forward
                        # budget; we must not open a forward-continuation
                        # session that would race with it.
                        if (
                            not cave_entry_phase.is_active
                            and _forward_continuation_is_eligible(
                                decision_accepted=decision.accepted,
                                decision_action=decision.action.action,
                                executed_action=action_to_execute.action,
                            )
                        ):
                            forward_continuation_eligible = True
                            forward_continuation_session_active = False
                            forward_continuation_macro_pending = False
                            forward_continuation_remaining_ticks = (
                                LOCAL_FORWARD_CONTINUATION_MAX_TICKS
                            )
                    else:
                        rejected_decisions += 1
                        has_effect = False
                        executed_has_effect = False
                    # A real planner decision always takes priority over a
                    # short local recovery scan; abandon any unfinished scan
                    # rather than resume it later out of context.
                    pending_turn_scan_actions = []
                    awaiting_turn_scan_observation = False
                    if cave_completion_requested:
                        _cancel_forward_continuation(
                            "cave_completion", tick=completed_ticks
                        )
                        executor.request_cave_completion()
                        logger.event(
                            "cave_completion_requested",
                            tick=completed_ticks,
                            observation_tick=decision.observation_tick,
                            target=target_before_decision.to_log_dict(),
                            evidence=cave_completion_evidence[-1],
                        )
                    elif cave_entry_phase.is_active:
                        # The environment owner is running the local entry
                        # forward block; we must not submit the (suppressed)
                        # model decision into the executor on top of it.
                        pass
                    else:
                        _submit_action(action_to_execute)
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
                        water_hazard_override=water_override if decision.accepted else False,
                        water_hazard_direction=water_direction if decision.accepted else None,
                        camera_pitch_guard_applied=camera_pitch_guard_applied,
                        commanded_camera_pitch_degrees=commanded_camera_pitch_degrees,
                        cave_visible=decision.action.cave_visible,
                        cave_text_evidence_complete=cave_text_evidence_complete,
                        cave_frame_plausible=cave_frame_plausible,
                        cave_candidate_validated=cave_candidate_validated,
                        candidate_gate_direction=candidate_gate_direction,
                        candidate_direction_source=candidate_direction_source,
                        candidate_gate_frame_tick=candidate_gate_frame_tick,
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
                stuck_recovery = (
                    change.low_change
                    and progress_action_ticks_since_observation > 0
                    and previous_action_context is not None
                    and previous_action_context["action"] == "move_forward"
                    and not decision_applied_this_tick
                )
                if stuck_recovery:
                    consecutive_forward_stall_periods += 1
                    if (
                        consecutive_forward_stall_periods
                        >= CONSECUTIVE_FORWARD_STALL_TURN_SCAN_THRESHOLD
                    ):
                        # Two or more consecutive forward stalls: run one
                        # fixed, capped local turn scan instead of continuing
                        # to alternate sidesteps without limit.
                        _cancel_forward_continuation("turn_scan", tick=completed_ticks)
                        if cave_entry_phase.is_active:
                            _cancel_forward_continuation(
                                "cave_entry_interrupted", tick=completed_ticks
                            )
                            cave_entry_phase.abort(
                                reason="turn_scan", tick=completed_ticks
                            )
                            cave_entry_cancellation_reason = "turn_scan"
                            logger.event(
                                "cave_entry_phase_aborted",
                                tick=completed_ticks,
                                reason="turn_scan",
                                snapshot=cave_entry_phase.snapshot().to_log_dict(),
                            )
                        pending_turn_scan_actions = safe_turn_scan_recovery(
                            turn_scan_recoveries
                        )
                        turn_scan_recoveries += 1
                        action_to_execute = pending_turn_scan_actions.pop(0)
                        _submit_action(action_to_execute)
                        previous_action_context = _action_context(action_to_execute)
                        awaiting_turn_scan_observation = True
                        logger.event(
                            "bounded_turn_scan_recovery_started",
                            tick=completed_ticks,
                            visual_change=visual_change,
                            consecutive_forward_stall_periods=(
                                consecutive_forward_stall_periods
                            ),
                            remaining_scan_steps=len(pending_turn_scan_actions),
                            executed=action_to_execute.to_log_dict(),
                        )
                    else:
                        _cancel_forward_continuation("low_progress", tick=completed_ticks)
                        if cave_entry_phase.is_active:
                            _cancel_forward_continuation(
                                "cave_entry_interrupted", tick=completed_ticks
                            )
                            cave_entry_phase.abort(
                                reason="low_progress", tick=completed_ticks
                            )
                            cave_entry_cancellation_reason = "low_progress"
                            logger.event(
                                "cave_entry_phase_aborted",
                                tick=completed_ticks,
                                reason="low_progress",
                                snapshot=cave_entry_phase.snapshot().to_log_dict(),
                            )
                        action_to_execute = safe_stuck_recovery(stuck_recovery_actions)
                        _submit_action(action_to_execute)
                        previous_action_context = _action_context(action_to_execute)
                        stuck_recovery_actions += 1
                        logger.event(
                            "low_progress_recovery",
                            tick=completed_ticks,
                            visual_change=visual_change,
                            executed=action_to_execute.to_log_dict(),
                        )
                else:
                    consecutive_forward_stall_periods = 0
                frame_cache.put(episode_id, completed_ticks, observation["pov"])
                _submit_observation(
                    completed_ticks,
                    observation["pov"],
                    previous_action_context,
                    visual_change,
                )
                periodic_observations += 1
                logger.event(
                    "observation_published",
                    tick=completed_ticks,
                    frame=f"decision_frames/{frame_name}",
                    visual_change=visual_change,
                    progress_action_ticks=progress_action_ticks_since_observation,
                    source="periodic",
                )
                progress_action_ticks_since_observation = 0

            if pending_turn_scan_actions and executor.needs_action:
                action_to_execute = pending_turn_scan_actions.pop(0)
                _submit_action(action_to_execute)
                previous_action_context = _action_context(action_to_execute)

            tick_action = executor.next_tick()
            commanded_camera_pitch_degrees = min(
                CAMERA_PITCH_GUARD_MAX_DEGREES,
                max(
                    CAMERA_PITCH_GUARD_MIN_DEGREES,
                    commanded_camera_pitch_degrees + float(tick_action["camera"][0]),
                ),
            )
            camera_changed = bool(
                float(tick_action["camera"][0]) or float(tick_action["camera"][1])
            )
            action_changed = bool(
                camera_changed
                or tick_action["attack"]
                or tick_action["back"]
                or tick_action["forward"]
                or tick_action["jump"]
                or tick_action["left"]
                or tick_action["right"]
                or tick_action["sprint"]
            )
            no_op_ticks += int(not action_changed)
            executing_forward_tick = bool(tick_action["forward"])
            forward_ticks += int(executing_forward_tick)
            if executing_forward_tick:
                cave_target.observe_forward_tick()
            # Count only ticks actually handed to MineRL while a macro that
            # the continuation layer itself submitted was executing -- never
            # the size of a macro merely allocated/queued. This keeps
            # forward_continuation_ticks a strict subset of forward_ticks.
            if forward_continuation_macro_pending and executing_forward_tick:
                forward_continuation_ticks += 1
            # Phase 5: count the real forward ticks that the entry phase
            # itself drove. Like forward_continuation_ticks, this is a
            # strict subset of forward_ticks and is incremented per real
            # tick, never at macro-allocation time.
            if cave_entry_phase.is_active and executing_forward_tick:
                cave_entry_phase.record_forward_tick()
            retreat_ticks += int(bool(tick_action["back"]))
            sidestep_left_ticks += int(bool(tick_action["left"]))
            sidestep_right_ticks += int(bool(tick_action["right"]))
            progress_action_ticks_since_observation += int(action_changed)
            esc_nonzero += int(bool(tick_action["ESC"]))
            step_started = time.perf_counter()
            step = adapter.step(tick_action)
            if watch:
                adapter.render()
            step_elapsed = time.perf_counter() - step_started
            step_latencies.append(step_elapsed)
            completed_ticks += 1
            watchdog.after_tick()
            reward_sum += step.reward
            observation = step.observation

            # The local forward-continuation safety layer runs without a
            # fresh model decision, so it re-checks the existing water-hazard
            # guard every tick (not just at decision time) and immediately
            # overrides whatever is currently executing, mid-macro if needed.
            if forward_continuation_eligible or cave_entry_phase.is_active:
                continuation_water_direction = water_hazard_direction(
                    observation["pov"]
                )
                if continuation_water_direction is not None:
                    if forward_continuation_eligible:
                        _cancel_forward_continuation(
                            "water_hazard", tick=completed_ticks
                        )
                    if cave_entry_phase.is_active:
                        _cancel_forward_continuation(
                            "cave_entry_interrupted", tick=completed_ticks
                        )
                        cave_entry_phase.abort(
                            reason="water_hazard", tick=completed_ticks
                        )
                        cave_entry_cancellation_reason = "water_hazard"
                        logger.event(
                            "cave_entry_phase_aborted",
                            tick=completed_ticks,
                            reason="water_hazard",
                            snapshot=cave_entry_phase.snapshot().to_log_dict(),
                        )
                    water_recovery_action = safe_water_recovery(
                        continuation_water_direction, water_hazard_overrides
                    )
                    water_hazard_overrides += 1
                    water_hazard_directions.append(continuation_water_direction)
                    _submit_action(water_recovery_action)
                    previous_action_context = _action_context(water_recovery_action)
                    logger.event(
                        "forward_continuation_water_override",
                        tick=completed_ticks,
                        water_hazard_direction=continuation_water_direction,
                        executed=water_recovery_action.to_log_dict(),
                    )

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

            # A short forward macro can finish long before the next fixed
            # sampling slot. Start the next inference from its actual
            # post-action frame immediately, without changing the macro length
            # or waiting in the MineRL loop. Camera-only actions retain the
            # periodic visual-change feedback to avoid a turn loop.
            if _should_publish_macro_completion_observation(
                action_completed=executor.needs_action,
                completed_action=executor.current.action,
                planner_idle=planner.idle.is_set(),
            ):
                frame_name = f"tick-{completed_ticks:04d}.png"
                Image.fromarray(observation["pov"]).save(frames_dir / frame_name)
                frame_cache.put(episode_id, completed_ticks, observation["pov"])
                _submit_observation(
                    completed_ticks,
                    observation["pov"],
                    previous_action_context,
                )
                macro_completion_observations += 1
                logger.event(
                    "observation_published",
                    tick=completed_ticks,
                    frame=f"decision_frames/{frame_name}",
                    visual_change=None,
                    progress_action_ticks=0,
                    source="macro_completion",
                )

            # Local forward-continuation safety layer: once the current
            # forward macro (the model's original decision or a previous
            # continuation step) is fully executed and the bounded budget is
            # not exhausted, keep making limited forward progress instead of
            # idling while the next (slow) decision is still pending. This
            # never waits on the planner; it only checks already-available
            # local state on this same tick.
            if (
                forward_continuation_eligible
                and not cave_entry_phase.is_active
                and executor.needs_action
                and executor.current.action == "move_forward"
                and forward_continuation_remaining_ticks > 0
            ):
                if not forward_continuation_session_active:
                    forward_continuation_session_active = True
                    forward_continuation_sessions_started += 1
                    logger.event(
                        "forward_continuation_started",
                        tick=completed_ticks,
                        budget_ticks=LOCAL_FORWARD_CONTINUATION_MAX_TICKS,
                    )
                continuation_action = safe_forward_continuation(
                    _forward_continuation_next_duration(
                        forward_continuation_remaining_ticks
                    )
                )
                forward_continuation_remaining_ticks -= continuation_action.duration_ticks
                _submit_action(continuation_action)
                # This macro is now the one executing; forward_continuation_ticks
                # is incremented per real tick below, never here at allocation
                # time. The session remains active (eligible + pending) even
                # once remaining_ticks reaches 0, until this exact macro either
                # finishes on its own (handled below) or is pre-empted by a
                # real cancellation reason.
                forward_continuation_macro_pending = True
                previous_action_context = _action_context(continuation_action)
                logger.event(
                    "forward_continuation_extended",
                    tick=completed_ticks,
                    duration_ticks=continuation_action.duration_ticks,
                    remaining_ticks=forward_continuation_remaining_ticks,
                )
            elif (
                forward_continuation_eligible
                and forward_continuation_macro_pending
                and forward_continuation_remaining_ticks == 0
                and executor.needs_action
            ):
                # The last allocated continuation macro has now fully
                # executed and no budget remains to extend it further: a
                # natural, non-cancelled end of the session. Until this
                # branch (or a real cancellation reason above) fires, the
                # session stayed active even though remaining_ticks was
                # already 0, so any pre-emption while this final macro was
                # still running was correctly counted as a cancellation.
                forward_continuation_eligible = False
                forward_continuation_session_active = False
                forward_continuation_macro_pending = False
                logger.event(
                    "forward_continuation_completed",
                    tick=completed_ticks,
                    reason="budget_exhausted",
                )

            # Phase 5 entry phase: once the local forward block (originally
            # submitted at activation time) finishes and budget remains, keep
            # walking forward. This borrows the deterministic safety-layer
            # pattern used by forward_continuation above: it never blocks
            # on the planner, only re-reads already-available local state.
            # Entry and forward_continuation are intentionally exclusive
            # (checked above); this branch only runs when the entry phase
            # owns the forward budget.
            if (
                cave_entry_phase.is_active
                and executor.needs_action
                and cave_entry_phase.remaining_budget() > 0
            ):
                entry_duration = min(
                    CAVE_ENTRY_PHASE_MACRO_TICKS,
                    cave_entry_phase.remaining_budget(),
                )
                granted = cave_entry_phase.consume_budget(entry_duration)
                if granted > 0:
                    entry_action = safe_forward_continuation(granted)
                    _submit_action(entry_action)
                    cave_entry_local_forward_actions += 1
                    previous_action_context = _action_context(entry_action)
                    logger.event(
                        "cave_entry_forward_extended",
                        tick=completed_ticks,
                        duration_ticks=granted,
                        remaining_budget=cave_entry_phase.remaining_budget(),
                        snapshot=cave_entry_phase.snapshot().to_log_dict(),
                    )
            elif (
                cave_entry_phase.is_active
                and executor.needs_action
                and cave_entry_phase.remaining_budget() == 0
            ):
                # All budget consumed and the last local macro has fully
                # executed. Always capture the post-frame evidence first --
                # the file is the only artifact left for human review when
                # the local plausibility check refuses to fire ESC. The
                # single local ESC tick is then queued *only* when the
                # evidence is plausible. When it is not, the phase is
                # sealed in the ``unverified`` terminal state and the
                # episode must continue without an ESC so the operator
                # can review the evidence frame manually.
                _cancel_forward_continuation("cave_entry_complete", tick=completed_ticks)
                post_luminance = _world_strip_mean_luminance(observation["pov"])
                cave_entry_post_luminance = post_luminance
                evidence_name = f"post-tick-{completed_ticks:04d}.png"
                evidence_path = entry_evidence_dir / evidence_name
                Image.fromarray(observation["pov"]).save(evidence_path)
                relative_evidence = f"entry_evidence/{evidence_name}"
                cave_entry_evidence_path = relative_evidence
                cave_entry_completion_tick = completed_ticks
                entry_snapshot = cave_entry_phase.complete(
                    tick=completed_ticks,
                    evidence_frame_path=relative_evidence,
                    post_entry_luminance=post_luminance,
                )
                if entry_snapshot.state == "entered":
                    cave_completion_requested = True
                    executor.request_cave_completion()
                    logger.event(
                        "cave_entry_phase_completed",
                        tick=completed_ticks,
                        snapshot=entry_snapshot.to_log_dict(),
                        evidence=relative_evidence,
                        pre_luminance=cave_entry_pre_luminance,
                        post_luminance=post_luminance,
                    )
                else:
                    # The post-entry frame was not plausibly inside the
                    # cave. The single local ESC tick is suppressed so a
                    # human can review the saved evidence before any
                    # environment-level completion. ``cave_completion_
                    # requested`` stays False; termination_reason will
                    # fall through to ``tick_budget`` or
                    # ``environment_done`` when the loop exits.
                    cave_entry_cancellation_reason = "plausibility_failed"
                    logger.event(
                        "cave_entry_phase_unverified",
                        tick=completed_ticks,
                        snapshot=entry_snapshot.to_log_dict(),
                        evidence=relative_evidence,
                        pre_luminance=cave_entry_pre_luminance,
                        post_luminance=post_luminance,
                    )

            # The bounded turn scan is a short, deterministic sequence of
            # camera-only ticks; once its capped total rotation is fully
            # executed, submit a fresh observation immediately rather than
            # waiting for the next periodic sampling slot, without blocking
            # the MineRL step loop on Qwen inference.
            if (
                awaiting_turn_scan_observation
                and executor.needs_action
                and not pending_turn_scan_actions
            ):
                awaiting_turn_scan_observation = False
                consecutive_forward_stall_periods = 0
                frame_name = f"tick-{completed_ticks:04d}.png"
                Image.fromarray(observation["pov"]).save(frames_dir / frame_name)
                frame_cache.put(episode_id, completed_ticks, observation["pov"])
                _submit_observation(
                    completed_ticks,
                    observation["pov"],
                    previous_action_context,
                )
                turn_scan_completion_observations += 1
                logger.event(
                    "observation_published",
                    tick=completed_ticks,
                    frame=f"decision_frames/{frame_name}",
                    visual_change=None,
                    progress_action_ticks=0,
                    source="turn_scan_completion",
                )

            peak_rss = max(peak_rss, process.memory_info().rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            logger.event(
                "tick",
                tick=completed_ticks,
                action={
                    "ESC": tick_action["ESC"],
                    "attack": tick_action["attack"],
                    "back": tick_action["back"],
                    "camera": tick_action["camera"],
                    "forward": tick_action["forward"],
                    "jump": tick_action["jump"],
                    "left": tick_action["left"],
                    "right": tick_action["right"],
                    "sprint": tick_action["sprint"],
                },
                reward=step.reward,
                done=step.done,
                step_seconds=step_elapsed,
            )
            if tick_action["ESC"]:
                termination_reason = "cave_completion_requested"
                terminal_info = step.info
                logger.event(
                    "cave_completion_sent",
                    tick=completed_ticks,
                    environment_done=step.done,
                    info=step.info,
                )
                break
            if step.done:
                _cancel_forward_continuation("environment_done", tick=completed_ticks)
                if cave_entry_phase.is_active:
                    _cancel_forward_continuation(
                        "cave_entry_interrupted", tick=completed_ticks
                    )
                    cave_entry_phase.abort(
                        reason="environment_done", tick=completed_ticks
                    )
                    cave_entry_cancellation_reason = "environment_done"
                    logger.event(
                        "cave_entry_phase_aborted",
                        tick=completed_ticks,
                        reason="environment_done",
                        snapshot=cave_entry_phase.snapshot().to_log_dict(),
                    )
                early_done = completed_ticks < tick_budget
                termination_reason = "environment_done"
                terminal_info = step.info
                logger.event(
                    "done",
                    tick=completed_ticks,
                    early=early_done,
                    info=step.info,
                )
                break

            if completed_ticks < tick_budget:
                sleep_seconds = _tick_sleep_seconds(tick_started, time.perf_counter())
                if sleep_seconds > 0:
                    sleep_started = time.perf_counter()
                    stop_all.wait(sleep_seconds)
                    paced_sleep_seconds += time.perf_counter() - sleep_started

        # Idempotent: a no-op if a decision, water hazard, low-progress, turn
        # scan, or environment_done already cancelled the session above.
        _cancel_forward_continuation("max_ticks", tick=completed_ticks)
        if cave_entry_phase.is_active:
            _cancel_forward_continuation(
                "cave_entry_interrupted", tick=completed_ticks
            )
            cave_entry_phase.abort(reason="max_ticks", tick=completed_ticks)
            cave_entry_cancellation_reason = "max_ticks"
            logger.event(
                "cave_entry_phase_aborted",
                tick=completed_ticks,
                reason="max_ticks",
                snapshot=cave_entry_phase.snapshot().to_log_dict(),
            )
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
        cave_completion_requested=cave_completion_requested,
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
        "retreat_ticks": retreat_ticks,
        "sidestep_left_ticks": sidestep_left_ticks,
        "sidestep_right_ticks": sidestep_right_ticks,
        "forward_continuation_ticks": forward_continuation_ticks,
        "forward_continuation_max_ticks": LOCAL_FORWARD_CONTINUATION_MAX_TICKS,
        "forward_continuation_sessions_started": forward_continuation_sessions_started,
        "forward_continuation_cancellations": forward_continuation_cancellations,
        "frame_change_samples": len(frame_changes),
        "periodic_observations": periodic_observations,
        "macro_completion_observations": macro_completion_observations,
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
        "water_hazard_overrides": water_hazard_overrides,
        "water_hazard_directions": water_hazard_directions,
        "low_progress_recovery_actions": stuck_recovery_actions,
        "bounded_turn_scan_recoveries": turn_scan_recoveries,
        "turn_scan_completion_observations": turn_scan_completion_observations,
        "camera_pitch_guard_overrides": camera_pitch_guard_overrides,
        "final_commanded_camera_pitch_degrees": commanded_camera_pitch_degrees,
        "cave_candidate_decisions": cave_candidate_decisions,
        "raw_cave_visible_decisions": raw_cave_visible_decisions,
        "cave_candidate_frame_vetoes": cave_candidate_frame_vetoes,
        "cave_candidate_frame_missing": cave_candidate_frame_missing,
        "cave_candidate_observation_ticks": cave_candidate_observation_ticks,
        "cave_candidate_evidence": cave_candidate_evidence,
        "cave_target_acquisitions": cave_target_acquisitions,
        "cave_target_reconfirmations": cave_target_reconfirmations,
        "final_cave_target": cave_target.snapshot().to_log_dict(),
        "cave_completion_requested": cave_completion_requested,
        "cave_completion_evidence": cave_completion_evidence,
        "cave_entry_phase": {
            "enabled": cave_entry_phase_enabled,
            "state": cave_entry_phase.state,
            "is_terminal": cave_entry_phase.is_terminal,
            "activation_tick": cave_entry_phase.snapshot().activation_tick,
            "completion_tick": cave_entry_completion_tick,
            "max_ticks": cave_entry_phase.snapshot().entry_budget_ticks,
            "entry_forward_ticks": cave_entry_phase.snapshot().entry_forward_ticks,
            "entry_local_forward_actions": cave_entry_local_forward_actions,
            "cancellation_reason": cave_entry_cancellation_reason,
            "evidence_frame": cave_entry_evidence_path,
            "pre_entry_luminance": cave_entry_pre_luminance,
            "post_entry_luminance": cave_entry_post_luminance,
            "plausible": cave_entry_phase.snapshot().plausible,
            "decisions_during_phase": cave_entry_decisions_during_phase,
            "decisions_suppressed": cave_entry_decisions_suppressed,
        },
        "peak_process_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "esc_nonzero_ticks": esc_nonzero,
        "termination_reason": (
            "tick_budget"
            if completed_ticks == tick_budget
            else termination_reason or watchdog.reason
        ),
        "terminal_info": terminal_info,
        "manual_review": {
            "required": True,
            "status": "pending",
            "cave_found": None,
            "evidence": [
                "initial.png",
                "final.png",
                "decision_frames/",
                *([cave_entry_evidence_path] if cave_entry_evidence_path else []),
                *cave_candidate_evidence,
            ],
            "note": (
                "Cave candidates require frame review. A local ESC is emitted only after "
                "two validated cave frames separated by real forward progress. Phase 5 "
                "adds an optional bounded entry phase: when enabled, the agent walks a "
                "few more local forward ticks into the validated opening and records a "
                "post-entry evidence frame before the single local ESC tick."
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
    seed: int | None = None,
    watch: bool = False,
    mission_max_ticks: int | None = None,
    model_lock_path: Path | None = None,
    cave_entry_phase_enabled: bool = False,
    cave_entry_phase_max_ticks: int = CAVE_ENTRY_PHASE_MAX_TICKS,
) -> dict[str, Any]:
    if episodes < 1 or ticks < 1 or observation_interval < 1:
        raise ValueError("episodes, ticks, and observation_interval must be positive")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    if mission_max_ticks is not None and (
        type(mission_max_ticks) is not int or mission_max_ticks < 1
    ):
        raise ValueError("mission_max_ticks must be a positive integer or None")
    if model_lock_path is not None and not model_lock_path.is_file():
        raise ValueError(f"model lock does not exist: {model_lock_path}")
    if not isinstance(cave_entry_phase_enabled, bool):
        raise ValueError("cave_entry_phase_enabled must be a boolean")
    if (
        type(cave_entry_phase_max_ticks) is not int
        or cave_entry_phase_max_ticks < 1
    ):
        raise ValueError("cave_entry_phase_max_ticks must be a positive integer")
    output_root = output_root or ROOT / "runs" / "episodes"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    stop_all = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_all.set())
    planner = QwenPlannerWorker(lock_path=model_lock_path)
    results: list[dict[str, Any]] = []
    planner.start()
    try:
        if not planner.ready.wait(30):
            raise TimeoutError("Qwen planner did not load within 30 seconds")
        if planner.error:
            raise RuntimeError(planner.error)
        with MineRLEnvAdapter(max_episode_steps=mission_max_ticks) as adapter:
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
                        seed=seed,
                        watch=watch,
                        mission_max_ticks=mission_max_ticks,
                        model_lock_path=model_lock_path,
                        cave_entry_phase_enabled=cave_entry_phase_enabled,
                        cave_entry_phase_max_ticks=cave_entry_phase_max_ticks,
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
        "seed": seed,
        "ticks_per_episode": ticks,
        "mission_max_ticks": mission_max_ticks,
        "model_lock_path": str(model_lock_path) if model_lock_path else "model.lock.json",
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
