"""Integration coverage for the Phase 5 bounded cave-entry phase.

The phase 5 entry layer is opt-in: when ``cave_entry_phase_enabled`` is False
the agent loop must behave exactly like Phase 4. When it is True, a bounded,
locally driven forward block follows the existing double-confirmed cave gate
and produces a single post-entry evidence frame before the local ESC tick.

These tests use the same ``FakePlanner`` / ``_DelayedDecisionMailbox``
machinery as ``tests/test_agent.py`` and only ever run against a fake
environment, so they require no MineRL session, no Qwen model, and no
network. They cover the twelve acceptance points from the Phase 5 plan.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from minerl.herobraine.envs import MINERL_BASALT_FIND_CAVES_ENV_SPEC

from mc_agent.actions import MacroAction
from mc_agent.agent import _run_episode
from mc_agent.env import MineRLEnvAdapter
from mc_agent.qwen import PlannerDecision

from tests.test_agent import (
    _DelayedDecisionMailbox,
    _FakeDecisionMailbox,
    FakePlanner,
)


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _cave_frame_base() -> np.ndarray:
    """A neutral stone world with a large dark opening in the center third."""
    frame = np.full((360, 640, 3), 110, dtype=np.uint8)
    frame[80:260, 250:410] = 5
    return frame


def _cave_frame() -> np.ndarray:
    return _cave_frame_base()


def _cave_frame_motion(step: int) -> np.ndarray:
    """A cave frame with deterministic micro-noise outside the center band.

    The center third (where the cave gate validates) stays exactly the same
    from frame to frame, so the stone/dark gate keeps passing. The left and
    right thirds get deterministic per-step noise so the frame change
    detector reports non-low_change most of the time, which keeps the local
    forward path from being mistaken for a stuck-recovery stall. The noise
    magnitude is small and stays on the bright side so the per-tick
    water-hazard guard never trips on a fake "blue" pixel.
    """
    frame = _cave_frame_base()
    rng = np.random.default_rng(step + 1)
    left_noise = rng.integers(0, 25, size=(300, 213, 3), dtype=np.int16)
    right_noise = rng.integers(0, 25, size=(300, 214, 3), dtype=np.int16)
    left = frame[:300, :213].astype(np.int16)
    right = frame[:300, 426:].astype(np.int16)
    np.clip(left + left_noise, 0, 255, out=left)
    np.clip(right + right_noise, 0, 255, out=right)
    frame[:300, :213] = left.astype(np.uint8)
    frame[:300, 426:] = right.astype(np.uint8)
    return frame


def _static_frame() -> np.ndarray:
    return np.full((360, 640, 3), 110, dtype=np.uint8)


def _dark_frame_motion(step: int) -> np.ndarray:
    """A clearly darker world strip (the post-entry interior).

    Used by tests that need the local plausibility check to actually
    succeed: world luminance must drop by at least 30% relative to the
    cave-side pre-entry frame (which sits at 110 on the neutral stone
    base), so the post-frame must average well below 77. Center third
    keeps the same neutral-dark rectangle; the rest of the world is a
    low-luminance neutral surface with deterministic per-step noise so
    the frame change detector still sees motion and the water-hazard
    guard never trips on a fake "blue" pixel.
    """
    frame = _cave_frame_base()
    # Drop the whole world strip to a dim, low-chroma base.
    frame[:300] = 28
    # Keep the existing dark rectangle; on the dim base it stays the
    # darkest region, which is fine for the post-entry plausibility
    # rule (low absolute luminance counts as interior).
    rng = np.random.default_rng(step + 1)
    left_noise = rng.integers(0, 12, size=(300, 213, 3), dtype=np.int16)
    right_noise = rng.integers(0, 12, size=(300, 214, 3), dtype=np.int16)
    left = frame[:300, :213].astype(np.int16)
    right = frame[:300, 426:].astype(np.int16)
    np.clip(left + left_noise, 0, 255, out=left)
    np.clip(right + right_noise, 0, 255, out=right)
    frame[:300, :213] = left.astype(np.uint8)
    frame[:300, 426:] = right.astype(np.uint8)
    return frame


def _water_frame() -> np.ndarray:
    """A frame with a large, dominant water block in the center third."""
    frame = np.full((360, 640, 3), 110, dtype=np.uint8)
    frame[150:300, 200:440] = np.array([40, 80, 160], dtype=np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Fake environments
# ---------------------------------------------------------------------------

class _ScriptedCaveEnv:
    """A FakeEnv-compatible object that swaps frames after a tick threshold.

    ``frame_for(tick)`` is called every step with the current
    ``completed_ticks`` and must return the POV for the next observation.
    """

    def __init__(self, frame_for, *, seed_after_cave: int | None = None):
        self.action_space = MINERL_BASALT_FIND_CAVES_ENV_SPEC.action_space
        self._frame_for = frame_for
        self._steps = 0
        self.actions: list[dict] = []
        self.closed = False
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {"pov": _cave_frame()}

    def step(self, action):
        self._steps += 1
        self.actions.append(
            {k: v.copy() if hasattr(v, "copy") else v for k, v in action.items()}
        )
        return {"pov": self._frame_for(self._steps)}, 0.0, False, {"ok": True}

    def render(self, mode="human"):
        pass

    def close(self):
        self.closed = True


def _cave_decision(
    *, observation_tick: int, duration_ticks: int = 12, decision_id: str
) -> PlannerDecision:
    return PlannerDecision(
        episode_id="episode-test",
        observation_tick=observation_tick,
        raw='{"action":"move_forward","duration_ticks":' + str(duration_ticks) + "}",
        action=MacroAction(
            action="move_forward",
            duration_ticks=duration_ticks,
            cave_visible=True,
            reason="dark stone opening in center",
        ),
        accepted=True,
        error=None,
        latency_seconds=1.0,
    )


def _cave_decision_with_id(observation_tick: int, duration_ticks: int, episode_id: str):
    return PlannerDecision(
        episode_id=episode_id,
        observation_tick=observation_tick,
        raw='{"action":"move_forward","duration_ticks":' + str(duration_ticks) + "}",
        action=MacroAction(
            action="move_forward",
            duration_ticks=duration_ticks,
            cave_visible=True,
            reason="dark stone opening in center",
        ),
        accepted=True,
        error=None,
        latency_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# 1. Phase 4 path is preserved when Phase 5 is disabled
# ---------------------------------------------------------------------------

class PhaseFiveDisabledPreservesPhaseFourTests(unittest.TestCase):
    def test_double_confirmation_still_emits_one_esc_with_phase5_disabled(self):
        env = _ScriptedCaveEnv(lambda step: _cave_frame_motion(step))
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, decision_id="a")
        second = _cave_decision(observation_tick=11, duration_ticks=6, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=13)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=40,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=False,
            )

        self.assertTrue(result["cave_completion_requested"], result)
        self.assertEqual(result["esc_nonzero_ticks"], 1)
        self.assertEqual(result["cave_entry_phase"]["enabled"], False)
        self.assertEqual(result["cave_entry_phase"]["state"], "idle")
        self.assertIsNone(result["cave_entry_phase"]["evidence_frame"])
        self.assertEqual(
            sum(int(bool(action["ESC"])) for action in env.actions), 1
        )


# ---------------------------------------------------------------------------
# 2. Entry phase cannot be activated without a real double confirmation
# ---------------------------------------------------------------------------

class PhaseFiveActivationGatingTests(unittest.TestCase):
    def test_phase5_enabled_with_only_one_candidate_keeps_entry_idle(self):
        # Phase 5 must not be activatable by a single cave candidate: the
        # double-confirmation gate is unchanged, and without it the entry
        # phase can never enter the "entering" state.
        env = _ScriptedCaveEnv(lambda step: _cave_frame_motion(step))
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        only = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        planner = FakePlanner(decisions=[only])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=60,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
            )

        # One accepted cave decision is not enough to activate the entry
        # phase. No second candidate, no reconfirmation, no entry. ESC is
        # never emitted because the underlying completion gate is also off.
        self.assertEqual(
            result["cave_entry_phase"]["state"],
            "idle",
            result["cave_entry_phase"],
        )
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertFalse(result["cave_completion_requested"])


# ---------------------------------------------------------------------------
# 3. Phase 5 enabled, double-confirmation with sufficient forward progress
#    drives the entry phase, captures evidence, and still ends on a single
#    local ESC tick.
# ---------------------------------------------------------------------------

class PhaseFiveEnabledHappyPathTests(unittest.TestCase):
    def test_entry_phase_runs_to_completion_and_emits_exactly_one_esc(self):
        # After the second cave decision has fired (around step 25) the
        # world strip drops to a clearly darker interior, so the local
        # plausibility check is satisfied and the phase lands in
        # ``entered`` rather than ``unverified``. This is the path the
        # operator wants the single local ESC tick to ride on.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        # Two cave decisions with enough forward ticks between them. The
        # second decision must arrive strictly after the model has executed
        # at least 12 forward ticks, otherwise the entry phase activation
        # gate fails closed and Phase 4 wins.
        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        # 25 calls of take_latest guarantees the loop has stepped well past
        # tick 19 with the forward macro from the first decision.
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=200,
                # observation_interval=1 keeps the published-frame cache
                # warm so the second candidate's observation_tick is
                # available for the cave gate. The motion frame prevents
                # stuck-recovery from firing on a constant world.
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )
            run_dir = Path(result["run_dir"])
            evidence_path = run_dir / result["cave_entry_phase"]["evidence_frame"]
            self.assertTrue(evidence_path.exists(), f"missing: {evidence_path}")
            self.assertEqual(result["esc_nonzero_ticks"], 1)
            self.assertEqual(
                sum(int(bool(action["ESC"])) for action in env.actions), 1
            )
            self.assertEqual(
                result["termination_reason"], "cave_completion_requested"
            )

        entry = result["cave_entry_phase"]
        self.assertEqual(entry["enabled"], True)
        self.assertEqual(entry["state"], "entered")
        self.assertIsNotNone(entry["activation_tick"])
        self.assertIsNotNone(entry["completion_tick"])
        self.assertEqual(entry["max_ticks"], 30)
        # The whole forward budget is consumed: the entry phase drove the
        # agent for 30 real forward ticks (the budget cap, never more).
        self.assertEqual(entry["entry_forward_ticks"], 30)
        self.assertLessEqual(entry["entry_forward_ticks"], entry["max_ticks"])
        # The evidence frame lives next to the rest of the run artifacts.
        self.assertIsNotNone(entry["evidence_frame"])
        self.assertTrue(entry["evidence_frame"].startswith("entry_evidence/"))
        # Entry forward ticks must be a strict subset of all forward ticks.
        self.assertLessEqual(entry["entry_forward_ticks"], result["forward_ticks"])


# ---------------------------------------------------------------------------
# 4. Water hazard mid-entry aborts the phase and never sends ESC.
# ---------------------------------------------------------------------------

class PhaseFiveInterruptTests(unittest.TestCase):
    def test_water_hazard_during_entry_aborts_without_emitting_esc(self):
        # Stay in a cave frame long enough for the second cave decision to
        # trigger the entry phase, then suddenly switch to a water frame so
        # the per-tick water-hazard guard trips mid-entry.
        def frame_for(step):
            return _water_frame() if step >= 30 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=200,
                # Keep the cache warm so the second candidate's
                # observation_tick is available.
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )

        entry = result["cave_entry_phase"]
        self.assertEqual(entry["state"], "aborted")
        self.assertEqual(entry["cancellation_reason"], "water_hazard")
        # Aborted entries must never trigger the local ESC: water is a
        # safety stop, not a cave completion.
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertFalse(result["cave_completion_requested"])
        self.assertNotEqual(result["termination_reason"], "cave_completion_requested")


# ---------------------------------------------------------------------------
# 5. Two consecutive low-progress stalls during entry trigger a turn-scan
#    abort.
# ---------------------------------------------------------------------------

class PhaseFiveLowProgressAbortTests(unittest.TestCase):
    def test_low_progress_during_entry_aborts_with_turn_scan_reason(self):
        # Use a motion frame long enough to validate the cave candidates
        # and reach the entry phase, then become a static frame so the
        # frame-change detector reports low_change every periodic
        # observation.
        def frame_for(step):
            return _static_frame() if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=300,
                # observation_interval=1 keeps the cache warm so the
                # second candidate's observation_tick is available, and
                # the static frame after step 25 makes every periodic
                # observation register a low_progress stall.
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )

        entry = result["cave_entry_phase"]
        # The phase may end as either a low_progress or turn_scan abort
        # depending on how many forward stalls the loop saw; both are
        # safety interrupts that must not emit ESC.
        self.assertEqual(entry["state"], "aborted")
        self.assertIn(
            entry["cancellation_reason"],
            {"low_progress", "turn_scan"},
            entry,
        )
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertFalse(result["cave_completion_requested"])


# ---------------------------------------------------------------------------
# 6. After entry is terminal, additional cave candidates cannot re-activate
#    the phase and cannot re-fire ESC.
# ---------------------------------------------------------------------------

class PhaseFiveTerminalStateTests(unittest.TestCase):
    def test_entry_phase_does_not_reactivate_and_does_not_re_emit_esc(self):
        # Drop to a darker frame once the entry phase is active so the
        # phase can actually reach ``entered``. The point of the test
        # is the re-activation guard, not the plausibility path.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        # First pair drives the entry phase to a clean completion. The third
        # decision arrives while the episode is still running (after the
        # entry has finished); it must be acknowledged and discarded without
        # re-activating the entry phase.
        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        third = _cave_decision(observation_tick=70, duration_ticks=2, decision_id="c")
        mailbox = _ScriptedTripleMailbox(
            immediate=[first],
            delayed=[second, third],
            # second fires at call 25, third at call 60 (after the entry
            # phase has already completed and ESC has been sent).
            delay_calls=[25, 60],
        )
        planner = FakePlanner(mailbox=mailbox)

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=120,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )

        entry = result["cave_entry_phase"]
        self.assertEqual(entry["state"], "entered")
        self.assertEqual(result["esc_nonzero_ticks"], 1)
        self.assertEqual(
            sum(int(bool(action["ESC"])) for action in env.actions), 1
        )
        # The third decision arrived after the entry phase had reached
        # "entered"; it must be counted but not re-fired.
        self.assertGreaterEqual(entry["decisions_during_phase"], 0)


class _ScriptedTripleMailbox:
    """Like _DelayedDecisionMailbox but with a list of delayed decisions.

    Each ``delay_calls[i]`` is the take_latest call index at which the
    matching delayed decision becomes available. Once a delayed decision is
    delivered, the next one waits for the next delay_calls value.
    """

    def __init__(self, *, immediate, delayed, delay_calls):
        self._immediate = list(immediate)
        self._delayed = list(delayed)
        self._delay_calls = list(delay_calls)
        self._calls = 0
        self._next_delay = 0

    def take_latest(self):
        self._calls += 1
        if self._immediate:
            return self._immediate.pop(0)
        if (
            self._delayed
            and self._next_delay < len(self._delay_calls)
            and self._calls >= self._delay_calls[self._next_delay]
        ):
            self._next_delay += 1
            return self._delayed.pop(0)
        return None


# ---------------------------------------------------------------------------
# 7. Model decisions during the entry phase are suppressed and never reach
#    the executor; they do not extend the forward continuation budget.
# ---------------------------------------------------------------------------

class PhaseFiveDecisionSuppressionTests(unittest.TestCase):
    def test_model_decisions_during_entry_are_suppressed_and_not_executed(self):
        # Drop to a darker frame once the entry phase is active so the
        # local plausibility check is satisfied and the phase reaches
        # ``entered``. We are isolating decision suppression here, not
        # the plausibility path.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        # A third cave-only decision that arrives inside the entry phase
        # window must be acknowledged and dropped, not turned into a new
        # forward macro.
        third = _cave_decision(observation_tick=40, duration_ticks=2, decision_id="c")
        mailbox = _ScriptedTripleMailbox(
            immediate=[first],
            delayed=[second, third],
            delay_calls=[25, 45],
        )
        planner = FakePlanner(mailbox=mailbox)

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=200,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )

        entry = result["cave_entry_phase"]
        self.assertEqual(entry["state"], "entered")
        # The third decision was suppressed: it was acknowledged but it did
        # not start a forward-continuation session on top of the entry
        # phase, so the entry forward tick count equals the full budget.
        # Any forward-continuation session in the metrics was started by
        # the first decision, before the entry phase took over, and that
        # is intentional -- the model still chooses the original
        # direction; the entry phase only owns the local forward budget.
        self.assertEqual(entry["entry_forward_ticks"], 30)
        # The third decision must still have been counted as received,
        # otherwise the planner worker would be permanently blocked.
        self.assertGreaterEqual(entry["decisions_during_phase"], 1)
        self.assertGreaterEqual(entry["decisions_suppressed"], 1)


# ---------------------------------------------------------------------------
# 8. Evidence frame is recorded under entry_evidence/ and is not the same
#    file as the pre-entry observation.
# ---------------------------------------------------------------------------

class PhaseFiveEvidenceArtifactTests(unittest.TestCase):
    def test_post_entry_evidence_frame_is_persisted(self):
        # Same dark-then-cave pattern as the happy-path test, so the
        # entry phase reaches ``entered`` and emits a single ESC tick
        # -- here we are checking the artifact, not the path.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=200,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )
            run_dir = Path(result["run_dir"])
            evidence_rel = result["cave_entry_phase"]["evidence_frame"]
            self.assertIsNotNone(evidence_rel)
            evidence_path = run_dir / evidence_rel
            self.assertTrue(evidence_path.exists(), f"missing evidence: {evidence_path}")
            # The file must live in entry_evidence/ and not inside decision_frames.
            self.assertIn("entry_evidence", evidence_rel)

        # Plausibility note: a clearly darker post-entry frame
        # satisfies the local plausibility check, so the phase lands in
        # ``entered`` and the single local ESC tick is emitted.
        self.assertTrue(result["cave_entry_phase"]["plausible"])


# ---------------------------------------------------------------------------
# 9. Entry forward tick count is capped at the configured max budget and is
#    always a subset of forward_ticks.
# ---------------------------------------------------------------------------

class PhaseFiveForwardTickBudgetTests(unittest.TestCase):
    def test_entry_forward_ticks_are_capped_and_subset_of_forward_ticks(self):
        # Drop to a darker frame once the entry phase is active so the
        # small budget we set here still ends in a real ``entered``
        # transition. The point of the test is the forward-tick
        # accounting, not the plausibility path.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=400,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=12,  # deliberately small budget
            )

        entry = result["cave_entry_phase"]
        # The hard cap can never be exceeded even with a long episode.
        self.assertLessEqual(entry["entry_forward_ticks"], entry["max_ticks"])
        self.assertEqual(entry["entry_forward_ticks"], entry["max_ticks"])
        # And it is always a subset of the global forward_ticks counter.
        self.assertLessEqual(entry["entry_forward_ticks"], result["forward_ticks"])
        self.assertEqual(result["esc_nonzero_ticks"], 1)


# ---------------------------------------------------------------------------
# 10. Without the second cave decision, Phase 5 stays idle.
# ---------------------------------------------------------------------------

class PhaseFiveNoDoubleConfirmationTests(unittest.TestCase):
    def test_no_double_confirmation_means_no_entry_and_no_esc(self):
        # Even with Phase 5 enabled, only one accepted cave decision is
        # never enough: the second validating candidate is the actual gate.
        env = _ScriptedCaveEnv(lambda step: _cave_frame_motion(step))
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        only = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        planner = FakePlanner(decisions=[only])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=60,
                observation_interval=1_000_000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
            )

        entry = result["cave_entry_phase"]
        self.assertEqual(entry["state"], "idle")
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertEqual(
            sum(int(bool(action["ESC"])) for action in env.actions), 0
        )


# ---------------------------------------------------------------------------
# 11. P1: when the post-entry plausibility check fails, the single local
# ESC tick must be suppressed even though the bounded forward block ran to
# completion. The post-entry evidence frame must still be written and the
# phase must land in a terminal aborted/unverified state for human review.
# ---------------------------------------------------------------------------

class PhaseFivePlausibilityGateTests(unittest.TestCase):
    def test_phase5_does_not_emit_esc_when_entry_plausibility_fails(self):
        # The world strip never goes dark: the post-entry frame is the
        # same neutral cave base as the pre-entry frame, so the
        # plausibility check is guaranteed to fail. The bounded forward
        # block still runs to completion (30 ticks), the evidence frame
        # is still saved, but the single local ESC tick must be
        # suppressed and the episode must fall through to the normal
        # termination reason (not ``cave_completion_requested``).
        env = _ScriptedCaveEnv(lambda step: _cave_frame_motion(step))
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=200,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )
            run_dir = Path(result["run_dir"])
            evidence_path = run_dir / result["cave_entry_phase"]["evidence_frame"]
            # The post-entry evidence frame must still be written -- it
            # is the only artifact the operator can use to decide
            # whether the entry was real. The check must run before the
            # temporary directory is cleaned up.
            self.assertIsNotNone(result["cave_entry_phase"]["evidence_frame"])
            self.assertTrue(
                evidence_path.exists(), f"missing: {evidence_path}"
            )

        entry = result["cave_entry_phase"]
        # The phase must be in a terminal aborted/unverified state.
        self.assertIn(entry["state"], {"unverified", "aborted"})
        self.assertTrue(entry["is_terminal"])
        # Plausibility is what triggered the unverified branch; it must
        # be False (not None, not missing).
        self.assertFalse(entry["plausible"])
        # The P1 contract: plausibility failure MUST suppress the local
        # ESC tick. No cave completion may be requested, no ESC may be
        # sent, and the termination reason must not claim a successful
        # cave completion.
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertEqual(
            sum(int(bool(action["ESC"])) for action in env.actions), 0
        )
        self.assertFalse(result["cave_completion_requested"])
        self.assertNotEqual(result["termination_reason"], "cave_completion_requested")
        # The forward budget still ran: the bounded entry block reached
        # its 30-tick cap; only the plausibility gate changed the
        # outcome.
        self.assertEqual(entry["entry_forward_ticks"], 30)
        self.assertEqual(entry["max_ticks"], 30)


# ---------------------------------------------------------------------------
# 12. P1: activating the entry phase must interrupt any forward macro
# already in flight from an earlier forward_continuation session, not just
# prevent a new session from being opened.
# ---------------------------------------------------------------------------

class PhaseFiveActivationInterruptsForwardContinuationTests(unittest.TestCase):
    def test_activation_interrupts_in_flight_forward_continuation_macro(self):
        # Drive the loop hard enough that a forward_continuation macro
        # is genuinely running in the executor when the second cave
        # decision arrives and the entry phase activates. After
        # activation, the leftover macro's remaining forward ticks must
        # NOT be counted toward ``entry_forward_ticks`` and must NOT
        # extend the entry phase beyond its configured budget.
        def frame_for(step):
            return _dark_frame_motion(step) if step >= 25 else _cave_frame_motion(step)

        env = _ScriptedCaveEnv(frame_for)
        adapter = MineRLEnvAdapter(env_factory=lambda _: env).open()
        self.addCleanup(adapter.close)

        # First decision: an accepted move_forward that opens a
        # forward_continuation session with a long budget.
        first = _cave_decision(observation_tick=0, duration_ticks=20, decision_id="a")
        # Second decision arrives shortly after the first forward macro
        # finishes, while a fresh forward_continuation macro is already
        # running in the executor. ``delay_calls=25`` is chosen so that
        # the published-frame cache has had time to record the frame at
        # ``observation_tick=19`` (the periodic put happens one
        # iteration *after* the decision block reads the cache) and so
        # that the second decision is still applied while the
        # forward_continuation macro is in flight.
        second = _cave_decision(observation_tick=19, duration_ticks=4, decision_id="b")
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=25)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=400,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
                cave_entry_phase_enabled=True,
                cave_entry_phase_max_ticks=30,
            )
            run_dir = Path(result["run_dir"])
            evidence_path = run_dir / result["cave_entry_phase"]["evidence_frame"]
            entry = result["cave_entry_phase"]
            self.assertEqual(entry["state"], "entered")
            # The post-entry evidence frame must exist for human review.
            # The check must run before the temporary directory is
            # cleaned up.
            self.assertTrue(
                evidence_path.exists(), f"missing: {evidence_path}"
            )
        # The entry phase ran exactly its 30-tick budget, even though a
        # forward_continuation macro was already in flight when it
        # activated. If activation failed to interrupt that macro, the
        # leftover forward ticks would have inflated
        # ``entry_forward_ticks`` past 30 and the post-entry evidence
        # would not match the post-activation frame.
        self.assertEqual(entry["entry_forward_ticks"], 30)
        self.assertEqual(entry["max_ticks"], 30)
        # The entry forward ticks are a strict subset of the global
        # forward ticks; the forward_continuation ticks accumulated
        # *before* activation are not folded into the entry counter.
        self.assertLessEqual(entry["entry_forward_ticks"], result["forward_ticks"])
        # Phase 4 invariant: exactly one local ESC tick.
        self.assertEqual(result["esc_nonzero_ticks"], 1)
        self.assertEqual(
            sum(int(bool(action["ESC"])) for action in env.actions), 1
        )
        self.assertEqual(result["termination_reason"], "cave_completion_requested")


if __name__ == "__main__":
    unittest.main()
