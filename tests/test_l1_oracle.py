"""Offline checks for the Gate 1 Scripted Oracle stability primitives.

Does not start Minecraft. Uses a minimal fake env implementing only the
``step``/``observe``/``hidden_state`` surface :class:`OracleSession`
needs. Live 3/3 stability is the only source of a live-success claim —
see ``obsidianlink/experiments/run_l1_oracle.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.experiments.l1_oracle import EpisodeAborted, OracleSession


class _FakeObservation:
    def __init__(self, inventory: dict[str, int] | None = None) -> None:
        self.inventory = inventory or {}
        self.frame = None
        self.selected_item = None


class _FakeEnv:
    """Minimal stand-in for ``L1ControlledEnv`` for pure-logic tests."""

    def __init__(
        self,
        *,
        done_after: int | None = None,
        raise_after: int | None = None,
        inventory: dict[str, int] | None = None,
    ) -> None:
        self._hidden: dict[str, Any] = {
            "xpos": 0.5,
            "ypos": 4.0,
            "zpos": 0.5,
            "yaw": 0.0,
            "pitch": 25.0,
        }
        self._obs = _FakeObservation(inventory=inventory or {"bucket": 1})
        self.step_calls = 0
        self.done_after = done_after
        self.raise_after = raise_after

    def step(self, action: Action) -> _FakeObservation:
        self.step_calls += 1
        if self.raise_after is not None and self.step_calls >= self.raise_after:
            raise RuntimeError("simulated backend failure")
        if self.done_after is not None and self.step_calls >= self.done_after:
            self._hidden["done"] = True
        return self._obs

    def observe(self) -> _FakeObservation:
        return self._obs

    @property
    def hidden_state(self) -> dict[str, Any]:
        return dict(self._hidden)


def _session(**env_kwargs: Any) -> OracleSession:
    return OracleSession(_FakeEnv(**env_kwargs), "/tmp/l1_oracle_test_frames")


def test_forbidden_action_types_never_reach_env_step() -> None:
    env = _FakeEnv()
    session = OracleSession(env, "/tmp/l1_oracle_test_frames")
    with pytest.raises(RuntimeError):
        session.step(Action(type=ActionType.EQUIP, target="bucket"))
    with pytest.raises(RuntimeError):
        session.step(Action(type=ActionType.PLACE, target="cobblestone"))
    assert env.step_calls == 0


def test_action_counts_increment_per_verb() -> None:
    session = _session()
    session.step(Action(type=ActionType.MOVE, dx=1))
    session.step(Action(type=ActionType.MOVE, dx=1))
    session.step(Action(type=ActionType.CAMERA, yaw=5.0))
    session.step(Action(type=ActionType.WAIT))
    assert session.action_counts["move"] == 2
    assert session.action_counts["camera"] == 1
    assert session.action_counts["wait"] == 1
    assert session.steps == 4
    assert len(session.step_durations) == 4


def test_stage_context_manager_records_elapsed_and_action_delta() -> None:
    session = _session()
    with session.stage("navigate_to_lava"):
        session.step(Action(type=ActionType.MOVE, dx=1))
        session.step(Action(type=ActionType.MOVE, dx=1))
    with session.stage("scoop_lava_1"):
        session.step(Action(type=ActionType.USE))
    assert len(session.stage_log) == 2
    first, second = session.stage_log
    assert first["name"] == "navigate_to_lava"
    assert first["action_counts"] == {"move": 2}
    assert first["elapsed_seconds"] >= 0.0
    assert second["name"] == "scoop_lava_1"
    assert second["action_counts"] == {"use": 1}


def test_stage_records_error_and_still_reraises_on_exception() -> None:
    session = _session(raise_after=1)
    with pytest.raises(RuntimeError):
        with session.stage("pour_lava_1"):
            session.step(Action(type=ActionType.USE))
    assert session.stage_log[-1]["error"] is not None
    assert "simulated backend failure" in session.stage_log[-1]["error"]


def test_done_in_hidden_state_aborts_and_blocks_further_steps() -> None:
    env = _FakeEnv(done_after=1)
    session = OracleSession(env, "/tmp/l1_oracle_test_frames")
    session.step(Action(type=ActionType.WAIT))
    assert session.aborted is True
    assert session.abort_reason is not None
    with pytest.raises(EpisodeAborted):
        session.step(Action(type=ActionType.WAIT))
    # The second call must not re-enter env.step() on a finished episode.
    assert env.step_calls == 1


def test_step_exception_aborts_and_blocks_further_steps() -> None:
    env = _FakeEnv(raise_after=1)
    session = OracleSession(env, "/tmp/l1_oracle_test_frames")
    with pytest.raises(RuntimeError):
        session.step(Action(type=ActionType.WAIT))
    assert session.aborted is True
    with pytest.raises(EpisodeAborted):
        session.step(Action(type=ActionType.WAIT))
    # No second real env.step() call after the first raised.
    assert env.step_calls == 1


def test_scoop_lava_at_retry_is_bounded_when_it_never_succeeds() -> None:
    # Inventory never changes -> scoop never detected -> must not loop
    # forever; the retry budget must be small (retry-on-failure only).
    session = _session(inventory={"bucket": 1})
    scooped, _snap = session.scoop_lava_at(
        (0, 3, 5), aim_from=(0.5, 3.9, 5.5), retries=2
    )
    assert scooped is False
    # 2 retries * (<=4 look_at + 3 use + 1 wait) is well under 50 steps;
    # this bounds the primitive, it does not pin an exact step count.
    assert session.steps < 50


def test_gate1_geometry_matches_portal_reference_bottom_row() -> None:
    from obsidianlink.experiments.run_l1_oracle import GEOMETRY
    from obsidianlink.tasks.portal import PORTAL_FRAME_BLOCK_COUNT

    bottom = [c for c in GEOMETRY.frame if c[1] == GEOMETRY.base_y]
    assert len(bottom) == 2
    xs = sorted(c[0] for c in bottom)
    assert xs == [GEOMETRY.base_x + 1, GEOMETRY.base_x + 2]
    assert len(GEOMETRY.frame) == PORTAL_FRAME_BLOCK_COUNT == 10
    # All frame cells share one z-plane (a single vertical portal wall).
    assert len({c[2] for c in GEOMETRY.frame}) == 1


def test_inventory_trace_records_each_step_and_flags_changes() -> None:
    env = _FakeEnv(inventory={"bucket": 1})
    session = OracleSession(env, "/tmp/l1_oracle_test_frames")
    session.begin_inventory_trace("hold_wait")
    session.step(Action(type=ActionType.WAIT))
    env._obs.inventory = {"bucket": 0, "water_bucket": 1}
    env._obs.selected_item = "water_bucket"
    session.inv_trace_phase = "look_at_verify"
    session.step(Action(type=ActionType.CAMERA, yaw=5.0))
    trace = session.end_inventory_trace()
    assert trace[0]["type"] == "baseline"
    assert trace[0]["changed"] is True
    wait_tick = next(t for t in trace if t["type"] == "wait")
    assert wait_tick["changed"] is False
    assert wait_tick["phase"] == "hold_wait"
    cam_tick = next(t for t in trace if t["type"] == "camera")
    assert cam_tick["changed"] is True
    assert cam_tick["phase"] == "look_at_verify"
    assert cam_tick["water_bucket"] == 1
    assert session._trace_inventory is False


def test_first_water_bucket_loss_helper_pins_drop_tick() -> None:
    from obsidianlink.experiments.run_l1_water_rollback_probe import _first_wb_loss

    trace = [
        {"step": 10, "phase": "recover_use", "type": "use", "water_bucket": 0, "sneak": False},
        {"step": 11, "phase": "recover_use", "type": "use", "water_bucket": 1, "sneak": False},
        {"step": 12, "phase": "hold_wait", "type": "wait", "water_bucket": 1, "sneak": False},
        {"step": 13, "phase": "look_at_verify", "type": "camera", "water_bucket": 0, "sneak": False},
    ]
    loss = _first_wb_loss(trace)
    assert loss is not None
    assert loss["step"] == 13
    assert loss["phase"] == "look_at_verify"
    assert loss["type"] == "camera"
    assert _first_wb_loss(trace[:3]) is None


def test_gate1_success_is_one_obsidian_not_portal_frame() -> None:
    from obsidianlink.experiments.run_l1_oracle import gate1_cast_obsidian, gate1_bottom_row

    assert gate1_cast_obsidian.__doc__ is not None
    assert "exact_block_truth" in gate1_cast_obsidian.__doc__
    assert "ObservationFromGrid" in gate1_cast_obsidian.__doc__
    assert "portal frame" in gate1_cast_obsidian.__doc__.lower()
    assert gate1_bottom_row.__doc__ is not None
