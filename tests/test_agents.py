"""Offline tests for the Agent interface layer.

Does not start Minecraft. Live wiring is
``obsidianlink/experiments/run_agent.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.agents.random_agent import LEGAL_TYPES, RandomAgent
from obsidianlink.agents.reactive_agent import ReactiveAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation, observation_field_names
from obsidianlink.experiments.run_agent import run_episode


class _FakeEnv(Environment):
    """Minimal Environment: inventory-only, no hidden fields on Observation."""

    def __init__(self, inventory: dict[str, int] | None = None) -> None:
        self._obs = Observation(
            frame=None,
            inventory=dict(inventory or {"bucket": 1, "water_bucket": 1}),
            selected_item="water_bucket",
        )
        self._hidden: dict[str, Any] = {"reward": 0.0, "done": False}
        self.step_calls = 0
        self.actions: list[Action] = []

    @property
    def hidden_state(self) -> dict[str, Any]:
        return dict(self._hidden)

    def reset(self) -> Observation:
        self.step_calls = 0
        self.actions = []
        self._hidden = {"reward": 0.0, "done": False}
        return self._obs

    def observe(self) -> Observation:
        return self._obs

    def step(self, action: Action) -> Observation:
        self.step_calls += 1
        self.actions.append(action)
        inv = dict(self._obs.inventory or {})
        selected = self._obs.selected_item
        if action.type is ActionType.EQUIP:
            target = action.target
            if target and inv.get(target, 0) >= 1:
                selected = target
        if action.type is ActionType.USE:
            if selected == "bucket" and inv.get("bucket", 0) >= 1:
                inv["bucket"] = inv.get("bucket", 0) - 1
                if inv["bucket"] <= 0:
                    inv.pop("bucket", None)
                inv["lava_bucket"] = inv.get("lava_bucket", 0) + 1
                selected = "lava_bucket"
            elif selected == "lava_bucket" and inv.get("lava_bucket", 0) >= 1:
                inv["lava_bucket"] = inv.get("lava_bucket", 0) - 1
                if inv["lava_bucket"] <= 0:
                    inv.pop("lava_bucket", None)
                inv["bucket"] = inv.get("bucket", 0) + 1
                selected = "bucket"
            elif selected == "water_bucket" and inv.get("water_bucket", 0) >= 1:
                inv["water_bucket"] = inv.get("water_bucket", 0) - 1
                if inv["water_bucket"] <= 0:
                    inv.pop("water_bucket", None)
                inv["bucket"] = inv.get("bucket", 0) + 1
                selected = "bucket"
        self._obs = Observation(frame=None, inventory=inv, selected_item=selected)
        if self.step_calls >= 80:
            self._hidden["done"] = True
        return self._obs

    def close(self) -> None:
        return None


def test_base_agent_act_is_not_implemented() -> None:
    agent = BaseAgent()
    agent.reset()
    with pytest.raises(NotImplementedError):
        agent.act(Observation())


def test_observation_has_no_hidden_state_fields() -> None:
    obs = Observation(frame=None, inventory={"bucket": 1}, selected_item="bucket")
    assert observation_field_names() == frozenset(
        {"frame", "inventory", "selected_item", "x", "y", "z", "yaw", "pitch"}
    )
    assert not hasattr(obs, "hidden_state")
    assert not hasattr(obs, "reward")
    assert not hasattr(obs, "biome_id")
    assert hasattr(obs, "x")


def test_random_agent_actions_are_legal() -> None:
    import random

    agent = RandomAgent(rng=random.Random(0))
    agent.reset()
    obs = Observation(inventory={"bucket": 1, "water_bucket": 1}, selected_item="bucket")
    seen: set[ActionType] = set()
    for _ in range(80):
        action = agent.act(obs)
        assert isinstance(action, Action)
        assert action.type in LEGAL_TYPES
        assert action.type not in {ActionType.HOTBAR, ActionType.INVENTORY}
        if action.type is ActionType.EQUIP:
            assert action.target in {"bucket", "water_bucket"}
        seen.add(action.type)
    assert ActionType.WAIT in seen or ActionType.MOVE in seen


def test_random_agent_refuses_hotbar_inventory_in_constructor() -> None:
    with pytest.raises(ValueError):
        RandomAgent(types=(ActionType.HOTBAR, ActionType.WAIT))


def test_reactive_agent_runs_full_fake_episode() -> None:
    env = _FakeEnv()
    agent = ReactiveAgent()
    report = run_episode(agent, env, max_steps=64)
    assert report["error"] is None
    assert report["steps"] >= 1
    assert report["success"] is True
    assert report["agent_finished"] is True
    assert env.step_calls == report["steps"]
    assert all(isinstance(a, Action) for a in env.actions)
    assert all(a.type not in {ActionType.HOTBAR, ActionType.INVENTORY} for a in env.actions)
    # Agent-visible Observation never grew evaluator fields.
    obs = env.observe()
    assert set(obs.__dataclass_fields__) == {
        "frame",
        "inventory",
        "selected_item",
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
    }


def test_reactive_agent_uses_only_observation_inventory() -> None:
    agent = ReactiveAgent()
    agent.reset()
    first = agent.act(
        Observation(inventory={"bucket": 1, "water_bucket": 1}, selected_item="water_bucket")
    )
    assert first.type is ActionType.EQUIP
    assert first.target == "bucket"
    scooped = agent.act(
        Observation(
            inventory={"lava_bucket": 1, "water_bucket": 1},
            selected_item="lava_bucket",
        )
    )
    assert scooped.type in {ActionType.MOVE, ActionType.CAMERA, ActionType.USE, ActionType.EQUIP}


def test_run_episode_does_not_pass_hidden_state_to_agent() -> None:
    seen: list[Observation] = []

    class _Spy(BaseAgent):
        def act(self, observation: Observation) -> Action:
            seen.append(observation)
            return Action(type=ActionType.WAIT)

    env = _FakeEnv()
    run_episode(_Spy(), env, max_steps=3)
    assert seen
    for obs in seen:
        assert not hasattr(obs, "hidden_state")
        assert not hasattr(obs, "reward")
