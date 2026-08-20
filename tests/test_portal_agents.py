"""Offline contracts for the formal Portal reference agents."""

from obsidianlink.agents.portal_agent import (
    OraclePortalAgent,
    PortalState,
    RuleBasedPortalAgent,
)
from obsidianlink.env.actions import ActionType
from obsidianlink.env.environment import Observation


def test_rule_portal_agent_has_explicit_fsm_and_uses_observation_only() -> None:
    agent = RuleBasedPortalAgent()
    assert agent.state is PortalState.FIND_RESOURCE
    first = agent.act(Observation(inventory={"bucket": 1}, selected_item="bucket"))
    assert first.type is ActionType.HOTBAR
    assert agent.state is PortalState.COLLECT

    build = agent.act(
        Observation(inventory={"lava_bucket": 1, "water_bucket": 1}, selected_item="lava_bucket")
    )
    assert agent.state is PortalState.BUILD
    assert build.type is ActionType.CAMERA


def test_oracle_portal_agent_is_deterministic_and_never_emits_unsafe_verbs() -> None:
    first = OraclePortalAgent()
    second = OraclePortalAgent()
    actions_one = [first.act(Observation()) for _ in range(25)]
    actions_two = [second.act(Observation()) for _ in range(25)]
    assert actions_one == actions_two
    assert all(action.type not in {ActionType.EQUIP, ActionType.PLACE} for action in actions_one)
    assert {action.type for action in actions_one} >= {
        ActionType.HOTBAR,
        ActionType.MOVE,
        ActionType.USE,
    }
