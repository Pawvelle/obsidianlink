from __future__ import annotations

from dataclasses import dataclass

import pytest

from obsidianlink.agents.agent import AutonomousMinecraftAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision, parse_planner_decision
from obsidianlink.agents.wiki import WikiKnowledge
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills import default_skill_library
from obsidianlink.skills.mining import (
    _lava_ahead,
    _tree_horizontal_offset,
    _trunk_horizontal_offset,
    _trunk_under_crosshair,
)
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool


class FakeWoodEnv(Environment):
    def __init__(self) -> None:
        self.inventory: dict[str, int] = {}
        self.selected: str | None = None
        self.attacks = 0
        self.table_placed = False
        self.gui_open = False
        self.table_gui = False
        self.last: Observation | None = None

    def _obs(self) -> Observation:
        self.last = Observation(inventory=dict(self.inventory), selected_item=self.selected)
        return self.last

    def reset(self) -> Observation:
        self.inventory = {}
        self.selected = None
        self.attacks = 0
        self.table_placed = False
        self.gui_open = False
        self.table_gui = False
        return self._obs()

    def observe(self) -> Observation:
        assert self.last is not None
        return self.last

    def step(self, action: Action) -> Observation:
        if action.type is ActionType.ATTACK and not self.gui_open and not self.table_gui:
            self.attacks += 1
            if self.attacks % 2 == 0 and self.inventory.get("oak_log", 0) < 3:
                self.inventory["oak_log"] = self.inventory.get("oak_log", 0) + 1
        elif action.type is ActionType.HOTBAR:
            slot_items = {
                "1": "iron_axe",
                "2": "oak_log",
                "3": "oak_planks",
                "4": "crafting_table",
                "5": "stick",
                "6": "wooden_pickaxe",
            }
            candidate = slot_items.get(action.target)
            self.selected = candidate if candidate and self.inventory.get(candidate, 0) else None
        elif action.type is ActionType.INVENTORY and self.table_gui:
            self.inventory["oak_planks"] -= 3
            self.inventory["stick"] -= 2
            self.inventory["wooden_pickaxe"] = 1
            self.table_gui = False
        elif action.type is ActionType.INVENTORY and self.gui_open:
            self.inventory["oak_log"] = 0
            self.inventory["oak_planks"] = 6
            self.inventory["stick"] = 4
            self.inventory["crafting_table"] = 1
            self.gui_open = False
        elif action.type is ActionType.INVENTORY:
            self.gui_open = True
        elif (
            action.type is ActionType.USE
            and self.selected == "crafting_table"
            and not self.table_placed
        ):
            self.inventory["crafting_table"] = 0
            self.table_placed = True
            self.selected = None
        elif action.type is ActionType.USE and self.table_placed:
            self.table_gui = True
        return self._obs()

    def close(self) -> None:
        pass


@dataclass
class SequencePlanner:
    decisions: list[PlannerDecision]

    def plan(self, memory, observation, skill_descriptions):
        assert "move" not in skill_descriptions
        return self.decisions.pop(0)


def test_planner_parser_accepts_skills_and_rejects_low_level_actions() -> None:
    decision = parse_planner_decision(
        '{"type":"skill","name":"collect_wood","arguments":{"quantity":3}}',
        frozenset({"collect_wood"}),
    )
    assert decision.name == "collect_wood"
    with pytest.raises(ValueError, match="forbidden"):
        parse_planner_decision(
            '{"type":"skill","name":"move","arguments":{"dx":1}}',
            frozenset({"collect_wood"}),
        )


def test_tree_visual_servo_points_toward_green_upper_mass() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:48, 145:170] = (35, 110, 30)
    offset = _tree_horizontal_offset(frame)
    assert offset is not None
    assert offset > 0.25


def test_trunk_servo_points_toward_vertical_brown_mass() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[5:78, 145:175] = (90, 58, 25)
    offset = _trunk_horizontal_offset(frame)
    assert offset is not None
    assert offset > 0.25


def test_near_field_lava_cue_ignores_small_accents() -> None:
    import numpy as np

    safe = np.zeros((100, 200, 3), dtype=np.uint8)
    safe[70:73, 98:102] = (240, 90, 10)
    assert _lava_ahead(safe) is False

    hazard = np.zeros((100, 200, 3), dtype=np.uint8)
    hazard[60:95, 70:130] = (240, 90, 10)
    assert _lava_ahead(hazard) is True


def test_trunk_contact_requires_brown_mass_under_crosshair() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:80, 80:120] = (90, 58, 25)
    assert _trunk_under_crosshair(frame) is True

    frame[:, :] = (90, 58, 42)
    assert _trunk_under_crosshair(frame) is False

    frame[:, :] = (65, 100, 60)
    assert _trunk_under_crosshair(frame) is False


def test_autonomous_loop_queries_wiki_collects_and_crafts() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision("wiki", query="how to craft wooden pickaxe"),
            PlannerDecision("skill", name="collect_wood", arguments={"quantity": 3}),
            PlannerDecision(
                "skill", name="craft_item", arguments={"item": "wooden_pickaxe"}
            ),
        ]
    )
    wiki_tool = MinecraftWikiTool(
        transport=lambda _url: {
            "query": {"search": [{"title": "Pickaxe", "snippet": "Use planks and sticks."}]}
        }
    )
    memory = AgentMemory()
    agent = AutonomousMinecraftAgent(
        planner,
        MinecraftController(FakeWoodEnv(), max_steps=100),
        skills=default_skill_library(),
        wiki=WikiKnowledge(wiki_tool),
        memory=memory,
    )

    result = agent.run()

    assert result.success is True
    assert result.inventory["wooden_pickaxe"] == 1
    assert result.wiki_queries == ("how to craft wooden pickaxe",)
    assert "how to craft wooden pickaxe" in memory.known_knowledge
    assert [step.skill for step in result.completed_steps] == [
        "collect_wood",
        "craft_item",
    ]


def test_finish_is_rejected_until_inventory_verifies_goal() -> None:
    planner = SequencePlanner([PlannerDecision("finish"), PlannerDecision("finish")])
    agent = AutonomousMinecraftAgent(
        planner,
        MinecraftController(FakeWoodEnv(), max_steps=10),
        max_planning_cycles=2,
    )
    result = agent.run()
    assert result.success is False
    assert "wooden_pickaxe is absent" in (agent.memory.last_error or "")
