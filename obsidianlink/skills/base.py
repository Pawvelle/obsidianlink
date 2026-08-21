"""Small skill contracts and capability registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController


@dataclass(frozen=True)
class SkillResult:
    success: bool
    message: str
    steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Skill(Protocol):
    name: str
    description: str

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, Any],
    ) -> SkillResult:
        ...


class SkillLibrary:
    def __init__(self, skills: list[Skill] | tuple[Skill, ...]) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"duplicate skill: {skill.name}")
            self._skills[skill.name] = skill

    @property
    def descriptions(self) -> dict[str, str]:
        return {name: skill.description for name, skill in self._skills.items()}

    def execute(
        self,
        name: str,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, Any] | None = None,
    ) -> SkillResult:
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise ValueError(f"unknown skill: {name}") from exc
        return skill.execute(controller, memory, dict(arguments or {}))


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


__all__ = ["Skill", "SkillLibrary", "SkillResult", "bounded_int"]
