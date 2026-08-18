"""Minimal environment interface. No MineRL backend in this phase."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from obsidianlink.env.actions import Action


@dataclass(frozen=True)
class Observation:
    frame: object | None = None
    inventory: object | None = None
    selected_item: object | None = None


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
