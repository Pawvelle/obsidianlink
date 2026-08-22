"""Current MineDojo adapters and offline test environments.

MineRL adapters remain available only through explicit legacy modules; new code
must use :class:`MineDojoEnvironment`.
"""

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.fake import FakeMinecraftEnv
from obsidianlink.env.live_view import LiveDesktopView
from obsidianlink.env.minedojo import MineDojoEnvironment

__all__ = [
    "Action",
    "ActionType",
    "Environment",
    "FakeMinecraftEnv",
    "LiveDesktopView",
    "MineDojoEnvironment",
    "Observation",
]
