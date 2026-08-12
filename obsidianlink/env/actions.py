"""Strict action facade for the v2 environment boundary."""

from obsidianlink.actions.protocol import ParsedAction, parse_macro_action
from obsidianlink.core.types import MacroAction

__all__ = ["MacroAction", "ParsedAction", "parse_macro_action"]
