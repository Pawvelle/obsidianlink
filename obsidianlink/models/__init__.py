"""External LLM clients. No LangChain, no tool calling."""

from obsidianlink.models.base_client import BaseLLMClient
from obsidianlink.models.minimax_client import MiniMaxClient

__all__ = ["BaseLLMClient", "MiniMaxClient"]
