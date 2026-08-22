"""External LLM clients. No LangChain, no tool calling."""

from obsidianlink.models.base_client import BaseLLMClient
from obsidianlink.models.minimax_client import MiniMaxClient
from obsidianlink.models.qwen_client import QwenLLMClient, default_qwen_model_path

__all__ = [
    "BaseLLMClient",
    "MiniMaxClient",
    "QwenLLMClient",
    "default_qwen_model_path",
]
