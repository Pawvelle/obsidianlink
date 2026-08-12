"""Agent boundary with lazy v1 policy/model compatibility exports."""

from obsidianlink.agents.base import Agent
from obsidianlink.agents.protocols import AgentDecision


__all__ = [
    "Agent",
    "AgentDecision",
    "A0PolicyDecision",
    "AsyncA0PolicyWorker",
    "DirectA0Policy",
    "LocalQwenResponder",
    "MiniMaxM3Responder",
    "MiniMaxRequestRecord",
    "PendingA0Decision",
    "QwenRequestRecord",
    "WorkflowA0Policy",
    "prompt_text",
]


_A0_NAMES = {
    "A0PolicyDecision",
    "AsyncA0PolicyWorker",
    "DirectA0Policy",
    "PendingA0Decision",
    "WorkflowA0Policy",
}
_QWEN_NAMES = {"LocalQwenResponder", "QwenRequestRecord", "prompt_text"}
_MINIMAX_NAMES = {"MiniMaxM3Responder", "MiniMaxRequestRecord"}


def __getattr__(name: str):
    if name in _A0_NAMES:
        from obsidianlink.agents import a0_policy

        return getattr(a0_policy, name)
    if name in _QWEN_NAMES:
        from obsidianlink.agents import local_qwen

        return getattr(local_qwen, name)
    if name in _MINIMAX_NAMES:
        from obsidianlink.agents import minimax_m3

        return getattr(minimax_m3, name)
    raise AttributeError(name)
