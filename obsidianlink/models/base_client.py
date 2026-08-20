"""Minimal LLM client: prompt in, text out. No tools, no memory."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Vendor-neutral text generation interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


__all__ = ["BaseLLMClient"]
