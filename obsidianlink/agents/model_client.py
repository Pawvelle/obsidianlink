"""Vendor-neutral model client. No provider implementations in this phase."""

from typing import Protocol


class ModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...
