"""Local Qwen3-VL adapter for the GeneralAgent planner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from obsidianlink.agents.qwen_vl import QwenVLModelClient
from obsidianlink.models.base_client import BaseLLMClient

DEFAULT_QWEN_2B = Path("models/Qwen3-VL-2B-Instruct")
DEFAULT_QWEN_4B = Path("models/Qwen3-VL-4B-Instruct")


def default_qwen_model_path() -> Path:
    """Prefer the local 2B checkpoint; use 4B only when 2B is missing."""
    if DEFAULT_QWEN_2B.exists():
        return DEFAULT_QWEN_2B.resolve()
    if DEFAULT_QWEN_4B.exists():
        return DEFAULT_QWEN_4B.resolve()
    return DEFAULT_QWEN_2B.resolve()


class QwenLLMClient(BaseLLMClient):
    """Planner-facing wrapper around the local Qwen3-VL checkpoint."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 768,
    ) -> None:
        path = Path(model_path) if model_path is not None else default_qwen_model_path()
        self.model_path = str(path)
        self.model = path.name
        self._inner = QwenVLModelClient(
            self.model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    @property
    def completions(self) -> int:
        return int(self._inner.completions)

    @property
    def vision_completions(self) -> int:
        return int(self._inner.vision_completions)

    def generate(self, prompt: str) -> str:
        return self._inner.complete(prompt)

    def generate_with_vision(self, prompt: str, *, frame: Any) -> str:
        return self._inner.complete_with_vision(prompt, frame=frame)


__all__ = [
    "DEFAULT_QWEN_2B",
    "DEFAULT_QWEN_4B",
    "QwenLLMClient",
    "default_qwen_model_path",
]
