"""Vendor-neutral model client.

Text and vision are different calls. Vision must receive the RGB frame.
Fallback to text-only is never silent: :class:`ModelCall` records it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from obsidianlink.env.environment import Observation


@runtime_checkable
class ModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@runtime_checkable
class VisionModelClient(Protocol):
    def complete_with_vision(self, prompt: str, *, frame: Any) -> str:
        ...


@dataclass(frozen=True)
class ModelCall:
    text: str
    used_vision: bool
    fallback_reason: str | None = None


def call_model(
    model: ModelClient,
    prompt: str,
    *,
    observation: Observation | None = None,
) -> ModelCall:
    """Dispatch to vision when the model and frame are both present.

    Fallback reasons (``used_vision=False``):

    * ``no_observation``
    * ``no_frame``
    * ``text_only_model``
    """
    if observation is None:
        return ModelCall(
            text=model.complete(prompt),
            used_vision=False,
            fallback_reason="no_observation",
        )
    frame = getattr(observation, "frame", None)
    if frame is None:
        return ModelCall(
            text=model.complete(prompt),
            used_vision=False,
            fallback_reason="no_frame",
        )
    if isinstance(model, VisionModelClient):
        return ModelCall(
            text=model.complete_with_vision(prompt, frame=frame),
            used_vision=True,
            fallback_reason=None,
        )
    return ModelCall(
        text=model.complete(prompt),
        used_vision=False,
        fallback_reason="text_only_model",
    )


__all__ = [
    "ModelCall",
    "ModelClient",
    "VisionModelClient",
    "call_model",
]
