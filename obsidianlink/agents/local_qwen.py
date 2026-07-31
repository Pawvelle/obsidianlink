"""Local Qwen3-VL responder for the bounded Phase 3 policy worker."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def prompt_text(prompt: Mapping[str, object]) -> str:
    """Serialize only public non-image prompt fields for the VLM instruction."""
    observation = dict(prompt.get("observation", {}))
    observation.pop("frame", None)
    sections = [
        "Return exactly one JSON MacroAction object. Do not use markdown, code, "
        "commands, paths, or explanations.",
        f"Task: {prompt.get('instruction', '')}",
        f"Observation: {observation}",
    ]
    if "workflow" in prompt:
        sections.append(f"Workflow: {prompt['workflow']}")
        sections.append(f"Current stage: {prompt.get('current_stage')}")
    return "\n".join(sections)


class LocalQwenResponder:
    """Lazy, local-only Qwen responder; intended to execute in a worker thread."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_new_tokens: int = 64,
        device: str = "auto",
    ) -> None:
        if max_new_tokens < 1 or max_new_tokens > 64:
            raise ValueError("max_new_tokens must be between 1 and 64")
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._processor = None

    @property
    def device(self) -> str:
        return self._device

    def __call__(self, prompt: Mapping[str, object]) -> str:
        self._ensure_loaded()
        observation = dict(prompt["observation"])
        frame = observation.pop("frame")
        if not isinstance(frame, np.ndarray):
            raise ValueError("Qwen responder requires an RGB numpy frame")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt_text(prompt)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[frame],
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        generated = self._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self._max_new_tokens,
        )
        completion = generated[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def prepare(self) -> None:
        """Load the fixed local model before an environment episode begins."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._model_path.is_dir():
            raise RuntimeError(f"local Qwen model is missing: {self._model_path}")
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if self._device == "auto":
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self._device not in {"mps", "cpu"}:
            raise ValueError("Qwen device must be auto, mps, or cpu")

        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            local_files_only=True,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_path,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()
