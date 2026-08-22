"""Local Qwen3-VL VisionModelClient.

Lazy-loads the checkpoint on first complete / complete_with_vision.
Importing this module does not import torch.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.agents.model_client import VisionModelClient


# A short, deterministic cap on generated tokens for a single
# perception call. D1 needs at most ~80 tokens of JSON; this cap
# gives the model room to add stray prose (which we strip) without
# runaway latency.
_MAX_NEW_TOKENS = 128


class QwenVLModelClient(VisionModelClient):
    """Vision-capable ModelClient backed by a local Qwen3-VL checkpoint.

    Parameters
    ----------
    model_path:
        Path to a local Qwen3-VL checkpoint directory (the same
        layout as ``models/Qwen3-VL-2B-Instruct/``). Must contain
        ``config.json`` and the safetensors weights.
    device:
        ``"auto"`` (default) picks ``mps`` when available, else
        ``cpu``. Explicit ``"cpu"`` or ``"mps"`` is also accepted.
    dtype:
        ``"auto"`` (default) lets transformers pick based on
        device. ``"float16"`` / ``"bfloat16"`` for explicit control.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = _MAX_NEW_TOKENS,
    ) -> None:
        if not model_path:
            raise ValueError("QwenVLModelClient requires a non-empty model_path")
        self._model_path = model_path
        self._device = _resolve_device(device)
        self._dtype = dtype
        self._max_new_tokens = max(16, int(max_new_tokens))
        # Lazy-loaded handles; populated on first vision call.
        self._model: Any = None
        self._processor: Any = None
        self.completions = 0  # text-only calls
        self.vision_completions = 0  # vision calls

    # ------------------------------------------------------------------
    # VisionModelClient / ModelClient
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Text-only completion. The frame is *not* sent.

        Provided so the :class:`ModelClient` base contract is honoured
        (e.g. for unit tests that pass a stub observation with
        ``frame=None``). For D1 perception use
        :meth:`complete_with_vision`.
        """
        self.completions += 1
        # Lazy-load the same model + processor as the vision path; we
        # just skip the image part of the chat template.
        model, processor = self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return self._generate(model, processor, messages)

    def complete_with_vision(self, prompt: str, *, frame: Any) -> str:
        """Vision + text completion. Sends the frame as a single image."""
        from PIL import Image  # local import: keep module cheap to import

        self.vision_completions += 1
        if frame is None:
            raise TypeError(
                "QwenVLModelClient.complete_with_vision requires a non-None frame"
            )
        pil_image = _to_pil(frame)
        model, processor = self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(model, processor, messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        # Local imports: torch / transformers / qwen_vl_utils are
        # heavy and only needed here.
        import torch  # type: ignore[import-untyped]
        from transformers import AutoProcessor  # type: ignore[import-untyped]

        # Lazy import of the model class. ``AutoModelForImageTextToText``
        # is the current name (replaced ``AutoModelForVision2Seq`` in
        # transformers 4.50+); we fall back to the legacy name for
        # older transformers installs.
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForImageTextToText as _AutoModel,
            )
        except ImportError:  # pragma: no cover - older transformers
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForVision2Seq as _AutoModel,
            )

        torch_dtype = self._resolve_torch_dtype(torch)

        # transformers 4.55+ renamed the kwarg ``torch_dtype`` to
        # ``dtype``. Pass the new name and fall back to the old one
        # for older installs.
        load_kwargs: dict[str, Any] = {"dtype": torch_dtype}
        try:
            self._model = _AutoModel.from_pretrained(
                self._model_path, **load_kwargs
            ).to(self._device)
        except TypeError:  # pragma: no cover - older transformers
            load_kwargs.pop("dtype")
            self._model = _AutoModel.from_pretrained(
                self._model_path, torch_dtype=torch_dtype, **load_kwargs
            ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        return self._model, self._processor

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self._dtype == "auto":
            # MPS supports float16 cleanly; float32 is the safe CPU
            # default. transformers will pick float16 when device is
            # MPS if we hand it "auto".
            return "auto"
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if self._dtype not in mapping:
            raise ValueError(
                f"Unknown dtype {self._dtype!r}; expected auto/float16/float32/bfloat16"
            )
        return mapping[self._dtype]

    def _generate(
        self,
        model: Any,
        processor: Any,
        messages: list[dict[str, Any]],
    ) -> str:
        # Local imports: qwen_vl_utils is an optional dep the
        # Qwen-VL ecosystem ships separately; some Qwen3-VL
        # checkpoints also work without it. We import lazily so a
        # missing dep is a runtime error, not an import-time one.
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "qwen_vl_utils is required for QwenVLModelClient; "
                "install it with `pip install qwen-vl-utils`"
            ) from exc

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # ``padding=True`` is required when the chat template produces
        # a left-padded prompt; we pass the full list so the processor
        # can infer the padding side from the tokenizer.
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to the model's device. ``to(self._device)`` is
        # a no-op when the input is already there, so this is safe for
        # both MPS and CPU.
        inputs = inputs.to(self._device)
        gen_kwargs: dict[str, Any] = dict(max_new_tokens=self._max_new_tokens)
        # ``use_cache=True`` is the transformers default; explicit for
        # clarity and to keep the MPS path fast.
        gen_kwargs["use_cache"] = True
        generated_ids = model.generate(**inputs, **gen_kwargs)
        # Trim the prompt tokens so we only decode the newly generated
        # part.
        input_ids = inputs["input_ids"]
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            return ""
        return decoded[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch  # type: ignore[import-untyped]

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    if device in ("cpu", "mps"):
        return device
    if device == "cuda":
        # Explicit CUDA is rejected on this codebase: see AGENTS.md
        # (this is a Mac-only dev env). Fail loud rather than
        # silently falling back.
        raise ValueError(
            "QwenVLModelClient does not support CUDA in this dev env; "
            "use device='auto' / 'mps' / 'cpu'."
        )
    raise ValueError(f"Unknown device {device!r}; expected auto/mps/cpu/cuda")


def _to_pil(frame: Any) -> Any:
    """Convert an RGB HWC frame (``ndarray shape=(H, W, 3) uint8``)
    to a PIL Image. Falls back to a no-op when the frame is already a
    PIL Image (e.g. tests that hand-build one).
    """
    from PIL import Image  # local import

    if isinstance(frame, Image.Image):
        return frame
    # ``np.ndarray`` duck-typing: we don't import numpy at module
    # level to keep this module cheap to import.
    shape = getattr(frame, "shape", None)
    dtype = getattr(frame, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError(
            f"QwenVLModelClient expects an ndarray or PIL.Image frame, "
            f"got {type(frame).__name__}"
        )
    if len(shape) != 3 or shape[-1] not in (3, 4):
        raise ValueError(
            f"Frame must have shape (H, W, 3) or (H, W, 4); got {shape!r}"
        )
    # PIL.Image.fromarray accepts a contiguous uint8 array. The
    # MineDojo adapter view is usually already contiguous.
    return Image.fromarray(frame)


__all__ = ["QwenVLModelClient"]
