"""Bounded MiniMax-M3 image responder for the Phase 3 policy worker."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image

from obsidianlink.agents.local_qwen import prompt_text


MINIMAX_MESSAGES_URL = "https://api.minimax.io/anthropic/v1/messages"
SYSTEM_PROMPT = (
    "You are a Minecraft control policy. Return exactly one JSON MacroAction "
    "object and nothing else. Never use tools, commands, code, paths, or prose."
)


@dataclass(frozen=True)
class MiniMaxRequestRecord:
    model: str
    latency_seconds: float
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    image_bytes: int


class MiniMaxM3Responder:
    """One non-streaming M3 vision request; no retries or remote tools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        max_tokens: int = 96,
        model: str = "MiniMax-M3",
        url: str = MINIMAX_MESSAGES_URL,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be between 1 and 256")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._model = model
        self._url = url
        self.last_request: MiniMaxRequestRecord | None = None

    def __call__(self, prompt: Mapping[str, object]) -> str:
        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is required for MiniMax-M3")
        image = self._encode_image(self._frame_from_prompt(prompt))
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "service_tier": "standard",
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text(prompt)},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(image).decode("ascii"),
                            },
                        },
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # Token Plan subscription keys (sk-cp-...) use bearer auth
                # on MiniMax's Anthropic-compatible endpoint.
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                request_id = response.headers.get("request-id")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"MiniMax HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"MiniMax network error: {error.reason}") from error
        text = self._response_text(body)
        usage = body.get("usage", {})
        self.last_request = MiniMaxRequestRecord(
            model=str(body.get("model", self._model)),
            latency_seconds=time.monotonic() - started,
            request_id=request_id,
            input_tokens=self._integer_or_none(usage.get("input_tokens")),
            output_tokens=self._integer_or_none(usage.get("output_tokens")),
            image_bytes=len(image),
        )
        return text

    @staticmethod
    def _frame_from_prompt(prompt: Mapping[str, object]) -> np.ndarray:
        observation = prompt.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("MiniMax prompt must contain an observation")
        frame = observation.get("frame")
        if not isinstance(frame, np.ndarray):
            raise ValueError("MiniMax prompt requires an RGB numpy frame")
        return frame

    @staticmethod
    def _encode_image(frame: np.ndarray) -> bytes:
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((512, 512))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) > 10 * 1024 * 1024:
            raise ValueError("MiniMax image exceeds the 10 MB API limit")
        return encoded

    @staticmethod
    def _response_text(body: Mapping[str, Any]) -> str:
        content = body.get("content")
        if not isinstance(content, list):
            raise RuntimeError("MiniMax response is missing content blocks")
        text = "".join(
            block["text"]
            for block in content
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if not text:
            raise RuntimeError("MiniMax response contains no text action")
        return text

    @staticmethod
    def _integer_or_none(value: object) -> int | None:
        return value if type(value) is int else None
