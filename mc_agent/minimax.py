"""Asynchronous MiniMax remote vision planner with capacity-one mailboxes.

This module mirrors :class:`mc_agent.qwen.QwenPlannerWorker`'s public
lifecycle (``start``, ``stop``, ``begin_episode``, ``submit``,
``acknowledge_decision``, ``wait_until_idle``) and reuses the existing
``LatestObservationMailbox``, ``LatestDecisionMailbox``,
``ObservationRequest`` and ``PlannerDecision`` records unchanged. It
performs all HTTP I/O on the planner thread; the MineRL step loop never
blocks on the network and never sees an unparsed provider payload.

Every raw MiniMax response still passes through
:func:`mc_agent.actions.parse_macro_action`. Any network failure, HTTP
error, empty body, non-JSON, schema-invalid, or out-of-allowlist
response becomes a rejected :class:`PlannerDecision` (a one-tick
``wait`` no-op). No provider payload is ever executed directly, and the
adapter never asks the model for tools, code, shell, attack, or ``ESC``.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mc_agent.actions import parse_macro_action
from mc_agent.qwen import (
    LatestDecisionMailbox,
    LatestObservationMailbox,
    ObservationRequest,
    PlannerDecision,
    QwenPlannerWorker,
    _prompt,
)


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ENDPOINT = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_THINKING = "disabled"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_COMPLETION_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95

# Preregistered prompt configurations. The V2 variant is the same suffix
# registered in ``scripts/benchmark_minimax.py``. The MiniMax worker
# MUST use a registered name; ``baseline`` is allowed but the smoke
# integration explicitly selects ``prompt_v2_cave_salience``.
PROMPT_CONFIG_BASELINE = "baseline"
PROMPT_CONFIG_V2 = "prompt_v2_cave_salience"
PROMPT_V2_CAVE_SALIENCE_SUFFIX = (
    " Final cave check before returning JSON: independently inspect the "
    "left, center, and right image thirds for cave evidence; do not skip "
    "this check just because the center route looks walkable. Set "
    "cave_visible=true only when a continuous dark recessed area is "
    "visibly surrounded by gray stone or rock and looks enterable. Set "
    "cave_visible=false for shadows, dark patches under trees, water "
    "surface, dirt walls or dirt pits, flat nighttime darkness, and small "
    "distant dark spots. When cave_visible=true, the reason must follow "
    "the exact pattern 'dark stone opening on the left|center|right' and "
    "must name the third where the dark opening itself sits, not the "
    "walking direction."
)


def list_prompt_configs() -> tuple[str, ...]:
    return (PROMPT_CONFIG_BASELINE, PROMPT_CONFIG_V2)


def build_prompt(
    prompt_config: str,
    *,
    previous_action: dict[str, Any] | None = None,
    visual_change: dict[str, Any] | None = None,
    cave_target: dict[str, Any] | None = None,
) -> str:
    """Return the user-text prompt for the chosen preregistered config.

    The base prompt is the same one used by the local Qwen worker and
    varies with the same context fields (``previous_action``,
    ``visual_change``, ``cave_target``); only the V2 variant appends the
    preregistered visual-salience suffix. The base prompt is never
    modified: MiniMax-specific prompt tuning is allowed only as a
    separately named configuration.
    """
    base = _prompt(previous_action, visual_change, cave_target)
    if prompt_config == PROMPT_CONFIG_V2:
        return base + PROMPT_V2_CAVE_SALIENCE_SUFFIX
    if prompt_config == PROMPT_CONFIG_BASELINE:
        return base
    raise ValueError(f"unknown prompt config: {prompt_config!r}")


def _data_url_from_pov(pov: np.ndarray) -> str:
    """Encode an in-memory RGB frame as a JPEG data URL.

    The base Qwen worker preprocesses its image to a fixed
    ``(336, 336)`` pad; we keep the original ``(360, 640)`` POV for
    MiniMax so the model sees the same content it would see in a
    single-frame offline benchmark. Using JPEG (not PNG) keeps the
    encoded body small for the typical plains biome frames.
    """
    if not isinstance(pov, np.ndarray) or pov.ndim != 3 or pov.shape[2] != 3:
        raise ValueError("pov must be an HxWx3 RGB array")
    image = Image.fromarray(pov).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _extract_response_text(response: Any) -> str:
    """Return the assistant text content from a MiniMax response payload.

    Strictly structural: no tool/function calls are read, no ``reasoning``
    is exposed, and a non-text response is treated as empty. The caller
    is expected to feed the result through ``parse_macro_action``.
    """
    if not isinstance(response, dict):
        raise ValueError("response body is not a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("response choice is invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no text content")
    return content.strip()


def _categorize_error(error: BaseException) -> str:
    """Map a transport / parse failure to a coarse category for the event log.

    Categories intentionally avoid leaking the API key, the request
    body, or any provider payload text.
    """
    if isinstance(error, urllib.error.HTTPError):
        try:
            code = int(getattr(error, "code", 0))
        except (TypeError, ValueError):
            code = 0
        if code in (401, 403):
            return "http_auth"
        if code == 429:
            return "http_rate_limit"
        if 500 <= code < 600:
            return "http_server"
        return f"http_{code}" if code else "http_error"
    if isinstance(error, urllib.error.URLError):
        return "network_error"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, OSError):
        return "network_error"
    if isinstance(error, json.JSONDecodeError):
        return "json_decode"
    if isinstance(error, ValueError):
        return "schema_violation"
    return type(error).__name__.lower()


class MiniMaxPlannerWorker(QwenPlannerWorker):
    """Asynchronous MiniMax remote planner with the Qwen worker contract.

    The worker thread is the only place that issues HTTP requests and
    touches the network. The MineRL step loop only ever reads the
    latest-observation/latest-decision mailboxes, hands decisions to the
    macro executor, and acknowledges the planner at the end of the
    tick. There is no synchronous network call on the MineRL thread.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        thinking: str = DEFAULT_THINKING,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        prompt_config: str = PROMPT_CONFIG_V2,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> None:
        if thinking not in {"disabled", "adaptive", "enabled"}:
            raise ValueError("thinking must be one of: disabled, adaptive, enabled")
        if prompt_config not in list_prompt_configs():
            raise ValueError(f"unknown prompt config: {prompt_config!r}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        # The Qwen base class expects a ``lock_path``; the MiniMax worker
        # never loads a local model so the value is ignored past
        # construction. The base __init__ still creates the lifecycle
        # state we share unchanged.
        super().__init__(lock_path=None)
        self.provider = "minimax"
        self.endpoint = endpoint
        self.model = model
        self.thinking = thinking
        self.timeout_seconds = timeout_seconds
        self.prompt_config = prompt_config
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._api_key = api_key
        self._diagnostics_lock = threading.Lock()
        self.request_diagnostics: list[dict[str, Any]] = []
        self.error_categories: dict[str, int] = {}
        self.last_request_error: str | None = None
        self.total_requests = 0
        self.failed_requests = 0
        self.parsed_requests = 0
        self.rejected_requests = 0

    @property
    def api_key(self) -> str | None:
        return self._api_key

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("planner worker already started")
        self._thread = threading.Thread(
            target=self._run, name="minimax-planner", daemon=True
        )
        self._thread.start()

    def _load_backend(self):
        # No local model to load; mark the load step as instantaneous so
        # the ready event can fire immediately.
        self.load_seconds = 0.0
        return None, None

    def _update_peak_memory(self) -> None:
        # Remote HTTP planner has no MPS allocation to track.
        return

    def _infer(self, model, processor, request: ObservationRequest) -> tuple[str, float]:
        del model, processor
        raw, elapsed, error_category, request_id, usage = self._http_request(request)
        with self._diagnostics_lock:
            self.total_requests += 1
            if error_category is not None:
                self.failed_requests += 1
                self.error_categories[error_category] = (
                    self.error_categories.get(error_category, 0) + 1
                )
                self.last_request_error = error_category
            self.request_diagnostics.append(
                {
                    "observation_tick": request.tick,
                    "episode_id": request.episode_id,
                    "latency_seconds": round(elapsed, 3),
                    "error_category": error_category,
                    "request_id": request_id,
                    "usage": usage,
                }
            )
        return raw, elapsed

    def _http_request(
        self, request: ObservationRequest
    ) -> tuple[str, float, str | None, str | None, Any]:
        """Run one MiniMax request.

        Returns ``(raw_text, elapsed_seconds, error_category, request_id, usage)``.
        On any transport, parse, or schema failure ``raw_text`` is the
        empty string, ``error_category`` is the coarse bucket name, and
        ``usage`` / ``request_id`` are whatever the provider returned
        before the failure (typically ``None``). The empty raw text is
        then routed through :func:`parse_macro_action` by ``_run`` and
        produces a one-tick no-op rejected decision.
        """
        if not self._api_key:
            return "", 0.0, "missing_api_key", None, None
        try:
            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url_from_pov(request.pov)},
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    self.prompt_config,
                                    previous_action=request.previous_action,
                                    visual_change=request.visual_change,
                                    cave_target=request.cave_target,
                                ),
                            },
                        ],
                    }
                ],
                "stream": False,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_completion_tokens": self.max_completion_tokens,
                "thinking": {"type": self.thinking},
            }
            request_obj = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
        except (TypeError, ValueError) as error:
            return "", 0.0, _categorize_error(error), None, None
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request_obj, timeout=self.timeout_seconds
            ) as response:
                payload = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
            elapsed = time.perf_counter() - started
            return "", elapsed, _categorize_error(error), None, None
        elapsed = time.perf_counter() - started
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            return "", elapsed, _categorize_error(error), None, None
        try:
            raw_text = _extract_response_text(decoded)
        except ValueError as error:
            return "", elapsed, _categorize_error(error), None, decoded.get("usage") if isinstance(decoded, dict) else None
        usage = decoded.get("usage") if isinstance(decoded, dict) else None
        request_id = decoded.get("id") if isinstance(decoded, dict) else None
        return raw_text, elapsed, None, request_id, usage
