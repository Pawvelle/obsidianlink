"""MiniMax text client. API key is read from the environment only."""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from obsidianlink.models.base_client import BaseLLMClient

DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_URL = "https://api.minimax.io/v1/chat/completions"
CHINA_URL = "https://api.minimaxi.com/v1/chat/completions"
API_KEY_ENV = "MINIMAX_API_KEY"
BASE_URL_ENV = "MINIMAX_BASE_URL"
MODEL_ENV = "MINIMAX_MODEL"
_ALT_URL = {
    DEFAULT_URL: CHINA_URL,
    CHINA_URL: DEFAULT_URL,
}

GenerateTransport = Callable[
    [str, dict[str, str], dict[str, Any], float],
    Mapping[str, Any],
]


class MiniMaxClient(BaseLLMClient):
    """OpenAI-compatible ``chat/completions`` against MiniMax.

    Credentials: ``MINIMAX_API_KEY``. Optional ``MINIMAX_BASE_URL``
    (full chat-completions URL) and ``MINIMAX_MODEL`` (default
    ``MiniMax-M3``).

    After each ``generate`` call, ``last_raw_response`` holds the API
    JSON (never the Authorization header). Exceptions never include the
    API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        url: str | None = None,
        timeout_s: float = 60.0,
        transport: GenerateTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        key = key.strip() if isinstance(key, str) else ""
        if not key:
            raise RuntimeError(f"{API_KEY_ENV} is not set")
        self._api_key = key
        env_url = os.environ.get(BASE_URL_ENV, "").strip()
        self._url = url or env_url or DEFAULT_URL
        env_model = os.environ.get(MODEL_ENV, "").strip()
        self._model = model or env_model or DEFAULT_MODEL
        self._timeout_s = float(timeout_s)
        self._transport = transport
        self.completions = 0
        self.last_raw_response: dict[str, Any] | None = None
        self.last_text: str | None = None
        self.last_error: str | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        return self._url

    def generate(self, prompt: str) -> str:
        self.completions += 1
        self.last_text = None
        self.last_error = None
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "max_completion_tokens": 256,
            "thinking": {"type": "disabled"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        transport = self._transport or _http_post_json
        urls = [self._url]
        alt = _ALT_URL.get(self._url)
        if alt and alt not in urls:
            urls.append(alt)
        last_exc: BaseException | None = None
        for index, url in enumerate(urls):
            try:
                data = transport(url, headers, payload, self._timeout_s)
                if not isinstance(data, Mapping):
                    raise RuntimeError("MiniMax API returned a non-object JSON body")
                self.last_raw_response = dict(data)
                _raise_if_api_error(data, self._api_key)
                text = _message_text(data)
                if text is None or not str(text).strip():
                    raise RuntimeError("MiniMax API returned empty message content")
                self._url = url
                self.last_text = text
                return text
            except Exception as exc:
                last_exc = exc
                if self.last_raw_response is None:
                    payload_hint = getattr(exc, "payload", None)
                    if isinstance(payload_hint, Mapping):
                        self.last_raw_response = dict(payload_hint)
                if index + 1 < len(urls) and _is_auth_error(exc):
                    continue
                break
        message = _public_error(
            last_exc if last_exc is not None else RuntimeError("MiniMax API call failed"),
            self._api_key,
            self._timeout_s,
        )
        self.last_error = message
        raise RuntimeError(message) from None


class MiniMaxHTTPError(RuntimeError):
    """HTTP-layer MiniMax failure with optional JSON body."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = dict(payload) if isinstance(payload, Mapping) else None


def _is_auth_error(exc: BaseException) -> bool:
    if isinstance(exc, MiniMaxHTTPError) and exc.status == 401:
        return True
    text = str(exc).lower()
    return "401" in text or "invalid api key" in text or "authorized_error" in text


def _public_error(exc: BaseException, api_key: str, timeout_s: float) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"MiniMax API timeout after {timeout_s:.0f}s"
    if isinstance(exc, URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
        return f"MiniMax API timeout after {timeout_s:.0f}s"
    if isinstance(exc, MiniMaxHTTPError):
        return redact(str(exc), api_key)
    if isinstance(exc, RuntimeError):
        return redact(str(exc), api_key)
    return redact(f"MiniMax API call failed: {type(exc).__name__}: {exc}", api_key)


def _raise_if_api_error(data: Mapping[str, Any], api_key: str) -> None:
    err = data.get("error")
    if isinstance(err, Mapping):
        msg = err.get("message") or err.get("type") or json.dumps(dict(err))
        raise RuntimeError(redact(f"MiniMax API error: {msg}", api_key))
    if isinstance(err, str) and err.strip():
        raise RuntimeError(redact(f"MiniMax API error: {err}", api_key))
    base = data.get("base_resp")
    if isinstance(base, Mapping):
        code = base.get("status_code", 0)
        if code not in (0, 200, None):
            msg = base.get("status_msg") or str(code)
            raise RuntimeError(redact(f"MiniMax API error {code}: {msg}", api_key))


def _message_text(data: Mapping[str, Any]) -> str | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if isinstance(message, Mapping):
        text = _content_to_text(message.get("content"))
        if text:
            return text
    return _content_to_text(first.get("text"))


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                piece = item.get("text")
                if isinstance(piece, str):
                    parts.append(piece)
        joined = "".join(parts)
        return joined if joined else None
    return None


def redact(text: str, api_key: str) -> str:
    """Strip the API key from any debug string."""
    if not api_key:
        return text
    return text.replace(api_key, "<redacted>")


def _http_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - HTTPS API
            raw = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw_body = ""
        parsed: Mapping[str, Any] | None = None
        try:
            raw_body = exc.read().decode("utf-8", errors="replace")[:4000]
            loaded = json.loads(raw_body)
            if isinstance(loaded, Mapping):
                parsed = loaded
        except Exception:
            parsed = None
        snippet = raw_body.replace("\n", " ")[:300]
        message = f"MiniMax API HTTP {exc.code}"
        if snippet:
            message = f"{message}: {snippet}"
        raise MiniMaxHTTPError(message, status=int(exc.code), payload=parsed) from None
    except (TimeoutError, socket.timeout) as exc:
        raise TimeoutError(str(exc) or "timed out") from None
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise TimeoutError(str(exc.reason) or "timed out") from None
        raise RuntimeError(f"MiniMax API request failed: {exc.reason}") from None
    if not isinstance(raw, Mapping):
        raise RuntimeError("MiniMax API returned a non-object JSON body")
    return raw


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "CHINA_URL",
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "GenerateTransport",
    "MODEL_ENV",
    "MiniMaxClient",
    "MiniMaxHTTPError",
    "redact",
]
