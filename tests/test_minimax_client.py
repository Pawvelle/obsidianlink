"""Offline MiniMax client tests. No live API, no Minecraft."""

from __future__ import annotations

from urllib.error import URLError

import pytest

from obsidianlink.models.minimax_client import MiniMaxClient, MiniMaxHTTPError, redact


@pytest.fixture(autouse=True)
def _clear_minimax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)


def test_missing_api_key_raises() -> None:
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY is not set"):
        MiniMaxClient()


def test_generate_stores_raw_response() -> None:
    payload = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"content": '{"action": "wait"}'}}],
    }

    def transport(url, headers, body, timeout_s):
        del url, headers, body, timeout_s
        return payload

    client = MiniMaxClient(api_key="sk-test-not-real", transport=transport)
    text = client.generate("hello")
    assert text == '{"action": "wait"}'
    assert client.last_raw_response == payload
    assert client.last_text == text
    assert client.last_error is None
    assert client.model == "MiniMax-M3"
    assert client.url == "https://api.minimax.io/v1/chat/completions"


def test_minimax_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M3")
    client = MiniMaxClient(
        api_key="sk-test-not-real",
        transport=lambda *a, **k: {"choices": [{"message": {"content": "ok"}}]},
    )
    assert client.model == "MiniMax-M3"


def test_http_error_is_explicit_and_keeps_payload() -> None:
    err = MiniMaxHTTPError(
        "MiniMax API HTTP 401: unauthorized",
        status=401,
        payload={"error": {"message": "unauthorized"}},
    )

    def transport(url, headers, body, timeout_s):
        del url, headers, body, timeout_s
        raise err

    client = MiniMaxClient(api_key="sk-test-not-real", transport=transport)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.generate("hello")
    assert client.last_raw_response == {"error": {"message": "unauthorized"}}
    assert client.last_error is not None


def test_http_401_retries_china_host() -> None:
    seen: list[str] = []

    def transport(url, headers, body, timeout_s):
        del headers, body, timeout_s
        seen.append(url)
        if "minimax.io" in url:
            raise MiniMaxHTTPError(
                "MiniMax API HTTP 401: invalid api key",
                status=401,
                payload={"error": {"message": "invalid api key"}},
            )
        return {"choices": [{"message": {"content": '{"action":"wait"}'}}]}

    client = MiniMaxClient(api_key="sk-test-not-real", transport=transport)
    text = client.generate("hello")
    assert text == '{"action":"wait"}'
    assert any("minimax.io" in u for u in seen)
    assert any("minimaxi.com" in u for u in seen)
    assert "minimaxi.com" in client.url


def test_timeout_message() -> None:
    def transport(url, headers, body, timeout_s):
        del url, headers, body
        raise TimeoutError(f"timed out after {timeout_s}")

    client = MiniMaxClient(
        api_key="sk-test-not-real",
        timeout_s=12,
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="timeout after 12s"):
        client.generate("hello")


def test_error_does_not_include_api_key() -> None:
    key = "sk-secret-must-not-leak"

    def transport(url, headers, body, timeout_s):
        del url, body, timeout_s
        raise URLError(f"failed Authorization Bearer {key} {headers.get('Authorization')}")

    client = MiniMaxClient(api_key=key, transport=transport)
    with pytest.raises(RuntimeError) as caught:
        client.generate("hello")
    assert key not in str(caught.value)
    assert "Bearer" not in str(caught.value) or key not in str(caught.value)
    assert key not in (client.last_error or "")


def test_redact() -> None:
    assert redact("token sk-abc used", "sk-abc") == "token <redacted> used"


def test_api_error_object_in_200_body() -> None:
    def transport(url, headers, body, timeout_s):
        del url, headers, body, timeout_s
        return {"error": {"message": "quota exceeded"}}

    client = MiniMaxClient(api_key="sk-test-not-real", transport=transport)
    with pytest.raises(RuntimeError, match="quota exceeded"):
        client.generate("hello")
    assert client.last_raw_response == {"error": {"message": "quota exceeded"}}
