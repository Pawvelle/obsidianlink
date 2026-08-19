"""A small, live Minecraft Wiki search tool for agents.

This intentionally uses the public MediaWiki search endpoint directly.  It
does not snapshot, crawl, embed, or otherwise mirror the wiki.  Network and
protocol failures are returned in :class:`WikiResult` so an agent can decide
how to proceed without silently receiving built-in game knowledge.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

_API_URL = "https://minecraft.wiki/api.php"
_ARTICLE_URL = "https://minecraft.wiki/w/"
_TAG_RE = re.compile(r"<[^>]+>")

WikiTransport = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class WikiResult:
    query: str
    title: str | None = None
    url: str | None = None
    content: str = ""
    error: str | None = None


class MinecraftWikiTool:
    """Search the live Minecraft Wiki and return one bounded relevant excerpt."""

    def __init__(
        self,
        *,
        transport: WikiTransport | None = None,
        api_url: str = _API_URL,
        content_limit: int = 1_000,
    ) -> None:
        if content_limit <= 0:
            raise ValueError("content_limit must be > 0")
        self._transport = transport or _http_get_json
        self._api_url = api_url
        self._content_limit = content_limit

    def search(self, query: str) -> WikiResult:
        normalized = query.strip() if isinstance(query, str) else ""
        if not normalized:
            return WikiResult(query="", error="query must be non-empty")
        url = f"{self._api_url}?{urlencode(_search_params(normalized))}"
        try:
            payload = self._transport(url)
        except Exception as exc:
            return WikiResult(
                query=normalized,
                error=f"wiki request failed: {type(exc).__name__}: {exc}",
            )
        try:
            entries = payload["query"]["search"]
            first = entries[0] if isinstance(entries, list) and entries else None
            if not isinstance(first, Mapping):
                return WikiResult(query=normalized, error="no wiki results")
            title = first.get("title")
            if not isinstance(title, str) or not title.strip():
                return WikiResult(query=normalized, error="no wiki results")
            snippet = first.get("snippet", "")
            content = _bounded_text(snippet, self._content_limit)
            return WikiResult(
                query=normalized,
                title=title,
                url=f"{_ARTICLE_URL}{quote(title.replace(' ', '_'))}",
                content=content,
            )
        except (KeyError, TypeError):
            return WikiResult(query=normalized, error="invalid wiki response")


def _search_params(query: str) -> dict[str, str | int]:
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "srprop": "snippet",
    }


def _bounded_text(value: Any, limit: int) -> str:
    raw = value if isinstance(value, str) else ""
    text = html.unescape(_TAG_RE.sub("", raw))
    text = " ".join(text.split())
    return text[:limit]


def _http_get_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "ObsidianLink/0.1 (research tool)"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("wiki response is not a JSON object")
    return payload


__all__ = ["MinecraftWikiTool", "WikiResult", "WikiTransport"]
