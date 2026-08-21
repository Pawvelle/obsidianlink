"""Live Minecraft Wiki retrieval with bounded structured extraction.

The tool performs a search and then requests the selected article body.  It
does not crawl, snapshot, embed, or mirror the Wiki.  Parsing is deliberately
small and deterministic so retrieved evidence remains inspectable.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_API_URL = "https://minecraft.wiki/api.php"
_ARTICLE_URL = "https://minecraft.wiki/w/"
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

WikiTransport = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class StructuredKnowledge:
    """A planner-usable fact extracted from one Wiki article."""

    knowledge_type: str
    subject: str
    summary: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WikiResult:
    query: str
    title: str | None = None
    url: str | None = None
    content: str = ""
    error: str | None = None
    knowledge: StructuredKnowledge | None = None
    from_cache: bool = False


class MinecraftWikiTool:
    """Search a live article and return bounded text plus structured knowledge."""

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
        search_url = f"{self._api_url}?{urlencode(_search_params(normalized))}"
        try:
            payload = self._transport(search_url)
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
            snippet = _clean_text(first.get("snippet", ""))
        except (KeyError, TypeError):
            return WikiResult(query=normalized, error="invalid wiki response")

        article_html = ""
        sections: dict[str, str] = {}
        rows: list[tuple[str, str]] = []
        parse_url = f"{self._api_url}?{urlencode(_parse_params(title))}"
        try:
            article_payload = self._transport(parse_url)
            article_html = _parse_html_value(article_payload)
            if article_html:
                parser = _WikiHTMLParser()
                parser.feed(article_html)
                sections = parser.sections
                rows = parser.rows
        except Exception:
            # Search snippets remain useful if the article endpoint is unavailable.
            article_html = ""

        article_text = _article_text(sections) if sections else _clean_text(article_html)
        content = _bounded_text(article_text or snippet, self._content_limit)
        knowledge = _extract_knowledge(normalized, title, content, sections, rows)
        return WikiResult(
            query=normalized,
            title=title,
            url=f"{_ARTICLE_URL}{quote(title.replace(' ', '_'))}",
            content=content,
            knowledge=knowledge,
        )


class _WikiHTMLParser(HTMLParser):
    """Collect readable sections and two-column infobox/recipe table rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, str] = {"summary": ""}
        self.rows: list[tuple[str, str]] = []
        self._section = "summary"
        self._heading: list[str] | None = None
        self._cell: list[str] | None = None
        self._row: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "sup"}:
            self._ignored += 1
        elif tag in {"h2", "h3"}:
            self._heading = []
        elif tag == "tr":
            self._row = []
        elif tag in {"th", "td"}:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "sup"} and self._ignored:
            self._ignored -= 1
        elif tag in {"h2", "h3"} and self._heading is not None:
            heading = _clean_text(" ".join(self._heading))
            if heading:
                self._section = heading.casefold()
                self.sections.setdefault(self._section, "")
            self._heading = None
        elif tag in {"th", "td"} and self._cell is not None:
            cell = _clean_text(" ".join(self._cell))
            if cell:
                self._row.append(cell)
            self._cell = None
        elif tag == "tr":
            if len(self._row) >= 2:
                self.rows.append((self._row[0], " | ".join(self._row[1:])))
            self._row = []

    def handle_data(self, data: str) -> None:
        if self._ignored or not data.strip():
            return
        if self._heading is not None:
            self._heading.append(data)
        elif self._cell is not None:
            self._cell.append(data)
        else:
            previous = self.sections.get(self._section, "")
            self.sections[self._section] = f"{previous} {data}".strip()


def _extract_knowledge(
    query: str,
    title: str,
    content: str,
    sections: Mapping[str, str],
    rows: list[tuple[str, str]],
) -> StructuredKnowledge:
    kind = _knowledge_type(query, sections)
    attributes: dict[str, Any] = {}
    row_map = {_clean_text(key).casefold(): _clean_text(value) for key, value in rows}
    if kind == "recipe":
        crafting = next(
            (value for key, value in sections.items() if "craft" in key or "recipe" in key),
            "",
        )
        attributes["ingredients"] = _recipe_ingredients(crafting, rows)
        attributes["result"] = title
    elif kind == "item":
        for key in ("renewable", "stackable", "rarity", "tool", "blast resistance"):
            if key in row_map:
                attributes[key.replace(" ", "_")] = row_map[key]
    elif kind == "mechanic":
        rules = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(content)
            if re.search(r"\b(can|cannot|must|requires?|when|if|only|causes?)\b", sentence, re.I)
        ]
        attributes["rules"] = tuple(rules[:6])
    return StructuredKnowledge(kind, title, content, attributes)


def _knowledge_type(query: str, sections: Mapping[str, str]) -> str:
    lowered = query.casefold()
    headings = " ".join(sections).casefold()
    if re.search(r"\b(recipe|craft|crafting|ingredients?|make)\b", lowered):
        return "recipe"
    if re.search(r"\b(how|mechanic|works?|behavior|interaction|rule)\b", lowered):
        return "mechanic"
    if "crafting" in headings and re.search(r"\b(recipe|craft)\b", headings):
        return "recipe"
    return "item"


def _recipe_ingredients(section: str, rows: list[tuple[str, str]]) -> tuple[str, ...]:
    values: list[str] = []
    for key, value in rows:
        if re.search(r"ingredient|input|material", key, re.I):
            values.extend(re.split(r"\s*(?:\+|,|\|| and )\s*", value))
    if not values and section:
        match = re.search(r"(?:crafted|made) (?:from|with|using) ([^.]+)", section, re.I)
        if match:
            values.extend(re.split(r"\s*(?:\+|,| and )\s*", match.group(1)))
    return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))


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


def _parse_params(title: str) -> dict[str, str | int]:
    return {
        "action": "parse",
        "format": "json",
        "formatversion": "2",
        "page": title,
        "prop": "text",
        "disableeditsection": 1,
    }


def _parse_html_value(payload: Mapping[str, Any]) -> str:
    parsed = payload.get("parse")
    if not isinstance(parsed, Mapping):
        return ""
    value = parsed.get("text", "")
    if isinstance(value, Mapping):
        value = value.get("*", "")
    return value if isinstance(value, str) else ""


def _article_text(sections: Mapping[str, str]) -> str:
    ordered = [sections.get("summary", "")]
    preferred = ("usage", "obtaining", "crafting", "behavior", "mechanics")
    ordered.extend(value for key, value in sections.items() if any(name in key for name in preferred))
    return _clean_text(" ".join(ordered))


def _clean_text(value: Any) -> str:
    raw = value if isinstance(value, str) else ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    return " ".join(text.split())


def _bounded_text(value: Any, limit: int) -> str:
    return _clean_text(value)[:limit]


def _http_get_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "ObsidianLink/0.1 (research tool)"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("wiki response is not a JSON object")
    return payload


__all__ = [
    "MinecraftWikiTool",
    "StructuredKnowledge",
    "WikiResult",
    "WikiTransport",
]
