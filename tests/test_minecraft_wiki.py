from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool


def _payload(*, title: str = "Nether portal", snippet: str = "A <b>Nether portal</b> is useful."):
    return {"query": {"search": [{"title": title, "snippet": snippet}]}}


def test_search_parses_bounded_result() -> None:
    seen: list[str] = []
    tool = MinecraftWikiTool(
        transport=lambda url: (seen.append(url) or _payload()),
        content_limit=20,
    )

    result = tool.search(" nether portal ")

    assert "srsearch=nether+portal" in seen[0]
    assert result.query == "nether portal"
    assert result.title == "Nether portal"
    assert result.url == "https://minecraft.wiki/w/Nether_portal"
    assert result.content == "A Nether portal is u"
    assert result.error is None


def test_search_rejects_empty_query_without_transport() -> None:
    tool = MinecraftWikiTool(transport=lambda _url: (_ for _ in ()).throw(AssertionError()))
    result = tool.search("  ")
    assert result.error == "query must be non-empty"


def test_search_returns_network_failure() -> None:
    def fail(_url: str):
        raise OSError("offline")

    result = MinecraftWikiTool(transport=fail).search("obsidian")
    assert result.title is None
    assert result.error == "wiki request failed: OSError: offline"


def test_search_returns_no_result() -> None:
    tool = MinecraftWikiTool(transport=lambda _url: {"query": {"search": []}})
    result = tool.search("nothing")
    assert result.error == "no wiki results"


def test_search_strips_markup_and_limits_content() -> None:
    tool = MinecraftWikiTool(
        transport=lambda _url: _payload(snippet="<i>one</i> &amp; two three"),
        content_limit=9,
    )
    result = tool.search("portal")
    assert result.content == "one & two"
