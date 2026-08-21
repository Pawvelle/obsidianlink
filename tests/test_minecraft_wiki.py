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


def test_search_fetches_article_and_extracts_recipe() -> None:
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        if "action=parse" in url:
            return {
                "parse": {
                    "text": (
                        "<p>A wooden pickaxe is a tool.</p><h2>Crafting</h2>"
                        "<p>It is crafted using planks and sticks.</p>"
                        "<table><tr><th>Ingredients</th><td>3 Planks + 2 Sticks</td></tr></table>"
                    )
                }
            }
        return _payload(title="Wooden Pickaxe", snippet="A tool")

    result = MinecraftWikiTool(transport=transport).search("wooden pickaxe recipe")

    assert len(calls) == 2
    assert result.knowledge is not None
    assert result.knowledge.knowledge_type == "recipe"
    assert result.knowledge.attributes["ingredients"] == ("3 Planks", "2 Sticks")
    assert "crafted using planks and sticks" in result.content


def test_search_extracts_mechanic_rules() -> None:
    def transport(url: str):
        if "action=parse" in url:
            return {
                "parse": {
                    "text": "<p>Water can convert lava when it touches a source block. Flowing lava cannot create the same block.</p>"
                }
            }
        return _payload(title="Lava", snippet="Lava behavior")

    result = MinecraftWikiTool(transport=transport).search("how water lava interaction works")

    assert result.knowledge is not None
    assert result.knowledge.knowledge_type == "mechanic"
    assert len(result.knowledge.attributes["rules"]) == 2
