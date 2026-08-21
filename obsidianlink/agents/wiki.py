"""Memory-aware wrapper around the live Minecraft Wiki tool."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool, WikiResult


class WikiKnowledge:
    def __init__(self, tool: MinecraftWikiTool | None = None) -> None:
        self.tool = tool or MinecraftWikiTool()

    def search_wiki(self, query: str, memory: AgentMemory) -> WikiResult:
        cached = memory.find_knowledge(query)
        if cached is not None:
            memory.record_knowledge_use(cached, cache_hit=True)
            memory.last_error = None
            return WikiResult(
                query=cached.query,
                title=cached.subject or None,
                url=cached.source_url,
                content=cached.summary,
                knowledge=None,
                from_cache=True,
            )
        result = self.tool.search(query)
        if result.error:
            memory.last_error = result.error
        else:
            rendered = result.content
            if result.title:
                rendered = f"{result.title}: {rendered}"
            knowledge = result.knowledge
            memory.remember_knowledge(
                result.query,
                rendered,
                knowledge_type=knowledge.knowledge_type if knowledge else "general",
                subject=(knowledge.subject if knowledge else result.title) or "",
                attributes=knowledge.attributes if knowledge else {},
                source_url=result.url,
            )
            memory.last_error = None
        return result

    @staticmethod
    def has_cached(query: str, memory: AgentMemory) -> bool:
        return memory.find_knowledge(query) is not None


__all__ = ["WikiKnowledge"]
