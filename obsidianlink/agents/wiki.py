"""Memory-aware wrapper around the live Minecraft Wiki tool."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool, WikiResult


class WikiKnowledge:
    def __init__(self, tool: MinecraftWikiTool | None = None) -> None:
        self.tool = tool or MinecraftWikiTool()

    def search_wiki(self, query: str, memory: AgentMemory) -> WikiResult:
        result = self.tool.search(query)
        if result.error:
            memory.last_error = result.error
        else:
            rendered = result.content
            if result.title:
                rendered = f"{result.title}: {rendered}"
            memory.remember_knowledge(result.query, rendered)
            memory.last_error = None
        return result


__all__ = ["WikiKnowledge"]
