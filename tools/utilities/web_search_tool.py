"""
Utility tool: general-purpose web search (TinyFish-backed).

Distinct from search_knowledge_base (which only searches the user's
ingested documents in Qdrant) -- this searches the live web for anything
outside the knowledge base: current events, docs for a library, prices,
anything time-sensitive.

NOTE: named `internet_search`, NOT `web_search` -- deliberately. A tool
literally named "web_search" is silently broken by the LLM gateway: it
collides with a reserved/built-in tool name (Gemini's native grounding
search / OpenAI's Responses API built-in "web_search" tool type), and the
model returns an EMPTY response with ZERO tool calls whenever a tool named
exactly "web_search" is bound -- confirmed via isolated testing: an
otherwise-identical tool (same schema, same docstring, same async logic)
works immediately once renamed. Do not rename this back to "web_search".
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rag.tools.web_search import search_web


class InternetSearchArgs(BaseModel):
    query: str = Field(description="The web search query")


@tool(args_schema=InternetSearchArgs)
async def internet_search(query: str):
    """Search the web for a query and return results."""
    docs = await search_web(query, max_results=3)
    if not docs:
        return [{"info": "No web results found (or web search is unavailable right now)."}]
    return [
        {
            "content": d.page_content[:4000],
            "source": d.metadata.get("source", ""),
            "title": d.metadata.get("title", ""),
        }
        for d in docs
    ]
