"""
LLM Registry — singleton LLM instances per model name, wrapped with circuit breaker.

WHY: ChatGoogleGenerativeAI() instantiates an HTTP client on creation.
     Creating it fresh on every LangGraph node execution wastes time and
     resources. This registry creates each model once and reuses it.

WHY circuit breaker: If Gemini API is down, we fail fast instead of
     letting 100 concurrent requests each wait through 30s backoff = system hangs.

Usage:
    from graph.llm_registry import get_llm
    llm = get_llm("gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)  # bind_tools does NOT mutate the original
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.circuit_breaker import gemini_breaker

logger = logging.getLogger(__name__)

# Registry: model_name -> ChatGoogleGenerativeAI instance
_registry: dict[str, ChatGoogleGenerativeAI] = {}


def get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance for the given model name.
    Creates it on first call. Thread-safe for async (single event loop).
    """
    if model_name not in _registry:
        logger.info(f"LLM registry: creating instance for {model_name}")
        _registry[model_name] = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            max_tokens=None,
            max_retries=2,
            streaming=True,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return _registry[model_name]


async def invoke_with_breaker(model_name: str, messages: list) -> object:
    """
    Invoke an LLM call wrapped in the gemini circuit breaker.
    Use this for critical paths where cascading failures must be prevented.
    Falls back gracefully when the breaker is OPEN.
    """
    llm = get_llm(model_name)
    return await gemini_breaker.call(llm.ainvoke, messages)


def clear_registry() -> None:
    """Clear all cached LLM instances. Useful for testing."""
    _registry.clear()
