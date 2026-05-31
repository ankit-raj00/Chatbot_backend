"""
LLM Registry — singleton LLM instances per model name.

WHY: ChatGoogleGenerativeAI() instantiates an HTTP client on creation.
     Creating it fresh on every LangGraph node execution wastes time and
     resources. This registry creates each model once and reuses it.

WHY NOT a module-level global: Module-level globals are created at import time
     before the API key is loaded from .env. The lazy-init dict pattern below
     creates the instance on first use, by which time the env var is loaded.

Usage:
    from graph.llm_registry import get_llm
    llm = get_llm("gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)  # bind_tools does NOT mutate the original
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI

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
        )
    return _registry[model_name]


def clear_registry() -> None:
    """Clear all cached LLM instances. Useful for testing."""
    _registry.clear()
