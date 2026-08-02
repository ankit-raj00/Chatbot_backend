"""
LLM Registry — singleton LLM instances per model name, wrapped with circuit breaker.

Models are served via OmniRoute, an OpenAI-compatible gateway, so we use
LangChain's ChatOpenAI pointed at OMNIROUTE_BASE_URL. Config is env-driven so the
same code targets a local OmniRoute now and a deployed one later.

WHY a registry: ChatOpenAI instantiates an HTTP client on creation. Creating it
     fresh on every LangGraph node execution wastes time and resources; this
     registry creates each model once and reuses it.

WHY circuit breaker: If the gateway is down, we fail fast instead of letting many
     concurrent requests each wait through backoff = system hangs.

Usage:
    from graph.llm_registry import get_llm
    from config.model_config import ModelConfig
    llm = get_llm(ModelConfig.DEFAULT_MODEL)
    llm_with_tools = llm.bind_tools(tools)  # bind_tools does NOT mutate the original
"""

import os
import logging
from langchain_openai import ChatOpenAI
from utils.circuit_breaker import gemini_breaker

logger = logging.getLogger(__name__)

# OmniRoute (OpenAI-compatible) connection — env-driven so local/deployed both work.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
OMNIROUTE_API_KEY  = os.getenv("OMNIROUTE_API_KEY", "")

# Registry: model_name -> ChatOpenAI instance
_registry: dict[str, ChatOpenAI] = {}


def get_llm(model_name: str) -> ChatOpenAI:
    """
    Returns a cached LLM instance for the given model name.
    Creates it on first call. Thread-safe for async (single event loop).
    """
    if model_name not in _registry:
        logger.info(f"LLM registry: creating instance for {model_name} via {OMNIROUTE_BASE_URL}")
        _registry[model_name] = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            max_retries=2,
            streaming=True,
            stream_usage=True,          # emit token usage on the streamed response
            base_url=OMNIROUTE_BASE_URL,
            api_key=OMNIROUTE_API_KEY,
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
