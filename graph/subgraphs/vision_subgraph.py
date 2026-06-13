"""
Vision Subgraph — multimodal image analysis using Gemini Flash.

Handles: image Q&A, screenshot analysis, diagram reading, chart interpretation.
Uses gemini-2.5-flash (full model, not lite) because vision needs higher capacity.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)

VISION_SYSTEM = """You are AgentX Vision Agent — a specialist in understanding images.

You can:
- Describe images in detail
- Read text from screenshots or photos (OCR-like)
- Analyze charts and graphs and extract their data
- Interpret technical diagrams and architecture drawings
- Compare multiple images

Be precise and specific. When analyzing charts, extract actual numbers."""


async def vision_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    # Vision always uses the full flash model (not lite)
    from graph.llm_registry import get_llm

    model = "gemini-2.5-flash"   # Force full model for vision quality
    skill_body = state.get("skill_body", "")

    system_content = VISION_SYSTEM
    if skill_body:
        system_content += f"\n\n## SKILL INSTRUCTIONS\n{skill_body}"

    messages = list(state["messages"])
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_content)
    else:
        messages = [SystemMessage(content=system_content)] + messages

    llm = get_llm(model)
    response = await llm.ainvoke(messages)

    text = str(response.content) if not isinstance(response.content, list) else \
           "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in response.content)

    logger.info("vision_subgraph.done")
    return {"messages": [response], "final_response": text}
