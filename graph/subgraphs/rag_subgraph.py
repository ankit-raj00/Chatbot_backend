"""
RAG Subgraph — wraps the existing RAGWorkflow for knowledge base search.

Uses the full agentic RAG pipeline:
  parallel_retrieve → embedding_grade → web_search (if needed) → generate → hallucination_check
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)


async def rag_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    from rag.graph.workflow import RAGWorkflow

    messages = list(state["messages"])
    user_id  = state.get("user_id", "")

    # Extract last human question
    question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if isinstance(m.content, list):
                question = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in m.content
                )
            else:
                question = str(m.content)
            break

    if not question:
        return {"messages": messages, "final_response": "No question found."}

    try:
        workflow = RAGWorkflow()
        result = await workflow.app.ainvoke({
            "question":           question,
            "documents":          [],
            "generation":         "",
            "web_search_needed":  False,
            "hallucination_count": 0,
            "user_id":            user_id,
            "selected_files":     state.get("selected_files") or [],
        })
        answer = result.get("generation", "No answer found in knowledge base.")
    except Exception as e:
        logger.error(f"rag_subgraph error: {e}")
        answer = f"Knowledge base search failed: {e}"

    response = AIMessage(content=answer)
    logger.info("rag_subgraph.done")
    return {"messages": [response], "final_response": answer}


# Local import needed after class definition
from langchain_core.messages import AIMessage
