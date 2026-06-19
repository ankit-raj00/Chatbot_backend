"""
Agent Routes — interrupt/resume for human-in-the-loop shell agent.

Endpoints:
  GET  /api/agent/status/{thread_id}        — get current agent state
  POST /api/agent/resume/{thread_id}        — resume a paused agent with user approval
  POST /api/agent/cancel/{thread_id}        — cancel a paused agent
  GET  /api/agent/threads                   — list active conversation threads
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from core.middleware import get_current_user

router = APIRouter(prefix="/api/agent", tags=["Agent"])


class ResumePayload(BaseModel):
    approved: bool
    feedback: str = ""


@router.get("/status/{thread_id}")
async def get_agent_status(
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the current state of an agent thread (for interrupt/resume UX)."""
    from graph.builder import get_agent_graph
    agent_graph = await get_agent_graph()

    try:
        config    = {"configurable": {"thread_id": thread_id}}
        state     = await agent_graph.aget_state(config)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        values = state.values if hasattr(state, "values") else {}
        return {
            "thread_id":   thread_id,
            "agent":       "agentx",
            "interrupted": bool(state.next),
            "next_node":   list(state.next) if state.next else [],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/{thread_id}")
async def resume_agent(
    thread_id: str,
    payload: ResumePayload,
    current_user: dict = Depends(get_current_user),
):
    """Resume a paused agent after human approval."""
    from graph.builder import get_agent_graph
    from langchain_core.messages import HumanMessage

    agent_graph = await get_agent_graph()
    config     = {"configurable": {"thread_id": thread_id}}

    try:
        if not payload.approved:
            # Inject a cancellation message and let agent wrap up
            await agent_graph.aupdate_state(
                config,
                {"messages": [HumanMessage(content="[USER CANCELLED] Do not execute the command.")]},
            )
        else:
            if payload.feedback:
                await agent_graph.aupdate_state(
                    config,
                    {"messages": [HumanMessage(content=f"[USER APPROVED] {payload.feedback}")]},
                )

        return {"success": True, "approved": payload.approved, "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{thread_id}")
async def cancel_agent(
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel a paused agent thread."""
    from graph.builder import get_agent_graph
    from langchain_core.messages import HumanMessage

    agent_graph = await get_agent_graph()
    config     = {"configurable": {"thread_id": thread_id}}
    try:
        await agent_graph.aupdate_state(
            config,
            {"messages": [HumanMessage(content="[CANCELLED]")]},
        )
        return {"success": True, "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
