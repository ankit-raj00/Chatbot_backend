from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import langgraph.prebuilt as prebuilt
from config.rag_config import RAG_MODEL
from rag.graph.state import RAGGraphState
from rag.tools.retrieval_tool import search_knowledge_base
from rag.tools.doc_store_tools import read_document_page
import logging
import os

import structlog
logger = structlog.get_logger(__name__)


def _bind_search_tool(user_id: Optional[str]):
    """
    Returns a search_knowledge_base wrapper with user_id pre-bound to the
    authenticated request's id, so the model's tool schema never exposes
    user_id as a model-fillable argument.

    SECURITY: mirrors graph/nodes/agent_tool_node.py's forced-injection
    principle (never trust the LLM to supply a tenancy-critical value) —
    adapted for create_react_agent, which (unlike agent_tool_node's own
    manual tool-call loop) gives no per-call hook to inject arguments before
    a bound tool executes. Confirmed live: without this, search_knowledge_base
    self-rejected with "called without user_id — refusing (tenancy guard)"
    whenever the model called it directly (e.g. after the initial retrieval
    got filtered out by the grader), since the LLM was never given a real
    user_id to pass and the tool refuses to search with none.
    """
    @tool("search_knowledge_base")
    def search_knowledge_base_bound(
        query: str,
        selected_files: Optional[List[str]] = None,
        limit: int = 5,
        offset: int = 0,
    ) -> list:
        """
        Searches the knowledge base (Vector DB) for relevant context.

        Args:
            query: The semantic search query.
            selected_files: List of file UUIDs to restrict search to. If empty
                             or None, searches all of the current user's files.
            limit: Number of chunks to return (default: 5).
            offset: Pagination offset (default: 0).

        Returns chunks with content/source/section/page/figures/score — cite
        the source and section when answering, and embed any figure as
        markdown: ![filename](url).
        """
        return search_knowledge_base.func(
            query=query, selected_files=selected_files, limit=limit,
            offset=offset, user_id=user_id,
        )
    return search_knowledge_base_bound


class AgentNode:
    """
    The 'Brain' of the Agentic RAG.
    Replaces the simple GenerationNode.
    Uses a ReAct Loop to reason about data and call tools if needed.
    """

    def __init__(self):
        # 1. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=RAG_MODEL,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )
        # Tools/agent_executor are built per-call in generate() now, not here —
        # search_knowledge_base must be bound to the authenticated user_id of
        # each individual request, and this AgentNode instance is a shared
        # singleton (one RAGWorkflow serves every request).

    async def generate(self, state: RAGGraphState) -> RAGGraphState:
        """
        Executes the Agentic Loop.
        Input context is provided as the "Initial Observation".
        """
        logger.info("🤖 AgentNode: Starting Reasoning Loop...")

        question = state["question"]
        documents = state.get("documents", [])
        user_id = state.get("user_id")

        # Built per-call: search_knowledge_base_bound closes over THIS
        # request's user_id (see _bind_search_tool's docstring).
        agent_executor = create_react_agent(self.llm, [_bind_search_tool(user_id), read_document_page])

        # Construct Initial Context String
        # The agent sees what the RetrievalNode found first.
        context_str = "\n\n".join([
            f"Chunk {i} (Source: {doc.metadata.get('source')}):\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        sys_prompt = """You are an expert Research Assistant. 
        You have access to a Knowledge Base and a Document Reader.
        
        Your Goal: Answer the user's question accurately using the provided context.
        
        Instructions:
        1. ANALYZE the 'Initial Context' provided below.
        2. IF the context is sufficient, answer the question directly.
        3. IF the context is cut off, vague, or missing tables -> USE YOUR TOOLS.
           - Use 'search_knowledge_base' to find more chunks (use offset for pagination).
           - Use 'read_document_page' to see the full page content if a chunk mentions a specific page/table. 
             YOU MUST get the 'doc_id' for this tool from the 'json_id' property in the chunk's metadata.
        4. Always cite your sources.
        """
        
        user_input = f"""
        Question: {question}
        
        Initial Context from Retrieval Step:
        {context_str}
        """
        
        try:
            # Run the Agent
            # The agent will loop until it decides to stop.
            # We pass the input as a messages list because create_react_agent expects it.
            inputs = {"messages": [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_input)
            ]}
            
            # ainvoke returns a dictionary with 'messages' (the full history)
            result = await agent_executor.ainvoke(inputs)
            
            # Extract Final Answer
            # The last message in the history is the AI's final response
            final_message = result["messages"][-1]
            answer = final_message.content
            
            logger.info("🤖 AgentNode: Finished.")
            
            return {
                **state,
                "generation": answer,
                "messages": result["messages"] # Persist thought process to global state if needed
            }
            
        except Exception as e:
            logger.error(f"Agent Loop Failed: {e}")
            return {
                **state,
                "generation": "I encountered an error while thinking. Please try again."
            }
