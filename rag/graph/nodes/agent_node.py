from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent # This is still correct in modern LangGraph, the warning might be based on older langchain-agents move.
# However, if I want to be safe and use exactly what the warning said:
# from langchain.agents import create_agent 
# BUT LangGraph's prebuilt is what we want for this flow.
import langgraph.prebuilt as prebuilt
from rag.graph.state import RAGGraphState
from rag.tools.retrieval_tool import search_knowledge_base
from rag.tools.doc_store_tools import read_document_page
import logging
import os

logger = logging.getLogger(__name__)

class AgentNode:
    """
    The 'Brain' of the Agentic RAG.
    Replaces the simple GenerationNode.
    Uses a ReAct Loop to reason about data and call tools if needed.
    """
    
    def __init__(self):
        # 1. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", # Fast, supports tools
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True 
        )
        
        # 2. Bind Tools
        # We need to wrap our python functions as LangChain Tools
        # Using the @tool decorator or StructuredTool is standard.
        # Since our functions are already clean, we can just pass them if they have type hints.
        # But 'create_react_agent' expects proper Tools.
        
        self.tools = [search_knowledge_base, read_document_page]
        
        # 3. Create Internal Agent Graph
        # This graph runs the "Reason -> Act -> Observe" loop
        self.agent_executor = create_react_agent(self.llm, self.tools)
        
    async def generate(self, state: RAGGraphState) -> RAGGraphState:
        """
        Executes the Agentic Loop.
        Input context is provided as the "Initial Observation".
        """
        logger.info("🤖 AgentNode: Starting Reasoning Loop...")
        
        question = state["question"]
        documents = state.get("documents", [])
        
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
            result = await self.agent_executor.ainvoke(inputs)
            
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
