import logging
from langgraph.graph import END, StateGraph
from rag.graph.state import RAGGraphState
from rag.graph.nodes.retrieval_node import RetrievalNode, parallel_retrieve_node
from rag.graph.nodes.grader_node import GraderNode
from rag.graph.nodes.agent_node import AgentNode
from rag.graph.nodes.hallucination_node import HallucinationNode
from rag.graph.nodes.web_search_node import WebSearchNode

logging.basicConfig(level=logging.INFO)
import structlog
logger = structlog.get_logger(__name__)

class RAGWorkflow:
    """
    Constructs the Agentic RAG Graph (Architecture v11.1).
    Nodes: Retrieve -> Grade -> Generate -> Hallucinate Check
    Edges: Conditional Logic for Loops & Fallbacks.
    """
    
    def __init__(self):
        # Initialize Nodes
        self.retriever = RetrievalNode()
        self.grader = GraderNode()
        self.agent = AgentNode()
        self.hallucinator = HallucinationNode()
        self.web_search = WebSearchNode()
        
        self.workflow = StateGraph(RAGGraphState)
        self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self):
        # 1. Add Nodes
        self.workflow.add_node("parallel_retrieve", parallel_retrieve_node)
        self.workflow.add_node("grade_documents", self.grader.grade_documents)
        self.workflow.add_node("agent", self.agent.generate)
        self.workflow.add_node("hallucination_check", self.hallucinator.check_hallucination)
        self.workflow.add_node("web_search", self.web_search.search) # Kept for backward compatibility

        # 2. Build Edges
        self.workflow.set_entry_point("parallel_retrieve")
        self.workflow.add_edge("parallel_retrieve", "grade_documents")
        
        # Conditional Edge: Grade -> Agent
        self.workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "web_search": "agent",
                "generate": "agent",
            },
        )
        
        # Web Search -> Agent
        self.workflow.add_edge("web_search", "agent")
        
        # Agent -> Hallucination Check
        self.workflow.add_edge("agent", "hallucination_check")
        
        # Conditional Edge: Hallucination Check → End
        # Note: retry loop removed — see _decide_to_retry for rationale.
        self.workflow.add_conditional_edges(
            "hallucination_check",
            self._decide_to_retry,
            {
                "end": END,
            },
        )

    # --- Conditional Logic Helpers ---
    
    def _decide_to_generate(self, state: RAGGraphState):
        """After parallel retrieval, always go to agent (web search already ran)."""
        docs = state.get("documents", [])
        if not docs:
            logger.warning("No documents after parallel retrieval — agent will rely on training data")
        return "generate"

    def _decide_to_retry(self, state: RAGGraphState):
        """
        Determines whether to retry based on hallucination check.

        DECISION: Always END — never retry.
        Reason: Retrying calls gemini-2.0-flash-lite again immediately, which:
          1. Hits 429 rate limits (cascading SDK retries = 1+2+4+8+16s per call)
          2. Multiplies total latency by retry_count (up to 3x)
          3. Causes 120s+ response times → client timeout

        Instead we return the best available answer with a hallucination_warning flag.
        The API response already includes `hallucination_warning` for the frontend to display.
        """
        hallucination_count = state.get("hallucination_count", 0)

        if hallucination_count == 0:
            logger.info("Decision: END (Verified ✅)")
        else:
            logger.warning(
                "Decision: END (Hallucination detected — returning best answer with warning)"
            )
        return "end"

            
    def get_app(self):
        return self.app
