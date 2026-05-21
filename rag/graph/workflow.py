import logging
from langgraph.graph import END, StateGraph
from rag.graph.state import RAGGraphState
from rag.graph.nodes.retrieval_node import RetrievalNode
from rag.graph.nodes.grader_node import GraderNode
from rag.graph.nodes.agent_node import AgentNode
from rag.graph.nodes.hallucination_node import HallucinationNode
from rag.graph.nodes.web_search_node import WebSearchNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.workflow.add_node("retrieve", self.retriever.retrieve)
        self.workflow.add_node("grade_documents", self.grader.grade_documents)
        self.workflow.add_node("agent", self.agent.generate)
        self.workflow.add_node("hallucination_check", self.hallucinator.check_hallucination)
        self.workflow.add_node("web_search", self.web_search.search)

        # 2. Build Edges
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional Edge: Grade -> (Agent OR Web Search)
        self.workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "web_search": "web_search",
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
        """
        Determines whether to generate (documents valid) or fall back to web search.
        """
        if state.get("web_search_needed"):
            logger.info("Decision: WEB SEARCH")
            return "web_search"
        else:
            logger.info("Decision: AGENT")
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
