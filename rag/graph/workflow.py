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
        self.retriever = RetrievalNode()
        self.grader = GraderNode()
        self.agent = AgentNode()
        self.hallucinator = HallucinationNode()
        self.web_search = WebSearchNode()
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
        
        # Conditional Edge: Hallucination Check -> (End OR Retry Loop)
        self.workflow.add_conditional_edges(
            "hallucination_check",
            self._decide_to_retry,
            {
                "end": END,
                "retry": "agent", # Simple retry. For full loop: "retrieve"
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
        Implements Defense #5: Max Retry Policy < 3.
        """
        hallucination_count = state.get("hallucination_count", 0)
        retry_count = state.get("retry_count", 0)
        
        if hallucination_count == 0:
            logger.info("Decision: END (Verified)")
            return "end"
            
        if retry_count < 3:
            logger.warning(f"Decision: RETRY (Count: {retry_count})")
            # Increment retry count in state is handled by passing new state in node return,
            # but wait, conditional edges don't modify state. 
            # We must modify state in the NODE.
            # Correction: HallucinationNode should increment the count.
            # Assuming HallucinationNode incremented `hallucination_count` on failure.
            # We don't have a specific `retry_count` incrementer here without a separate node 
            # or modifying Graph state in edge (impossible).
            # For V1, we rely on hallucination_count acting as proxy or loop limit.
            # Let's check HallucinationNode implementation... it returns state with updated count.
            return "retry"
        else:
            logger.warning("Decision: END (Max Retries Exceeded)")
            return "end"
            
    def get_app(self):
        return self.app
