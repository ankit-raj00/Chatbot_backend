import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from rag.graph.state import RAGGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HallucinationResult(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class HallucinationNode:
    """
    The 'Hallucination' node that fact-checks the answer.
    Implements Defense #6: Prevents ungrounded answers.
    """
    
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.structured_llm = llm.with_structured_output(HallucinationResult)
        
        # System Prompt
        self.system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in and supported by the set of facts."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def check_hallucination(self, state: RAGGraphState) -> RAGGraphState:
        """
        Determines whether the generation is grounded in the document.
        """
        logger.info("Checking for hallucinations...")
        documents = state["documents"]
        generation = state["generation"]
        
        if not generation:
            return state
            
        try:
            # Format docs
            docs_text = "\n\n".join([doc.page_content for doc in documents])
            
            score = self.chain.invoke({"documents": docs_text, "generation": generation})
            grade = score.binary_score
            
            if grade == "yes":
                logger.info("Answer is grounded (No Hallucination).")
                # Reset counts on success
                return {
                    **state,
                    "hallucination_count": 0
                }
            else:
                logger.warning("Hallucination detected! Answer not grounded.")
                return {
                    **state,
                    "hallucination_count": state.get("hallucination_count", 0) + 1
                }
                
        except Exception as e:
            logger.error(f"Hallucination check failed: {str(e)}")
            return state # Fail open
