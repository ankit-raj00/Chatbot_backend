import logging
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from rag.graph.state import RAGGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Structured Output Schema ---
class GradeResult(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class GraderNode:
    """
    The 'Grader' node filtering retrieval noise.
    Implements Defense #6: Prevents garbage-in to Generator.
    """
    
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", # Fast & Cheap for grading
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.structured_llm = llm.with_structured_output(GradeResult)
        
        # System Prompt
        self.system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def grade_documents(self, state: RAGGraphState) -> RAGGraphState:
        """
        Determines whether the retrieved documents are relevant to the question.
        Filters out irrelevant documents. 
        """
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = False
        
        for d in documents:
            try:
                score = self.chain.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
                
                if grade == "yes":
                    logger.info(f"Document relevant: {d.metadata.get('source', 'unknown')}")
                    filtered_docs.append(d)
                else:
                    logger.info(f"Document filtered out: {d.metadata.get('source', 'unknown')}")
                    continue
            except Exception as e:
                logger.warning(f"Grading failed for doc, keeping it safe: {str(e)}")
                filtered_docs.append(d) # Fail open (keep doc) on error
                
        # If no documents are left, we need web search
        if not filtered_docs:
            logger.warning("No relevant documents found. Web search needed.")
            web_search = True
            
        return {
            **state,
            "documents": filtered_docs,
            "web_search_needed": web_search
        }
