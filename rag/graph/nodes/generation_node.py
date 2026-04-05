import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.graph.state import RAGGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationNode:
    """
    The 'Generator' node that synthesizes the final answer.
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # System Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.
            
            Context:
            {context}"""),
            ("human", "{question}"),
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, state: RAGGraphState) -> RAGGraphState:
        """
        Generates answer using retrieved documents.
        """
        logger.info("Generating answer...")
        question = state["question"]
        documents = state["documents"]
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in documents])
        
        try:
            generation = self.chain.invoke({"context": context, "question": question})
            logger.info("Answer generated successfully.")
            
            return {
                **state,
                "generation": generation
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                **state, 
                "generation": "I'm sorry, I encountered an error while generating the answer."
            }
