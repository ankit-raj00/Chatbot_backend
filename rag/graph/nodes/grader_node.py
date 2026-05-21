import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.graph.state import RAGGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraderNode:
    """
    Grades retrieved documents for relevance.

    KEY CHANGE: All k docs are graded in ONE batched LLM call.
    Before: k=5 docs → 5 sequential LLM calls → 429 cascade
    After:  k=5 docs → 1 LLM call → 5x fewer API hits
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-09-2025",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Single prompt grades ALL chunks at once
        self.system_prompt = (
            "You are a relevance grader. Given a user question and a numbered list "
            "of retrieved document chunks, decide which are relevant.\n"
            "Respond with ONLY a comma-separated list of 'yes' or 'no', one per chunk, in order.\n"
            "Example for 3 chunks: yes,no,yes\n"
            "Do NOT include any other text."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Question: {question}\n\nChunks:\n{chunks}"),
        ])

    def grade_documents(self, state: RAGGraphState) -> RAGGraphState:
        """
        Grades all retrieved documents in a SINGLE LLM call.

        API calls per question:
          Before refactor: k grader calls + 1 generate + 1 hallucination = k+2
          After refactor:  1 grader call  + 1 generate + 1 hallucination = 3
          For k=5: 7 → 3 calls  (57% reduction → far fewer 429s)
        """
        question  = state["question"]
        documents = state["documents"]

        if not documents:
            logger.warning("No documents to grade → web search needed.")
            return {**state, "documents": [], "web_search_needed": True}

        try:
            # Pack all chunks into one prompt (cap each at 500 chars to avoid token bloat)
            chunks_text = "\n\n".join(
                f"[{i+1}] {doc.page_content[:500]}"
                for i, doc in enumerate(documents)
            )

            response = (self.prompt | self.llm).invoke({
                "question": question,
                "chunks": chunks_text
            })
            raw    = response.content.strip().lower()
            grades = [g.strip() for g in raw.split(",")]

            filtered_docs = []
            for i, doc in enumerate(documents):
                grade = grades[i] if i < len(grades) else "yes"  # fail-open on parse error
                if grade == "yes":
                    logger.info(f"   ✅ Doc [{i+1}] relevant")
                    filtered_docs.append(doc)
                else:
                    logger.info(f"   ❌ Doc [{i+1}] filtered out")

        except Exception as e:
            logger.warning(f"Batch grading failed ({e}) — keeping all docs (fail-open)")
            filtered_docs = documents  # fail open on any error

        web_search = len(filtered_docs) == 0
        if web_search:
            logger.warning("No relevant docs after grading → web search needed.")

        return {
            **state,
            "documents": filtered_docs,
            "web_search_needed": web_search
        }
