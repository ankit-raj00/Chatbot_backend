import asyncio
from dotenv import load_dotenv

load_dotenv()

from rag.graph.workflow import RAGWorkflow

async def test_full_rag():
    workflow = RAGWorkflow()
    app = workflow.get_app()
    
    inputs = {
        "question": "What is LangGraph?",
        "retry_count": 0,
        "hallucination_count": 0,
        "selected_file_ids": None
    }
    
    print("Running end-to-end RAG workflow...")
    final_state = await app.ainvoke(inputs)
    
    answer = final_state.get("generation", "No answer generated.")
    sources = list(set([doc.metadata.get("source", "unknown") for doc in final_state.get("documents", [])]))
    
    print("✅ RAG Workflow completed successfully.")
    print("-" * 40)
    print("Answer snippet:", str(answer)[:250] + "...")
    print("-" * 40)
    print("Sources:", sources)
    print("Hallucination Warning:", final_state.get("hallucination_count", 0) > 0)

if __name__ == "__main__":
    asyncio.run(test_full_rag())
