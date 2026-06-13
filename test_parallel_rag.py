import asyncio, time
from dotenv import load_dotenv
load_dotenv()

from rag.graph.nodes.retrieval_node import parallel_retrieve_node
from rag.graph.state import RAGGraphState

async def test():
    state: RAGGraphState = {
        'question': 'Latest FastAPI performance benchmarks 2025', 
        'documents': [],
        'generation': None, 
        'web_search_needed': False, 
        'hallucination_count': 0,
        'retry_count': 0, 
        'selected_file_ids': None, 
        'messages': []
    }
    
    start = time.monotonic()
    result = await parallel_retrieve_node(state)
    elapsed = time.monotonic() - start
    
    docs = result.get("documents", [])
    print(f'✅ Parallel retrieval completed in {elapsed:.2f}s with {len(docs)} docs')
    
    if len(docs) > 1:
        contents = [d.page_content[:200] for d in docs]
        assert len(contents) == len(set(contents)), 'Duplicates found!'
        print('✅ No duplicate documents in results')
    
    for doc in docs:
        assert 'source' in doc.metadata, 'Missing source in metadata'
    print('✅ All documents have source metadata')

if __name__ == "__main__":
    asyncio.run(test())
